"""Open Face AI Studio: a local face AI web demo built only on commercially licensed models."""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import engine
from engine import Face, FaceAnalysis, IoUTracker, PersonDetector, ShotCutDetector, appearance

ROOT = Path(__file__).resolve().parent
PHOTOS = ROOT / "photos"
VIDEOS = ROOT / "video"
WORK = ROOT / "work"
THUMBS = WORK / "thumbs"
UPLOADS = WORK / "uploads"
PEOPLE = PHOTOS / "people"
DELETED = WORK / "deleted"
NAME_FILE = "name.txt"
OUTPUTS = WORK / "outputs"
for d in (THUMBS, UPLOADS, OUTPUTS):
    d.mkdir(parents=True, exist_ok=True)

MAX_SIDE = 1600
VERIFY_THRESHOLD = 0.35
IDENTIFY_THRESHOLD = 0.35
CLUSTER_THRESHOLD = 0.30
BODY_THRESHOLD = 0.80
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

app = FastAPI(title="Open Face AI Studio")
infer_lock = threading.Lock()
jobs: dict[str, dict] = {}
job_pool = ThreadPoolExecutor(max_workers=1)

# Models -----------------------------------------------------------------

_fa: FaceAnalysis | None = None
_pd: PersonDetector | None = None


def face_model() -> FaceAnalysis:
    global _fa
    if _fa is None:
        engine.ensure_models()
        _fa = FaceAnalysis()
    return _fa


def person_model() -> PersonDetector:
    global _pd
    if _pd is None:
        _pd = PersonDetector()
    return _pd


# Image helpers ------------------------------------------------------------


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise HTTPException(400, f"Cannot read image {path.name}")
    h, w = img.shape[:2]
    scale = MAX_SIDE / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def safe_media_path(rel: str, base: Path) -> Path:
    p = (ROOT / rel).resolve()
    if not str(p).startswith(str(base.resolve())) or not p.is_file():
        raise HTTPException(404, f"Unknown media {rel}")
    return p


async def resolve_image(file: UploadFile | None, sample: str | None) -> tuple[np.ndarray, str]:
    if file is not None and file.filename:
        dest = UPLOADS / f"{uuid.uuid4().hex}{Path(file.filename).suffix.lower() or '.jpg'}"
        dest.write_bytes(await file.read())
        return read_image(dest), str(dest.relative_to(ROOT))
    if sample:
        p = safe_media_path(sample, ROOT)
        return read_image(p), sample
    raise HTTPException(400, "Provide an image file or a sample path")


def encode_jpg(img: np.ndarray, quality: int = 88) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def aligned_crop(img: np.ndarray, face: Face) -> str:
    return encode_jpg(engine.align(img, face.kps, 112), 90)


def largest(faces: list[Face]) -> Face | None:
    return max(faces, key=lambda f: f.area) if faces else None


def simulate_screen_replay(img: np.ndarray) -> np.ndarray:
    """Approximate a photo of a screen: resolution loss, RGB subpixel stripes, moire, colour cast, glare, recompression."""
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * 0.35), int(h * 0.35)), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    subpixel_period, stripe_strength = 3.0, 0.35
    for c in range(3):
        out[..., c] *= 1 - stripe_strength + stripe_strength * (0.5 + 0.5 * np.cos(2 * np.pi * (xx / subpixel_period - c / 3)))
    moire = 0.5 + 0.5 * np.sin((xx * 0.93 + yy * 0.31) * 0.35) * np.sin((yy * 0.97 - xx * 0.27) * 0.33)
    out *= (0.8 + 0.2 * moire)[..., None]
    out = out * np.array([1.12, 1.0, 0.82], np.float32) + 18
    glare = np.exp(-(((xx - w * 0.7) / (w * 0.35)) ** 2 + ((yy - h * 0.25) / (h * 0.3)) ** 2))
    out = np.clip(out + (glare * 80)[..., None], 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 40])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def tolist(v, nd: int = 2):
    return None if v is None else np.asarray(v).round(nd).tolist()


# Gallery (1:N identification) ------------------------------------------

gallery: dict[str, dict] = {}


def person_label(slug: str) -> str:
    return slug.replace("_", " ").title()


def build_gallery():
    for d in sorted(PEOPLE.iterdir()):
        if d.is_dir():
            try:
                enroll(d.name, person_photos(d))
            except HTTPException as e:
                print(f"Skipping {d.name}: {e.detail}")


def person_photos(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def display_name(d: Path) -> str:
    name_file = d / NAME_FILE
    return name_file.read_text().strip() if name_file.exists() else person_label(d.name)


def enroll(slug: str, paths: list[Path]) -> dict:
    fa = face_model()
    embs, crops, used = [], [], []
    for p in paths:
        try:
            img = read_image(p)
        except HTTPException:
            continue
        with infer_lock:
            f = largest(fa.get(img, "fast"))
        if f is None or f.embedding is None:
            continue
        embs.append(f.normed_embedding)
        crops.append(aligned_crop(img, f))
        used.append(str(p.relative_to(ROOT)))
    if not embs:
        raise HTTPException(400, "No usable face found in the enrollment photos")
    centroid = np.mean(embs, axis=0)
    gallery[slug] = {
        "name": display_name(PEOPLE / slug),
        "embeddings": np.array(embs),
        "centroid": centroid / np.linalg.norm(centroid),
        "crops": crops,
        "photos": used,
    }
    return gallery[slug]


def identify(emb: np.ndarray, only: list[str] | None = None) -> list[dict]:
    scores = []
    for slug, g in gallery.items():
        if only is None or slug in only:
            scores.append({"id": slug, "name": g["name"], "score": float(np.max(g["embeddings"] @ emb))})
    return sorted(scores, key=lambda s: -s["score"])


# Routes: catalogue ------------------------------------------------------


def thumb_for(p: Path) -> str:
    t = THUMBS / (str(p.relative_to(ROOT)).replace("/", "__") + ".jpg")
    if not t.exists():
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        s = 360 / max(h, w)
        cv2.imwrite(str(t), cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA))
    return "/" + str(t.relative_to(ROOT))


def video_poster(p: Path) -> str:
    t = THUMBS / (p.stem + "_poster.jpg")
    if not t.exists():
        cap = cv2.VideoCapture(str(p))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.3))
        ok, fr = cap.read()
        cap.release()
        if ok:
            cv2.imwrite(str(t), cv2.resize(fr, (480, int(fr.shape[0] * 480 / fr.shape[1]))))
    return "/" + str(t.relative_to(ROOT))


@app.get("/api/info")
def info():
    fa = face_model()
    return {
        "provider": fa.provider(),
        "models": [{"role": k, "name": m["name"], "license": m["license"], "source": m["source"]} for k, m in engine.MODELS.items()],
        "thresholds": {"verify": VERIFY_THRESHOLD, "identify": IDENTIFY_THRESHOLD, "cluster": CLUSTER_THRESHOLD},
        "mesh_contours": fa.landmarker.contours,
        "mesh_irises": fa.landmarker.irises,
    }


@app.get("/api/credits")
def credits_info():
    from importlib.metadata import distributions

    packages = []
    for d in distributions():
        m = d.metadata
        if m["Name"] in ("pip", "setuptools"):
            continue
        lic = m.get("License-Expression") or "; ".join(
            c.split(" :: ")[-1] for c in (m.get_all("Classifier") or []) if c.startswith("License ::")) or (m.get("License") or "").split("\n")[0][:80]
        packages.append({"name": m["Name"], "version": d.version, "license": lic or "see package"})
    photos = [{k: c.get(k) for k in ("file", "title", "author", "license", "source")} for c in json.loads((PHOTOS / "CREDITS.json").read_text())]
    return {
        "models": [{"role": k, **{f: m.get(f) for f in ("name", "license", "authors", "copyright", "source", "modified")},
                    "license_url": "/licenses/" + m["license_file"]} for k, m in engine.MODELS.items()],
        "packages": sorted(packages, key=lambda p: p["name"].lower()),
        "photos": photos,
        "videos": [p.name for p in sorted(VIDEOS.glob("*.mp4"))],
    }


@app.get("/notices")
def notices():
    return FileResponse(ROOT / "THIRD_PARTY_NOTICES.md", media_type="text/plain; charset=utf-8")


@app.get("/api/samples")
def samples():
    credits = {c["file"]: c for c in json.loads((PHOTOS / "CREDITS.json").read_text())}
    items = []
    for p in sorted(PHOTOS.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = str(p.relative_to(ROOT))
        group = "people" if "/people/" in rel else "groups"
        c = credits.get(rel, {})
        items.append({
            "path": rel,
            "url": "/" + rel,
            "thumb": thumb_for(p),
            "group": group,
            "person": p.parent.name if group == "people" else None,
            "title": c.get("title", p.name).removeprefix("File:"),
            "source": c.get("source", ""),
            "license": c.get("license", "user upload"),
        })
    vids = [{"path": str(p.relative_to(ROOT)), "url": "/" + str(p.relative_to(ROOT)), "poster": video_poster(p), "name": p.stem}
            for p in sorted(VIDEOS.glob("*.mp4"))]
    return {"photos": items, "videos": vids}


@app.get("/api/gallery")
def get_gallery():
    return [{"id": k, "name": g["name"], "count": len(g["photos"]), "crops": g["crops"][:4], "photos": g["photos"]}
            for k, g in gallery.items()]


@app.post("/api/gallery/enroll")
async def gallery_enroll(name: str = Form(...), files: list[UploadFile] = File(...)):
    """Save the photos to photos/people/<slug>/ (appending to an existing person) so the gallery survives restarts."""
    slug = "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower()).strip("_")
    if not slug:
        raise HTTPException(400, "Name is required")
    d = PEOPLE / slug
    is_new = not d.exists()
    d.mkdir(parents=True, exist_ok=True)
    added = []
    for f in files:
        suffix = Path(f.filename or "").suffix.lower()
        p = d / f"upload_{uuid.uuid4().hex[:12]}{suffix if suffix in IMAGE_EXTS else '.jpg'}"
        p.write_bytes(await f.read())
        added.append(p)
    try:
        g = enroll(slug, person_photos(d))
    except HTTPException:
        g = None
    used = set(g["photos"]) if g else set()
    unusable = [p for p in added if str(p.relative_to(ROOT)) not in used]
    for p in unusable:
        p.unlink()
    if g is None or len(unusable) == len(added):
        if is_new:
            shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "No usable face found in the enrollment photos")
    if is_new or name.strip() != g["name"]:
        (d / NAME_FILE).write_text(name.strip())
        g["name"] = name.strip()
    return {"id": slug, "name": g["name"], "count": len(g["photos"]), "added": len(added) - len(unusable),
            "skipped": len(unusable)}


@app.delete("/api/gallery/{slug}")
def gallery_delete(slug: str):
    """Remove a person from the gallery; their folder is moved to work/deleted/ so it can be restored."""
    if slug not in gallery:
        raise HTTPException(404, "Unknown person")
    gallery.pop(slug)
    d = PEOPLE / slug
    if d.is_dir():
        DELETED.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d), str(DELETED / f"{slug}_{int(time.time())}"))
    return {"ok": True}


# Routes: image analysis -------------------------------------------------


def top_expressions(bs: dict | None, n: int = 3) -> list | None:
    if not bs:
        return None
    return [[k, round(v, 2)] for k, v in sorted(bs.items(), key=lambda kv: -kv[1]) if k != "_neutral"][:n]


@app.post("/api/analyze")
async def analyze(file: UploadFile | None = File(None), sample: str | None = Form(None), model: str = Form("full"),
                  spoof: bool = Form(False)):
    if model not in FaceAnalysis.PACKS:
        raise HTTPException(400, "Unsupported model")
    img, src = await resolve_image(file, sample)
    if spoof:
        img = simulate_screen_replay(img)
    fa = face_model()
    t0 = time.perf_counter()
    with infer_lock:
        faces = fa.get(img, model)
    elapsed = time.perf_counter() - t0
    out = []
    for i, f in enumerate(sorted(faces, key=lambda f: f.bbox[0])):
        out.append({
            "index": i,
            "bbox": tolist(f.bbox),
            "det_score": f.det_score,
            "kps": tolist(f.kps),
            "mesh": tolist(f.mesh[:, :2], 1) if f.mesh is not None else None,
            "pose": tolist(f.pose),
            "age": f.age,
            "age_score": f.age_score,
            "sex": f.sex,
            "sex_score": f.sex_score,
            "expressions": top_expressions(f.blendshapes),
            "liveness": f.liveness,
            "embedding_dim": None if f.embedding is None else int(f.embedding.shape[0]),
            "embedding_norm": None if f.embedding is None else round(f.embedding_norm, 2),
            "embedding_preview": None if f.embedding is None else f.normed_embedding[:48].round(3).tolist(),
            "crop": aligned_crop(img, f),
            "matches": identify(f.normed_embedding)[:3] if f.embedding is not None else [],
        })
    h, w = img.shape[:2]
    return {"source": src, "image": encode_jpg(img), "width": w, "height": h, "faces": out,
            "elapsed_ms": round(elapsed * 1000, 1), "model": model, "provider": fa.provider()}


@app.post("/api/compare")
async def compare(file_a: UploadFile | None = File(None), sample_a: str | None = Form(None),
                  file_b: UploadFile | None = File(None), sample_b: str | None = Form(None)):
    img_a, src_a = await resolve_image(file_a, sample_a)
    img_b, src_b = await resolve_image(file_b, sample_b)
    fa = face_model()
    t0 = time.perf_counter()
    with infer_lock:
        fa_a, fa_b = largest(fa.get(img_a, "fast")), largest(fa.get(img_b, "fast"))
    if fa_a is None or fa_b is None:
        raise HTTPException(422, "No face found in " + ("image A" if fa_a is None else "image B"))
    sim = float(np.dot(fa_a.normed_embedding, fa_b.normed_embedding))
    return {
        "similarity": sim,
        "threshold": VERIFY_THRESHOLD,
        "same_person": sim >= VERIFY_THRESHOLD,
        "a": {"source": src_a, "crop": aligned_crop(img_a, fa_a), "image": encode_jpg(img_a, 80), "bbox": tolist(fa_a.bbox), "width": img_a.shape[1], "height": img_a.shape[0]},
        "b": {"source": src_b, "crop": aligned_crop(img_b, fa_b), "image": encode_jpg(img_b, 80), "bbox": tolist(fa_b.bbox), "width": img_b.shape[1], "height": img_b.shape[0]},
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


_cluster_cache: dict | None = None


@app.post("/api/cluster")
def cluster(threshold: float = Form(CLUSTER_THRESHOLD)):
    global _cluster_cache
    from sklearn.cluster import AgglomerativeClustering

    fa = face_model()
    t0 = time.perf_counter()
    if _cluster_cache is None:
        rows = []
        for p in sorted(PHOTOS.rglob("*")):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            img = read_image(p)
            with infer_lock:
                faces = fa.get(img, "fast", min_score=0.8)
            for f in faces:
                rows.append({"photo": str(p.relative_to(ROOT)), "crop": aligned_crop(img, f), "emb": f.normed_embedding})
        _cluster_cache = {"rows": rows, "embed_ms": round((time.perf_counter() - t0) * 1000)}
    rows = _cluster_cache["rows"]
    if len(rows) < 2:
        raise HTTPException(422, "Need at least two faces to cluster")
    X = np.array([r["emb"] for r in rows])
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=1 - threshold).fit_predict(X)
    clusters: dict[int, list] = {}
    for r, lab in zip(rows, labels):
        clusters.setdefault(int(lab), []).append(r)
    result = []
    for lab, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        c = np.mean([m["emb"] for m in members], axis=0)
        guess = identify(c / np.linalg.norm(c))
        name = guess[0]["name"] if guess and guess[0]["score"] >= 0.4 and len(members) > 1 else None
        result.append({"size": len(members), "photos": len({m["photo"] for m in members}), "guess": name,
                       "faces": [{"crop": m["crop"], "photo": m["photo"]} for m in members]})
    return {"faces": len(rows), "clusters": result, "embed_ms": _cluster_cache["embed_ms"],
            "cluster_ms": round((time.perf_counter() - t0) * 1000)}


# Routes: video jobs -----------------------------------------------------


def new_job(kind: str) -> dict:
    jid = uuid.uuid4().hex[:10]
    jobs[jid] = {"id": jid, "kind": kind, "status": "queued", "progress": 0.0, "stage": "queued", "result": None, "error": None, "started": time.time()}
    return jobs[jid]


def run_job(job: dict, fn):
    def wrapper():
        job["status"] = "running"
        try:
            job["result"] = fn(job)
            job["status"] = "done"
            job["progress"] = 1.0
        except Exception as e:  # surfaced to the UI
            job["status"] = "error"
            job["error"] = f"{type(e).__name__}: {e}"
        job["elapsed_s"] = round(time.time() - job["started"], 1)
    job_pool.submit(wrapper)


async def resolve_video(file: UploadFile | None, sample: str | None) -> Path:
    if file is not None and file.filename:
        dest = UPLOADS / f"{uuid.uuid4().hex}{Path(file.filename).suffix.lower() or '.mp4'}"
        dest.write_bytes(await file.read())
        return dest
    if sample:
        return safe_media_path(sample, ROOT)
    raise HTTPException(400, "Provide a video file or sample path")


def reference_photos(people: list[str]) -> list[Path]:
    paths = []
    for slug in people:
        g = gallery.get(slug)
        if g:
            paths += [ROOT / p for p in g["photos"]]
    return paths


def to_browser_mp4(src: Path, dest: Path, audio_from: Path | None = None):
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src)]
    if audio_from:
        cmd += ["-i", str(audio_from), "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-shortest"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "22", "-movflags", "+faststart", str(dest)]
    subprocess.run(cmd, check=True)


def open_video(src: Path) -> tuple[cv2.VideoCapture, dict]:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {src.name}")
    meta = {"fps": cap.get(cv2.CAP_PROP_FPS) or 30.0, "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1,
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}
    return cap, meta


def redact(frame: np.ndarray, box, method: str, pad: float = 0.2):
    """Blur or pixelate an elliptical region around a face box, in place."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = max(0, int(x1 - bw * pad)), max(0, int(y1 - bh * pad * 1.3))
    x2, y2 = min(w, int(x2 + bw * pad)), min(h, int(y2 + bh * pad))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return
    roi = frame[y1:y2, x1:x2]
    if method == "mosaic":
        block = max(2, min(roi.shape[:2]) // 10)
        small = cv2.resize(roi, (max(1, roi.shape[1] // block), max(1, roi.shape[0] // block)), interpolation=cv2.INTER_LINEAR)
        cover = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        k = max(3, (min(roi.shape[:2]) // 3) | 1)
        cover = cv2.GaussianBlur(roi, (k, k), 0)
        cover = cv2.GaussianBlur(cover, (k, k), 0)
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.ellipse(mask, (roi.shape[1] // 2, roi.shape[0] // 2), (roi.shape[1] // 2, roi.shape[0] // 2), 0, 0, 360, 255, -1)
    if method != "mosaic":
        mask = cv2.GaussianBlur(mask, (0, 0), max(1, min(roi.shape[:2]) / 20))
    a = (mask.astype(np.float32) / 255)[..., None]
    frame[y1:y2, x1:x2] = (cover * a + roi * (1 - a)).astype(np.uint8)


def fill_track_gaps(obs_by_track: dict[int, list[dict]], frame_count: int, cuts: list[int], hold: int = 4) -> dict[int, dict]:
    """Per frame face boxes for every track: interpolate across missed detections, hold a few frames at the ends
    (never across a shot cut)."""
    import bisect

    boxes: dict[int, dict] = {}
    for tid, obs in obs_by_track.items():
        obs = sorted(obs, key=lambda o: o["frame"])
        for a, b in zip(obs, obs[1:]):
            gap = b["frame"] - a["frame"]
            for k in range(gap):
                t = k / gap
                boxes.setdefault(a["frame"] + k, {})[tid] = [ai + (bi - ai) * t for ai, bi in zip(a["bbox"], b["bbox"])]
        first, last = obs[0], obs[-1]
        k = bisect.bisect_right(cuts, first["frame"])
        shot_start = cuts[k - 1] if k else 0
        k = bisect.bisect_right(cuts, last["frame"])
        shot_end = cuts[k] if k < len(cuts) else frame_count
        for fi in range(max(shot_start, first["frame"] - hold), first["frame"]):
            boxes.setdefault(fi, {})[tid] = first["bbox"]
        for fi in range(last["frame"], min(shot_end, last["frame"] + hold + 1)):
            boxes.setdefault(fi, {})[tid] = last["bbox"]
    return boxes


@app.post("/api/video/privateframe")
async def privateframe(file: UploadFile | None = File(None), sample: str | None = Form(None),
                       mode: str = Form("all"), method: str = Form("gaussian"), people: str = Form("")):
    if mode not in ("all", "blur_only", "exempt"):
        raise HTTPException(400, "mode must be all, blur_only or exempt")
    if method not in ("gaussian", "mosaic"):
        raise HTTPException(400, "method must be gaussian or mosaic")
    src = await resolve_video(file, sample)
    selected = [p for p in people.split(",") if p]
    if mode != "all" and not selected:
        raise HTTPException(400, "Pick at least one reference person for this mode")
    job = new_job("privateframe")

    def work(job):
        fa = face_model()
        out_dir = OUTPUTS / job["id"]
        out_dir.mkdir(parents=True, exist_ok=True)

        # Pass 1: detect + track every frame, embed faces every few frames per track.
        job["stage"] = "detecting and tracking faces"
        cap, meta = open_video(src)
        tracker = IoUTracker(min_iou=0.2, max_age=int(meta["fps"] // 2))
        shots = ShotCutDetector()
        embed_every = max(1, round(meta["fps"] / 5))
        obs, track_scores, track_emb, cuts = [], {}, {}, []
        idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if shots(frame):
                tracker.reset()
                cuts.append(idx)
            with infer_lock:
                faces = [f for f in fa.detector.detect(frame, sides=(None, 640)) if f.det_score >= 0.6]
                ids = tracker.update([f.bbox.tolist() for f in faces], idx)
                if mode != "all":
                    todo = [f for f, tid in zip(faces, ids) if (idx + tid) % embed_every == 0 and f.bbox[2] - f.bbox[0] >= 24]
                    fa.recognizer.embed(frame, todo)
            for i, f in enumerate(faces):
                if f.embedding is None:
                    continue
                e, tid = f.normed_embedding, ids[i]
                if tid in track_emb and float(e @ track_emb[tid]) / np.linalg.norm(track_emb[tid]) < 0.2:
                    ids[i] = tid = tracker.split(tid)  # same box, different person
                track_emb[tid] = track_emb.get(tid, 0) + e
            for f, tid in zip(faces, ids):
                obs.append({"frame": idx, "track_id": tid, "bbox": tolist(f.bbox, 1), "score": round(f.det_score, 3)})
                if f.embedding is not None:
                    m = identify(f.normed_embedding, selected)
                    if m:
                        track_scores.setdefault(tid, []).append((m[0]["id"], m[0]["score"]))
            job["progress"] = idx / meta["frame_count"] * 0.6
        cap.release()
        meta["frame_count"] = idx + 1

        # Track-level decision: the best person must clear the threshold on the mean of its top-3 frames.
        tracks = {}
        for tid in {o["track_id"] for o in obs}:
            per: dict[str, list[float]] = {}
            for pid, s in track_scores.get(tid, []):
                per.setdefault(pid, []).append(s)
            best = max(((pid, float(np.mean(sorted(v)[-3:]))) for pid, v in per.items()), key=lambda x: x[1], default=(None, 0.0))
            confirmed = best[1] >= IDENTIFY_THRESHOLD
            tracks[str(tid)] = {"status": "CONFIRMED" if confirmed else "UNMATCHED", "person": best[0] if confirmed else None,
                                "score": round(best[1], 3), "observations": sum(o["track_id"] == tid for o in obs)}

        def redacted(tid) -> bool:
            if mode == "all":
                return True
            return (tracks[str(tid)]["status"] == "CONFIRMED") == (mode == "blur_only")

        # Pass 2: render.
        job["stage"] = "rendering"
        by_track: dict[int, list[dict]] = {}
        for o in obs:
            by_track.setdefault(o["track_id"], []).append(o)
        per_frame = fill_track_gaps(by_track, meta["frame_count"], cuts)
        cap, _ = open_video(src)
        raw = out_dir / "raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), meta["fps"], (meta["width"], meta["height"]))
        idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            for tid, box in per_frame.get(idx, {}).items():
                if redacted(tid):
                    redact(frame, box, method)
            writer.write(frame)
            job["progress"] = 0.6 + idx / meta["frame_count"] * 0.35
        cap.release()
        writer.release()
        job["stage"] = "encoding"
        result_video = out_dir / "privateframe.mp4"
        to_browser_mp4(raw, result_video, audio_from=src)
        raw.unlink(missing_ok=True)

        data = {"source_video": {"path": str(src.relative_to(ROOT)), "metadata": meta},
                "config": {"mode": mode, "method": method, "people": selected, "identify_threshold": IDENTIFY_THRESHOLD,
                           "models": {k: engine.MODELS[k]["name"] for k in ("detector", "recognition")}},
                "recognition": {"tracks": tracks}, "observations": obs}
        result_json = out_dir / "privateframe.json"
        result_json.write_text(json.dumps(data, indent=1))
        n_redacted = sum(redacted(o["track_id"]) for o in obs)
        summary = {"face_tracks": len(tracks), "face_regions": len(obs), "redacted_face_regions": n_redacted,
                   "kept_face_regions": len(obs) - n_redacted,
                   "matched_tracks": sum(t["status"] == "CONFIRMED" for t in tracks.values()), "frame_count": meta["frame_count"]}
        return {"video": "/" + str(result_video.relative_to(ROOT)), "json": "/" + str(result_json.relative_to(ROOT)),
                "source": "/" + str(src.relative_to(ROOT)), "summary": summary,
                "mode": mode, "method": method, "people": [person_label(p) for p in selected]}

    run_job(job, work)
    return job


@app.post("/api/video/person")
async def person_video(file: UploadFile | None = File(None), sample: str | None = Form(None),
                       people: str = Form(""), analysis_fps: float = Form(10.0)):
    src = await resolve_video(file, sample)
    selected = [p for p in people.split(",") if p] or list(gallery)
    job = new_job("person")

    def work(job):
        fa, pd = face_model(), person_model()
        out_dir = OUTPUTS / job["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        cap, meta = open_video(src)
        fps, total, w, h = meta["fps"], meta["frame_count"], meta["width"], meta["height"]
        step = max(1, round(fps / analysis_fps))
        raw = out_dir / "raw.mp4"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        registrations = [{"id": s, "name": gallery[s]["name"], "accepted": len(gallery[s]["photos"]), "rejected": []}
                         for s in selected if s in gallery]
        body_bank: dict[str, list[np.ndarray]] = {s: [] for s in selected}
        track_identity: dict[int, str] = {}
        tracker = IoUTracker(min_iou=0.3, max_age=step * 8)
        shots = ShotCutDetector()
        timeline: dict[str, list[float]] = {}
        stats = {"face": 0, "body": 0, "track": 0, "unmatched": 0}
        job["stage"] = "analyzing"
        idx, last, infer_s, analyzed = -1, [], 0.0, 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % step == 0:
                if shots(frame):
                    tracker.reset()
                t0 = time.perf_counter()
                with infer_lock:
                    last = analyze_people(frame, fa, pd, selected, tracker, idx, body_bank, track_identity)
                infer_s += time.perf_counter() - t0
                analyzed += 1
                for m in last:
                    stats[m["matched_by"] or "unmatched"] += 1
                    if m["person_id"]:
                        timeline.setdefault(m["person_id"], []).append(round(idx / fps, 2))
            writer.write(draw_person(frame, last, idx / fps))
            job["progress"] = idx / total * 0.95
        cap.release()
        writer.release()
        job["stage"] = "encoding"
        out = out_dir / "person_analysis.mp4"
        to_browser_mp4(raw, out, audio_from=src)
        raw.unlink(missing_ok=True)
        duration = (idx + 1) / fps
        appearances = []
        for pid, ts in timeline.items():
            appearances.append({"id": pid, "name": gallery[pid]["name"] if pid in gallery else person_label(pid),
                                "first": ts[0], "last": ts[-1], "seconds": round(len(set(ts)) * step / fps, 1), "times": sorted(set(ts))})
        return {"video": "/" + str(out.relative_to(ROOT)), "source": "/" + str(src.relative_to(ROOT)),
                "duration": round(duration, 2), "registrations": registrations,
                "appearances": sorted(appearances, key=lambda a: a["first"]), "stats": stats,
                "analyzed_frames": analyzed, "avg_ms_per_frame": round(infer_s / max(1, analyzed) * 1000, 1)}

    run_job(job, work)
    return job


def analyze_people(frame, fa: FaceAnalysis, pd: PersonDetector, selected: list[str], tracker: IoUTracker, idx: int,
                   body_bank: dict[str, list[np.ndarray]], track_identity: dict[int, str]) -> list[dict]:
    """One analysed frame: bodies (YOLOX) + faces (YuNet/AuraFace), matched by face, then body appearance, then track continuity."""
    bodies = pd.detect(frame)
    faces = [f for f in fa.detector.detect(frame, sides=(None, 640)) if f.det_score >= 0.6]
    fa.recognizer.embed(frame, faces)

    # Attach each face to the smallest body whose upper half contains the face centre.
    obs = [{"body": b, "body_score": s, "face": None} for b, s in bodies]
    for f in faces:
        cx, cy = (f.bbox[0] + f.bbox[2]) / 2, (f.bbox[1] + f.bbox[3]) / 2
        cands = [o for o in obs if o["face"] is None and o["body"][0] <= cx <= o["body"][2]
                 and o["body"][1] <= cy <= o["body"][1] + (o["body"][3] - o["body"][1]) * 0.5]
        if cands:
            min(cands, key=lambda o: (o["body"][2] - o["body"][0]) * (o["body"][3] - o["body"][1]))["face"] = f
        else:
            obs.append({"body": None, "body_score": None, "face": f})

    ids = tracker.update([(o["body"] if o["body"] is not None else o["face"].bbox).tolist() for o in obs], idx)
    for o, tid in zip(obs, ids):
        o.update(track_id=tid, person_id=None, similarity=0.0, matched_by=None,
                 app=appearance(frame, o["body"]) if o["body"] is not None else None)
        if o["face"] is not None:
            m = identify(o["face"].normed_embedding, selected)
            if m and m[0]["score"] >= IDENTIFY_THRESHOLD:
                o.update(person_id=m[0]["id"], similarity=m[0]["score"], matched_by="face")

    # One person per frame: keep the strongest face match.
    seen = set()
    for o in sorted(obs, key=lambda o: -o["similarity"]):
        if o["person_id"] in seen:
            o.update(person_id=None, similarity=0.0, matched_by=None)
        elif o["person_id"]:
            seen.add(o["person_id"])

    for o in obs:
        if o["matched_by"] == "face":
            track_identity[o["track_id"]] = o["person_id"]
            if o["app"] is not None:
                body_bank[o["person_id"]] = (body_bank[o["person_id"]] + [o["app"]])[-40:]
            continue
        if o["app"] is not None:
            scores = sorted(((float(np.max(np.array(bank) @ o["app"])), pid) for pid, bank in body_bank.items()
                             if bank and pid not in seen), reverse=True)
            if scores and scores[0][0] >= BODY_THRESHOLD and (len(scores) == 1 or scores[0][0] - scores[1][0] >= 0.05):
                o.update(person_id=scores[0][1], similarity=scores[0][0], matched_by="body")
                seen.add(scores[0][1])
                continue
        pid = track_identity.get(o["track_id"])
        if pid and pid not in seen:
            o.update(person_id=pid, similarity=0.0, matched_by="track")
            seen.add(pid)
    return obs


PALETTE = [(255, 140, 66), (90, 200, 120), (230, 90, 200), (80, 190, 250), (60, 220, 230), (200, 120, 255)]


def color_for(pid: str | None):
    if pid is None:
        return (0, 165, 255)
    ids = list(gallery)
    return PALETTE[(ids.index(pid) if pid in ids else sum(map(ord, str(pid)))) % len(PALETTE)]


def draw_person(frame, matches, t):
    out = frame.copy()
    for m in matches:
        color = color_for(m["person_id"])
        label = (gallery[m["person_id"]]["name"] if m["person_id"] in gallery else person_label(m["person_id"])) if m["person_id"] else "unknown"
        if m["person_id"]:
            label += f" {m['similarity']:.2f} ({m['matched_by']})" if m["matched_by"] != "track" else " (track)"
        if m["body"] is not None:
            x1, y1, x2, y2 = map(int, m["body"])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        if m["face"] is not None:
            fx1, fy1, fx2, fy2 = map(int, m["face"].bbox)
            cv2.rectangle(out, (fx1, fy1), (fx2, fy2), color, 1)
        box = m["body"] if m["body"] is not None else m["face"].bbox
        x1, y1 = int(box[0]), int(box[1])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y = max(th + 6, y1)
        cv2.rectangle(out, (x1, y - th - 6), (x1 + tw + 8, y), color, -1)
        cv2.putText(out, label, (x1 + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(out, f"{t:5.1f}s  YOLOX + YuNet + AuraFace", (10, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


@app.get("/api/jobs/{jid}")
def job_status(jid: str):
    if jid not in jobs:
        raise HTTPException(404, "Unknown job")
    return jobs[jid]


# Static -----------------------------------------------------------------


@app.on_event("startup")
def startup():
    t0 = time.perf_counter()
    face_model()
    build_gallery()
    person_model()
    print(f"Models ready and gallery enrolled in {time.perf_counter() - t0:.1f}s ({len(gallery)} people)")


app.mount("/photos", StaticFiles(directory=PHOTOS), name="photos")
app.mount("/video", StaticFiles(directory=VIDEOS), name="video")
app.mount("/work", StaticFiles(directory=WORK), name="work")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
app.mount("/licenses", StaticFiles(directory=ROOT / "licenses"), name="licenses")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")
