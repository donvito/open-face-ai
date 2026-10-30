"""Commercially licensed face/person models behind a small InsightFace-like API.

Every model here is under a license that permits commercial use (MIT, Apache 2.0, CC BY 4.0).
See MODELS for sources and licenses; nothing from InsightFace (code or weights) is used.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import requests

MODEL_DIR = Path(os.environ.get("OPEN_FACE_MODELS", Path(__file__).resolve().parent / "models"))

MODELS = {
    "detector": {
        "name": "YuNet (2023mar)", "file": "face_detection_yunet_2023mar.onnx", "license": "MIT",
        "url": "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
        "source": "https://huggingface.co/opencv/face_detection_yunet",
        "authors": "Shiqi Yu et al. (libfacedetection); ONNX via OpenCV Zoo", "copyright": "Copyright (c) 2020 Shiqi Yu",
        "license_file": "YuNet-MIT.txt",
    },
    "recognition": {
        "name": "AuraFace v1 (ResNet100, 512-d)", "file": "auraface_v1_glintr100.onnx", "license": "Apache-2.0",
        "url": "https://huggingface.co/fal/AuraFace-v1/resolve/main/glintr100.onnx",
        "source": "https://huggingface.co/fal/AuraFace-v1",
        "authors": "fal (fal.ai)", "copyright": None, "license_file": "AuraFace-Apache-2.0.txt",
    },
    "landmarks": {
        "name": "MediaPipe Face Landmarker v2 (478 pts + blendshapes)", "file": "face_landmarker.task", "license": "Apache-2.0",
        "url": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        "source": "https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker",
        "authors": "Google LLC (The MediaPipe Authors)", "copyright": "Copyright The MediaPipe Authors (Google LLC)", "license_file": "MediaPipe-Apache-2.0.txt",
    },
    "attributes": {
        "name": "FairFace ResNet34 (age group, gender)", "file": "fairface.onnx", "license": "CC-BY-4.0",
        "url": "https://github.com/yakhyo/fairface-onnx/releases/download/weights/fairface.onnx",
        "source": "https://github.com/joojs/fairface",
        "authors": "Kimmo Kärkkäinen and Jungseock Joo (FairFace); ONNX export by Yakhyokhuja Valikhujaev",
        "copyright": None, "license_file": "FairFace-CC-BY-4.0.txt",
        "modified": "Converted from PyTorch to ONNX (yakhyo/fairface-onnx). This app uses only the gender and age outputs; the race output is discarded.",
    },
    "liveness_a": {
        "name": "MiniFASNetV2 @2.7", "file": "minifasnet_v2_2.7.onnx", "license": "Apache-2.0",
        "url": "https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/raw/main/onnx/2.7_80x80_MiniFASNetV2.onnx",
        "source": "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing",
        "authors": "Minivision AI; ONNX export by QingHeYang", "copyright": "Copyright 2020 Minivision",
        "license_file": "Silent-Face-Anti-Spoofing-Apache-2.0.txt", "modified": "Converted from PyTorch to ONNX (QingHeYang/Silent-Face-Anti-Spoofing-onnx).",
    },
    "liveness_b": {
        "name": "MiniFASNetV1SE @4.0", "file": "minifasnet_v1se_4.0.onnx", "license": "Apache-2.0",
        "url": "https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx/raw/main/onnx/4_0_0_80x80_MiniFASNetV1SE.onnx",
        "source": "https://github.com/minivision-ai/Silent-Face-Anti-Spoofing",
        "authors": "Minivision AI; ONNX export by QingHeYang", "copyright": "Copyright 2020 Minivision",
        "license_file": "Silent-Face-Anti-Spoofing-Apache-2.0.txt", "modified": "Converted from PyTorch to ONNX (QingHeYang/Silent-Face-Anti-Spoofing-onnx).",
    },
    "person": {
        "name": "YOLOX-S (COCO person class)", "file": "object_detection_yolox_2022nov.onnx", "license": "Apache-2.0",
        "url": "https://huggingface.co/opencv/object_detection_yolox/resolve/main/object_detection_yolox_2022nov.onnx",
        "source": "https://huggingface.co/opencv/object_detection_yolox",
        "authors": "Megvii Inc. (YOLOX); ONNX via OpenCV Zoo", "copyright": "Copyright (c) 2021-2022 Megvii Inc.",
        "license_file": "YOLOX-Apache-2.0.txt",
    },
}


def model_path(key: str) -> Path:
    return MODEL_DIR / MODELS[key]["file"]


def ensure_models():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for key, m in MODELS.items():
        dest = model_path(key)
        if dest.exists():
            continue
        print(f"Downloading {m['name']} ({m['license']}) -> {dest}")
        tmp = dest.with_suffix(dest.suffix + ".part")
        with requests.get(m["url"], stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tmp.rename(dest)


def providers() -> list[str]:
    wanted = os.environ.get("ORT_PROVIDERS")
    if wanted:
        return wanted.split(",")
    avail = ort.get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider") if p in avail]


def session(key: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(str(model_path(key)), opts, providers=providers())


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# Alignment ----------------------------------------------------------------

# Standard 5-point face template for 112x112 crops (eyes, nose tip, mouth corners; image-left first).
TEMPLATE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                         [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def align(img: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    """Similarity-warp the face so its 5 keypoints land on the template."""
    M, _ = cv2.estimateAffinePartial2D(np.asarray(kps, np.float32), TEMPLATE_112 * (size / 112.0), method=cv2.LMEDS)
    return cv2.warpAffine(img, M, (size, size), borderValue=0)


# Face object ----------------------------------------------------------------


@dataclass
class Face:
    bbox: np.ndarray  # x1, y1, x2, y2
    kps: np.ndarray  # 5x2
    det_score: float
    embedding: np.ndarray | None = None
    mesh: np.ndarray | None = None  # 478x3 (x, y in image px, z relative)
    pose: tuple[float, float, float] | None = None  # pitch, yaw, roll (degrees)
    blendshapes: dict[str, float] | None = None
    age: str | None = None
    age_score: float | None = None
    sex: str | None = None
    sex_score: float | None = None
    liveness: dict | None = None
    extra: dict = field(default_factory=dict)

    @property
    def normed_embedding(self) -> np.ndarray | None:
        return None if self.embedding is None else self.embedding / np.linalg.norm(self.embedding)

    @property
    def embedding_norm(self) -> float | None:
        return None if self.embedding is None else float(np.linalg.norm(self.embedding))

    @property
    def area(self) -> float:
        return float((self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1]))


# Models -------------------------------------------------------------------


class Detector:
    """YuNet, run at several scales because it is trained for faces of roughly 10-300 px."""

    def __init__(self, score=0.6, nms=0.3):
        self.score, self.nms = score, nms
        self.net = cv2.FaceDetectorYN.create(str(model_path("detector")), "", (320, 320), score, nms, 5000)

    def detect(self, img: np.ndarray, sides=(None, 640, 320)) -> list[Face]:
        h, w = img.shape[:2]
        found = []
        for side in sides:
            s = 1.0 if side is None else min(1.0, side / max(h, w))
            if side is not None and s >= 1.0:
                continue
            im = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA) if s < 1 else img
            self.net.setInputSize((im.shape[1], im.shape[0]))
            _, f = self.net.detect(im)
            if f is not None:
                f = f.copy()
                f[:, :14] /= s
                found.append(f)
        if not found:
            return []
        a = np.concatenate(found)
        keep = np.array(cv2.dnn.NMSBoxes(a[:, :4].tolist(), a[:, 14].tolist(), self.score, self.nms)).flatten()
        faces = []
        for r in a[keep]:
            x, y, bw, bh = r[:4]
            faces.append(Face(bbox=np.array([x, y, x + bw, y + bh], np.float32), kps=r[4:14].reshape(5, 2), det_score=float(r[14])))
        return faces


class Recognizer:
    """AuraFace v1: ArcFace-style ResNet100, 112x112 aligned RGB input, 512-d embedding."""

    def __init__(self):
        self.sess = session("recognition")
        self.input = self.sess.get_inputs()[0].name

    def embed(self, img: np.ndarray, faces: list[Face], batch: int = 1):  # CoreML recompiles on every new batch shape
        for i in range(0, len(faces), batch):
            chunk = faces[i:i + batch]
            x = np.stack([align(img, f.kps)[:, :, ::-1] for f in chunk]).astype(np.float32)
            x = ((x - 127.5) / 127.5).transpose(0, 3, 1, 2)
            for f, e in zip(chunk, self.sess.run(None, {self.input: x})[0]):
                f.embedding = e


class Landmarker:
    """MediaPipe Face Landmarker on a crop around each detected face: 478 3D points, head pose, 52 blendshapes."""

    def __init__(self):
        from mediapipe.tasks.python import BaseOptions, vision

        self.vision = vision
        self.lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path("landmarks"))),
            running_mode=vision.RunningMode.IMAGE, num_faces=1, min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3, output_face_blendshapes=True, output_facial_transformation_matrixes=True))
        C = vision.FaceLandmarksConnections
        self.contours = [[c.start, c.end] for c in C.FACE_LANDMARKS_CONTOURS]
        self.irises = [[c.start, c.end] for c in C.FACE_LANDMARKS_LEFT_IRIS + C.FACE_LANDMARKS_RIGHT_IRIS]

    def run(self, img: np.ndarray, face: Face, scale: float = 1.8):
        import mediapipe as mp

        h, w = img.shape[:2]
        x1, y1, x2, y2 = face.bbox
        cx, cy, side = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * scale
        sx, sy = int(round(cx - side / 2)), int(round(cy - side / 2))
        size = int(round(side))
        crop = np.zeros((size, size, 3), np.uint8)  # pad so faces near the border stay centred
        ox1, oy1, ox2, oy2 = max(sx, 0), max(sy, 0), min(sx + size, w), min(sy + size, h)
        if ox2 <= ox1 or oy2 <= oy1:
            return
        crop[oy1 - sy:oy2 - sy, ox1 - sx:ox2 - sx] = img[oy1:oy2, ox1:ox2]
        res = self.lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(crop[:, :, ::-1])))
        if not res.face_landmarks:
            return
        pts = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], np.float32) * size
        pts[:, 0] += sx
        pts[:, 1] += sy
        face.mesh = pts
        if res.facial_transformation_matrixes:
            R = np.asarray(res.facial_transformation_matrixes[0])[:3, :3]
            pitch = math.degrees(math.atan2(R[2, 1], R[2, 2]))
            yaw = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
            roll = math.degrees(math.atan2(R[1, 0], R[0, 0]))
            face.pose = (pitch, yaw, roll)
        if res.face_blendshapes:
            face.blendshapes = {b.category_name: float(b.score) for b in res.face_blendshapes[0]}


class Attributes:
    """FairFace ResNet34: gender and one of 9 age groups (race head is ignored on purpose)."""

    AGES = ["0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+"]
    MEAN = np.array([0.485, 0.456, 0.406], np.float32)
    STD = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self):
        self.sess = session("attributes")
        self.input = self.sess.get_inputs()[0].name

    def run(self, img: np.ndarray, faces: list[Face]):
        for f in faces:  # one at a time: fixed input shape keeps CoreML from recompiling
            x = align(img, f.kps, 224)[:, :, ::-1].astype(np.float32) / 255.0
            x = ((x - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
            _, gender, age = self.sess.run(None, {self.input: x})
            g, a = softmax(gender)[0], softmax(age)[0]
            f.sex, f.sex_score = ("M", float(g[0])) if g[0] >= g[1] else ("F", float(g[1]))
            f.age, f.age_score = self.AGES[int(a.argmax())], float(a.max())


class Liveness:
    """Silent-Face MiniFASNet ensemble (two crops at 2.7x and 4.0x the face box, 80x80 BGR, raw 0-255).

    Classes: 0 = print attack, 1 = real, 2 = screen replay. live_score is the averaged 'real' probability.
    """

    MIN_FACE = 48

    def __init__(self):
        self.nets = [(session("liveness_a"), 2.7), (session("liveness_b"), 4.0)]

    @staticmethod
    def _crop(img, bbox, scale):
        src_h, src_w = img.shape[:2]
        x, y, bw, bh = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
        scale = min((src_h - 1) / bh, (src_w - 1) / bw, scale)
        nw, nh, cx, cy = bw * scale, bh * scale, x + bw / 2, y + bh / 2
        l, t, r, b = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
        if l < 0:
            r, l = r - l, 0
        if t < 0:
            b, t = b - t, 0
        if r > src_w - 1:
            l, r = l - (r - src_w + 1), src_w - 1
        if b > src_h - 1:
            t, b = t - (b - src_h + 1), src_h - 1
        crop = img[int(t):int(b) + 1, int(l):int(r) + 1]
        return cv2.resize(crop, (80, 80)).astype(np.float32).transpose(2, 0, 1)

    def run(self, img: np.ndarray, faces: list[Face]):
        for f in faces:
            w = f.bbox[2] - f.bbox[0]
            if w < self.MIN_FACE:
                f.liveness = {"status": "rejected", "is_live": None, "live_score": None,
                              "reason": f"face too small ({int(w)} px < {self.MIN_FACE} px)"}
                continue
            probs = np.zeros(3)
            for sess, scale in self.nets:
                x = self._crop(img, f.bbox, scale)[None]
                probs += softmax(sess.run(None, {sess.get_inputs()[0].name: x})[0])[0]
            probs /= len(self.nets)
            f.liveness = {"status": "ok", "is_live": bool(probs.argmax() == 1), "live_score": float(probs[1]),
                          "print_score": float(probs[0]), "replay_score": float(probs[2]), "reason": None}


class PersonDetector:
    """YOLOX-S (COCO). Only the 'person' class is kept."""

    def __init__(self, score=0.45, nms=0.5):
        self.score, self.nms = score, nms
        self.sess = session("person")
        grids, strides = [], []
        for s in (8, 16, 32):
            n = 640 // s
            xv, yv = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), s))
        self.grids, self.strides = np.concatenate(grids), np.concatenate(strides)

    def detect(self, img: np.ndarray) -> list[tuple[np.ndarray, float]]:
        h, w = img.shape[:2]
        r = min(640 / h, 640 / w)
        pad = np.full((640, 640, 3), 114, np.float32)
        pad[:int(h * r), :int(w * r)] = cv2.resize(img, (int(w * r), int(h * r)))
        out = self.sess.run(None, {"images": pad.transpose(2, 0, 1)[None]})[0][0]
        xy = (out[:, :2] + self.grids) * self.strides
        wh = np.exp(out[:, 2:4]) * self.strides
        score = out[:, 4] * out[:, 5]  # objectness * P(person)
        m = score >= self.score
        if not m.any():
            return []
        boxes = np.concatenate([xy[m] - wh[m] / 2, xy[m] + wh[m] / 2], 1) / r
        s = score[m]
        keep = np.array(cv2.dnn.NMSBoxes([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes.tolist()], s.tolist(),
                                         self.score, self.nms)).flatten()
        return [(np.clip(boxes[k], 0, [w, h, w, h]), float(s[k])) for k in keep]


def appearance(img: np.ndarray, box) -> np.ndarray | None:
    """Model-free body appearance signature: HSV colour histograms of the upper and lower body, L2-normalised."""
    x1, y1, x2, y2 = map(int, box)
    bh = y2 - y1
    if bh < 40 or x2 - x1 < 16:
        return None
    feats = []
    for a, b in ((0.2, 0.55), (0.55, 0.95)):
        part = img[y1 + int(bh * a):y1 + int(bh * b), x1:x2]
        if part.size == 0:
            return None
        hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
        feats.append(cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten())
    v = np.sqrt(np.concatenate(feats))  # Hellinger kernel makes cosine behave well on histograms
    return v / (np.linalg.norm(v) + 1e-9)


# Pipeline -----------------------------------------------------------------


class FaceAnalysis:
    """Detection + optional embedding, landmarks/pose, attributes and liveness, per pack."""

    PACKS = {
        "full": ("detection", "recognition", "landmarks", "attributes", "liveness"),
        "fast": ("detection", "recognition"),
    }

    def __init__(self):
        self.detector = Detector()
        self.recognizer = Recognizer()
        self.landmarker = Landmarker()
        self.attributes = Attributes()
        self.liveness = Liveness()

    def get(self, img: np.ndarray, pack: str = "full", min_score: float = 0.0) -> list[Face]:
        parts = self.PACKS[pack]
        faces = [f for f in self.detector.detect(img) if f.det_score >= min_score]
        if "recognition" in parts:
            self.recognizer.embed(img, faces)
        if "landmarks" in parts:
            for f in faces:
                self.landmarker.run(img, f)
        if "attributes" in parts:
            self.attributes.run(img, faces)
        if "liveness" in parts:
            self.liveness.run(img, faces)
        return faces

    def provider(self) -> str:
        return self.recognizer.sess.get_providers()[0]


# Tracking -----------------------------------------------------------------


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class ShotCutDetector:
    """Flags hard cuts by comparing colour histograms of consecutive (analysed) frames."""

    def __init__(self, min_corr: float = 0.85):
        self.min_corr, self.prev = min_corr, None

    def __call__(self, frame: np.ndarray) -> bool:
        hsv = cv2.cvtColor(cv2.resize(frame, (96, 54), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        hist = cv2.normalize(cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256]), None).flatten()
        cut = self.prev is not None and cv2.compareHist(self.prev, hist, cv2.HISTCMP_CORREL) < self.min_corr
        self.prev = hist
        return cut


class IoUTracker:
    """Greedy IoU tracker: good enough for short clips analysed at >= 5 fps."""

    def __init__(self, min_iou: float = 0.25, max_age: int = 10):
        self.min_iou, self.max_age = min_iou, max_age
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    def reset(self):
        """Forget live tracks (e.g. at a shot cut) so a new face in the same spot gets a new id."""
        self.tracks = {}

    def split(self, tid: int) -> int:
        """Move a live track to a fresh id (its box now belongs to someone else)."""
        new = self.next_id
        self.next_id += 1
        if tid in self.tracks:
            self.tracks[new] = self.tracks.pop(tid)
        return new

    def update(self, boxes: list, t: int) -> list[int]:
        pairs = sorted(((iou(tr["box"], b), tid, i) for tid, tr in self.tracks.items() for i, b in enumerate(boxes)), reverse=True)
        ids: list[int | None] = [None] * len(boxes)
        used = set()
        for score, tid, i in pairs:
            if score < self.min_iou:
                break
            if tid in used or ids[i] is not None:
                continue
            ids[i] = tid
            used.add(tid)
        for i, b in enumerate(boxes):
            if ids[i] is None:
                ids[i] = self.next_id
                self.next_id += 1
            self.tracks[ids[i]] = {"box": b, "last": t}
        self.tracks = {k: v for k, v in self.tracks.items() if t - v["last"] <= self.max_age}
        return ids
