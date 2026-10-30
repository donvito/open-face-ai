# Open Face AI Studio

A local face AI web app with the same features as [face-ai-studio](https://github.com/donvito/face-ai-studio), rebuilt
without InsightFace. Every model is under a license that permits commercial use (MIT, Apache 2.0, CC BY 4.0).

| Tab | How it works |
| --- | --- |
| Face analysis | YuNet detection + 5 keypoints, MediaPipe 478-point 3D mesh with head pose and 52 expression blendshapes, FairFace age group + gender, 512-d AuraFace embedding. `full` or `fast` (detection + embedding only) |
| Verify 1:1 | 5-point alignment to 112x112, AuraFace embeddings, cosine similarity with adjustable threshold |
| Identify 1:N | Gallery search; enroll your own people from uploads (saved to `photos/people/`) |
| Liveness | Silent-Face MiniFASNet two-model ensemble (print / replay / real), with an optional simulated screen-replay attack |
| Album clustering | Agglomerative clustering of every face's AuraFace embedding in the sample album |
| Video privacy | Per-frame detection, IoU tracking with shot-cut detection, per-track recognition vote; blur everyone, only selected people, or keep selected people visible (Gaussian or mosaic). Writes an analysis JSON |
| Person tracking | YOLOX body detection + face matching, then body-appearance matching (colour signature learned from face-confirmed frames) or track continuity when the face is hidden. Annotated video and appearance timeline |

## Run

```bash
./run.sh            # creates .venv on first run, then serves http://127.0.0.1:7860
```

Requires Python 3.10+ and `ffmpeg` on `PATH`. The first start downloads the models (~390 MB) to `./models`
(override with `OPEN_FACE_MODELS=/path`). ONNX Runtime uses CoreML on Apple Silicon, CUDA if `onnxruntime-gpu` is
installed, else CPU; force a choice with `ORT_PROVIDERS=CPUExecutionProvider`.

## Models and licenses

| Role | Model | License | Training data |
| --- | --- | --- | --- |
| Face detection | [YuNet 2023mar](https://huggingface.co/opencv/face_detection_yunet) (OpenCV Zoo) | MIT | WIDER FACE |
| Face recognition | [AuraFace v1](https://huggingface.co/fal/AuraFace-v1) (`glintr100.onnx` only) | Apache 2.0 | Commercially licensed data, per fal |
| Mesh, pose, expressions | [MediaPipe Face Landmarker v2](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) | Apache 2.0 | Google |
| Age group, gender | [FairFace ResNet34](https://github.com/joojs/fairface) ([ONNX export](https://github.com/yakhyo/fairface-onnx)) | CC BY 4.0 (attribution required) | FairFace (CC BY 4.0) |
| Liveness | [Silent-Face MiniFASNetV2 + V1SE](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) ([ONNX export](https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx)) | Apache 2.0 | Minivision |
| Person detection | [YOLOX-S](https://huggingface.co/opencv/object_detection_yolox) (OpenCV Zoo) | Apache 2.0 | COCO |
| Body appearance | HSV colour histograms (no model) | n/a | n/a |

What was deliberately left out:

- The AuraFace repo also contains `scrfd_10g_bnkps.onnx`, `2d106det.onnx`, `1k3d68.onnx` and `genderage.onnx`. Those are
  InsightFace models (non-commercial) despite the repo's Apache tag; only `glintr100.onnx` is downloaded.
- OpenCV Zoo SFace and YoutuReID are Apache 2.0 but trained on MS-Celeb-1M / CASIA-WebFace and person ReID research
  datasets, so they are not used.
- The `insightface_sample_t1.jpg` group photo from the original repo is not included.

Before you ship, have counsel review the remaining points: YuNet is trained on WIDER FACE and YOLOX on COCO, whose
image terms are research-oriented (this is the industry-standard situation for detectors); FairFace's CC BY 4.0 needs
attribution in your product; and facial recognition, age and gender estimation are regulated in some places
(e.g. BIPA in Illinois, GDPR biometric data, EU AI Act).

## Credits and license notices

[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) lists every model, Python package and media source with authors,
copyright lines and changes made; full model license texts are in [`licenses/`](licenses/). The app shows the same
information under **Credits & licenses** (tab 08, also linked from the sidebar footer), including the CC BY 4.0
attribution FairFace requires. Keep both when you ship the app.

Note: the macOS `opencv-contrib-python` wheel (required by MediaPipe) bundles a GPLv3 FFmpeg build. That doesn't matter
for SaaS, but read the notices file before distributing the app as a desktop or on-prem package.

## Media

- `photos/people/<name>/`: four NASA portraits each of the Artemis II crew, used as the identification gallery.
- `photos/groups/`: group photos for detection, identification and clustering.
- `video/`: two short Artemis II clips for video privacy and person tracking.

All photos and videos are public domain (NASA) from Wikimedia Commons; sources are in `photos/CREDITS.json`.
`fetch_photos.py` re-downloads the photos.

## Notes

- Default thresholds (0.35 verify / identify) are calibrated on the bundled gallery: same-person pairs score 0.39-0.83 and different people at most 0.34.
  Full profiles (e.g. `jeremy_hansen_looks_back_home.jpg`) fall below the threshold, as with any 5-point aligned recogniser.
- FairFace predicts age groups (e.g. `40-49`), not exact ages.
- MiniFASNet is trained on camera captures. Real web photos score about 0.94 to 1.0, simulated replays about 0.27 to 0.94; the UI defaults to 0.95. The screen-replay simulation is synthetic, not a real presentation attack.
- Body-appearance matching uses colour only, so people in identical outfits fall back to track continuity.
- People enrolled in the app are saved to `photos/people/<name>/` and reloaded on start; removing one moves the folder to `work/deleted/`.
- Uploads and outputs go to `work/`.
