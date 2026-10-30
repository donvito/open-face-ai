# Third-party notices

Open Face AI Studio uses the third-party models, libraries and media listed below. Full license texts for the models
are in [`licenses/`](licenses/). Python packages ship their own license files inside their installed distributions
(`.venv/lib/python*/site-packages/*.dist-info/`). The same information is shown in the app under
**Credits & licenses**.

## Models

The model files are downloaded unmodified from the URLs in `engine.py` (`MODELS`) to `models/`. Only the AuraFace file is
renamed (`glintr100.onnx` to `auraface_v1_glintr100.onnx`); its contents are unchanged.

### YuNet face detector (2023mar)
- Authors: Shiqi Yu et al. ([libfacedetection.train](https://github.com/ShiqiYu/libfacedetection.train)); ONNX distributed by [OpenCV Zoo](https://huggingface.co/opencv/face_detection_yunet)
- Copyright (c) 2020 Shiqi Yu
- License: MIT, see [`licenses/YuNet-MIT.txt`](licenses/YuNet-MIT.txt)

### AuraFace v1 face recognition
- Authors: fal ([fal/AuraFace-v1](https://huggingface.co/fal/AuraFace-v1)). Only `glintr100.onnx` is used; the other
  files in that repository are InsightFace models and are not downloaded.
- License: Apache License 2.0, see [`licenses/AuraFace-Apache-2.0.txt`](licenses/AuraFace-Apache-2.0.txt) (upstream supplies no copyright line or NOTICE file)

### MediaPipe Face Landmarker v2 (`face_landmarker.task`) and the `mediapipe` Python package
- Authors: Google LLC / The MediaPipe Authors ([google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe), [model page](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker))
- License: Apache License 2.0, see [`licenses/MediaPipe-Apache-2.0.txt`](licenses/MediaPipe-Apache-2.0.txt)
- "MediaPipe" is a trademark of Google LLC.

### FairFace age and gender model
- **"FairFace" by Kimmo Kärkkäinen and Jungseock Joo**, [github.com/joojs/fairface](https://github.com/joojs/fairface),
  licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), see [`licenses/FairFace-CC-BY-4.0.txt`](licenses/FairFace-CC-BY-4.0.txt).
- Paper: Kärkkäinen, K., & Joo, J. (2021). *FairFace: Face Attribute Dataset for Balanced Race, Gender, and Age for Bias
  Measurement and Mitigation.* WACV 2021, pp. 1548-1558.
- Changes: converted from PyTorch (`res34_fair_align_multi_7_20190809.pt`) to ONNX by Yakhyokhuja Valikhujaev
  ([yakhyo/fairface-onnx](https://github.com/yakhyo/fairface-onnx)). This app uses only the gender and age-group
  outputs; the race output is discarded.
- The licensor provides the material as-is, with no warranties (CC BY 4.0, Section 5).

### Silent-Face-Anti-Spoofing (MiniFASNetV2 @2.7, MiniFASNetV1SE @4.0)
- Authors: Minivision AI ([minivision-ai/Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing))
- Copyright 2020 Minivision
- License: Apache License 2.0, see [`licenses/Silent-Face-Anti-Spoofing-Apache-2.0.txt`](licenses/Silent-Face-Anti-Spoofing-Apache-2.0.txt)
- Changes: converted from PyTorch to ONNX by QingHeYang
  ([QingHeYang/Silent-Face-Anti-Spoofing-onnx](https://github.com/QingHeYang/Silent-Face-Anti-Spoofing-onnx), Apache 2.0, same license text).

### YOLOX-S person detector
- Authors: Megvii Inc. ([Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)); ONNX distributed by [OpenCV Zoo](https://huggingface.co/opencv/object_detection_yolox)
- Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
- License: Apache License 2.0, see [`licenses/YOLOX-Apache-2.0.txt`](licenses/YOLOX-Apache-2.0.txt)

## Python packages

Direct dependencies (`requirements.txt`) and everything they pull in, as installed in `.venv`:

| Package | Version | License |
| --- | --- | --- |
| absl-py | 2.5.1 | Apache-2.0 |
| annotated-doc | 0.0.5 | MIT |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.15.1 | MIT |
| certifi | 2026.10.23 | MPL-2.0 |
| cffi | 2.1.1 | MIT-0 |
| charset-normalizer | 3.5.2 | MIT |
| click | 8.5.0 | BSD-3-Clause |
| cloudpickle | 3.1.2 | BSD-3-Clause |
| contourpy | 1.4.0 | BSD-3-Clause |
| cycler | 0.12.1 | BSD-3-Clause |
| fastapi | 0.162.0 | MIT |
| flatbuffers | 25.12.19 | Apache-2.0 |
| fonttools | 4.68.0 | MIT |
| h11 | 0.16.0 | MIT |
| idna | 3.20 | BSD-3-Clause |
| joblib | 1.6.0 | BSD-3-Clause |
| kiwisolver | 1.5.1 | BSD-3-Clause |
| matplotlib | 3.11.2 | PSF-based (matplotlib license) |
| mediapipe | 1.1.1 | Apache-2.0 |
| narwhals | 2.28.0 | MIT |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| onnxruntime | 1.31.0 | MIT |
| opencv-contrib-python | 5.0.0.93 | Apache-2.0 (bundled third-party binaries: see below) |
| opentelemetry-api | 1.45.2 | Apache-2.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pillow | 13.0.0 | MIT-CMU |
| protobuf | 7.36.2 | BSD-3-Clause |
| pycparser | 3.12 | BSD-3-Clause |
| pydantic | 2.14.2 | MIT |
| pydantic_core | 2.50.2 | MIT |
| pyparsing | 3.3.3 | MIT |
| python-dateutil | 3.0.0 | Apache-2.0 AND BSD-3-Clause |
| python-multipart | 0.0.32 | Apache-2.0 |
| requests | 2.34.2 | Apache-2.0 |
| scikit-learn | 1.9.1 | BSD-3-Clause |
| scipy | 1.18.1 | BSD-3-Clause |
| six | 1.17.0 | MIT |
| sounddevice | 0.5.6 | MIT |
| starlette | 1.10.0 | BSD-3-Clause |
| threadpoolctl | 3.7.0 | BSD-3-Clause |
| typing-inspection | 0.4.4 | MIT |
| typing_extensions | 4.16.0 | PSF-2.0 |
| tzdata | 2026.6 | Apache-2.0 |
| urllib3 | 2.8.0 | MIT |
| uvicorn | 0.54.0 | BSD-3-Clause |

The live list, read from the running environment, is shown in the app (`/api/credits`).

### FFmpeg inside opencv-contrib-python (important if you distribute the app)

The `opencv-contrib-python` wheel bundles FFmpeg and other native libraries in `cv2/.dylibs` (macOS) and documents
them in `cv2/LICENSE-3RD-PARTY.txt`. That file describes FFmpeg as LGPL, but the macOS arm64 wheel installed here
bundles an FFmpeg built with `--enable-gpl --enable-version3` together with `libx264`, `libx265`, `libpostproc`,
`librubberband` and `libvidstab`, so `libavcodec` reports **"GPL version 3 or later"**. `mediapipe` depends on
`opencv-contrib-python`, so the package can't simply be dropped.

- Running the app on your own servers (SaaS) does not distribute these binaries.
- Shipping the app with this environment (desktop installer, Docker image, on-prem) means distributing GPLv3 binaries.
  Get legal advice first, or ship a build of OpenCV / FFmpeg without GPL components.

## System tools (not bundled)

- `ffmpeg` on `PATH` is used to encode browser-playable H.264 MP4s. Builds with `libx264` are GPL, and H.264 is
  covered by patent pools (Via LA / Access Advance). This project does not distribute ffmpeg.

## Media

- Photos in `photos/` are public domain NASA images from Wikimedia Commons. Title, author, source page and license
  for each are listed in [`photos/CREDITS.json`](photos/CREDITS.json) and in the app.
- Videos in `video/` are short Artemis II clips described by the original project as public domain NASA footage from
  Wikimedia Commons. **Their exact source pages were not recorded**; add them before redistributing the clips.
- NASA material is not copyrighted, but NASA's media guidelines prohibit implying NASA endorsement, and the people
  shown keep their rights of publicity. Use the samples for demonstration only.
- Photos you enroll through the app are saved under `photos/people/` and are your responsibility.

## Code

This project's own code (`app.py`, `engine.py`, `static/`) is derived from
[donvito/face-ai-studio](https://github.com/donvito/face-ai-studio). It contains no InsightFace code. The
5-point alignment template in `engine.py` is the standard ArcFace 112x112 landmark layout (10 coordinates), which is
also used by OpenCV's face recognizer.
