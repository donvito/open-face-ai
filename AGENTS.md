# AGENTS.md

- Run: `./run.sh` (or `PORT=7861 ./run.sh`). Uses `.venv`, serves FastAPI + static UI.
- Code: `engine.py` (model wrappers, download registry `MODELS`), `app.py` (routes, video jobs), `static/` (vanilla JS UI).
- Hard rule: only commercially usable models (MIT / Apache 2.0 / CC BY). Never add InsightFace code or weights
  (SCRFD, buffalo_*, 2d106det, 1k3d68, genderage), and check training-data provenance before adding a model.
- When adding or changing a model or dependency, update its `engine.MODELS` metadata (authors, copyright,
  license_file, modified), add the license text to `licenses/`, and update `THIRD_PARTY_NOTICES.md`. The in-app
  Credits tab reads `/api/credits`.
- CoreML recompiles on every new input shape: keep ONNX inputs at a fixed batch size of 1.
- Smoke test (server running):
  `curl -s -F sample=photos/groups/artemis_2_crew_portrait.jpg -F model=full http://127.0.0.1:7861/api/analyze`
  should report 4 faces, each matching the right crew member at a score of 0.7 or higher.
- Video jobs: POST `/api/video/privateframe` or `/api/video/person`, then poll `/api/jobs/{id}`.
