"""HTTP vision service: face detection + embeddings (InsightFace antelopev2) and image tags (RAM++).

Run: spt serve   (or: uvicorn synology_photos_tagger.vision_service:app --host 127.0.0.1 --port 8095)
Every endpoint except /health needs `Authorization: Bearer $SPT_VISION_TOKEN`.
Set SPT_LOW_VRAM=1 on small GPUs (tested on a 4 GB GTX 1650 shared with Plex).
"""
from __future__ import annotations

import hmac
import io
import logging
import os
import threading
import time
from typing import Any

import numpy as np
import onnxruntime as ort

try:
    ort.preload_dlls()
except Exception:  # noqa: BLE001 - older onnxruntime or system CUDA
    pass

import cv2  # noqa: E402
import pillow_heif  # noqa: E402
from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from insightface.app import FaceAnalysis  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from . import config  # noqa: E402

log = logging.getLogger("spt.vision")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
pillow_heif.register_heif_opener()

CFG = config.load()
MAX_BODY = 60 * 1024 * 1024
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
LANDMARKS = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]

if CFG.low_vram:
    CUDA = ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested",
                                      "cudnn_conv_algo_search": "HEURISTIC",
                                      "cudnn_conv_use_max_workspace": "0"})
else:
    CUDA = "CUDAExecutionProvider"
PROVIDERS = [CUDA, "CPUExecutionProvider"]

state: dict[str, Any] = {}
face_lock, tag_lock = threading.Lock(), threading.Lock()


def decode(raw: bytes, max_side: int) -> tuple[Image.Image, float, int, int]:
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    return img, scale, w, h


def run_tags(img: Image.Image) -> list[dict]:
    x = np.asarray(img.resize((384, 384), Image.BICUBIC), dtype=np.float32) / 255.0
    x = ((x - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    opts = ort.RunOptions()
    if CFG.low_vram:
        opts.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:0")
    with tag_lock:
        try:
            s = state["ram"].run(["output"], {"input": x}, opts)[0][0]
        except Exception as exc:  # noqa: BLE001 - typically GPU out of memory on 4 GB cards
            if "CUDAExecutionProvider" not in state["ram"].get_providers():
                raise
            log.warning("RAM++ failed on GPU (%s); switching tags to CPU", str(exc)[:120])
            state["ram"] = ort.InferenceSession(str(CFG.model_dir / "ram-plus" / "ram_plus.onnx"),
                                                providers=["CPUExecutionProvider"])
            s = state["ram"].run(["output"], {"input": x})[0][0]
    if s.min() < 0 or s.max() > 1:
        s = 1.0 / (1.0 + np.exp(-s))
    idx = np.nonzero(s >= state["thresholds"])[0]
    return sorted(({"tag": state["tags"][i], "score": round(float(s[i]), 4)} for i in idx),
                  key=lambda t: -t["score"])


def run_faces(img: Image.Image, scale: float, w: int, h: int) -> list[dict]:
    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    with face_lock:
        faces = state["faces"].get(bgr)
    out = []
    for f in faces:
        x1, y1, x2, y2 = (float(v) / scale for v in f.bbox)
        out.append({
            "bbox_norm": {"x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h},
            "landmarks": {n: {"x": float(f.kps[i][0]) / scale / w, "y": float(f.kps[i][1]) / scale / h}
                          for i, n in enumerate(LANDMARKS)},
            "score": float(f.det_score),
            "embedding": [round(float(v), 6) for v in f.normed_embedding],
        })
    return out


app = FastAPI(title="synology-photos-tagger vision")


@app.on_event("startup")
def startup() -> None:
    fa = FaceAnalysis(name="antelopev2", root=str(CFG.model_dir / "insightface"),
                      allowed_modules=["detection", "recognition"], providers=PROVIDERS)
    fa.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.5)
    ram_dir = CFG.model_dir / "ram-plus"
    state["faces"] = fa
    state["ram"] = ort.InferenceSession(str(ram_dir / "ram_plus.onnx"), providers=PROVIDERS)
    state["tags"] = (ram_dir / "ram_tag_list.txt").read_text(encoding="utf-8").splitlines()
    state["thresholds"] = np.array([float(x) for x in (ram_dir / "ram_tag_list_threshold.txt")
                                    .read_text().split()], dtype=np.float32)
    if len(state["tags"]) != len(state["thresholds"]):
        raise RuntimeError("RAM++ tag list and thresholds differ in length")
    log.info("ready: %s", providers())


def providers() -> dict:
    fa = state.get("faces")
    return {"faces": sorted({p for m in fa.models.values() for p in m.session.get_providers()}) if fa else [],
            "tags": state["ram"].get_providers() if "ram" in state else []}


def auth(req: Request) -> None:
    h = req.headers.get("Authorization", "")
    if not h.startswith("Bearer ") or not hmac.compare_digest(h[7:], CFG.vision_token):
        raise HTTPException(401, "unauthorized")


async def body(req: Request) -> bytes:
    if int(req.headers.get("content-length") or 0) > MAX_BODY:
        raise HTTPException(413, "body exceeds 60 MB")
    b = await req.body()
    if len(b) > MAX_BODY:
        raise HTTPException(413, "body exceeds 60 MB")
    return b


@app.get("/health")
def health() -> dict:
    return {"ok": True, "providers": providers()}


@app.post("/analyze")
async def analyze(req: Request, max_side: int = Query(1600, ge=320, le=8000),
                  top_k: int = Query(60, ge=1, le=500)) -> dict:
    auth(req)
    try:
        img, scale, w, h = decode(await body(req), max_side)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(422, f"cannot decode image: {exc}")
    t = time.perf_counter()
    faces = await run_in_threadpool(run_faces, img, scale, w, h)
    tags = await run_in_threadpool(run_tags, img)
    log.info("analyze %d faces %d tags %.3fs", len(faces), len(tags), time.perf_counter() - t)
    return {"width": w, "height": h, "faces": faces, "tags": tags[:top_k]}


def serve(host: str = "127.0.0.1", port: int = 8095) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=int(os.environ.get("SPT_VISION_PORT", port)))
