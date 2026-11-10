"""Settings, read from environment variables (or a .env file next to the working directory)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip("'\""))


@dataclass(frozen=True)
class Config:
    pg_dsn: str
    pg_password: str
    photos_root: Path
    syno_user: str
    data_dir: Path
    vision_url: str
    vision_token: str
    model_dir: Path
    low_vram: bool
    cluster_dist: float
    attach: float
    max_tags: int
    min_face_score: float
    min_face_width: float


def load() -> Config:
    _load_dotenv(Path(os.environ.get("SPT_ENV_FILE", ".env")))
    e = os.environ.get

    def req(key: str) -> str:
        v = e(key)
        if not v:
            raise SystemExit(f"missing setting {key} (see .env.example)")
        return v

    return Config(
        pg_dsn=req("SPT_PG_DSN"),
        pg_password=e("SPT_PG_PASSWORD", ""),
        photos_root=Path(req("SPT_PHOTOS_ROOT")),
        syno_user=req("SPT_SYNO_USER"),
        data_dir=Path(e("SPT_DATA_DIR", "./data")),
        vision_url=e("SPT_VISION_URL", "http://127.0.0.1:8095"),
        vision_token=req("SPT_VISION_TOKEN"),
        model_dir=Path(e("SPT_MODEL_DIR", "./models")),
        low_vram=e("SPT_LOW_VRAM", "0") == "1",
        cluster_dist=float(e("SPT_CLUSTER_DIST", "0.65")),
        attach=float(e("SPT_ATTACH", "0.45")),
        max_tags=int(e("SPT_MAX_TAGS", "15")),
        min_face_score=float(e("SPT_MIN_FACE_SCORE", "0.65")),
        min_face_width=float(e("SPT_MIN_FACE_WIDTH", "0.035")),
    )
