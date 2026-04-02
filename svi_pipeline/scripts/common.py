from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
import yaml


def load_config(config_path: str | Path) -> Dict:
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_dataset(xlsx_path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, engine="openpyxl")
    return df


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def infer_id_width(svi_dir: str | Path) -> int:
    svi_dir = Path(svi_dir)
    for p in sorted(svi_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            return len(p.stem)
    return 5


def id_to_filename(img_id: int, width: int, ext: str = ".jpg") -> str:
    return f"{int(img_id):0{width}d}{ext}"


def safe_json_loads(text: str) -> Dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.replace("json", "", 1).strip()
    return json.loads(text)
