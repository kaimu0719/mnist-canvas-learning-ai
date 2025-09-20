import io
import numpy as np
import urllib.request
from fastapi import HTTPException
from PIL import Image

def fetch_image_bytes_from_url(url: str):
    "指定された URL から画像をダウンロードし、バイト列として返す。"
    try:
        with urllib.request.urlopen(url, timeout=10) as image:
            img_bytes = image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to fetch image_url: {e}")
  
    return img_bytes

def preprocess_to_tensor(img_bytes: bytes) -> np.ndarray:
    """PILで2グレースケール化 → [N,C,H,W] に整形"""
  
    # convert("L")でmodeをRGBA→Lに変換
    # Lとはグレースケール（白黒）のこと
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    # グレースケールは0~255の値を持っており、255.0で割ることによって正規化している
    # これによって推論精度を上げることができる
    x = np.array(img, dtype=np.float32) / 255.0

    x = x.reshape(1, 1, 28, 28)

    return x