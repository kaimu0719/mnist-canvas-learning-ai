import urllib.request
from fastapi import HTTPException

def fetch_image_bytes_from_url(url: str):
    "指定された URL から画像をダウンロードし、バイト列として返す。"
    try:
        with urllib.request.urlopen(url, timeout=10) as image:
            img_bytes = image.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"failed to fetch image_url: {e}")
  
    return img_bytes
