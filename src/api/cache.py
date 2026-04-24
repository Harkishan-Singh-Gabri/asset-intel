import hashlib
import json
from PIL import Image
import io

_cache = {}

def get_image_hash(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.resize((32, 32)).save(buf, format="JPEG", quality=50)
    return hashlib.md5(buf.getvalue()).hexdigest()

def get_cached(img: Image.Image):
    key = get_image_hash(img)
    return _cache.get(key)

def set_cached(img: Image.Image, result: dict):
    key = get_image_hash(img)
    _cache[key] = result