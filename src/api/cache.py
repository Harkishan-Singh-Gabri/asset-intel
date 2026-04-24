import redis
import hashlib
import json
from PIL import Image
import io
import os
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL")

if not REDIS_URL:
    REDIS_AVAILABLE = False
    _cache = {}
    print("REDIS_URL not set — using in-memory cache")
else:
    try:
        r = redis.from_url(REDIS_URL, socket_connect_timeout=2)
        r.ping()
        REDIS_AVAILABLE = True
        print("Redis connected")
    except Exception as e:
        REDIS_AVAILABLE = False
        _cache = {}
        print(f"Redis not available — using in-memory cache: {e}")

def get_image_hash(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.resize((32, 32)).save(buf, format="JPEG", quality=50)
    return hashlib.md5(buf.getvalue()).hexdigest()

def get_cached(img: Image.Image):
    key = f"scan:{get_image_hash(img)}"
    if REDIS_AVAILABLE:
        cached = r.get(key)
        return json.loads(cached) if cached else None
    return _cache.get(key)

def set_cached(img: Image.Image, result: dict, ttl=3600):
    key = f"scan:{get_image_hash(img)}"
    if REDIS_AVAILABLE:
        r.setex(key, ttl, json.dumps(result))
    else:
        _cache[key] = result