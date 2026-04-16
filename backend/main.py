from fastapi import FastAPI, UploadFile, File
import json
import math
import os
from google.cloud import vision

app = FastAPI()

# ---------------------------------------
# CARICA BORGHI
# ---------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
borghi_path = os.path.join(BASE_DIR, "..", "data", "borghi_italia_clean.json")
with open(borghi_path, "r", encoding="utf-8") as f:
    BORGHI = json.load(f)

if isinstance(BORGHI, dict):
    for key in ["items", "data", "borghi", "results"]:
        if key in BORGHI and isinstance(BORGHI[key], list):
            BORGHI = BORGHI[key]
            break

if not isinstance(BORGHI, list):
    BORGHI = []

# ---------------------------------------
# HELPERS
# ---------------------------------------

def get_borgo_name(b):
    for key in ["name", "title", "nome"]:
        value = b.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Borgo sconosciuto"


def get_borgo_coords(b):
    location = b.get("location")
    if isinstance(location, dict):
        lat = location.get("lat")
        lon = location.get("lon")
        if lat is not None and lon is not None:
            try:
                return float(lat), float(lon)
            except:
                pass

    for lat_key, lon_key in [("lat", "lng"), ("latitude", "longitude")]:
        lat = b.get(lat_key)
        lon = b.get(lon_key)
        if lat is not None and lon is not None:
            try:
                return float(lat), float(lon)
            except:
                pass

    return None, None

# ---------------------------------------
# DISTANZA
# ---------------------------------------

def distance(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2 +
        math.cos(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ---------------------------------------
# DIREZIONE
# ---------------------------------------

def calculate_bearing(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)

    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = (
        math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
        - math.sin(math.radians(lat1)) *
        math.cos(math.radians(lat2)) *
        math.cos(dlon)
    )

    bearing = math.degrees(math.atan2(y, x))
    return (bearing + 360) % 360

# ---------------------------------------
# GOOGLE VISION
# ---------------------------------------

def analyze_image_bytes(image_bytes):
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)

    response = client.label_detection(image=image)
    return [l.description.lower() for l in response.label_annotations]

# ---------------------------------------
# 🔥 FILTRO AVANZATO
# ---------------------------------------

def is_borgo_scene(labels):
    labels = [l.lower() for l in labels]

    strong_positive = [
        "village", "town", "city",
        "building", "architecture", "house",
        "street", "road",
        "mountain", "hill", "landscape",
        "sea", "coast"
    ]

    weak_positive = [
        "sky", "tree", "nature", "outdoor"
    ]

    strong_negative = [
        "comic", "cartoon", "animation",
        "fictional character", "toy",
        "poster", "book", "cosplay",
        "costume"
    ]

    weak_negative = [
        "laptop", "phone", "electronics",
        "indoor"
    ]

    pos_score = 0
    neg_score = 0

    for l in labels:
        if any(k in l for k in strong_positive):
            pos_score += 2
        elif any(k in l for k in weak_positive):
            pos_score += 1

        if any(k in l for k in strong_negative):
            neg_score += 3
        elif any(k in l for k in weak_negative):
            neg_score += 1

    if neg_score >= 3:
        return False

    if pos_score >= 2 and pos_score > neg_score:
        return True

    return False

# ---------------------------------------
# SCORING
# ---------------------------------------

def score_borgo(b, lat, lng, heading):
    b_lat, b_lng = get_borgo_coords(b)
    if b_lat is None:
        return 9999

    dist = distance(lat, lng, b_lat, b_lng)

    bearing = calculate_bearing(lat, lng, b_lat, b_lng)
    diff = abs(bearing - heading)
    if diff > 180:
        diff = 360 - diff

    return dist * 1.5 + diff / 20

# ---------------------------------------
# FALLBACK
# ---------------------------------------

def get_closest_borghi(lat, lng):
    results = []

    for b in BORGHI:
        b_lat, b_lng = get_borgo_coords(b)
        if b_lat is None:
            continue

        d = distance(lat, lng, b_lat, b_lng)

        results.append({
            "name": get_borgo_name(b),
            "lat": b_lat,
            "lng": b_lng,
            "score": d
        })

    results.sort(key=lambda x: x["score"])
    return results[:3]

# ---------------------------------------
# ENDPOINT
# ---------------------------------------

@app.post("/recognize")
async def recognize(
    file: UploadFile = File(...),
    lat: float = 0,
    lng: float = 0,
    heading: float = 0
):
    image_bytes = await file.read()
    labels = analyze_image_bytes(image_bytes)

    # 🔥 BLOCCO INTELLIGENTE
    if not is_borgo_scene(labels):
        return {
            "labels": labels,
            "candidates": [],
            "message": "Nessun borgo riconoscibile nella foto"
        }

    # scorri tutti i borghi
    results = []

    for b in BORGHI:
        s = score_borgo(b, lat, lng, heading)

        if s < 100:
            b_lat, b_lng = get_borgo_coords(b)

            results.append({
                "name": get_borgo_name(b),
                "lat": b_lat,
                "lng": b_lng,
                "score": s
            })

    results.sort(key=lambda x: x["score"])

    if results:
        candidates = results[:3]
    else:
        candidates = get_closest_borghi(lat, lng)

    return {
        "labels": labels,
        "candidates": candidates
    }