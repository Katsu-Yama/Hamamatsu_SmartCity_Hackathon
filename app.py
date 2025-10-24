import csv
import math
import os
from functools import lru_cache
from typing import Dict, List, Tuple

import requests
from flask import Flask, jsonify, render_template, request


app = Flask(__name__)


def project_path(*parts: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *parts)


@lru_cache(maxsize=1)
def load_shelters() -> List[Dict]:
    """Load shelters from CSV with flexible headers.
    Accepts UTF-8 (with BOM) and CP932 as fallback.
    Recognizes header variants for name/address/lat/lon including Japanese.
    """
    csv_path = project_path("data", "shelters.csv")
    rows: List[Dict] = []
    if not os.path.exists(csv_path):
        return rows

    def find_key(candidates: List[str], headers: List[str]) -> str:
        lower_map = {h.lower(): h for h in headers}
        # direct lower match
        for c in candidates:
            if c.lower() in lower_map:
                return lower_map[c.lower()]
        # substring match
        for h in headers:
            hl = h.lower()
            for c in candidates:
                if c.lower() in hl:
                    return h
        return ""

    encodings_to_try = ["utf-8-sig", "cp932", "utf-8"]
    for enc in encodings_to_try:
        try:
            with open(csv_path, "r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                headers = reader.fieldnames

                name_key = find_key([
                    "name", "名称", "施設名", "避難所名", "学校名"
                ], headers)
                addr_key = find_key([
                    "address", "住所", "所在地"
                ], headers)
                lat_key = find_key([
                    "lat", "latitude", "緯度", "緯度（世界測地系）"
                ], headers)
                lon_key = find_key([
                    "lon", "lng", "long", "longitude", "経度", "経度（世界測地系）"
                ], headers)

                for r in reader:
                    try:
                        name = (r.get(name_key) or "").strip() if name_key else ""
                        address = (r.get(addr_key) or "").strip() if addr_key else ""
                        lat_str = (r.get(lat_key) or "").strip() if lat_key else ""
                        lon_str = (r.get(lon_key) or "").strip() if lon_key else ""
                        if not (name and address and lat_str and lon_str):
                            continue
                        lat = float(str(lat_str).replace("\u3000", " ").replace(",", "."))
                        lon = float(str(lon_str).replace("\u3000", " ").replace(",", "."))
                        rows.append({
                            "name": name,
                            "address": address,
                            "lat": lat,
                            "lon": lon,
                        })
                    except Exception:
                        continue
            return rows
        except Exception:
            continue
    return rows


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points (meters)."""
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def nearest_shelter(lat: float, lon: float) -> Tuple[Dict, float]:
    shelters = load_shelters()
    best = None
    best_d = float("inf")
    for s in shelters:
        d = haversine(lat, lon, s["lat"], s["lon"])
        if d < best_d:
            best = s
            best_d = d
    return best, best_d


def fetch_route(from_lat: float, from_lon: float, to_lat: float, to_lon: float, profile: str = "walking") -> Dict:
    """Fetch route from public OSRM. Returns dict with distance (m), duration (s), geometry (GeoJSON LineString)."""
    base = "https://router.project-osrm.org/route/v1"
    url = f"{base}/{profile}/{from_lon},{from_lat};{to_lon},{to_lat}?overview=full&geometries=geojson"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        routes = data.get("routes", [])
        if routes:
            route = routes[0]
            return {
                "distance": route.get("distance"),
                "duration": route.get("duration"),
                "geometry": route.get("geometry"),
                "raw": route,
            }
    except Exception:
        pass
    # Fallback: straight line
    return {
        "distance": haversine(from_lat, from_lon, to_lat, to_lon),
        "duration": None,
        "geometry": {
            "type": "LineString",
            "coordinates": [
                [from_lon, from_lat],
                [to_lon, to_lat],
            ],
        },
        "raw": None,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/shelters")
def api_shelters():
    return jsonify(load_shelters())


@app.route("/api/nearest")
def api_nearest():
    try:
        lat = float(request.args.get("lat"))
        lon = float(request.args.get("lng") or request.args.get("lon"))
    except Exception:
        return jsonify({"error": "lat,lng are required"}), 400
    s, d = nearest_shelter(lat, lon)
    if not s:
        return jsonify({"error": "no shelters"}), 404
    return jsonify({"shelter": s, "straight_distance_m": d})


@app.route("/api/route")
def api_route():
    try:
        from_lat = float(request.args.get("from_lat"))
        from_lon = float(request.args.get("from_lng") or request.args.get("from_lon"))
    except Exception:
        return jsonify({"error": "from_lat,from_lng are required"}), 400

    to_lat = request.args.get("to_lat")
    to_lon = request.args.get("to_lng") or request.args.get("to_lon")
    profile = request.args.get("profile", "walking")

    if to_lat is None or to_lon is None:
        s, _ = nearest_shelter(from_lat, from_lon)
        if not s:
            return jsonify({"error": "no shelters"}), 404
        to_lat = s["lat"]
        to_lon = s["lon"]
        target = s
    else:
        to_lat = float(to_lat)
        to_lon = float(to_lon)
        target = {"lat": to_lat, "lon": to_lon}

    route = fetch_route(from_lat, from_lon, to_lat, to_lon, profile=profile)
    return jsonify({
        "from": {"lat": from_lat, "lon": from_lon},
        "to": target,
        "route": route,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=True)
