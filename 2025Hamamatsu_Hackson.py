# -*- coding: utf-8 -*-
import io
import math
import os
import typing as T
from dataclasses import dataclass

import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import LocateControl

# =========================
# 設定（既定ファイル名）
# =========================
DEFAULT_MOBILITY_CSV = "年齢層別移動能力.csv"
DEFAULT_SHELTER_CSV  = "避難所リスト.csv"

# =========================
# ユーティリティ
# =========================
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))

def try_float(x):
    try:
        return float(x)
    except:
        return None

def normalize_colname(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return name.strip().lower().replace(" ", "").replace("　", "")

def detect_columns(df: pd.DataFrame, candidates: T.Dict[str, T.List[str]]) -> T.Dict[str, str]:
    norm_map = {c: normalize_colname(c) for c in df.columns}
    inv_map = {v: k for k, v in norm_map.items()}
    resolved = {}
    for key, names in candidates.items():
        found = None
        for nm in names:
            if nm in df.columns:
                found = nm
                break
            nm_norm = normalize_colname(nm)
            if nm_norm in inv_map:
                found = inv_map[nm_norm]
                break
        if not found:
            for orig, norm in norm_map.items():
                if any(normalize_colname(nm) in norm for nm in names):
                    found = orig
                    break
        if found:
            resolved[key] = found
    return resolved

@dataclass
class Shelter:
    name: str
    address: str
    lat: float
    lon: float

@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSVが見つかりません: {path}")
    for enc in ["utf-8-sig", "utf-8", "cp932"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

def extract_timeband_distances(df: pd.DataFrame, age_col: str, act_col: str,
                               col_5: str, col_10: str, col_15: str,
                               age: str, act: str) -> tuple[float|None, float|None, float|None]:
    sub = df[(df[age_col].astype(str) == str(age)) & (df[act_col].astype(str) == str(act))]
    if sub.empty:
        return None, None, None
    r = sub.iloc[0]
    d5  = try_float(r[col_5])
    d10 = try_float(r[col_10])
    d15 = try_float(r[col_15])
    return d5, d10, d15

def parse_shelters(df: pd.DataFrame) -> T.List[Shelter]:
    cols = detect_columns(df, {
        "name": ["避難所名","名称","施設名","name","sheltername"],
        "addr": ["住所","address"],
        "lat":  ["緯度","latitude","lat","y"],
        "lon":  ["経度","longitude","lon","lng","x"],
    })
    if not all(k in cols for k in ["name","lat","lon"]):
        raise ValueError("避難所CSVの列名を解決できませんでした。『避難所名/名称/施設名』『緯度』『経度』が必要です。")
    name_c = cols["name"]
    addr_c = cols.get("addr")
    lat_c  = cols["lat"]
    lon_c  = cols["lon"]
    shelters = []
    for _, row in df.iterrows():
        lat = try_float(row[lat_c]); lon = try_float(row[lon_c])
        if lat is None or lon is None:
            continue
        shelters.append(Shelter(
            name=str(row[name_c]),
            address=str(row[addr_c]) if addr_c else "",
            lat=lat, lon=lon
        ))
    return shelters

def osrm_route_foot(start_lat, start_lon, end_lat, end_lon) -> dict | None:
    url = f"https://router.project-osrm.org/route/v1/foot/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if "routes" not in data or len(data["routes"]) == 0:
            return None
        return data["routes"][0]
    except Exception:
        return None

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="避難ルート・到達圏デモ（完全版）", layout="wide")
st.title("避難ルート・到達圏デモ（📍統一＋経路自動描画 完全版）")

with st.sidebar:
    st.header("ファイル設定")
    mobility_path = st.text_input("年齢層別移動能力 CSV", value=DEFAULT_MOBILITY_CSV)
    shelter_path  = st.text_input("避難所リスト CSV", value=DEFAULT_SHELTER_CSV)
    st.caption("※ 同じフォルダに配置してください。")

    st.divider()
    st.header("現在地設定")
    default_lat = st.number_input("緯度", value=34.706, step=0.000001, format="%.6f")
    default_lon = st.number_input("経度", value=137.735, step=0.000001, format="%.6f")

# ---- CSV読込
mob_df = load_csv(mobility_path)
sh_df  = load_csv(shelter_path)
shelters = parse_shelters(sh_df)

colmap = detect_columns(mob_df, {
    "age": ["年齢区分","年齢層","年齢","age"],
    "activity": ["活動種別","活動種類","活動","アクティビティ","activity"],
    "m5": ["5分(km)","5分","5min(km)","5min"],
    "m10": ["10分(km)","10分","10min(km)","10min"],
    "m15": ["15分(km)","15分","15min(km)","15min"],
})
age_col = colmap["age"]; act_col = colmap["activity"]
col_5 = colmap["m5"]; col_10 = colmap["m10"]; col_15 = colmap["m15"]

ages = list(pd.unique(mob_df[age_col].astype(str)))
acts = list(pd.unique(mob_df[act_col].astype(str)))
c1, c2 = st.columns(2)
with c1:
    age_selected = st.selectbox("年齢区分", ages)
with c2:
    act_selected = st.selectbox("活動種別", acts)

dist5_km, dist10_km, dist15_km = extract_timeband_distances(
    mob_df, age_col, act_col, col_5, col_10, col_15, age_selected, act_selected
)

# ---- 地図初期化
m = folium.Map(location=[default_lat, default_lon], zoom_start=13, control_scale=True)
LocateControl(auto_start=False, position="topleft",
              icon="fa fa-map-marker", strings={"title": "📍現在地を取得"}).add_to(m)

# 現在地マーカー
folium.Marker([default_lat, default_lon],
              tooltip="現在地",
              icon=folium.Icon(color="green", icon="home")).add_to(m)

# 同心円（ラベルごとに正確に）
for r_m, color, label in [
    (dist5_km*1000.0, "#2ecc71", f"5分 到達距離 {dist5_km:.2f} km"),
    (dist10_km*1000.0, "#f39c12", f"10分 到達距離 {dist10_km:.2f} km"),
    (dist15_km*1000.0, "#e74c3c", f"15分 到達距離 {dist15_km:.2f} km"),
]:
    folium.Circle(
        location=[default_lat, default_lon],
        radius=r_m,
        color=color, weight=2,
        fill=True, fill_opacity=0.08,
        tooltip=label, popup=label
    ).add_to(m)

# 避難所マーカー
for s in shelters:
    folium.Marker([s.lat, s.lon],
                  popup=f"<b>{s.name}</b><br>{s.address or ''}",
                  tooltip=s.name,
                  icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)

# ==== 上位3避難ルートを描画 ====
def draw_top3_routes(m, origin_lat, origin_lon, shelters):
    scored = []
    for s in shelters:
        route = osrm_route_foot(origin_lat, origin_lon, s.lat, s.lon)
        if route and ("distance" in route) and ("duration" in route):
            d_km = route["distance"]/1000.0
            t_min = route["duration"]/60.0
        else:
            d_km = haversine_km(origin_lat, origin_lon, s.lat, s.lon)
            t_min = (d_km/4.5)*60.0
            route = None
        scored.append((s, d_km, t_min, route))

    scored.sort(key=lambda x: x[1])
    top3 = scored[:3]

    colors = ["#0074D9", "#7FDBFF", "#001f3f"]
    summaries = []
    for i, (s, d_km, t_min, route) in enumerate(top3, start=1):
        color = colors[i-1]
        if route and "geometry" in route and route["geometry"].get("coordinates"):
            coords = route["geometry"]["coordinates"]
            latlngs = [[lat, lon] for lon, lat in coords]
            folium.PolyLine(latlngs, color=color, weight=5, opacity=0.8).add_to(m)
        else:
            folium.PolyLine([[origin_lat, origin_lon], [s.lat, s.lon]],
                            color=color, weight=3, opacity=0.6, dash_array="5,10").add_to(m)
        folium.Marker(
            [s.lat, s.lon],
            tooltip=f"{i}. {s.name}",
            popup=folium.Popup(
                f"<b>{i}. {s.name}</b><br>距離: {d_km:.2f} km<br>所要: {t_min:.0f} 分",
                max_width=250
            ),
            icon=folium.DivIcon(html=f"<div style='background:{color};color:white;border-radius:12px;padding:2px 6px;font-weight:bold'>{i}</div>")
        ).add_to(m)
        summaries.append({"順": i, "避難所名": s.name, "距離(km)": round(d_km,2), "所要時間(分)": int(round(t_min))})
    return summaries

route_summaries = draw_top3_routes(m, default_lat, default_lon, shelters)

# 地図描画
st_folium(m, height=650, width=None)

if route_summaries:
    st.subheader("最寄り3施設（徒歩・OSRM優先 / 失敗時は直線距離換算）")
    st.dataframe(pd.DataFrame(route_summaries), use_container_width=True)
