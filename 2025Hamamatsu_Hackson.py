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
DEFAULT_SHELTER_CSV  = "避難所リスト.csv"  # ← デモ用に “test” を既定に

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
        # 完全一致 or 正規化一致
        for nm in names:
            if nm in df.columns:
                found = nm
                break
            nm_norm = normalize_colname(nm)
            if nm_norm in inv_map:
                found = inv_map[nm_norm]
                break
        # ゆるい包含
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
    # エンコーディング自動化（日本語CSV想定のフォールバック付き）
    for enc in ["utf-8-sig", "utf-8", "cp932"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    # 最後にデフォルトで再挑戦（例外をそのまま投げる）
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

# OSRM（徒歩）ルート取得
def osrm_route_foot(start_lat, start_lon, end_lat, end_lon) -> dict | None:
    url = f"https://router.project-osrm.org/route/v1/foot/{start_lon},{start_lat};{end_lon},{end_lat}?overview=full&geometries=geojson"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if "routes" not in data or len(data["routes"]) == 0:
            return None
        return data["routes"][0]  # distance(m), duration(s), geometry
    except Exception:
        return None

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="避難ルート・到達圏デモ（固定CSV）", layout="wide")
st.title("避難ルート・到達圏デモ（固定CSV自動読込）")

with st.sidebar:
    st.header("ファイル設定（必要なら変更）")
    mobility_path = st.text_input("年齢層別移動能力 CSV パス", value=DEFAULT_MOBILITY_CSV)
    shelter_path  = st.text_input("避難所リスト CSV パス", value=DEFAULT_SHELTER_CSV)
    st.caption("※ リポジトリ直下に配置するのが簡単です。相対/絶対パスいずれも可。")

    st.divider()
    st.header("現在地（任意で手動設定）")
    default_lat = st.number_input("現在地 緯度（未入力なら地図の📍で取得）", value=35.681236, step=0.000001, format="%.6f")
    default_lon = st.number_input("現在地 経度（未入力なら地図の📍で取得）", value=139.767125, step=0.000001, format="%.6f")
    st.caption("※ 左上の📍ボタン（位置情報の許可が必要）でブラウザの現在地を取得できます。")

# ---- CSV 自動読込（年齢層別移動能力）
mob_df = None
dist5_km = dist10_km = dist15_km = None
age_selected = act_selected = None
try:
    mob_df = load_csv(mobility_path)
    colmap = detect_columns(mob_df, {
        "age":    ["年齢区分","年齢層","年齢","age"],
        "activity": ["活動種別","活動種類","活動","アクティビティ","activity"],
        "m5":     ["5分(km)","5分","5min(km)","5min"],
        "m10":    ["10分(km)","10分","10min(km)","10min"],
        "m15":    ["15分(km)","15分","15min(km)","15min"],
    })
    required = ["age","activity","m5","m10","m15"]
    if not all(k in colmap for k in required):
        st.error("年齢層別移動能力CSVの列名が認識できません。必要列：年齢区分 / 活動種別(活動種類) / 5分(km) / 10分(km) / 15分(km)")
    else:
        age_col = colmap["age"]
        act_col = colmap["activity"]
        col_5   = colmap["m5"]
        col_10  = colmap["m10"]
        col_15  = colmap["m15"]

        ages = list(pd.unique(mob_df[age_col].astype(str)))
        acts = list(pd.unique(mob_df[act_col].astype(str)))
        c1, c2 = st.columns(2)
        with c1:
            age_selected = st.selectbox("年齢区分", ages, index=0 if ages else None)
        with c2:
            act_selected = st.selectbox("活動種別（活動種類）", acts, index=0 if acts else None)

        if age_selected and act_selected:
            dist5_km, dist10_km, dist15_km = extract_timeband_distances(
                mob_df, age_col, act_col, col_5, col_10, col_15, age_selected, act_selected
            )
            if None in (dist5_km, dist10_km, dist15_km):
                st.warning("選択の組み合わせの5/10/15分距離が見つかりません。")
            else:
                st.success(f"到達距離（{age_selected} × {act_selected}）："
                           f"5分={dist5_km:.2f} km, 10分={dist10_km:.2f} km, 15分={dist15_km:.2f} km")
except Exception as e:
    st.error(f"年齢層別移動能力CSVの読み込みエラー: {e}")

# ---- CSV 自動読込（避難所）
shelters: T.List[Shelter] = []
try:
    sh_df = load_csv(shelter_path)
    shelters = parse_shelters(sh_df)
    st.success(f"避難所リストを読み込みました：{len(shelters)} 件")
except Exception as e:
    st.error(f"避難所リストCSVの読み込みエラー: {e}")

# 地図の作成
st.subheader("地図（現在地の📍ボタンで位置取得 → 同心円と避難所を確認）")
m = folium.Map(location=[default_lat, default_lon], zoom_start=12, control_scale=True)
LocateControl(auto_start=False, position="topleft").add_to(m)

# 同心円描画（CSVが有効な場合のみ）
def add_timeband_circles(lat, lon, d5_km, d10_km, d15_km):
    circle_defs = [
        (d5_km  * 1000.0,  "#2ecc71", f"5分 到達距離 {d5_km:.2f} km"),   # 緑
        (d10_km * 1000.0,  "#f39c12", f"10分 到達距離 {d10_km:.2f} km"), # 橙
        (d15_km * 1000.0,  "#e74c3c", f"15分 到達距離 {d15_km:.2f} km"), # 赤
    ]
    for r_m, color, label in circle_defs:
        folium.Circle(
            location=[lat, lon],
            radius=r_m,
            color=color,
            weight=2,
            fill=True,
            fill_opacity=0.08,
            tooltip=label
        ).add_to(m)

if all(v is not None for v in (dist5_km, dist10_km, dist15_km)):
    add_timeband_circles(default_lat, default_lon, dist5_km, dist10_km, dist15_km)

# 避難所マーカー
for s in shelters:
    popup = folium.Popup(f"<b>{s.name}</b><br/>{s.address or ''}", max_width=250)
    folium.Marker(
        location=[s.lat, s.lon],
        popup=popup,
        tooltip=s.name,
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)

map_state = st_folium(m, height=600, width=None, returned_objects=["center"])

# 現在地の決定（LocateControlで移動後のcenterを採用 / ない場合は手動入力値）
current_lat = map_state["center"]["lat"] if map_state and map_state.get("center") else default_lat
current_lon = map_state["center"]["lng"] if map_state and map_state.get("center") else default_lon
st.info(f"現在地推定: lat={current_lat:.6f}, lon={current_lon:.6f}")

# OSRM（徒歩）ルート取得
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

# 再描画＆ルート計算
if st.button("現在地を基準に再描画（同心円＆ルート計算）"):
    m2 = folium.Map(location=[current_lat, current_lon], zoom_start=12, control_scale=True)
    LocateControl(auto_start=False, position="topleft").add_to(m2)
    # 現在地マーカー
    folium.Marker([current_lat, current_lon], tooltip="現在地", icon=folium.Icon(color="green", icon="home")).add_to(m2)

    # 同心円（CSVが有効な場合のみ）
    if all(v is not None for v in (dist5_km, dist10_km, dist15_km)):
        for r_m, color, label in [
            (dist5_km*1000.0, "#2ecc71", f"5分 到達距離 {dist5_km:.2f} km"),
            (dist10_km*1000.0, "#f39c12", f"10分 到達距離 {dist10_km:.2f} km"),
            (dist15_km*1000.0, "#e74c3c", f"15分 到達距離 {dist15_km:.2f} km"),
        ]:
            folium.Circle(
                location=[current_lat, current_lon],
                radius=r_m,
                color=color, weight=2, fill=True, fill_opacity=0.08, tooltip=label
            ).add_to(m2)
    else:
        st.warning("5/10/15分の到達距離が未選択のため、同心円は描画されません。")

    # ルート（徒歩・OSRM優先 / 失敗時は直線距離）
    route_summaries = []
    if shelters:
        scored = []
        for s in shelters:
            route = osrm_route_foot(current_lat, current_lon, s.lat, s.lon)
            if route and ("distance" in route) and ("duration" in route):
                d_km = route["distance"] / 1000.0
                t_min = route["duration"] / 60.0
            else:
                d_km = haversine_km(current_lat, current_lon, s.lat, s.lon)
                t_min = (d_km / 4.5) * 60.0  # 徒歩時速4.5km仮定
                route = None
            scored.append((s, d_km, t_min, route))

        scored.sort(key=lambda x: x[1])
        top3 = scored[:3]

        for i, (s, d_km, t_min, route) in enumerate(top3, start=1):
            color = ["#0074D9", "#7FDBFF", "#001f3f"][i-1]
            if route and "geometry" in route and route["geometry"].get("coordinates"):
                coords = route["geometry"]["coordinates"]
                latlngs = [[lat, lon] for lon, lat in coords]
                folium.PolyLine(latlngs, color=color, weight=5, opacity=0.8).add_to(m2)
            else:
                folium.PolyLine([[current_lat, current_lon], [s.lat, s.lon]],
                                color=color, weight=3, opacity=0.6, dash_array="5,10").add_to(m2)

            folium.Marker(
                [s.lat, s.lon],
                tooltip=f"{i}. {s.name}",
                popup=folium.Popup(
                    f"<b>{i}. {s.name}</b><br/>推定距離: {d_km:.2f} km<br/>所要時間: {t_min:.0f} 分",
                    max_width=260
                ),
                icon=folium.DivIcon(html=f"""
                    <div style="background:{color};color:white;border-radius:12px;padding:2px 6px;font-weight:bold">{i}</div>
                """)
            ).add_to(m2)

            route_summaries.append({"順": i, "避難所名": s.name, "距離(km)": round(d_km, 2), "所要時間(分)": int(round(t_min))})

    st_folium(m2, height=650, width=None)

    if route_summaries:
        st.subheader("最寄り3施設（徒歩・OSRM優先 / 失敗時は直線距離換算）")
        st.dataframe(pd.DataFrame(route_summaries), use_container_width=True)

with st.expander("補足：到達圏の見方"):
    st.markdown("""
- 同心円は選択した **年齢区分 × 活動種別** の行から、**5/10/15分で移動できる距離（km）** を読み取り半径にしています。
- 厳密な等時間到達圏（Isochrone）が必要なら、OpenRouteService等の等時間APIへの切替も可能です。
""")

