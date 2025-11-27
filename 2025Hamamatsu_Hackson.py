# -*- coding: utf-8 -*-
import io
import math
import os
import typing as T
from dataclasses import dataclass

import pandas as pd
import time
import requests
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import LocateControl

# =========================
# 設定（既定ファイル名）
# =========================
DEFAULT_MOBILITY_CSV = "年齢層別移動能力.csv"
DEFAULT_SHELTER_CSV  = "221309_hamamatsu_tsunami_hinan.csv"

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

def parse_number_from_text(text: str) -> float | None:
    if text is None:
        return None
    try:
        # 全角・単位混在を許容して数値だけ抽出
        s = str(text)
        buf = []
        dot_used = False
        sign_used = False
        for ch in s:
            if ch.isdigit():
                buf.append(ch)
            elif ch in [".", "．", "｡"] and not dot_used:
                buf.append("."); dot_used = True
            elif ch in ["-", "−", "ー"] and not sign_used:
                buf.append("-"); sign_used = True
            else:
                continue
        if not buf:
            return None
        return float("".join(buf))
    except Exception:
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
    floors: str | None = None
    floors_x3m: str | None = None
    area: float | None = None

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
        "name": ["USER_施設名称","施設名","名称","建物名","name","sheltername"],
        "wing": ["USER_棟名","棟名"],
        "addr": ["USER_住所","住所","address"],
        "lat":  ["緯度","latitude","lat","y"],
        "lon":  ["経度","longitude","lon","lng","x"],
        "floors": ["階数"],
        "floors3": ["3階×ｍ","3階×m","3階xｍ","3階x m"],
        "area": ["USER_面積___","面積"],
    })
    if not all(k in cols for k in ["name","lat","lon"]):
        raise ValueError("避難所CSVの列名を解決できませんでした。『避難所名/名称/施設名』『緯度』『経度』が必要です。")
    name_c = cols["name"]
    wing_c = cols.get("wing")
    addr_c = cols.get("addr")
    lat_c  = cols["lat"]
    lon_c  = cols["lon"]
    floors_c = cols.get("floors")
    floors3_c = cols.get("floors3")
    area_c = cols.get("area")
    shelters = []
    for _, row in df.iterrows():
        lat = try_float(row[lat_c]); lon = try_float(row[lon_c])
        if lat is None or lon is None:
            continue
        name_val = str(row[name_c])
        if wing_c and str(row.get(wing_c, "")).strip():
            name_val = f"{name_val} {str(row[wing_c]).strip()}"
        floors_val = str(row[floors_c]) if floors_c and (row.get(floors_c) is not None) else None
        floors3_val = str(row[floors3_c]) if floors3_c and (row.get(floors3_c) is not None) else None
        area_val = try_float(row[area_c]) if area_c else None
        shelters.append(Shelter(
            name=name_val,
            address=str(row[addr_c]) if addr_c else "",
            lat=lat,
            lon=lon,
            floors=floors_val,
            floors_x3m=floors3_val,
            area=area_val
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
    st.header("年齢・移動手段")
    # 年齢
    age_selected = st.selectbox("年齢", list(pd.unique(load_csv(mobility_path)[detect_columns(load_csv(mobility_path), {
        "age": ["年齢区分","年齢層","年齢","age"]
    })["age"]].astype(str))) if False else None)
    # 上の一行はダミー（動的再読込を避けるため）。実際の age_selected は下で再設定します。

    # 手段選択（ピクトグラムボタン）
    if "transport_select" not in st.session_state:
        st.session_state["transport_select"] = None
    st.write("移動手段")
    t1, t2, t3 = st.columns(3)
    current_t = st.session_state.get("transport_select")
    with t1:
        if st.button("🚶‍♂️\n歩く", use_container_width=True, key="btn_walk_sidebar", type=("primary" if current_t == "歩く" else "secondary")):
            st.session_state["transport_select"] = "歩く"
            st.session_state["act_override"] = "歩く"
            st.rerun()
    with t2:
        if st.button("🏃‍♀️\n走る", use_container_width=True, key="btn_run_sidebar", type=("primary" if current_t == "走る" else "secondary")):
            st.session_state["transport_select"] = "走る"
            st.session_state["act_override"] = "走る"
            st.rerun()
    with t3:
        if st.button("🚲\n自転車", use_container_width=True, key="btn_bike_sidebar", type=("primary" if current_t == "自転車" else "secondary")):
            st.session_state["transport_select"] = "自転車"
            st.session_state["act_override"] = "自転車"
            st.rerun()
    if current_t:
        st.caption(f"選択中の移動手段: {current_t}")
    st.divider()
    st.header("到達圏の選択")
    if "reach_band" not in st.session_state:
        st.session_state["reach_band"] = "10分"
    st.session_state["reach_band"] = st.radio("到達圏", ["5分","10分","15分"], index=(1 if st.session_state["reach_band"]=="10分" else (0 if st.session_state["reach_band"]=="5分" else 2)), horizontal=True)

    st.divider()
    st.header("現在地設定")
    if "lat_input" not in st.session_state:
        st.session_state["lat_input"] = 34.685000   # 緯度（デフォルト）
    if "lon_input" not in st.session_state:
        st.session_state["lon_input"] = 137.718987  # 経度（デフォルト）
    default_lat = st.number_input("緯度", value=st.session_state["lat_input"], step=0.000001, format="%.6f", key="lat_input")
    default_lon = st.number_input("経度", value=st.session_state["lon_input"], step=0.000001, format="%.6f", key="lon_input")

    def set_demo_location():
        # デモ用の固定位置
        st.session_state["lat_input"] = 34.685000
        st.session_state["lon_input"] = 137.718987

    st.button("現在地を入力する", use_container_width=True, on_click=set_demo_location)

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

# サイドバー年齢を実データで再設定（上のダミー抑止）
with st.sidebar:
    age_selected = st.selectbox("年齢", ages, key="age_select")

# サイドバーの手段ボタンからの上書きを適用
def map_transport_to_act(label: str, available: list[str]) -> str | None:
    if not label:
        return None
    cands = []
    n = normalize_colname(label)
    if n in ("歩く","あるく"):
        cands = ["徒歩","歩く","walking","walk"]
    elif n in ("走る","はしる"):
        cands = ["走る","ラン","ランニング","run","running","jog"]
    elif n in ("自転車","じてんしゃ"):
        cands = ["自転車","bicycle","bike"]
    else:
        cands = [label]
    norm_av = {a: normalize_colname(str(a)) for a in available}
    for cand in cands:
        cn = normalize_colname(cand)
        for orig, norm in norm_av.items():
            if cn == norm or cn in norm or norm in cn:
                return str(orig)
    return None

act_selected = None
if st.session_state.get("act_override"):
    mapped = map_transport_to_act(st.session_state.get("act_override"), acts)
    if mapped:
        act_selected = mapped
if not act_selected:
    act_selected = acts[0] if acts else ""

# 入力の自動補完（未選択時のデフォルト設定）
transport_selected = st.session_state.get("transport_select")
if not transport_selected:
    st.session_state["transport_select"] = "歩く"
    st.session_state["act_override"] = "歩く"

dist5_km, dist10_km, dist15_km = extract_timeband_distances(
    mob_df, age_col, act_col, col_5, col_10, col_15, age_selected, act_selected
)

# None安全化（CSV未整合時でも地図を描画できるよう0で代替）
dist5_km = float(dist5_km) if isinstance(dist5_km, (int, float)) and math.isfinite(dist5_km) else 0.0
dist10_km = float(dist10_km) if isinstance(dist10_km, (int, float)) and math.isfinite(dist10_km) else 0.0
dist15_km = float(dist15_km) if isinstance(dist15_km, (int, float)) and math.isfinite(dist15_km) else 0.0

# 選択された到達圏（km）
reach_label = st.session_state.get("reach_band", "10分")
reach_km = dist10_km
if reach_label == "5分":
    reach_km = dist5_km
elif reach_label == "15分":
    reach_km = dist15_km

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
    popup_html = f"<b>{s.name}</b><br>{s.address or ''}"
    extra = []
    if s.floors:
        extra.append(f"階数: {s.floors}")
    if s.floors_x3m:
        extra.append(f"階数×3: {s.floors_x3m}")
    if s.area is not None and isinstance(s.area, (int, float)) and math.isfinite(s.area):
        extra.append(f"面積: {int(round(s.area))}")
    if extra:
        popup_html += "<br>" + " / ".join(extra)
    folium.Marker([s.lat, s.lon],
                  popup=folium.Popup(popup_html, max_width=350),
                  tooltip=s.name,
                  icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)

# 選択した到達圏内の標高上位3件を抽出（地図描画は先に行う）
@st.cache_data(show_spinner=False)
def fetch_elevations(locations: list[tuple[float, float]]) -> list[float|None]:
    if not locations:
        return []
    try:
        url = "https://api.open-elevation.com/api/v1/lookup"
        payload = {"locations": [{"latitude": lat, "longitude": lon} for lat, lon in locations]}
        r = requests.post(url, json=payload, timeout=15)
        if r.status_code != 200:
            return [None]*len(locations)
        data = r.json()
        results = data.get("results", [])
        out = []
        for i in range(len(locations)):
            h = None
            if i < len(results):
                h = try_float(results[i].get("elevation"))
            out.append(h)
        return out
    except Exception:
        return [None]*len(locations)

center_lat, center_lon = float(default_lat), float(default_lon)
within = []
for s in shelters:
    d_km = haversine_km(center_lat, center_lon, s.lat, s.lon)
    # reach_km が未取得または0の場合は全候補から選ぶ（フォールバック）
    if (reach_km is None) or (try_float(reach_km) == 0):
        within.append((s, d_km))
    else:
        if d_km <= float(reach_km):
            within.append((s, d_km))

top3_elev = []
if within:
    locs = [(s.lat, s.lon) for s, _ in within]
    hs = fetch_elevations(locs)
    rows = []
    for i, (s, d_km) in enumerate(within):
        h = hs[i] if i < len(hs) else None
        # 最高避難地点の階高寄与 = (階数 - 1) * 3 [m]
        floors_contrib_m = 0.0
        floors_num = parse_number_from_text(s.floors) if s.floors is not None else None
        if floors_num is not None:
            floors_contrib_m = max(0.0, (floors_num - 1.0) * 3.0)
        else:
            # 階数×3 が与えられている場合は 3[m] を減算して近似
            raw = parse_number_from_text(s.floors_x3m) if s.floors_x3m is not None else None
            if raw is not None:
                floors_contrib_m = max(0.0, raw - 3.0)
        base_h = h if isinstance(h, (int, float)) else 0.0
        highest = base_h + floors_contrib_m
        rows.append((s, d_km, h, floors_contrib_m, highest))
    # 最高避難高度で降順ソート
    rows.sort(key=lambda x: (-x[4]))
    top3_elev = rows[:3]
    # 地図上で強調
    for idx, (s, d_km, h, floors3_m, highest) in enumerate(top3_elev, start=1):
        folium.CircleMarker(
            location=[s.lat, s.lon],
            radius=8,
            color="#e74c3c",
            fill=True,
            fill_color="#e74c3c",
            fill_opacity=0.9,
            tooltip=f"TOP{idx}: {s.name} / {('%.2f km' % d_km)} / 最高避難高度 {int(round(highest))} m",
        ).add_to(m)
        # 大きな順位ラベルを重ねて表示
        folium.Marker(
            [s.lat, s.lon],
            icon=folium.DivIcon(html=(
                f"<div style=\"font-size:24px;font-weight:900;color:#e74c3c;"
                f"background:rgba(255,255,255,0.85);border:2px solid #e74c3c;"
                f"border-radius:14px;padding:2px 8px;line-height:1;\">{idx}</div>"
            )),
            tooltip=f"順位 {idx}"
        ).add_to(m)

# 地図描画（先に描画してブロッキングを避ける）
st_folium(m, height=650, width=None)

# ==== 地図下：Top3を大きく表示して選択 → 詳細/開始 ====
if "sim_started" not in st.session_state:
    st.session_state["sim_started"] = False
if "start_time" not in st.session_state:
    st.session_state["start_time"] = None
if "selected_top3_name" not in st.session_state:
    st.session_state["selected_top3_name"] = None
if "pred_seconds" not in st.session_state:
    st.session_state["pred_seconds"] = None
if "results" not in st.session_state:
    st.session_state["results"] = []

# 10分到達圏内 上位3（標高順）を表で表示
if top3_elev:
    st.subheader(f"到達圏内（{reach_label}） 最高避難高度 上位3施設")
    disp = []
    for idx, (s, d_km, h, floors3_m, highest) in enumerate(top3_elev, start=1):
        disp.append({
            "順位": idx,
            "避難所名": s.name,
            "距離(km)": round(d_km, 2),
            "標高(m)": (int(round(h)) if isinstance(h, (int, float)) else None),
            "階数×3(m)": int(round(floors3_m)) if isinstance(floors3_m, (int, float)) else None,
            "最高避難高度(m)": int(round(highest)),
            "住所": s.address,
        })
    st.dataframe(pd.DataFrame(disp), use_container_width=True)

if top3_elev:
    st.markdown(f"<div style='font-size:20px; font-weight:700; margin-top:12px;'>最高避難高度Top3（{reach_label}圏内）から選択</div>", unsafe_allow_html=True)
    elev_names = [s.name for (s, _, _, _, _) in top3_elev]
    options = []
    for i, (s, d_km, h, floors3_m, highest) in enumerate(top3_elev, start=1):
        options.append(f"{i}. {s.name}（{d_km:.2f} km / 最高避難高度 {int(round(highest))} m）")
    default_idx = 0
    if st.session_state.get("selected_top3_name") in elev_names:
        default_idx = elev_names.index(st.session_state.get("selected_top3_name"))
    choice = st.radio(" ", options=options, index=default_idx, label_visibility="collapsed")
    chosen_idx = options.index(choice)
    chosen_s, chosen_d_km, _h, _f3, _hi = top3_elev[chosen_idx]
    st.session_state["selected_top3_name"] = chosen_s.name

    # 距離・時間（OSRM優先、失敗時は直線距離÷4.5km/h）
    route = osrm_route_foot(default_lat, default_lon, chosen_s.lat, chosen_s.lon)
    if route and ("distance" in route) and ("duration" in route):
        d_km = route["distance"]/1000.0
        t_min = route["duration"]/60.0
    else:
        d_km = chosen_d_km
        t_min = (d_km/4.5)*60.0

    c1, c2 = st.columns(2)
    c1.metric("距離", f"{d_km:.2f} km")
    c2.metric("所要時間(予測)", f"{int(round(t_min))} 分")

    # 操作ボタン
    col_a, col_b = st.columns(2)
    with col_a:
        start_clicked = st.button("移動開始", disabled=st.session_state["sim_started"]) 
    with col_b:
        stop_clicked = st.button("到着（計測終了）", disabled=not st.session_state["sim_started"]) 

    if start_clicked and not st.session_state["sim_started"]:
        st.session_state["sim_started"] = True
        st.session_state["start_time"] = time.time()
        st.session_state["pred_seconds"] = int(round(t_min))*60
        st.success("計測を開始しました。安全に移動してください。")

    if stop_clicked and st.session_state["sim_started"]:
        elapsed_sec = int(time.time() - (st.session_state["start_time"] or time.time()))
        pred = st.session_state.get("pred_seconds") or 0
        over = elapsed_sec - pred
        name = st.session_state.get("selected_top3_name")
        if over > 0:
            st.error(f"到着: {elapsed_sec//60}分{elapsed_sec%60:02d}秒｜制限超過 +{over//60}分{over%60:02d}秒")
        else:
            st.success(f"到着: {elapsed_sec//60}分{elapsed_sec%60:02d}秒")
        # ログ保存
        st.session_state["results"].append({
            "目的地": name,
            "予想(分)": int(round(t_min)),
            "実測(秒)": elapsed_sec,
            "超過(秒)": over if over > 0 else 0,
        })
        st.session_state["sim_started"] = False
        st.session_state["start_time"] = None

    if st.session_state["sim_started"]:
        elapsed_sec = int(time.time() - (st.session_state["start_time"] or time.time()))
        pred = st.session_state.get("pred_seconds") or 0
        remain = max(0, pred - elapsed_sec)
        c1, c2, c3 = st.columns(3)
        c1.metric("経過", f"{elapsed_sec//60}:{elapsed_sec%60:02d}")
        c2.metric("残り", f"{remain//60}:{remain%60:02d}")
        c3.metric("制限", f"{pred//60}:{pred%60:02d}")
        over = elapsed_sec - pred
        if over > 0:
            st.markdown(f":red[制限超過 +{over//60}分{over%60:02d}秒]")

    if st.session_state.get("results"):
        st.subheader("移動ログ")
        df_hist = pd.DataFrame(st.session_state["results"]).copy()
        if not df_hist.empty:
            df_hist["実測(分)"] = (df_hist["実測(秒)"]//60).astype(int)
            st.dataframe(df_hist, use_container_width=True)
        # 浜松市に提供するボタン
        if st.button("浜松市に提供する", use_container_width=True):
            st.success("ありがとうございます！")
else:
    st.info("到達圏内（10分）で標高Top3を算出できませんでした。入力や到達距離をご確認ください。")
