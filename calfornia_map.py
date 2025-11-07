# California Housing 説明 + 緯度経度×住宅価格（Foliumマップ & HeatMap）/ Streamlit
# 依存: streamlit, scikit-learn, pandas, numpy, streamlit-folium, folium

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.cm as cm
import matplotlib.colors as colors

from sklearn.datasets import fetch_california_housing

st.set_page_config(page_title="California Housing 地図可視化", layout="wide")

# ---------------------------
# 0) データセットの説明（大学生向け）
# ---------------------------
st.markdown(
    """
    <h3 style="font-size:22px; margin-bottom:10px;">
    🏠 California Housing：緯度・経度 × 住宅価格の地図可視化（雑草研・システム研統計ゼミ2025年11月）
    </h3>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    **California Housing データセット** は、米カリフォルニア州の **18,940 地区**について、
    地域の属性と **地区の中央値住宅価格（MedHouseVal, 単位は ×100,000 USD）** をまとめたデータです。
    
    **主な列（説明変数）**：
    - `MedInc`：世帯中央値所得（$10,000 単位）
    - `HouseAge`：住宅の築年数（中央値）
    - `AveRooms`：平均部屋数（世帯あたり）
    - `AveBedrms`：平均寝室数（世帯あたり）
    - `Population`：人口
    - `AveOccup`：平均居住者数（世帯あたり）
    - `Latitude`（緯度）, `Longitude`（経度）
    
    **目的変数**：
    - `MedHouseVal`：地区の **中央値住宅価格（×100,000 USD）**
    
    ここでは、**緯度・経度の位置に住宅価格を重ねて**地理的傾向を直感的に可視化します。  
    表示方法は **サークルマーカー** と **HeatMap（密度重み付き）** を切り替えられます。
    """
)

# ---------------------------
# 1) データ読み込み & 前処理
# ---------------------------
cal = fetch_california_housing(as_frame=True)
X_full = cal.data.copy()
y = cal.target.copy()  # MedHouseVal (×100k USD)

df = X_full.copy()
df["MedHouseVal"] = y

# ---------------------------
# 2) サイドバー（表示数やモード）
# ---------------------------
st.sidebar.header("🧭 表示オプション")
max_show = st.sidebar.slider("表示点数（サンプリング）", min_value=1000, max_value=len(df), value=5000, step=1000)
random_state = st.sidebar.number_input("乱数シード", min_value=0, value=42, step=1)

mode = st.sidebar.radio("表示方法", ["サークルマーカー", "HeatMap（密度重み付き）"], index=0)

# HeatMap 用パラメータ
if mode == "HeatMap（密度重み付き）":
    radius = st.sidebar.slider("HeatMap: 半径（radius）", 3, 30, 12, 1)
    blur   = st.sidebar.slider("HeatMap: ぼかし（blur）", 3, 30, 18, 1)
    max_z  = st.sidebar.slider("HeatMap: max_zoom", 1, 18, 13, 1)

# ---------------------------
# 3) 表（先頭）と統計
# ---------------------------
st.markdown("### 📋 先頭プレビュー")
st.dataframe(df[["Latitude", "Longitude", "MedHouseVal"]].head(10), use_container_width=True)

st.caption(
    f"全 {len(df)} 地区から {max_show} 地区をランダム抽出して地図に描画します "
    "(重さ・負荷のためサンプリングしています)。"
)

# ---------------------------
# 4) Folium マップ描画
# ---------------------------
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap

# サンプリング
df_show = df.sample(max_show, random_state=int(random_state))

# ベースマップ（中心は平均位置）
center_lat = float(df["Latitude"].mean())
center_lon = float(df["Longitude"].mean())
m = folium.Map(location=[center_lat, center_lon], zoom_start=6, tiles="CartoDB positron")

# 価格スケール
vmin, vmax = float(df["MedHouseVal"].min()), float(df["MedHouseVal"].max())

if mode == "サークルマーカー":
    # カラーマップ
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap("viridis")

    for _, r in df_show.iterrows():
        color = colors.to_hex(cmap(norm(float(r["MedHouseVal"]))))
        folium.CircleMarker(
            location=[float(r["Latitude"]), float(r["Longitude"])],
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=f"MedHouseVal: {r['MedHouseVal']:.2f} (×100k USD)",
        ).add_to(m)

    st.caption(f"色スケール（サークル着色）: {vmin:.2f} 〜 {vmax:.2f} (×100k USD)")

else:
    # HeatMap 用データ：重みは価格を 0〜1 に正規化
    denom = (vmax - vmin) if (vmax - vmin) > 0 else 1.0
    heat_data = [
        [float(r["Latitude"]), float(r["Longitude"]), (float(r["MedHouseVal"]) - vmin) / denom]
        for _, r in df_show.iterrows()
    ]
    HeatMap(
        heat_data,
        radius=radius,
        blur=blur,
        max_zoom=max_z,
        min_opacity=0.2,
        max_val=1.0,
    ).add_to(m)
    st.caption("HeatMap の重み：MedHouseVal を 0〜1 に正規化（高価格ほど高強度）")

# マップ表示
st.markdown("### 🗺️ 地図表示")
st_folium(m, height=620, use_container_width=True)

# ---------------------------
# 5) 参考：価格の基本統計
# ---------------------------
with st.expander("📈 価格（MedHouseVal）の基本統計"):
    st.write(df["MedHouseVal"].describe().to_frame().T)


