import numpy as np
import scipy.stats as stats
import streamlit as st

# ==========================================
# 🎨 網頁前端介面設定
# ==========================================
st.set_page_config(page_title="職業足球 xG 狀態地利融合預測模型", layout="wide")
st.title("⚽ 職業足球 xG 狀態×地利融合預測系統")
st.markdown("---")

# 建立網頁左右兩大欄位 (左邊輸入數據，右邊看結果)
col_data, col_odds = st.columns([2, 1])

with col_data:
    st.header("📋 步驟 1：輸入球隊數據")
    
    # 1. 聯賽與核心設定
    st.subheader("🌐 聯賽與加權設定")
    c1, c2 = st.columns(2)
    with c1:
        league_goals = st.number_input("聯賽場均總進球數 (AVG)", min_value=1.5, max_value=4.5, value=3.24, step=0.01)
    with c2:
        weight_xg = st.slider("短期 xG 權重占比 (推薦 0.6)", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
    
    # 2. 主隊數據輸入
    st.subheader("🏠 主隊數據 (Home Team)")
    h_col1, h_col2 = st.columns(2)
    with h_col1:
        st.markdown("**近 5 場單場真實進球 (Scored)**")
        h_sc_str = st.text_input("格式：用逗號隔開", "1, 4, 3, 2, 1", key="h_sc")
        st.markdown("**近 5 場單場創造 xG**")
        h_xg_str = st.text_input("格式：用逗號隔開", "1.35, 2.25, 1.98, 1.48, 2.20", key="h_xg")
    with h_col2:
        st.markdown("**近 5 場單場真實失球 (Conceded)**")
        h_co_str = st.text_input("格式：用逗號隔開", "1, 0, 0, 1, 0", key="h_co")
        st.markdown("**近 5 場單場允許 xGA**")
        h_xga_str = st.text_input("格式：用逗號隔開", "1.10, 0.85, 1.20, 1.55, 0.90", key="h_xga")
    
    h1, h2 = st.columns(2)
    with h1:
        h_season_sc = st.number_input("🏠 賽季主場場均進球", value=3.00, step=0.01)
    with h2:
        h_season_co = st.number_input("🏠 賽季主場場均失球", value=0.25, step=0.01)
        
    st.markdown("---")
    
    # 3. 客隊數據輸入 (補全原先被截斷的部分)
    st.subheader("🚌 客隊數據 (Away Team)")
    a_col1, a_col2 = st.columns(2)
    with a_col1:
        st.markdown("**近 5 場單場真實進球 (Scored)**")
        a_sc_str = st.text_input("格式：用逗號隔開", "0, 2, 3, 1, 4", key="a_sc")
        st.markdown("**近 5 場單場創造 xG**")
        a_xg_str = st.text_input("格式：用逗號隔開", "0.88, 1.35, 1.72, 1.15, 2.01", key="a_xg")
    with a_col2:
        st.markdown("**近 5 場單場真實失球 (Conceded)**")
        a_co_str = st.text_input("格式：用逗號隔開", "1, 2, 1, 0, 1", key="a_co")
        st.markdown("**近 5 場單場允許 xGA**")
        a_xga_str = st.text_input("格式：用逗號隔開", "1.40, 1.10, 1.35, 0.95, 1.20", key="a_xga")
        
    a1, a2 = st.columns(2)
    with a1:
        a_season_sc = st.number_input("🚌 賽季客場場均進球", value=1.50, step=0.01)
    with a2:
        a_season_co = st.number_input("🚌 賽季客場場均失球", value=1.20, step=0.01)

# ==========================================
# 📊 步驟 2：數據解析與卜瓦松預測模型運算
# ==========================================
def parse_list(text_str):
    try:
        return [float(x.strip()) for x in text_str.split(",")]
    except Exception:
        return [0.0]

# 解析所有輸入字串為數字陣列
h_sc_list = parse_list(h_sc_str)
h_xg_list = parse_list(h_xg_str)
h_co_list = parse_list(h_co_str)
h_xga_list = parse_list(h_xga_str)

a_sc_list = parse_list(a_sc_str)
a_xg_list = parse_list(a_xg_str)
a_co_list = parse_list(a_co_str)
a_xga_list = parse_list(a_xga_str)

# 1. 計算近期狀態期望值 (結合真實進球與 xG 權重)
h_recent_attack = weight_xg * np.mean(h_xg_list) + (1 - weight_xg) * np.mean(h_sc_list)
h_recent_defense = weight_xg * np.mean(h_xga_list) + (1 - weight_xg) * np.mean(h_co_list)

a_recent_attack = weight_xg * np.mean(a_xg_list) + (1 - weight_xg) * np.mean(a_sc_list)
a_recent_defense = weight_xg * np.mean(a_xga_list) + (1 - weight_xg) * np.mean(a_co_list)

# 2. 融合「短期狀態」與「長期地利」
# 主隊進球期望值 = (主隊近期進攻 + 主隊賽季主場進球) / 2
# 客隊進球期望值 = (客隊近期進攻 + 客隊賽季客場進球) / 2
# 同時導入防守因子調整 (相乘並除以聯賽基調)
home_lambda = ((h_recent_attack + h_season_sc) / 2) * ((a_recent_defense + a_season_co) / 2) / (league_goals / 2)
away_lambda = ((a_recent_attack + a_season_sc) / 2) * ((h_recent_defense + h_season_co) / 2) / (league_goals / 2)

# 避免運算出現負數或 0
home_lambda = max(home_lambda, 0.01)
away_lambda = max(away_lambda, 0.01)

# 3. 計算卜瓦松機率分佈 (計算單隊 0-10 球的機率)
max_goals = 11
home_probs = [stats.poisson.pmf(i, home_lambda) for i in range(max_goals)]
away_probs = [stats.poisson.pmf(i, away_lambda) for i in range(max_goals)]

# 4. 計算勝負平矩陣機率
home_win = 0.0
draw = 0.0
away_win = 0.0

for h in range(max_goals):
    for a in range(max_goals):
        p = home_probs[h] * away_probs[a]
        if h > a:
            home_win += p
        elif h == a:
            draw += p
        else:
            away_win += p

# 歸一化確保總和為 1
total_p = home_win + draw + away_win
home_win /= total_p
draw /= total_p
away_win /= total_p

# ==========================================
# 📈 步驟 3：右側結果呈現
# ==========================================
with col_odds:
    st.header("🎯 預測分析結果")
    st.markdown("---")
    
    # 顯示雙方最終期望進球數
    st.metric(label="🏠 主隊最終預期進球 (Lambda)", value=f"{home_lambda:.2f}")
    st.metric(label="🚌 客隊最終預期進球 (Lambda)", value=f"{away_lambda:.2f}")
    
    st.markdown("---")
    st.subheader("📊 賽果勝率預估")
    
    # 用進度條或文字美化勝率
    st.write(f"🏠 **主勝勝率**: `{home_win*100:.1f}%`")
    st.progress(float(home_win))
    
    st.write(f"🤝 **和局機率**: `{draw*100:.1f}%`")
    st.progress(float(draw))
    
    st.write(f"🚌 **客勝勝率**: `{away_win*100:.1f}%`")
    st.progress(float(away_win))
    
    st.markdown("---")
    st.caption("💡 模型已融合：近5場真實進球、近5場xG走勢、賽季主客場地利以及對手防守實力。")
