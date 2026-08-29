import streamlit as st
import numpy as np
import scipy.stats as stats

# ==========================================
# 🎨 手機端極簡單頁排版設定
# ==========================================
st.set_page_config(page_title="足球 xG 預測系統", layout="centered")

st.title("⚽ 足球 xG 狀態×地利融合預測系統")
st.markdown("💡 *提示：模型已融合：近5場真實進球、近5場xG走勢、賽季主客場地利以及對手防守實力。*")

# 1. 聯賽背景基準設定
st.subheader("🌐 聯賽與加權基礎設定")
c1, c2 = st.columns(2)
with c1:
    league_goals = st.number_input("聯賽場均總進球數 (AVG)", min_value=1.5, max_value=4.5, value=3.24, step=0.01, key="unique_lg_g")
with c2:
    weight_xg = st.slider("短期 xG 權重占比 (推薦 0.6)", min_value=0.0, max_value=1.0, value=0.6, step=0.1, key="unique_w_xg")

st.markdown("---")

# 2. 用 Expander 把主客隊數據收納起來，確保每個 key 都是全球唯一
with st.expander("🏠 點擊展開/收起：輸入【主隊】數據", expanded=True):
    h_sc_str = st.text_input("📋 近 5 場單場【真實進球】(用英文逗號隔開)", "1, 4, 3, 2, 1", key="k_h_sc")
    h_xg_str = st.text_input("📋 近 5 場單場【創造 xG】(用英文逗號隔開)", "1.35, 2.25, 1.98, 1.48, 2.20", key="k_h_xg")
    h_co_str = st.text_input("📋 近 5 場單場【真實失球】(用英文逗號隔開)", "1, 0, 0, 1, 0", key="k_h_co")
    h_xga_str = st.text_input("📋 近 5 場單場【允許 xGA】(用英文逗號隔開)", "1.10, 0.85, 1.20, 1.55, 0.90", key="k_h_xga")
    h_season_sc = st.number_input("🏟️ 賽季【主場】場均進球", value=3.00, step=0.01, key="k_h_s_sc")
    h_season_co = st.number_input("🏟️ 賽季【主場】場均失球", value=0.25, step=0.01, key="k_h_s_co")

with st.expander("🚌 點擊展開/收起：輸入【客隊】數據", expanded=False):
    a_sc_str = st.text_input("📋 近 5 場單場【真實進球】(用英文逗號隔開)", "0, 2, 3, 1, 4", key="k_a_sc")
    a_xg_str = st.text_input("📋 近 5 場單場【創造 xG】(用英文逗號隔開)", "0.88, 1.35, 1.72, 1.15, 2.01", key="k_a_xg")
    a_co_str = st.text_input("📋 近 5 場單場【真實失球】(用英文逗號隔開)", "2, 1, 1, 5, 1", key="k_a_co")
    a_xga_str = st.text_input("📋 近 5 場單場【允許 xGA】(用英文逗號隔開)", "1.95, 2.10, 1.45, 2.80, 1.30", key="k_a_xga")
    a_season_sc = st.number_input("🛣️ 賽季【客場】場均進球", value=0.88, step=0.01, key="k_a_s_sc")
    a_season_co = st.number_input("🛣️ 賽季【客場】場均失球", value=1.50, step=0.01, key="k_a_s_co")

# 3. 獨立露出的莊家賠率區
st.markdown("---")
st.subheader("💰 步驟 2：輸入莊家即時賠率")
o1, o2, o3 = st.columns(3)
with o1:
    bookie_h = st.number_input("🏠 獨贏 - 主勝賠率", value=1.29, step=0.01, key="k_b_h")
with o2:
    bookie_d = st.number_input("🤝 獨贏 - 平局賠率", value=5.10, step=0.01, key="k_b_d")
with o3:
    bookie_a = st.number_input("🚌 獨贏 - 客勝賠率", value=6.00, step=0.01, key="k_b_a")

o4, o5 = st.columns(2)
with o4:
    bookie_over = st.number_input("🔥 2.5 大球賠率", value=1.60, step=0.01, key="k_b_o")
with o5:
    bookie_under = st.number_input("🔒 2.5 小球賠率", value=2.19, step=0.01, key="k_b_u")

st.markdown("---")
run_analysis = st.button("🚀 開始深度融合精算分析", type="primary", use_container_width=True)

# ==========================================
# 🧠 核心數學計算與精美輸出
# ==========================================
if run_analysis:
    try:
        h_xg = [float(x.strip()) for x in h_xg_str.split(",")]
        h_sc = [float(x.strip()) for x in h_sc_str.split(",")]
        h_xga = [float(x.strip()) for x in h_xga_str.split(",")]
        h_co = [float(x.strip()) for x in h_co_str.split(",")]

        a_xg = [float(x.strip()) for x in a_xg_str.split(",")]
        a_sc = [float(x.strip()) for x in a_sc_str.split(",")]
        a_xga = [float(x.strip()) for x in a_xga_str.split(",")]
        a_co = [float(x.strip()) for x in a_co_str.split(",")]

        h_xg_r, h_sc_r = np.mean(h_xg), np.mean(h_sc)
        h_xga_r, h_co_r = np.mean(h_xga), np.mean(h_co)
        a_xg_r, a_sc_r = np.mean(a_xg), np.mean(a_sc)
        a_xga_r, a_co_r = np.mean(a_xga), np.mean(a_co)

        w_actual = 1.0 - weight_xg
        state_h_att = (h_xg_r * weight_xg) + (h_sc_r * w_actual)
        state_h_def = (h_xga_r * weight_xg) + (h_co_r * w_actual)
        state_a_att = (a_xg_r * weight_xg) + (a_sc_r * w_actual)
        state_a_def = (a_xga_r * weight_xg) + (a_co_r * w_actual)

        base = league_goals / 2
        factor_h_att = h_season_sc / base
        factor_h_def = h_season_co / base
        factor_a_att = a_season_sc / base
        factor_a_def = a_season_co / base

        lambda_home = state_h_att * factor_a_def * base
        lambda_away = state_a_att * factor_h_def * base

        max_goals = 10
        home_poisson = [stats.poisson.pmf(i, lambda_home) for i in range(max_goals)]
        away_poisson = [stats.poisson.pmf(i, lambda_away) for i in range(max_goals)]
        score_matrix = np.outer(home_poisson, away_poisson)

        prob_h = np.sum(np.tril(score_matrix, -1))
        prob_d = np.sum(np.diag(score_matrix))
        prob_a = np.sum(np.triu(score_matrix, 1))

        prob_under = sum(score_matrix[h, a] for h in range(max_goals) for a in range(max_goals) if h + a < 2.5)
        prob_over = 1.0 - prob_under

        def de_margin(h, d, a):
            r_h, r_d, r_a = 1 / h, 1 / d, 1 / a
            margin = (r_h + r_d + r_a) - 1.0
            return r_h / (r_h + r_d + r_a), r_d / (r_h + r_d + r_a), r_a / (r_h + r_d + r_a), margin

        true_h, true_d, true_a, match_margin = de_margin(bookie_h, bookie_d, bookie_a)
        true_o = (1 / bookie_over) / ((1 / bookie_over) + (1 / bookie_under))

        # ==========================================
        # 🎯 輸出分析報告
        # ==========================================
        st.markdown("---")
        st.markdown("## 🎯 預測分析結果")
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("🏠 主隊最終預期進球 (Lambda)", f"{lambda_home:.2f}")
        with col_res2:
            st.metric("🚌 客隊最終預期進球 (Lambda)", f"{lambda_away:.2f}")
            
        st.info(f"💰 莊家抽水率: {match_margin*100:.1f}% | 2.5大球模型預估率: {prob_over*100:.1f}%")

        st.markdown("---")
        st.markdown("### 🔍 全盤口市場錯位機會深度分析")
        markets = [
            {"盤口": "🏠 獨贏 - 主勝", "m_p": prob_h, "t_p": true_h, "odds": bookie_h},
            {"盤口": "🤝 獨贏 - 平局", "m_p": prob_d, "t_p": true_d, "odds": bookie_d},
            {"盤口": "🚌 獨贏 - 客勝", "m_p": prob_a, "t_p": true_a, "odds": bookie_a},
            {"盤口": "🔥 總分 - 2.5大", "m_p": prob_over, "t_p": true_o, "odds": bookie_over},
            {"盤口": "🔒 總分 - 2.5小", "m_p": prob_under, "t_p": 1 - true_o, "odds": bookie_under},
        ]

        for m in markets:
            edge = m["m_p"] * m["odds"] - 1
            if edge >= 0.05:
                st.error(f"{m['盤口']} ｜ 模型: {m['m_p']*100:.1f}% vs 莊家: {m['t_p']*100:.1f}% ==> 🚨【黃金錯位！優勢: +{edge*100:.1f}%】")
            else:
                st.write(f"{m['盤口']} ｜ 模型: {m['m_p']*100:.1f}% vs 莊家: {m['t_p']*100:.1f}% —— 無明顯優勢")

    except Exception as e:
        st.error(f"數據格式錯誤！")
