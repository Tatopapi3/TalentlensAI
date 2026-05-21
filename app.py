import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

from core.logger import log_prediction, log_outcome, get_predictions
from core.drift import detect_drift, load_training_stats
from models.match_scorer import FEATURE_COLS as MATCH_FEATURES
from models.trend_predictor import FEATURE_COLS as TREND_FEATURES

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
DATA_DIR  = os.path.join(os.path.dirname(__file__), 'data')


@st.cache_resource
def load_models():
    match_model  = tf.keras.models.load_model(f'{MODEL_DIR}/match_model.h5')
    match_scaler = pickle.load(open(f'{MODEL_DIR}/match_scaler.pkl', 'rb'))
    trend_model  = tf.keras.models.load_model(f'{MODEL_DIR}/trend_model.h5')
    trend_scaler = pickle.load(open(f'{MODEL_DIR}/trend_scaler.pkl', 'rb'))
    meta         = json.load(open(f'{DATA_DIR}/meta.json'))
    return match_model, match_scaler, trend_model, trend_scaler, meta


st.set_page_config(page_title="TalentLensAI", page_icon="🔍", layout="wide")
st.markdown("""
<style>
.big-score  { font-size: 4.5rem; font-weight: 800; text-align: center; margin: 0; line-height: 1; }
.score-sub  { text-align: center; color: #888; font-size: 1rem; margin-top: 0.4rem; }
</style>
""", unsafe_allow_html=True)

st.title("🔍 TalentLensAI")
st.caption("ML-powered resume-to-role scoring + job market trend analysis with observability")

if not os.path.exists(f'{MODEL_DIR}/match_model.h5'):
    st.error("Models not found. Run `python train.py` first.")
    st.stop()

match_model, match_scaler, trend_model, trend_scaler, meta = load_models()
ALL_SKILLS   = meta['all_skills']
ROLES        = meta['roles']
EDUCATION    = meta['education']
TREND_SKILLS = meta['skills']
EDU_LEVEL    = {e: i for i, e in enumerate(EDUCATION)}

tab_match, tab_trends, tab_obs = st.tabs([
    "🎯 Resume Scorer", "📈 Market Trends", "📊 Observability"
])


# ── Tab 1: Resume Match Scorer ─────────────────────────────────────────────────
with tab_match:
    st.subheader("How well does your resume match this role?")

    col_res, col_job = st.columns(2)

    with col_res:
        st.markdown("**Your Resume**")
        resume_skills = st.multiselect("Your Skills", ALL_SKILLS,
                                        default=['Python', 'SQL', 'React'])
        resume_exp    = st.slider("Years of Experience", 0, 15, 3)
        resume_edu    = st.selectbox("Education Level", EDUCATION, index=2)

    with col_job:
        st.markdown("**Job Posting**")
        role            = st.selectbox("Target Role", ROLES)
        required_skills = st.multiselect("Required Skills", ALL_SKILLS,
                                          default=['Python', 'AWS', 'Docker'])
        req_exp         = st.slider("Min Experience Required (yrs)", 1, 10, 3)
        req_edu         = st.selectbox("Min Education Required", EDUCATION[:4], index=2)

    if st.button("⚡ Score My Match", type="primary", use_container_width=True):
        resume_set   = set(resume_skills)
        required_set = set(required_skills)

        overlap   = len(resume_set & required_set) / max(len(required_set), 1)
        missing   = len(required_set - resume_set)
        extra     = len(resume_set - required_set)
        exp_ratio = min(resume_exp / max(req_exp, 1), 2.0)
        edu_match = 1 if EDU_LEVEL.get(resume_edu, 0) >= EDU_LEVEL.get(req_edu, 2) else 0
        role_idx  = ROLES.index(role)

        inputs = {
            'skill_overlap':  overlap,
            'missing_skills': missing,
            'extra_skills':   min(extra, 10),
            'exp_ratio':      exp_ratio,
            'edu_match':      edu_match,
            'req_exp':        req_exp,
            'resume_exp':     resume_exp,
            'role_idx':       role_idx,
        }

        X     = np.array([[inputs[f] for f in MATCH_FEATURES]])
        X     = match_scaler.transform(X)
        score = float(match_model.predict(X, verbose=0)[0][0])
        pct   = round(score * 100, 1)

        drift     = detect_drift(inputs)
        any_drift = any(v['drifted'] for v in drift.values())

        pred_id = log_prediction('match_scorer', inputs, score,
                                 label=f"{role} | {pct}%")
        st.session_state['last_pred_id'] = pred_id

        color = "#22c55e" if pct >= 70 else "#f59e0b" if pct >= 45 else "#ef4444"
        st.markdown(f'<p class="big-score" style="color:{color}">{pct}%</p>',
                    unsafe_allow_html=True)
        st.markdown('<p class="score-sub">Match Score</p>', unsafe_allow_html=True)
        st.progress(score)

        m1, m2, m3 = st.columns(3)
        m1.metric("Skill Overlap", f"{overlap*100:.0f}%")
        m2.metric("Missing Skills", missing)
        m3.metric("Bonus Skills", extra)

        missing_list = sorted(required_set - resume_set)
        if missing_list:
            st.warning(f"**Skills to add:** {', '.join(missing_list)}")

        if pct >= 70:
            st.success("Strong match — you meet the core requirements.")
        elif pct >= 45:
            st.warning("Partial match — a few gaps to close.")
        else:
            st.error("Weak match — significant skill gaps for this role.")

        if any_drift:
            with st.expander("⚠️ Input drift detected"):
                st.caption("Some of your inputs are outside the range the model was trained on.")
                for feat, info in drift.items():
                    if info['drifted']:
                        st.write(f"**{feat}**: value `{info['value']}` — "
                                 f"train avg `{info['train_mean']} ± {info['train_std']}` "
                                 f"(z-score: {info['z_score']})")

        st.divider()
        st.markdown("**Did you get an interview?** *(helps calibrate the model)*")
        c1, c2, _ = st.columns([1, 1, 6])
        with c1:
            if st.button("✅ Got interview"):
                log_outcome(pred_id, 1.0)
                st.success("Logged!")
        with c2:
            if st.button("❌ No interview"):
                log_outcome(pred_id, 0.0)
                st.info("Logged.")


# ── Tab 2: Market Trends ───────────────────────────────────────────────────────
with tab_trends:
    st.subheader("Which skills are growing in demand?")

    selected = st.multiselect("Compare skills", TREND_SKILLS,
                               default=['Python', 'LLMs', 'RAG', 'PyTorch', 'Java'])

    if selected and st.button("📈 Predict Demand Trends", type="primary"):
        from data.generate_data import SKILL_TRENDS as SKILL_MAP

        quarters = [
            (y, q) for y in range(2020, 2027)
            for q in range(1, 5)
            if not (y == 2026 and q > 2)
        ]
        chart_data = {}

        for skill in selected:
            if skill not in SKILL_MAP:
                continue
            skill_idx = list(SKILL_MAP.keys()).index(skill)
            demands   = []
            for i, (year, quarter) in enumerate(quarters):
                X = np.array([[skill_idx, i, year, quarter]])
                X = trend_scaler.transform(X)
                d = float(trend_model.predict(X, verbose=0)[0][0])
                demands.append(round(d, 2))
            chart_data[skill] = demands
            log_prediction('trend_predictor', {'skill': skill, 'skill_idx': skill_idx},
                           demands[-1], label=skill)

        labels   = [f"{y} Q{q}" for y, q in quarters]
        df_chart = pd.DataFrame(chart_data, index=labels)

        st.markdown("#### Demand Score Over Time (0–10)")
        st.line_chart(df_chart, height=380)

        st.markdown("#### Current Demand Ranking")
        current  = {s: chart_data[s][-1] for s in selected if s in chart_data}
        rank_df  = (pd.DataFrame(list(current.items()), columns=['Skill', 'Demand Score'])
                    .sort_values('Demand Score', ascending=False))
        st.bar_chart(rank_df.set_index('Skill'))


# ── Tab 3: Observability Dashboard ────────────────────────────────────────────
with tab_obs:
    st.subheader("📊 Model Observability")
    st.caption("Every prediction logged — drift detection, calibration, outcome tracking.")

    if st.button("🔄 Refresh"):
        st.rerun()

    all_preds = get_predictions(limit=300)

    if not all_preds:
        st.info("No predictions yet. Use the Resume Scorer or Market Trends tabs to generate data.")
    else:
        df = pd.DataFrame([{
            'model':      p['model'],
            'created_at': p['created_at'],
            'prediction': p['prediction'],
            'actual':     p['actual'],
            'label':      p['label'],
        } for p in all_preds])

        match_df = df[df['model'] == 'match_scorer']
        trend_df = df[df['model'] == 'trend_predictor']

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Predictions", len(df))
        m2.metric("Resume Scores", len(match_df))
        m3.metric("Trend Queries", len(trend_df))
        rated = match_df[match_df['actual'].notna()]
        m4.metric("Outcomes Logged", len(rated))

        if not match_df.empty:
            st.markdown("#### Match Score Distribution")
            st.bar_chart(
                match_df['prediction']
                .apply(lambda x: round(x, 1))
                .value_counts().sort_index()
            )

        if len(rated) >= 3:
            st.markdown("#### Calibration — Predicted Score vs Actual Interview Rate")
            cal = (rated[['prediction', 'actual']].copy()
                   .assign(bucket=lambda d: (d['prediction'] * 10).astype(int) / 10)
                   .groupby('bucket')['actual'].mean()
                   .reset_index()
                   .rename(columns={'bucket': 'Predicted Score', 'actual': 'Interview Rate'}))
            st.line_chart(cal.set_index('Predicted Score'))

        st.markdown("#### Feature Drift — Training Baseline vs Recent Inputs")
        train_stats   = load_training_stats()
        recent_inputs = [p['inputs'] for p in all_preds if p['model'] == 'match_scorer'][:50]
        if train_stats and recent_inputs:
            recent_df = pd.DataFrame(recent_inputs)
            rows = []
            for feat in train_stats:
                if feat in recent_df.columns:
                    z = abs((recent_df[feat].mean() - train_stats[feat]['mean'])
                            / max(train_stats[feat]['std'], 1e-6))
                    rows.append({
                        'Feature':     feat,
                        'Train Mean':  round(train_stats[feat]['mean'], 3),
                        'Recent Mean': round(recent_df[feat].mean(), 3),
                        'Z-Score':     round(z, 2),
                        'Drifted':     '⚠️ Yes' if z > 2 else '✅ No',
                    })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.markdown("#### Recent Predictions Log")
        st.dataframe(
            df[['created_at', 'model', 'prediction', 'actual', 'label']].head(20),
            use_container_width=True,
        )
