"""Streamlit web UI for F1 race predictions.

Launch with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without pip install -e .
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from f1predict.config import get_config
from f1predict.data.schedule import get_schedule
from f1predict.i18n import t
from f1predict.pipeline import F1Pipeline
from f1predict import cache as C


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    lang = st.radio("🌐 Language / Lingua", ["EN", "IT"], horizontal=True)
    lang = lang.lower()

    st.title(t("title", lang))
    st.caption(t("subtitle", lang))
    st.divider()

    # Season selector
    current_year = pd.Timestamp.now().year
    year_options = list(range(2023, current_year + 2))
    selected_year = st.selectbox(t("select_year", lang), year_options,
                                 index=year_options.index(current_year))

    # GP selector (loaded from schedule)
    @st.cache_data(ttl=3600)
    def _load_schedule(year: int):
        try:
            schedule = get_schedule(year)
            return schedule[["RoundNumber", "EventName", "Location", "Country"]].copy()
        except Exception:
            return pd.DataFrame()

    schedule_df = _load_schedule(selected_year)
    if not schedule_df.empty:
        gp_options = [
            f"R{int(row['RoundNumber']):02d} — {row['EventName']}"
            for _, row in schedule_df.iterrows()
        ]
        selected_gp_str = st.selectbox(t("select_gp", lang), gp_options)
        selected_round = int(selected_gp_str.split("—")[0].strip()[1:])
    else:
        st.warning("Could not load schedule.")
        selected_round = 1

    st.divider()
    predict_btn = st.button(t("predict_btn", lang), type="primary", use_container_width=True)
    train_btn = st.button(t("train_btn", lang), use_container_width=True)

# ── Main area ─────────────────────────────────────────────────────────────────

st.header(t("title", lang))

cfg = get_config()

# ── Training ─────────────────────────────────────────────────────────────────

if train_btn:
    with st.spinner(t("training", lang, cfg["training"]["seasons"])):
        log_container = st.empty()
        log_lines: list[str] = []

        def _cb(msg: str):
            log_lines.append(msg)
            log_container.text("\n".join(log_lines[-20:]))

        pipe = F1Pipeline(cfg)
        try:
            pipe.train(cfg["training"]["seasons"], progress_cb=_cb)
            st.success(t("done", lang))
        except Exception as exc:
            st.error(str(exc))

# ── Prediction ────────────────────────────────────────────────────────────────

if predict_btn:
    if not C.models_exist(cfg):
        st.warning(t("no_model", lang))
        st.stop()

    progress_placeholder = st.empty()
    progress_placeholder.info(t("loading", lang))
    log_lines2: list[str] = []

    def _cb2(msg: str):
        log_lines2.append(msg)
        progress_placeholder.info("\n".join(log_lines2[-5:]))

    try:
        pipe = F1Pipeline(cfg)
        result = pipe.predict_race(selected_year, selected_round, progress_cb=_cb2)
        progress_placeholder.empty()
    except Exception as exc:
        progress_placeholder.error(str(exc))
        st.stop()

    # ── Event info ────────────────────────────────────────────────────────────
    ev = result["event"]
    col1, col2, col3 = st.columns(3)
    col1.metric(t("circuit", lang), ev.get("name", "?"))
    col2.metric(t("race_date", lang), str(ev.get("race_date", "?"))[:10])
    col3.metric("Mode", result.get("data_mode", "?").replace("_", " ").title())
    st.caption(t("data_source", lang))

    race_df = result["race"]
    quali_df = result.get("quali", pd.DataFrame())

    tab_race, tab_quali = st.tabs([t("tab_race", lang), t("tab_quali", lang)])

    # ── Race tab ─────────────────────────────────────────────────────────────
    with tab_race:
        col_left, col_right = st.columns([1.2, 1])

        with col_left:
            st.subheader("📋 " + t("tab_race", lang))
            display_race = race_df[[
                "predicted_pos", "driver_name", "team",
                "p_win", "p_podium", "p_top5",
            ]].copy()
            display_race.columns = [
                t("col_pred_pos", lang), t("col_driver", lang), t("col_team", lang),
                t("col_p_win", lang), t("col_p_podium", lang), t("col_p_top5", lang),
            ]
            for pct_col in [t("col_p_win", lang), t("col_p_podium", lang), t("col_p_top5", lang)]:
                display_race[pct_col] = display_race[pct_col].map(lambda x: f"{x:.1%}")

            st.dataframe(display_race, use_container_width=True, hide_index=True)

        with col_right:
            st.subheader("🥇 " + t("chart_title", lang))
            # Top 10 by podium probability
            chart_data = race_df.nlargest(10, "p_podium")[
                ["driver_name", "p_win", "p_podium", "p_top5"]
            ].copy()
            fig = go.Figure()
            fig.add_bar(
                y=chart_data["driver_name"],
                x=chart_data["p_win"],
                name=t("col_p_win", lang),
                orientation="h",
                marker_color="#e8c84f",
            )
            fig.add_bar(
                y=chart_data["driver_name"],
                x=chart_data["p_podium"] - chart_data["p_win"],
                name="P2-P3",
                orientation="h",
                marker_color="#c0c0c0",
            )
            fig.update_layout(
                barmode="stack",
                xaxis_tickformat=".0%",
                xaxis_title="Probability",
                yaxis={"categoryorder": "total ascending"},
                height=380,
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
                legend={"orientation": "h", "yanchor": "bottom", "y": 1.02},
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Qualifying tab ────────────────────────────────────────────────────────
    with tab_quali:
        if quali_df is not None and not quali_df.empty:
            st.subheader("🏁 " + t("tab_quali", lang))
            display_quali = quali_df[["predicted_quali_pos", "driver_name", "team"]].copy()
            display_quali.columns = [
                t("col_quali_pos", lang), t("col_driver", lang), t("col_team", lang)
            ]
            st.dataframe(display_quali, use_container_width=True, hide_index=True)
        else:
            st.info(t("warning_no_fp", lang))

else:
    st.info(
        "👈 Select a season and GP from the sidebar, then click **" +
        t("predict_btn", lang) + "**."
    )
    if not C.models_exist(cfg):
        st.warning(t("no_model", lang))
