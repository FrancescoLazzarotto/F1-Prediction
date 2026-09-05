"""F1 Race Predictor — web interface.

Launch with:
    streamlit run app/streamlit_app.py
or:
    f1predict serve
"""

from __future__ import annotations

import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
# Allow `streamlit run app/streamlit_app.py` without installing the package.
for path in (str(_APP_DIR), str(_APP_DIR.parent / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd
import streamlit as st

import components as ui
import theme
from f1predict import __version__
from f1predict import cache as C
from f1predict.config import get_config
from f1predict.data import repository as repo
from f1predict.data.schedule import event_status, get_next_event, season_events
from f1predict.i18n import LANGUAGES, grid_source_label, translator
from f1predict.models import registry
from f1predict.pipeline import F1Pipeline

st.set_page_config(
    page_title="F1 Race Predictor",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)
theme.inject()

CURRENT_YEAR = datetime.now(tz=timezone.utc).year
FIRST_YEAR = 2021


# ── Cached data access ────────────────────────────────────────────────────────
# Streamlit reruns the whole script on every interaction, so anything that
# touches the network or trains a model has to be memoised or the app would
# re-download the season on each click.

@st.cache_resource(show_spinner=False)
def get_pipeline() -> F1Pipeline:
    return F1Pipeline(get_config())


@st.cache_data(ttl=3600, show_spinner=False)
def load_events(year: int) -> list[dict]:
    return [e.to_dict() for e in season_events(year)]


@st.cache_data(ttl=1800, show_spinner=False)
def load_next_event() -> dict | None:
    try:
        return get_next_event().to_dict()
    except LookupError:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def load_standings(year: int, constructors: bool) -> pd.DataFrame:
    cfg = get_config()
    return (
        repo.constructor_standings(year, cfg=cfg) if constructors
        else repo.driver_standings(year, cfg=cfg)
    )


@st.cache_data(ttl=1800, show_spinner=False)
def load_status(year: int, round_num: int) -> str:
    return event_status(year, round_num)


def cache_key(year: int, round_num: int, simulations: int) -> str:
    """Identity of a prediction, so a rerun with the same inputs is free."""
    return f"{year}:{round_num}:{simulations}"


# ── Sidebar ───────────────────────────────────────────────────────────────────

def sidebar() -> dict:
    """Render controls and return the current selection."""
    with st.sidebar:
        st.markdown("### 🏎️ F1 Predictor")
        st.caption(f"v{__version__}")

        language = st.radio(
            "Language", list(LANGUAGES), horizontal=True,
            format_func=lambda code: LANGUAGES[code], key="language",
        )
        _ = translator(language)
        st.divider()

        next_event = load_next_event()
        default_year = next_event["year"] if next_event else CURRENT_YEAR
        years = list(range(FIRST_YEAR, CURRENT_YEAR + 2))
        year = st.selectbox(
            _("season"), years,
            index=years.index(default_year) if default_year in years else len(years) - 2,
        )

        events = load_events(year)
        if not events:
            st.error(f"No calendar available for {year}.")
            st.stop()

        default_round = (
            next_event["round"]
            if next_event and next_event["year"] == year
            else events[-1]["round"]
        )
        options = [e["round"] for e in events]
        labels = {e["round"]: f"R{e['round']:02d} · {e['name']}" for e in events}
        round_num = st.selectbox(
            _("grand_prix"), options,
            index=options.index(default_round) if default_round in options else 0,
            format_func=lambda r: labels[r],
        )

        status = load_status(year, round_num)
        st.caption(f"Weekend status: **{status.replace('_', ' ')}**")

        st.divider()
        predict_clicked = st.button(
            f"🏁 {_('predict')}", type="primary", width="stretch"
        )

        with st.expander(f"⚙️ {_('settings')}"):
            simulations = st.select_slider(
                _("simulations"), options=[2_000, 5_000, 20_000, 50_000, 100_000],
                value=20_000,
                help="More simulations mean smoother probabilities and a slower run.",
            )
            st.caption(
                "Probabilities come from replaying the race this many times with "
                "randomised pace and retirements."
            )

        st.divider()
        ready = registry.models_ready(get_config())
        if ready:
            st.success("Models trained and ready", icon="✅")
        else:
            st.warning("No trained model yet", icon="⚠️")

        train_clicked = st.button(
            f"🧠 {_('train')}", width="stretch",
            disabled=st.session_state.get("training", False),
        )

    return {
        "language": language, "year": year, "round": round_num,
        "predict": predict_clicked, "train": train_clicked,
        "simulations": simulations, "models_ready": ready, "status": status,
        "events": events, "next_event": next_event,
    }


# ── Training ──────────────────────────────────────────────────────────────────

def run_training(_) -> None:
    cfg = get_config()
    st.session_state["training"] = True
    st.markdown(f"### 🧠 {_('train')}")
    st.caption(_("training_on", ", ".join(str(s) for s in cfg.training.seasons)))

    log_box = st.empty()
    progress_bar = st.progress(0.0)
    lines: list[str] = []
    # Feature building dominates the wall clock, so drive the bar off how many
    # races have been assembled rather than off elapsed time.
    total_races = max(len(cfg.training.seasons) * 24, 1)

    def on_progress(message: str) -> None:
        lines.append(message)
        if "Building" in message:
            done = sum(1 for line in lines if "Building" in line)
            progress_bar.progress(min(done / total_races, 0.95))
        log_box.code("\n".join(lines[-12:]), language=None)

    try:
        reports = get_pipeline().train(progress=on_progress)
        progress_bar.progress(1.0)
        st.success(_("done"), icon="✅")

        summary = pd.DataFrame([
            {"Model": name, "Result": report.summary() if report else "not trained"}
            for name, report in reports.items()
        ])
        st.dataframe(summary, hide_index=True, width="stretch")

        race_report = reports.get("race")
        if race_report and race_report.feature_importance:
            st.plotly_chart(
                ui.importance_chart(race_report.top_features(12)),
                width="stretch",
            )
    except Exception as exc:
        st.error(f"Training failed: {exc}")
    finally:
        st.session_state["training"] = False
        # A fresh model invalidates any prediction still held in session state.
        st.session_state.pop("prediction_key", None)
        st.session_state.pop("prediction", None)


# ── Prediction ────────────────────────────────────────────────────────────────

def run_prediction(selection: dict, _):
    """Run the pipeline for the selected event, caching by inputs."""
    key = cache_key(selection["year"], selection["round"], selection["simulations"])
    if st.session_state.get("prediction_key") == key:
        return st.session_state.get("prediction")

    cfg = get_config()
    pipeline = get_pipeline()
    pipeline.cfg = replace(
        cfg, simulation=replace(cfg.simulation, n_simulations=selection["simulations"])
    )

    status_box = st.status(_("loading"), expanded=True)
    try:
        prediction = pipeline.predict_race(
            selection["year"], selection["round"],
            progress=lambda m: status_box.write(m),
        )
        status_box.update(label=_("done"), state="complete", expanded=False)
    except Exception as exc:
        status_box.update(label="Prediction failed", state="error")
        st.error(str(exc))
        return None

    st.session_state["prediction_key"] = key
    st.session_state["prediction"] = prediction
    return prediction


# ── Tabs ──────────────────────────────────────────────────────────────────────

def race_tab(prediction, _) -> None:
    table = prediction.table

    ui.confidence_chip(prediction.confidence, grid_source_label(prediction.grid_source))
    ui.podium_cards(table)

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown(f"#### {_('podium_chart')}")
        st.plotly_chart(ui.podium_probability_chart(table), width="stretch")
    with right:
        st.markdown(f"#### {_('outcome_spread')}")
        st.plotly_chart(ui.outcome_spread_chart(table), width="stretch")

    st.markdown("#### " + _("tab_race"))
    ui.race_table(table, _labels(_))

    with st.expander("📊 Full outcome distribution"):
        st.caption(
            "Each cell is the share of simulated races in which that driver "
            "finished in that position."
        )
        if prediction.simulation is not None:
            st.plotly_chart(
                ui.position_heatmap(prediction.simulation, table),
                width="stretch",
            )

    with st.expander("🔀 Grid to finish"):
        st.plotly_chart(ui.grid_to_finish_chart(table), width="stretch")

    with st.expander(f"🔍 {_('why')}"):
        codes = table["driver_code"].tolist()
        chosen = st.selectbox("Driver", codes, key="explain_driver")
        try:
            contributions = get_pipeline().explain(prediction, chosen)
            st.plotly_chart(
                ui.explanation_chart(contributions, chosen), width="stretch"
            )
        except KeyError:
            st.info("No explanation available for that driver.")

    with st.expander(f"⚔️ {_('vs')}"):
        codes = table["driver_code"].tolist()
        col_a, col_b = st.columns(2)
        driver_a = col_a.selectbox("Driver A", codes, index=0, key="h2h_a")
        driver_b = col_b.selectbox(
            "Driver B", codes, index=min(1, len(codes) - 1), key="h2h_b"
        )
        if prediction.simulation is not None and driver_a != driver_b:
            st.plotly_chart(
                ui.head_to_head(prediction.simulation, table, driver_a, driver_b),
                width="stretch",
            )

    st.download_button(
        "⬇️ Download predictions (CSV)",
        table.to_csv(index=False).encode("utf-8"),
        file_name=(
            f"f1predict_{prediction.event.get('year')}"
            f"_r{prediction.event.get('round'):02d}.csv"
        ),
        mime="text/csv",
    )


def quali_tab(prediction, _) -> None:
    if prediction.quali_table is None or prediction.quali_table.empty:
        if prediction.grid_source == "actual_quali":
            st.info(
                "Qualifying has already run for this event, so the real grid was "
                "used instead of a prediction."
            )
            actual = repo.quali_results(
                prediction.event["year"], prediction.event["round"], get_config()
            )
            if not actual.empty:
                st.markdown("#### Actual qualifying result")
                st.dataframe(
                    actual[["quali_pos", "driver_name", "team", "best_quali_s",
                            "quali_gap_to_pole_s"]],
                    hide_index=True, width="stretch",
                )
        else:
            st.warning(_("no_practice"))
        return

    st.markdown(f"#### {_('tab_quali')}")
    ui.note(
        f"Predicted from {prediction.practice_session or 'practice'} pace: best lap, "
        "long-run stints, theoretical best lap and the gap to each driver's teammate."
    )
    ui.quali_table(prediction.quali_table, _labels(_))


def championship_tab(selection: dict, _) -> None:
    year = selection["year"]
    mode = st.radio(
        "Championship", [_("drivers_title"), _("constructors_title")],
        horizontal=True, key="champ_mode", label_visibility="collapsed",
    )
    constructors = mode == _("constructors_title")

    if st.button("🔮 Simulate the rest of the season", type="primary"):
        st.session_state.pop(f"outlook_{year}_{constructors}", None)

    key = f"outlook_{year}_{constructors}"
    if key not in st.session_state:
        with st.spinner("Simulating remaining rounds…"):
            try:
                st.session_state[key] = get_pipeline().championship_outlook(
                    year, constructors=constructors
                )
            except Exception as exc:
                st.error(str(exc))
                return

    outlook = st.session_state[key]

    ui.stat_strip([
        {"label": _("races_left"), "value": str(outlook.races_remaining),
         "sub": f"{outlook.sprints_remaining} with a sprint"},
        {"label": "Simulated seasons", "value": f"{outlook.n_simulations:,}"},
        {"label": "Favourite",
         "value": str(outlook.table.iloc[0][
             "driver_name" if not constructors else "team"]),
         "sub": f"{outlook.table.iloc[0]['p_title']:.1%} title chance",
         "accent": theme.GOLD},
    ])

    if outlook.is_decided:
        st.success(_("title_decided"), icon="🏆")

    left, right = st.columns([1, 1])
    with left:
        st.markdown(f"#### {_('title_odds')}")
        st.plotly_chart(ui.championship_chart(outlook), width="stretch")
    with right:
        st.markdown(f"#### {_('projected_points')}")
        st.plotly_chart(ui.points_projection_chart(outlook), width="stretch")

    label_col = "team" if constructors else "driver_name"
    view = outlook.table[[
        label_col, "current_points", "p_title", "p_top3",
        "expected_points", "points_p10", "points_p90",
    ]].copy()
    view.columns = [
        _("driver") if not constructors else _("team"), _("current_points"),
        _("title_odds"), "Top 3", _("projected_points"), "P10", "P90",
    ]
    st.dataframe(
        view, hide_index=True, width="stretch",
        column_config={
            _("title_odds"): st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0.0, max_value=1.0
            ),
            "Top 3": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0.0, max_value=1.0
            ),
        },
    )

    st.markdown("#### Current standings")
    st.dataframe(
        load_standings(year, constructors), hide_index=True, width="stretch"
    )


def accuracy_tab(selection: dict, _) -> None:
    year = selection["year"]
    st.markdown(f"#### {_('tab_backtest')}")
    ui.note(
        "Backtesting re-predicts a race that has already happened, using only the "
        "data that existed beforehand, and scores the result against reality."
    )

    col_a, col_b = st.columns([1, 2])
    with col_a:
        scope = st.radio(
            "Scope", ["Single race", "Whole season"], key="bt_scope",
        )
    with col_b:
        rounds = [e["round"] for e in selection["events"]]
        chosen_round = st.selectbox(
            "Round", rounds, index=rounds.index(selection["round"])
            if selection["round"] in rounds else 0, key="bt_round",
            disabled=scope != "Single race",
        )

    if not st.button(f"📐 {_('run_backtest')}", type="primary"):
        return

    pipeline = get_pipeline()
    if scope == "Single race":
        with st.spinner("Scoring…"):
            try:
                result = pipeline.backtest(year, chosen_round)
            except Exception as exc:
                st.error(str(exc))
                return

        metrics = result.metrics
        ui.stat_strip([
            {"label": _("spearman"), "value": f"{metrics.spearman:+.3f}",
             "accent": theme.POSITIVE if metrics.spearman > 0.6 else theme.WARNING},
            {"label": _("mae"), "value": f"{metrics.mae:.2f}", "sub": "positions"},
            {"label": _("winner_hit"), "value": "✓" if metrics.top1 >= 1 else "✗"},
            {"label": _("podium_hit"), "value": f"{metrics.top3:.0%}"},
            {"label": _("within_3"), "value": f"{metrics.within_3:.0%}"},
        ])

        view = result.table[[
            "actual_pos", "predicted_pos", "error", "driver_name", "team",
            "p_win", "p_podium", "status",
        ]].copy()
        view.columns = ["Actual", "Predicted", "Δ", _("driver"), _("team"),
                        _("p_win"), _("p_podium"), "Status"]
        st.dataframe(
            view, hide_index=True, width="stretch",
            column_config={
                "Δ": st.column_config.NumberColumn(format="%+d"),
                _("p_win"): st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0.0, max_value=1.0
                ),
                _("p_podium"): st.column_config.ProgressColumn(
                    format="%.1f%%", min_value=0.0, max_value=1.0
                ),
            },
        )
        return

    progress = st.progress(0.0, text="Backtesting the season…")
    log = st.empty()
    lines: list[str] = []
    total = max(len(selection["events"]), 1)

    def on_progress(message: str) -> None:
        lines.append(message)
        progress.progress(min(len(lines) / total, 0.98), text=message)
        log.code("\n".join(lines[-6:]), language=None)

    try:
        overall, per_race = pipeline.backtest_season(year, progress=on_progress)
    except Exception as exc:
        st.error(str(exc))
        return
    progress.progress(1.0, text=_("done"))
    log.empty()

    ui.stat_strip([
        {"label": _("spearman"), "value": f"{overall.spearman:+.3f}",
         "sub": f"across {len(per_race)} races",
         "accent": theme.POSITIVE if overall.spearman > 0.6 else theme.WARNING},
        {"label": _("mae"), "value": f"{overall.mae:.2f}", "sub": "positions"},
        {"label": _("winner_hit"), "value": f"{overall.top1:.0%}"},
        {"label": _("podium_hit"), "value": f"{overall.top3:.0%}"},
        {"label": _("within_3"), "value": f"{overall.within_3:.0%}"},
    ])
    st.plotly_chart(ui.accuracy_chart(per_race), width="stretch")
    st.dataframe(per_race, hide_index=True, width="stretch")


def calendar_tab(selection: dict, _) -> None:
    events = selection["events"]
    now = datetime.now(tz=timezone.utc)

    rows = []
    for event in events:
        when = pd.to_datetime(event["race_date"], utc=True, errors="coerce") \
            if event["race_date"] else pd.NaT
        rows.append({
            "R": event["round"],
            _("grand_prix"): event["name"],
            _("circuit"): event["location"],
            "Country": event["country"],
            _("race_date"): when,
            "Format": "sprint" if "sprint" in event["format"].lower() else "conventional",
            "Status": "done" if (pd.notna(when) and when < now) else "upcoming",
        })

    st.dataframe(
        pd.DataFrame(rows), hide_index=True, width="stretch", height=760,
        column_config={
            _("race_date"): st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
        },
    )


def model_tab(_) -> None:
    cfg = get_config()
    card = registry.describe(cfg)

    st.markdown("#### Trained models")
    st.caption(f"Feature contract: `{card['feature_signature']}`")

    for name, entry in card["models"].items():
        if entry.get("status") != "ready":
            st.warning(f"**{name}** — {entry.get('status')}")
            continue

        with st.container(border=True):
            st.markdown(f"**{name}** · `{entry['kind']}`")
            cols = st.columns(4)
            cols[0].metric("Samples", f"{entry['n_samples']:,}")
            cols[1].metric("Features", entry["n_features"])
            if entry["cv_mae"] == entry["cv_mae"]:
                cols[2].metric("CV error", f"{entry['cv_mae']:.2f} pos")
            if entry["cv_spearman"] == entry["cv_spearman"]:
                cols[3].metric("Rank corr.", f"{entry['cv_spearman']:+.3f}")
            st.caption(
                f"Trained {entry['trained_at']} on seasons "
                f"{', '.join(str(s) for s in entry['seasons'])}"
            )
            if entry["top_features"]:
                st.plotly_chart(
                    ui.importance_chart(entry["top_features"]),
                    width="stretch", key=f"imp_{name}",
                )
            for line in entry.get("notes", []):
                st.caption(f"· {line}")

    st.markdown("#### Cache")
    usage = C.usage(cfg)
    cols = st.columns(len(usage) + 1)
    for i, (section, size) in enumerate(usage.items()):
        cols[i].metric(section, _human_bytes(size))
    cols[-1].metric("total", _human_bytes(sum(usage.values())))


# ── Helpers ───────────────────────────────────────────────────────────────────

def _labels(_) -> dict[str, str]:
    return {
        "predicted_pos": _("pos"), "driver_name": _("driver"), "team": _("team"),
        "grid_pos": _("grid"), "grid_delta": _("delta"), "p_win": _("p_win"),
        "p_podium": _("p_podium"), "p_points": _("p_points"), "p_dnf": _("p_dnf"),
        "expected_points": _("expected_points"),
        "predicted_quali_pos": _("quali_pos"), "approx_gap_s": _("gap"),
        "fp_best_gap_pct": "FP best gap %", "fp_pace_gap_pct": "FP race pace gap %",
    }


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def event_header(prediction, selection: dict, _) -> None:
    """Context strip above the tabs: circuit, date, grid source, weather."""
    event = prediction.event
    when = str(event.get("race_date", ""))[:16].replace("T", " ")

    ui.stat_strip([
        {"label": _("circuit"), "value": event.get("circuit") or event.get("name", "—"),
         "sub": f"{event.get('locality', '')} · {event.get('country', '')}".strip(" ·"),
         "accent": theme.ACCENT},
        {"label": _("race_date"), "value": when or "—",
         "sub": f"Round {event.get('round', '?')} of {len(selection['events'])}"},
        {"label": _("grid_source"), "value": grid_source_label(prediction.grid_source),
         "sub": f"{_('practice_used')}: {prediction.practice_session or '—'}",
         "accent": theme.CONFIDENCE_COLORS.get(prediction.confidence, theme.BORDER)},
    ])
    ui.weather_strip(prediction.weather)


# ── Page ──────────────────────────────────────────────────────────────────────

def main() -> None:
    selection = sidebar()
    _ = translator(selection["language"])

    ui.hero(_("app_title"), _("app_tagline"))

    if selection["train"]:
        run_training(_)
        return

    if not selection["models_ready"]:
        st.warning(_("no_model"), icon="🧠")
        st.markdown(
            "Training downloads a few seasons of practice timing the first time, "
            "which takes several minutes. Afterwards everything is cached and "
            "predictions are instant."
        )
        return

    if selection["predict"]:
        st.session_state.pop("prediction_key", None)

    prediction = None
    if selection["predict"] or st.session_state.get("prediction_key") == cache_key(
        selection["year"], selection["round"], selection["simulations"]
    ):
        prediction = run_prediction(selection, _)

    tabs = st.tabs([
        f"🏁 {_('tab_race')}", f"⏱️ {_('tab_quali')}", f"🏆 {_('tab_championship')}",
        f"📐 {_('tab_backtest')}", f"📅 {_('tab_calendar')}", f"🧠 {_('tab_model')}",
    ])

    with tabs[0]:
        if prediction is None:
            st.info(_("select_prompt"), icon="👈")
            _upcoming_teaser(selection, _)
        else:
            event_header(prediction, selection, _)
            race_tab(prediction, _)

    with tabs[1]:
        if prediction is None:
            st.info(_("select_prompt"), icon="👈")
        else:
            quali_tab(prediction, _)

    with tabs[2]:
        championship_tab(selection, _)
    with tabs[3]:
        accuracy_tab(selection, _)
    with tabs[4]:
        calendar_tab(selection, _)
    with tabs[5]:
        model_tab(_)


def _upcoming_teaser(selection: dict, _) -> None:
    """Something useful on screen before the first prediction is run."""
    next_event = selection.get("next_event")
    if not next_event:
        return

    when = pd.to_datetime(next_event["race_date"], utc=True, errors="coerce")
    countdown = "—"
    if pd.notna(when):
        delta = when - pd.Timestamp.now(tz="UTC")
        if delta.total_seconds() > 0:
            days = delta.days
            hours = int(delta.total_seconds() % 86400) // 3600
            countdown = f"{days}d {hours}h"

    ui.stat_strip([
        {"label": "Next race", "value": next_event["name"],
         "sub": f"{next_event['location']} · {next_event['country']}",
         "accent": theme.ACCENT},
        {"label": "Lights out in", "value": countdown,
         "sub": str(next_event["race_date"])[:16].replace("T", " ")},
        {"label": "Round", "value": f"{next_event['round']}",
         "sub": next_event["format"]},
    ])

    standings = load_standings(selection["year"], constructors=False)
    if not standings.empty:
        st.markdown(f"#### {_('drivers_title')} · {selection['year']}")
        st.dataframe(
            standings.head(10)[["position", "driver_name", "team", "points", "wins"]],
            hide_index=True, width="stretch",
        )


main()
