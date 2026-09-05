"""Reusable UI pieces: stat strips, podium cards, and every chart.

Each function takes plain DataFrames and returns either a Plotly figure or
writes directly to Streamlit, so the page module stays a thin arrangement of
these blocks.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from f1predict.constants import team_color
from theme import (
    ACCENT,
    BORDER,
    CONFIDENCE_COLORS,
    HEAT_SCALE,
    POSITIVE,
    TEXT,
    TEXT_DIM,
    WARNING,
    medal_color,
    style_figure,
)

# ── Layout blocks ─────────────────────────────────────────────────────────────

def hero(title: str, tagline: str) -> None:
    """Masthead with the accented product name."""
    head, tail = (*title.split(" ", 1), "")[:2]
    st.markdown(
        f'<div class="f1-hero"><h1><em>{html.escape(head)}</em> '
        f'{html.escape(tail)}</h1><span class="tag">{html.escape(tagline)}</span></div>'
        '<hr class="f1-rule">',
        unsafe_allow_html=True,
    )


def stat_strip(stats: list[dict]) -> None:
    """A row of compact metric cards.

    Each entry takes ``label``, ``value`` and optionally ``sub`` and ``accent``.
    """
    cards = []
    for stat in stats:
        accent = stat.get("accent", BORDER)
        sub = stat.get("sub", "")
        cards.append(
            f'<div class="f1-stat" style="--accent:{accent}">'
            f'<div class="k">{html.escape(str(stat["label"]))}</div>'
            f'<div class="v">{html.escape(str(stat["value"]))}</div>'
            + (f'<div class="s">{html.escape(str(sub))}</div>' if sub else "")
            + "</div>"
        )
    st.markdown(f'<div class="f1-stats">{"".join(cards)}</div>', unsafe_allow_html=True)


def podium_cards(table: pd.DataFrame) -> None:
    """The predicted top three, coloured by constructor."""
    if table.empty:
        return

    labels = ("Predicted winner", "Second", "Third")
    cards = []
    for i, (_, row) in enumerate(table.head(3).iterrows()):
        colour = team_color(row.get("team", ""))
        cards.append(
            f'<div class="f1-card" style="--team:{colour}; --medal:{medal_color(i + 1)}">'
            f'<div class="rank">{labels[i]}</div>'
            f'<div class="name">{html.escape(str(row.get("driver_name", "")))}</div>'
            f'<div class="team">{html.escape(str(row.get("team", "")))}</div>'
            '<div class="odds">'
            f'<span>Win<b>{row.get("p_win", 0):.0%}</b></span>'
            f'<span>Podium<b>{row.get("p_podium", 0):.0%}</b></span>'
            f'<span>Points<b>{row.get("expected_points", 0):.0f}</b></span>'
            "</div></div>"
        )
    st.markdown(f'<div class="f1-podium">{"".join(cards)}</div>', unsafe_allow_html=True)


def confidence_chip(level: str, description: str) -> None:
    colour = CONFIDENCE_COLORS.get(level, TEXT_DIM)
    st.markdown(
        f'<span class="f1-chip" style="color:{colour}">{html.escape(level.upper())}'
        f"</span> <span style='color:{TEXT_DIM};font-size:.85rem'>"
        f"{html.escape(description)}</span>",
        unsafe_allow_html=True,
    )


def note(text: str) -> None:
    st.markdown(f'<div class="f1-note">{html.escape(text)}</div>', unsafe_allow_html=True)


# ── Charts ────────────────────────────────────────────────────────────────────

def podium_probability_chart(table: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Stacked bars splitting each driver's podium chance into win / P2-P3."""
    data = table.head(top_n).iloc[::-1]
    colours = [team_color(t) for t in data["team"]]

    fig = go.Figure()
    fig.add_bar(
        y=data["driver_name"], x=data["p_win"], orientation="h", name="Win",
        marker={"color": colours, "line": {"color": TEXT, "width": 1}},
        hovertemplate="%{y}<br>Win %{x:.1%}<extra></extra>",
    )
    fig.add_bar(
        y=data["driver_name"], x=(data["p_podium"] - data["p_win"]).clip(lower=0),
        orientation="h", name="P2 or P3",
        marker={"color": colours, "opacity": 0.42},
        hovertemplate="%{y}<br>P2-P3 %{x:.1%}<extra></extra>",
    )
    fig.update_layout(barmode="stack", xaxis={"tickformat": ".0%", "title": ""})
    return style_figure(fig, height=max(320, 30 * len(data)))


def outcome_spread_chart(table: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """Expected finish with a P10-P90 whisker, showing how open each fight is."""
    data = table.head(top_n).iloc[::-1]
    colours = [team_color(t) for t in data["team"]]

    fig = go.Figure()
    for _, row in data.iterrows():
        colour = team_color(row["team"])
        fig.add_trace(go.Scatter(
            x=[row["pos_p10"], row["pos_p90"]], y=[row["driver_name"]] * 2,
            mode="lines", line={"color": colour, "width": 7},
            opacity=0.35, showlegend=False, hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=data["expected_pos"], y=data["driver_name"], mode="markers",
        marker={"color": colours, "size": 13, "line": {"color": TEXT, "width": 1.5}},
        name="Expected finish",
        customdata=np.stack([data["pos_p10"], data["pos_p90"]], axis=-1),
        hovertemplate=(
            "%{y}<br>Expected P%{x:.1f}"
            "<br>Range P%{customdata[0]:.0f}-P%{customdata[1]:.0f}<extra></extra>"
        ),
    ))
    fig.update_layout(xaxis={"title": "Finishing position", "autorange": "reversed"})
    return style_figure(fig, height=max(320, 30 * len(data)), legend=False)


def position_heatmap(simulation, table: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """P(finishing in each position) for every driver — the full distribution."""
    distribution = simulation.position_distribution(max_position=min(top_n, 20))
    order = table["driver_code"].tolist()[:top_n]
    distribution = distribution.reindex(order).dropna(how="all")

    names = dict(zip(table["driver_code"], table["driver_name"], strict=False))
    labels = [names.get(code, code) for code in distribution.index]

    fig = go.Figure(go.Heatmap(
        z=distribution.to_numpy(),
        x=[f"P{p}" for p in distribution.columns],
        y=labels,
        colorscale=HEAT_SCALE, zmin=0,
        colorbar={"title": {"text": "P", "side": "right"}, "tickformat": ".0%",
                  "thickness": 12, "outlinewidth": 0},
        hovertemplate="%{y}<br>%{x}: %{z:.1%}<extra></extra>",
    ))
    fig.update_layout(yaxis={"autorange": "reversed"}, xaxis={"side": "top"})
    return style_figure(fig, height=max(360, 26 * len(labels)), legend=False)


def grid_to_finish_chart(table: pd.DataFrame) -> go.Figure:
    """Slope chart from starting grid to predicted finish."""
    fig = go.Figure()
    for _, row in table.iterrows():
        colour = team_color(row["team"])
        gained = row["grid_delta"] > 0
        fig.add_trace(go.Scatter(
            x=["Grid", "Predicted finish"],
            y=[row["grid_pos"], row["predicted_pos"]],
            mode="lines+markers",
            line={"color": colour, "width": 2.5,
                  "dash": "solid" if gained else "dot"},
            marker={"size": 8, "line": {"color": TEXT, "width": 1}},
            name=str(row["driver_code"]),
            hovertemplate=(
                f"<b>{row['driver_name']}</b><br>Grid P{int(row['grid_pos'])}"
                f" → P{int(row['predicted_pos'])}"
                f" ({row['grid_delta']:+d})<extra></extra>"
            ),
        ))
    fig.update_layout(
        yaxis={"title": "Position", "autorange": "reversed"},
        xaxis={"title": ""},
    )
    return style_figure(fig, height=520, legend=False)


def championship_chart(outlook, top_n: int = 10) -> go.Figure:
    """Title probability bars with the projected points range beneath."""
    label_col = "driver_name" if "driver_name" in outlook.table.columns else "team"
    data = outlook.table.head(top_n).iloc[::-1]
    colours = [team_color(t) for t in data.get("team", data[label_col])]

    fig = go.Figure(go.Bar(
        y=data[label_col], x=data["p_title"], orientation="h",
        marker={"color": colours, "line": {"color": TEXT, "width": 1}},
        text=[f"{v:.1%}" for v in data["p_title"]],
        textposition="outside", textfont={"color": TEXT, "size": 11},
        customdata=np.stack([
            data["current_points"], data["expected_points"],
            data["points_p10"], data["points_p90"],
        ], axis=-1),
        hovertemplate=(
            "<b>%{y}</b><br>Title chance %{x:.1%}"
            "<br>Now %{customdata[0]:.0f} pts"
            "<br>Projected %{customdata[1]:.0f} pts"
            " (%{customdata[2]:.0f}–%{customdata[3]:.0f})<extra></extra>"
        ),
    ))
    fig.update_layout(xaxis={"tickformat": ".0%", "range": [0, 1.12], "title": ""})
    return style_figure(fig, height=max(320, 34 * len(data)), legend=False)


def points_projection_chart(outlook, top_n: int = 8) -> go.Figure:
    """Where each contender's final points total is likely to land."""
    label_col = "driver_name" if "driver_name" in outlook.table.columns else "team"
    data = outlook.table.head(top_n)

    fig = go.Figure()
    for i, (_, row) in enumerate(data.iterrows()):
        colour = team_color(row.get("team", row[label_col]))
        fig.add_trace(go.Scatter(
            x=[row["points_p10"], row["points_p90"]], y=[row[label_col]] * 2,
            mode="lines", line={"color": colour, "width": 9},
            opacity=0.4, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[row["current_points"]], y=[row[label_col]], mode="markers",
            marker={"color": TEXT_DIM, "size": 9, "symbol": "line-ns-open",
                    "line": {"width": 2.5, "color": TEXT_DIM}},
            name="Today" if i == 0 else None, showlegend=i == 0,
            hovertemplate="Today: %{x:.0f} pts<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[row["expected_points"]], y=[row[label_col]], mode="markers",
            marker={"color": colour, "size": 13, "line": {"color": TEXT, "width": 1.5}},
            name="Projected" if i == 0 else None, showlegend=i == 0,
            hovertemplate="Projected: %{x:.0f} pts<extra></extra>",
        ))

    fig.update_layout(xaxis={"title": "Championship points"},
                      yaxis={"autorange": "reversed"})
    return style_figure(fig, height=max(300, 40 * len(data)))


def accuracy_chart(per_race: pd.DataFrame) -> go.Figure:
    """Per-race rank correlation across a backtested season."""
    fig = go.Figure()
    colours = [POSITIVE if v >= 0.7 else (WARNING if v >= 0.4 else ACCENT)
               for v in per_race["spearman"]]
    fig.add_bar(
        x=per_race["round"], y=per_race["spearman"],
        marker={"color": colours},
        customdata=np.stack([per_race["race"], per_race["mae"]], axis=-1),
        hovertemplate=(
            "R%{x} %{customdata[0]}<br>Rank correlation %{y:.3f}"
            "<br>Mean error %{customdata[1]:.2f} places<extra></extra>"
        ),
        name="Rank correlation",
    )
    mean = per_race["spearman"].mean()
    fig.add_hline(
        y=mean, line={"color": TEXT_DIM, "dash": "dash", "width": 1.5},
        annotation={"text": f"season mean {mean:.3f}", "font": {"color": TEXT_DIM}},
    )
    fig.update_layout(
        xaxis={"title": "Round", "dtick": 1},
        yaxis={"title": "Spearman ρ", "range": [-0.2, 1.05]},
    )
    return style_figure(fig, height=340, legend=False)


def importance_chart(importances: list[tuple[str, float]]) -> go.Figure:
    """What the race model actually leans on."""
    if not importances:
        return style_figure(go.Figure(), height=200, legend=False)

    names = [n for n, _ in importances][::-1]
    values = [v for _, v in importances][::-1]
    fig = go.Figure(go.Bar(
        y=names, x=values, orientation="h",
        marker={"color": values, "colorscale": HEAT_SCALE, "showscale": False},
        text=[f"{v:.1%}" for v in values], textposition="outside",
        textfont={"color": TEXT_DIM, "size": 11},
        hovertemplate="%{y}<br>%{x:.1%} of model influence<extra></extra>",
    ))
    fig.update_layout(
        xaxis={"tickformat": ".0%", "title": "", "range": [0, max(values) * 1.25]}
    )
    return style_figure(fig, height=max(260, 26 * len(names)), legend=False)


def explanation_chart(contributions: list[dict], driver: str) -> go.Figure:
    """Diverging bars: what helps this driver, and what holds them back."""
    if not contributions:
        return style_figure(go.Figure(), height=200, legend=False)

    ordered = contributions[::-1]
    labels = [c["label"] for c in ordered]
    # Signed by direction so favourable factors read right, unfavourable left.
    values = [c["impact"] * (1 if c["direction"] == "strength" else -1) for c in ordered]
    colours = [POSITIVE if v > 0 else ACCENT for v in values]

    fig = go.Figure(go.Bar(
        y=labels, x=values, orientation="h", marker={"color": colours},
        customdata=np.stack([
            [c["value"] for c in ordered], [c["field_median"] for c in ordered],
        ], axis=-1),
        hovertemplate=(
            "%{y}<br>This driver: %{customdata[0]:.2f}"
            "<br>Field median: %{customdata[1]:.2f}<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line={"color": BORDER, "width": 1})
    fig.update_layout(
        title={"text": f"Why {driver} is rated where they are", "font": {"size": 13}},
        xaxis={"title": "← holds them back    ·    helps them →",
               "showticklabels": False},
    )
    return style_figure(fig, height=max(260, 34 * len(labels)), legend=False)


def head_to_head(simulation, table: pd.DataFrame, a: str, b: str) -> go.Figure:
    """Probability that one driver finishes ahead of another."""
    probability = simulation.head_to_head(a, b)
    names = dict(zip(table["driver_code"], table["driver_name"], strict=False))
    teams = dict(zip(table["driver_code"], table["team"], strict=False))

    fig = go.Figure(go.Bar(
        x=[probability, 1 - probability],
        y=["", ""], orientation="h",
        marker={"color": [team_color(teams.get(a, "")), team_color(teams.get(b, ""))]},
        text=[f"{names.get(a, a)}  {probability:.0%}",
              f"{names.get(b, b)}  {1 - probability:.0%}"],
        textposition="inside", insidetextanchor="middle",
        textfont={"color": TEXT, "size": 14},
        hovertemplate="%{text}<extra></extra>",
    ))
    fig.update_layout(
        barmode="stack",
        xaxis={"range": [0, 1], "showticklabels": False, "showgrid": False},
        yaxis={"showticklabels": False, "showgrid": False},
    )
    return style_figure(fig, height=110, legend=False)


def weather_strip(weather: dict) -> None:
    """Compact weather readout with the forecast source made explicit."""
    rain = float(weather.get("rain_prob", 0.0))
    accent = ACCENT if rain > 0.4 else (WARNING if rain > 0.15 else POSITIVE)
    source = {
        "forecast": "Open-Meteo forecast",
        "archive": "Open-Meteo archive",
        "measured": "Trackside sensors",
        "default": "Seasonal average (no data)",
    }.get(str(weather.get("source", "")), "Open-Meteo")

    stat_strip([
        {"label": "Rain risk", "value": f"{rain:.0%}", "sub": source, "accent": accent},
        {"label": "Air temp", "value": f"{weather.get('temperature', 0):.0f} °C"},
        {"label": "Wind", "value": f"{weather.get('wind_speed', 0):.1f} m/s"},
        {"label": "Humidity", "value": f"{weather.get('humidity', 0):.0f}%"},
    ])


# ── Table styling ─────────────────────────────────────────────────────────────

def race_table(table: pd.DataFrame, labels: dict[str, str]) -> None:
    """The full classification, with probability bars in-cell."""
    columns = [
        "predicted_pos", "driver_name", "team", "grid_pos", "grid_delta",
        "p_win", "p_podium", "p_points", "p_dnf", "expected_points",
    ]
    view = table[[c for c in columns if c in table.columns]].copy()
    view.columns = [labels.get(c, c) for c in view.columns]

    st.dataframe(
        view, hide_index=True, height=min(760, 38 * len(view) + 40),
        column_config={
            labels["predicted_pos"]: st.column_config.NumberColumn(width="small"),
            labels["grid_pos"]: st.column_config.NumberColumn(width="small"),
            labels["grid_delta"]: st.column_config.NumberColumn(
                format="%+d", width="small", help="Places gained or lost from the grid"
            ),
            labels["p_win"]: st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0.0, max_value=1.0
            ),
            labels["p_podium"]: st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0.0, max_value=1.0
            ),
            labels["p_points"]: st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0.0, max_value=1.0
            ),
            labels["p_dnf"]: st.column_config.NumberColumn(format="%.1f%%"),
            labels["expected_points"]: st.column_config.NumberColumn(format="%.1f"),
        },
    )


def quali_table(table: pd.DataFrame, labels: dict[str, str]) -> None:
    columns = ["predicted_quali_pos", "driver_name", "team", "approx_gap_s",
               "fp_best_gap_pct", "fp_pace_gap_pct"]
    view = table[[c for c in columns if c in table.columns]].copy()
    view.columns = [labels.get(c, c) for c in view.columns]
    st.dataframe(view, hide_index=True, height=min(760, 38 * len(view) + 40))
