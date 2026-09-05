"""Visual language for the web app: palette, CSS, and chart defaults.

Kept separate from the page logic so the look can change without touching the
behaviour, and so every chart inherits the same typography and grid treatment.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# ── Palette ───────────────────────────────────────────────────────────────────

INK = "#0B0D11"
SURFACE = "#14171D"
SURFACE_2 = "#1B1F27"
BORDER = "#2A2F3A"
TEXT = "#E8EAED"
TEXT_DIM = "#9BA1AC"
ACCENT = "#E8002D"
ACCENT_SOFT = "#FF3355"

GOLD = "#FFD700"
SILVER = "#C4C8CE"
BRONZE = "#CD7F32"

POSITIVE = "#2ECC71"
NEGATIVE = "#FF5A5F"
WARNING = "#F5A623"

#: Sequential ramp for probability heatmaps — dark for improbable, hot for likely.
HEAT_SCALE = [
    [0.00, "#14171D"],
    [0.20, "#232838"],
    [0.45, "#3D4A7A"],
    [0.70, "#8C4A6B"],
    [0.88, "#D4453F"],
    [1.00, "#FFB020"],
]

CONFIDENCE_COLORS = {"high": POSITIVE, "medium": WARNING, "low": NEGATIVE}


CSS = f"""
<style>
  /* ── Shell ───────────────────────────────────────────────────────────── */
  .stApp {{
    background:
      radial-gradient(1100px 520px at 12% -12%, #1E2230 0%, transparent 60%),
      radial-gradient(900px 420px at 92% 0%, #2A1220 0%, transparent 55%),
      {INK};
  }}
  .block-container {{ padding-top: 2.2rem; max-width: 1400px; }}
  section[data-testid="stSidebar"] {{
    background: {SURFACE};
    border-right: 1px solid {BORDER};
  }}

  /* ── Masthead ────────────────────────────────────────────────────────── */
  .f1-hero {{
    display: flex; align-items: baseline; gap: .8rem;
    padding: 0 0 .3rem 0; margin-bottom: .2rem;
  }}
  .f1-hero h1 {{
    font-size: 2.45rem; font-weight: 800; letter-spacing: -.03em;
    margin: 0; color: {TEXT};
    font-family: "Segoe UI", -apple-system, system-ui, sans-serif;
  }}
  .f1-hero h1 em {{
    font-style: normal;
    background: linear-gradient(92deg, {ACCENT} 0%, {ACCENT_SOFT} 55%, {WARNING} 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}
  .f1-hero .tag {{ color: {TEXT_DIM}; font-size: .93rem; }}
  .f1-rule {{
    height: 3px; border: 0; margin: .1rem 0 1.5rem 0; border-radius: 2px;
    background: linear-gradient(90deg, {ACCENT} 0%, {WARNING} 35%, transparent 78%);
  }}

  /* ── Stat strip ──────────────────────────────────────────────────────── */
  .f1-stats {{ display: flex; gap: .7rem; flex-wrap: wrap; margin-bottom: 1.3rem; }}
  .f1-stat {{
    flex: 1 1 150px; background: {SURFACE}; border: 1px solid {BORDER};
    border-radius: 12px; padding: .75rem .95rem; position: relative;
    overflow: hidden;
  }}
  .f1-stat::before {{
    content: ""; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: var(--accent, {BORDER});
  }}
  .f1-stat .k {{
    color: {TEXT_DIM}; font-size: .69rem; text-transform: uppercase;
    letter-spacing: .1em; font-weight: 600;
  }}
  .f1-stat .v {{
    color: {TEXT}; font-size: 1.28rem; font-weight: 700; margin-top: .18rem;
    line-height: 1.25;
  }}
  .f1-stat .s {{ color: {TEXT_DIM}; font-size: .74rem; margin-top: .1rem; }}

  /* ── Podium ──────────────────────────────────────────────────────────── */
  .f1-podium {{ display: flex; gap: .8rem; margin: .3rem 0 1.4rem 0; }}
  .f1-card {{
    flex: 1; background: linear-gradient(160deg, {SURFACE_2} 0%, {SURFACE} 100%);
    border: 1px solid {BORDER}; border-radius: 14px; padding: 1rem 1.1rem;
    position: relative; overflow: hidden;
  }}
  .f1-card::after {{
    content: ""; position: absolute; right: -28px; top: -28px;
    width: 96px; height: 96px; border-radius: 50%;
    background: var(--team, {BORDER}); opacity: .16;
  }}
  .f1-card .rank {{
    font-size: .68rem; font-weight: 800; letter-spacing: .14em;
    text-transform: uppercase; color: var(--medal, {TEXT_DIM});
  }}
  .f1-card .name {{
    font-size: 1.22rem; font-weight: 700; color: {TEXT};
    margin: .3rem 0 .1rem 0; line-height: 1.2;
  }}
  .f1-card .team {{ font-size: .82rem; color: {TEXT_DIM}; }}
  .f1-card .odds {{
    margin-top: .7rem; font-size: .78rem; color: {TEXT_DIM};
    display: flex; justify-content: space-between; gap: .5rem;
  }}
  .f1-card .odds b {{ color: {TEXT}; font-size: .95rem; display: block; }}

  /* ── Misc ────────────────────────────────────────────────────────────── */
  .f1-chip {{
    display: inline-block; padding: .16rem .6rem; border-radius: 999px;
    font-size: .72rem; font-weight: 700; letter-spacing: .04em;
    border: 1px solid currentColor;
  }}
  .f1-note {{
    color: {TEXT_DIM}; font-size: .82rem; border-left: 2px solid {BORDER};
    padding-left: .7rem; margin: .5rem 0;
  }}
  .stTabs [data-baseweb="tab-list"] {{ gap: .3rem; border-bottom: 1px solid {BORDER}; }}
  .stTabs [data-baseweb="tab"] {{
    height: 2.7rem; padding: 0 1rem; background: transparent;
    border-radius: 8px 8px 0 0; color: {TEXT_DIM}; font-weight: 600;
  }}
  .stTabs [aria-selected="true"] {{ background: {SURFACE}; color: {TEXT}; }}
  div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; }}
  #MainMenu, footer {{ visibility: hidden; }}
</style>
"""


def inject() -> None:
    """Apply the stylesheet. Safe to call on every rerun."""
    st.markdown(CSS, unsafe_allow_html=True)


def style_figure(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    """Apply the shared chart treatment to a Plotly figure."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Segoe UI, system-ui, sans-serif", "color": TEXT, "size": 12},
        height=height,
        margin={"l": 8, "r": 12, "t": 28, "b": 8},
        showlegend=legend,
        legend={
            "orientation": "h", "yanchor": "bottom", "y": 1.02,
            "xanchor": "right", "x": 1, "font": {"size": 11},
        },
        hoverlabel={"bgcolor": SURFACE_2, "bordercolor": BORDER,
                    "font": {"color": TEXT, "size": 12}},
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER)
    return fig


def medal_color(position: int) -> str:
    return {1: GOLD, 2: SILVER, 3: BRONZE}.get(position, TEXT_DIM)
