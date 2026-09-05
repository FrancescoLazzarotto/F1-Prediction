"""Rich renderers shared by the CLI.

Kept out of :mod:`f1predict.cli` so the command functions stay about argument
handling, and so the same tables can be reused by scripts or notebooks.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from f1predict.constants import PODIUM_COLORS
from f1predict.pipeline import BacktestResult, RacePrediction

console = Console()

_POSITION_STYLES = {1: "bold yellow", 2: "bold white", 3: "bold rgb(205,127,50)"}

_CONFIDENCE_STYLE = {"high": "green", "medium": "yellow", "low": "red"}


def position_style(position: int) -> str:
    return _POSITION_STYLES.get(position, "")


def render_race(prediction: RacePrediction, show_probabilities: bool = True) -> None:
    """Print the predicted classification with outcome probabilities."""
    console.print(event_panel(prediction))

    table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
    table.add_column("Pos", justify="right", width=4)
    table.add_column("Grid", justify="right", width=5, style="dim")
    table.add_column("Δ", justify="right", width=4)
    table.add_column("Driver", min_width=18)
    table.add_column("Team", min_width=16, style="dim")
    if show_probabilities:
        table.add_column("Win", justify="right", width=7)
        table.add_column("Podium", justify="right", width=7)
        table.add_column("Points", justify="right", width=7)
        table.add_column("DNF", justify="right", width=6, style="dim")
        table.add_column("Exp.pts", justify="right", width=8)

    for _, row in prediction.table.iterrows():
        position = int(row["predicted_pos"])
        delta = int(row.get("grid_delta", 0))
        cells = [
            str(position),
            str(int(row.get("grid_pos", 0))),
            _delta_text(delta),
            str(row.get("driver_name") or row.get("driver_code", "")),
            str(row.get("team", "")),
        ]
        if show_probabilities:
            cells += [
                _pct(row.get("p_win")), _pct(row.get("p_podium")),
                _pct(row.get("p_points")), _pct(row.get("p_dnf")),
                f"{row.get('expected_points', 0):.1f}",
            ]
        table.add_row(*cells, style=position_style(position))

    console.print(table)


def render_quali(quali: pd.DataFrame, title: str = "Predicted qualifying order") -> None:
    """Print a predicted qualifying classification."""
    if quali is None or quali.empty:
        console.print("[yellow]No qualifying prediction available.[/yellow]")
        return

    table = Table(title=title, show_header=True, header_style="bold magenta",
                  box=None, title_justify="left")
    table.add_column("Pos", justify="right", width=4)
    table.add_column("Driver", min_width=18)
    table.add_column("Team", min_width=16, style="dim")
    table.add_column("Gap", justify="right", width=8, style="dim")

    for _, row in quali.iterrows():
        position = int(row["predicted_quali_pos"])
        gap = row.get("approx_gap_s", 0.0)
        table.add_row(
            str(position),
            str(row.get("driver_name") or row.get("driver_code", "")),
            str(row.get("team", "")),
            "pole" if position == 1 else f"+{gap:.3f}s",
            style=position_style(position),
        )
    console.print(table)


def render_backtest(result: BacktestResult) -> None:
    """Print a scored prediction next to the real classification."""
    metrics = result.metrics
    header = Table.grid(padding=(0, 3))
    header.add_row(
        _metric("Rank correlation", f"{metrics.spearman:+.3f}"),
        _metric("Mean error", f"{metrics.mae:.2f} pos"),
        _metric("Winner", "hit" if metrics.top1 >= 1 else "miss"),
        _metric("Podium", f"{metrics.top3:.0%}"),
        _metric("Within 3", f"{metrics.within_3:.0%}"),
    )
    console.print(Panel(
        header,
        title=f"[bold]{result.event.get('name', 'Race')} {result.event.get('year', '')}[/bold]",
        border_style="magenta",
    ))
    if result.in_sample:
        console.print(
            "[yellow]Note:[/yellow] [dim]this season is in the model's training set, so "
            "these figures are optimistic. Run [/dim][bold]f1predict info[/bold][dim] for "
            "the cross-validated out-of-sample scores.[/dim]"
        )

    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("Actual", justify="right", width=7)
    table.add_column("Pred", justify="right", width=6)
    table.add_column("Δ", justify="right", width=5)
    table.add_column("Driver", min_width=18)
    table.add_column("Team", min_width=16, style="dim")
    table.add_column("P(win)", justify="right", width=8, style="dim")
    table.add_column("Status", min_width=10, style="dim")

    for _, row in result.table.iterrows():
        error = int(row["error"])
        table.add_row(
            str(int(row["actual_pos"])), str(int(row["predicted_pos"])),
            _error_text(error),
            str(row.get("driver_name") or row.get("driver_code", "")),
            str(row.get("team", "")),
            _pct(row.get("p_win")),
            str(row.get("status", "")),
            style=position_style(int(row["actual_pos"])),
        )
    console.print(table)


def render_championship(outlook, title: str, top_n: int = 12) -> None:
    """Print title probabilities from a season simulation."""
    label_col = "driver_name" if "driver_name" in outlook.table.columns else "team"

    table = Table(
        title=(
            f"{title} — {outlook.races_remaining} round(s) left, "
            f"{outlook.n_simulations:,} simulated seasons"
        ),
        show_header=True, header_style="bold cyan", box=None, title_justify="left",
    )
    table.add_column("#", justify="right", width=3)
    table.add_column("Name", min_width=20)
    table.add_column("Pts", justify="right", width=6)
    table.add_column("Title", justify="left", width=15, no_wrap=True)
    table.add_column("Top 3", justify="right", width=7)
    table.add_column("Projected", justify="right", width=12)
    table.add_column("Range (P10–P90)", justify="right", width=16, style="dim")

    for i, (_, row) in enumerate(outlook.table.head(top_n).iterrows(), start=1):
        table.add_row(
            str(i), str(row[label_col]),
            f"{row['current_points']:.0f}",
            _bar_pct(row["p_title"]), _pct(row["p_top3"]),
            f"{row['expected_points']:.0f}",
            f"{row['points_p10']:.0f} – {row['points_p90']:.0f}",
            style="bold yellow" if i == 1 else "",
        )
    console.print(table)

    if outlook.is_decided:
        console.print("[bold green]The championship is mathematically settled.[/bold green]")


def render_explanation(contributions: list[dict], driver: str) -> None:
    """Print the features that most distinguish one driver from the field."""
    if not contributions:
        console.print("[dim]No explanation available.[/dim]")
        return

    table = Table(title=f"What stands out about {driver}", box=None,
                  header_style="bold", title_justify="left")
    table.add_column("Factor", min_width=30)
    table.add_column("Value", justify="right", width=10)
    table.add_column("Field median", justify="right", width=13, style="dim")
    table.add_column("", width=12)

    for item in contributions:
        favourable = item["direction"] == "strength"
        table.add_row(
            item["label"],
            f"{item['value']:.2f}",
            f"{item['field_median']:.2f}",
            Text(
                "▲ in favour" if favourable else "▼ against",
                style="green" if favourable else "red",
            ),
        )
    console.print(table)


def render_schedule(events: list, highlight_round: int | None = None) -> None:
    """Print a season calendar with a countdown to the next race."""
    now = datetime.now(tz=timezone.utc)
    table = Table(show_header=True, header_style="bold cyan", box=None)
    table.add_column("R", justify="right", width=3)
    table.add_column("Grand Prix", min_width=24)
    table.add_column("Circuit", min_width=16, style="dim")
    table.add_column("Date (UTC)", min_width=17)
    table.add_column("Format", width=13, style="dim")
    table.add_column("", min_width=14)

    for event in events:
        when = event.race_date
        if when is None:
            status, style = "", ""
        elif when < now:
            status, style = "done", "dim"
        else:
            status, style = _countdown(when - now), "bold green"

        if highlight_round is not None and event.round == highlight_round:
            style = "bold cyan"

        table.add_row(
            str(event.round), event.name, event.location,
            when.strftime("%Y-%m-%d %H:%M") if when else "—",
            "sprint" if event.is_sprint else "conventional",
            status, style=style,
        )
    console.print(table)


def render_standings(standings: pd.DataFrame, title: str, top_n: int = 20) -> None:
    """Print a championship table."""
    if standings.empty:
        console.print("[yellow]No standings available.[/yellow]")
        return

    label_col = "driver_name" if "driver_name" in standings.columns else "team"
    table = Table(title=title, show_header=True, header_style="bold cyan",
                  box=None, title_justify="left")
    table.add_column("#", justify="right", width=3)
    table.add_column("Name", min_width=22)
    if "team" in standings.columns and label_col != "team":
        table.add_column("Team", min_width=18, style="dim")
    table.add_column("Points", justify="right", width=8)
    table.add_column("Wins", justify="right", width=6, style="dim")

    for _, row in standings.head(top_n).iterrows():
        position = int(row.get("position", 0))
        cells = [str(position), str(row[label_col])]
        if "team" in standings.columns and label_col != "team":
            cells.append(str(row["team"]))
        cells += [f"{row['points']:.0f}", str(int(row.get("wins", 0)))]
        table.add_row(*cells, style=position_style(position))
    console.print(table)


def event_panel(prediction: RacePrediction) -> Panel:
    """Context header: circuit, date, weather and how confident we are."""
    event, weather = prediction.event, prediction.weather

    grid_label = {
        "actual_quali": "actual qualifying result",
        "predicted_quali": "predicted from practice",
        "form_only": "form only, no session data",
    }.get(prediction.grid_source, prediction.grid_source)

    lines = Table.grid(padding=(0, 2))
    lines.add_row(
        Text("Circuit", style="dim"),
        Text(str(event.get("circuit") or event.get("name", "?"))),
        Text("Date", style="dim"),
        Text(str(event.get("race_date", "?"))[:16].replace("T", " ")),
    )
    lines.add_row(
        Text("Grid", style="dim"),
        Text(grid_label),
        Text("Confidence", style="dim"),
        Text(prediction.confidence.upper(),
             style=_CONFIDENCE_STYLE.get(prediction.confidence, "")),
    )
    lines.add_row(
        Text("Weather", style="dim"),
        Text(
            f"{weather.get('temperature', 0):.0f}°C · "
            f"rain {weather.get('rain_prob', 0):.0%} · "
            f"wind {weather.get('wind_speed', 0):.0f} m/s"
        ),
        Text("Practice", style="dim"),
        Text(prediction.practice_session or "—"),
    )

    podium = " · ".join(
        f"[{color}]{name}[/{color}]" for color, name in zip(PODIUM_COLORS, prediction.podium, strict=False)
    )

    return Panel(
        Group(lines, Text(""), Text.from_markup(f"Predicted podium: {podium}")),
        title=f"[bold cyan]{event.get('name', 'Race')} {event.get('year', '')}"
              f" · Round {event.get('round', '?')}[/bold cyan]",
        border_style="cyan",
    )


# ── Small helpers ─────────────────────────────────────────────────────────────

def _pct(value) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


def _bar_pct(value: float) -> str:
    """Percentage with a tiny inline bar, for scanning a column quickly."""
    try:
        pct = float(value)
    except (TypeError, ValueError):
        return "—"
    filled = round(pct * 8)
    return f"{'█' * filled}{'░' * (8 - filled)} {pct:.0%}"


def _delta_text(delta: int) -> Text:
    if delta > 0:
        return Text(f"+{delta}", style="green")
    if delta < 0:
        return Text(str(delta), style="red")
    return Text("=", style="dim")


def _error_text(error: int) -> Text:
    if error == 0:
        return Text("✓", style="bold green")
    style = "yellow" if abs(error) <= 2 else "red"
    return Text(f"{error:+d}", style=style)


def _metric(label: str, value: str) -> Text:
    return Text.assemble((f"{label}\n", "dim"), (value, "bold"))


def _countdown(delta) -> str:
    total = round(delta.total_seconds())
    days, rest = divmod(total, 86400)
    hours = rest // 3600
    if days > 0:
        return f"in {days}d {hours}h"
    minutes = (rest % 3600) // 60
    return f"in {hours}h {minutes}m"
