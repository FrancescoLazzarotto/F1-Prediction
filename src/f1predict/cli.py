"""CLI entry point: `f1predict <command> [options]`."""

from __future__ import annotations

import logging
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from f1predict.config import get_config
from f1predict.data.schedule import resolve_event
from f1predict.pipeline import F1Pipeline

app = typer.Typer(
    name="f1predict",
    help="F1 race and qualifying predictor — powered by FastF1, Jolpica, Open-Meteo.",
    add_completion=False,
)
console = Console()

logging.basicConfig(level=logging.WARNING, stream=sys.stderr)


def _progress(msg: str) -> None:
    console.print(f"  [dim]{msg}[/dim]")


def _resolve(year, round_num, gp, next_race):
    try:
        return resolve_event(year, round_num, gp, next_race)
    except Exception as exc:
        console.print(f"[red]Error resolving event: {exc}[/red]")
        raise typer.Exit(1)


# ── predict ───────────────────────────────────────────────────────────────────

@app.command("predict")
def predict(
    year: Optional[int] = typer.Option(None, "--year", "-y", help="Season year"),
    round_num: Optional[int] = typer.Option(None, "--round", "-r", help="Round number"),
    gp: Optional[str] = typer.Option(None, "--gp", "-g", help="GP name (substring match)"),
    next_race: bool = typer.Option(False, "--next", "-n", help="Use next upcoming race"),
):
    """Predict race outcome: finishing order + podium probabilities."""
    y, r, event = _resolve(year, round_num, gp, next_race)
    console.rule(f"[bold cyan]F1 Predictor — {event.get('name', '')} {y} (Round {r})[/bold cyan]")

    pipe = F1Pipeline(get_config())
    try:
        result = pipe.predict_race(y, r, progress_cb=_progress)
    except Exception as exc:
        console.print(f"[red]Prediction failed: {exc}[/red]")
        raise typer.Exit(1)

    # ── Qualifying prediction (if FP-based) ───────────────────────────────────
    if result.get("quali") is not None and len(result["quali"]) > 0:
        console.print("\n[bold yellow]Predicted Qualifying Order (from FP)[/bold yellow]")
        q_table = Table(show_header=True, header_style="bold yellow")
        q_table.add_column("Pos", justify="right", style="dim", width=4)
        q_table.add_column("Driver", min_width=20)
        q_table.add_column("Team", min_width=20)
        for _, row in result["quali"].iterrows():
            q_table.add_row(
                str(row.get("predicted_quali_pos", "")),
                str(row.get("driver_name", row.get("driver_code", ""))),
                str(row.get("team", "")),
            )
        console.print(q_table)

    # ── Race prediction ───────────────────────────────────────────────────────
    mode_label = {
        "actual_quali": "[green]actual qualifying data[/green]",
        "predicted_quali": "[yellow]predicted qualifying (FP-based)[/yellow]",
        "form_only": "[red]form/history only (no session data)[/red]",
    }.get(result.get("data_mode", ""), "")
    console.print(f"\n[bold]Data mode:[/bold] {mode_label}")
    console.print(f"[bold]Circuit:[/bold] {result['event'].get('name', '?')}")

    race_df = result["race"]
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Pos", justify="right", style="bold", width=4)
    table.add_column("Driver", min_width=20)
    table.add_column("Team", min_width=22)
    table.add_column("P(win)", justify="right", width=8)
    table.add_column("P(podium)", justify="right", width=10)
    table.add_column("P(top5)", justify="right", width=8)

    for _, row in race_df.iterrows():
        pos = int(row.get("predicted_pos", 0))
        style = "bold gold1" if pos == 1 else ("silver" if pos == 2 else
                ("dark_orange" if pos == 3 else ""))
        table.add_row(
            str(pos),
            str(row.get("driver_name", row.get("driver_code", ""))),
            str(row.get("team", "")),
            f"{row.get('p_win', 0):.1%}",
            f"{row.get('p_podium', 0):.1%}",
            f"{row.get('p_top5', 0):.1%}",
            style=style,
        )
    console.print("\n[bold cyan]Predicted Race Result[/bold cyan]")
    console.print(table)


# ── quali ─────────────────────────────────────────────────────────────────────

@app.command("quali")
def quali(
    year: Optional[int] = typer.Option(None, "--year", "-y"),
    round_num: Optional[int] = typer.Option(None, "--round", "-r"),
    gp: Optional[str] = typer.Option(None, "--gp", "-g"),
    next_race: bool = typer.Option(False, "--next", "-n"),
):
    """Predict qualifying order from Free Practice data."""
    y, r, event = _resolve(year, round_num, gp, next_race)
    console.rule(f"[bold yellow]Qualifying Prediction — {event.get('name', '')} {y} R{r}[/bold yellow]")

    pipe = F1Pipeline(get_config())
    try:
        result = pipe.predict_quali(y, r, progress_cb=_progress)
    except Exception as exc:
        console.print(f"[red]Prediction failed: {exc}[/red]")
        raise typer.Exit(1)

    if result.empty:
        console.print("[yellow]No qualifying prediction available (FP data missing?).[/yellow]")
        raise typer.Exit(0)

    table = Table(show_header=True, header_style="bold yellow")
    table.add_column("Pred pos", justify="right", width=8)
    table.add_column("Driver", min_width=20)
    table.add_column("Team", min_width=22)
    for _, row in result.iterrows():
        table.add_row(
            str(row.get("predicted_quali_pos", "")),
            str(row.get("driver_name", row.get("driver_code", ""))),
            str(row.get("team", "")),
        )
    console.print(table)


# ── train ─────────────────────────────────────────────────────────────────────

@app.command("train")
def train(
    seasons: list[int] = typer.Argument(default=None, help="Season years to train on"),
    force: bool = typer.Option(False, "--force", "-f", help="Re-download and retrain from scratch"),
):
    """Train race and qualifying models on historical seasons."""
    cfg = get_config()
    if not seasons:
        seasons = cfg["training"]["seasons"]

    console.rule(f"[bold green]Training on seasons: {seasons}[/bold green]")
    pipe = F1Pipeline(cfg)
    try:
        pipe.train(seasons, progress_cb=_progress, force_retrain=force)
    except Exception as exc:
        console.print(f"[red]Training failed: {exc}[/red]")
        raise typer.Exit(1)
    console.print("[bold green]Training complete![/bold green]")


# ── backtest ──────────────────────────────────────────────────────────────────

@app.command("backtest")
def backtest(
    year: int = typer.Argument(..., help="Season year"),
    round_num: int = typer.Argument(..., help="Round number"),
):
    """Evaluate prediction accuracy on a past race."""
    console.rule(f"[bold magenta]Backtest — {year} Round {round_num}[/bold magenta]")

    pipe = F1Pipeline(get_config())
    try:
        result = pipe.backtest(year, round_num)
    except Exception as exc:
        console.print(f"[red]Backtest failed: {exc}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Circuit:[/bold] {result['event'].get('name', '?')}")
    console.print(f"[bold]Spearman ρ:[/bold] {result['spearman_rho']:.3f}"
                  f"  [bold]Top-3 accuracy:[/bold] {result['top3_accuracy']:.0%}"
                  f"  [bold]MAE:[/bold] {result['mae_positions']:.1f} positions\n")

    df = result["prediction_df"]
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Actual", justify="right", width=7)
    table.add_column("Predicted", justify="right", width=9)
    table.add_column("Δ", justify="right", width=4)
    table.add_column("Driver", min_width=20)
    table.add_column("Team", min_width=22)
    for _, row in df.iterrows():
        delta = int(row.get("predicted_pos", 0)) - int(row.get("actual_pos", 0))
        delta_str = f"{delta:+d}" if delta != 0 else "✓"
        color = "green" if delta == 0 else ("yellow" if abs(delta) <= 2 else "red")
        table.add_row(
            str(int(row.get("actual_pos", 0))),
            str(int(row.get("predicted_pos", 0))),
            f"[{color}]{delta_str}[/{color}]",
            str(row.get("driver_name", row.get("driver_code", ""))),
            str(row.get("team", "")),
        )
    console.print(table)


if __name__ == "__main__":
    app()
