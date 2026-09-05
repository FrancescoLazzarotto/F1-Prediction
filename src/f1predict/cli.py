"""Command-line interface: ``f1predict <command> [options]``."""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from f1predict import __version__
from f1predict import cache as C
from f1predict.config import get_config, load_config, replace_cache_root, set_config
from f1predict.data import repository as repo
from f1predict.data.schedule import Event, resolve_event, season_events
from f1predict.models import registry
from f1predict.pipeline import F1Pipeline
from f1predict.reporting import (
    console,
    render_backtest,
    render_championship,
    render_explanation,
    render_quali,
    render_race,
    render_schedule,
    render_standings,
)


def _configure_stdio() -> None:
    """Make stdout and stderr UTF-8 safe.

    On Windows, Python picks cp1252 when output is redirected to a file or a
    pipe, so the box-drawing and typographic characters in our tables raise
    UnicodeEncodeError the moment anyone runs `f1predict train > log.txt`.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        # A detached or already-wrapped stream cannot be reconfigured; plain
        # ASCII output still works, so carry on.
        with contextlib.suppress(ValueError, OSError):
            reconfigure(encoding="utf-8", errors="replace")


_configure_stdio()

app = typer.Typer(
    name="f1predict",
    help="Formula 1 race and qualifying predictor, powered by FastF1, Jolpica and Open-Meteo.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

err_console = Console(stderr=True)


# ── Global options ────────────────────────────────────────────────────────────

@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show progress logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Errors only."),
    config_file: Path | None = typer.Option(
        None, "--config", help="Path to an alternative YAML config."
    ),
    cache_dir: Path | None = typer.Option(
        None, "--cache-dir", help="Override the cache root for this run."
    ),
) -> None:
    """Set up logging and configuration before any command runs."""
    level = logging.DEBUG if verbose else (logging.ERROR if quiet else logging.WARNING)
    logging.basicConfig(
        level=level, format="%(message)s", datefmt="[%X]",
        handlers=[RichHandler(console=err_console, rich_tracebacks=True,
                              show_path=False, markup=False)],
    )
    cfg = load_config(config_file) if config_file else get_config()
    if cache_dir:
        cfg = replace_cache_root(cfg, cache_dir)
    set_config(cfg)

    # FastF1 owns its own logging hierarchy and its own cache, and both have to
    # be configured before the first schedule lookup — otherwise it prints a
    # "DEFAULT CACHE ENABLED" warning and re-downloads into a temp directory.
    _quiet_fastf1(logging.INFO if verbose else logging.ERROR)
    C.ensure_dirs(cfg)
    C.setup_fastf1_cache(cfg)


def _quiet_fastf1(level: int) -> None:
    """Turn down FastF1's own loggers.

    Setting the parent logger is not enough: FastF1 assigns an explicit level to
    each child, so they ignore the parent and keep printing.
    """
    try:
        import fastf1

        fastf1.set_log_level(logging.getLevelName(level))
    except Exception:
        logging.getLogger("fastf1").setLevel(level)
    for name in list(logging.root.manager.loggerDict):
        if name == "fastf1" or name.startswith("fastf1."):
            logging.getLogger(name).setLevel(level)


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"f1predict {__version__}")
        raise typer.Exit()


@app.command("version")
def version() -> None:
    """Print the installed version."""
    console.print(f"[bold cyan]f1predict[/bold cyan] {__version__}")


# ── Shared option helpers ─────────────────────────────────────────────────────

def _resolve(
    year: int | None, round_num: int | None, gp: str | None,
    next_race: bool, last_race: bool,
) -> Event:
    try:
        return resolve_event(year, round_num, gp, next_race, last_race)
    except (LookupError, ValueError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc


def _progress(message: str) -> None:
    console.print(f"  [dim]{message}[/dim]")


def _fail(message: str, exc: Exception | None = None) -> None:
    err_console.print(f"[bold red]Error:[/bold red] {message}")
    if exc is not None:
        logging.getLogger(__name__).debug("Details", exc_info=exc)
    raise typer.Exit(1)


def _export(df: pd.DataFrame, path: Path | None) -> None:
    """Write a result table to CSV or JSON, inferred from the extension."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    else:
        df.to_csv(path, index=False)
    console.print(f"[green]Wrote[/green] {path}")


# ── predict ───────────────────────────────────────────────────────────────────

@app.command("predict")
def predict(
    year: int | None = typer.Option(None, "--year", "-y", help="Season."),
    round_num: int | None = typer.Option(None, "--round", "-r", help="Round number."),
    gp: str | None = typer.Option(None, "--gp", "-g", help="Grand Prix name."),
    next_race: bool = typer.Option(False, "--next", "-n", help="The next upcoming race."),
    last_race: bool = typer.Option(False, "--last", "-l", help="The most recent race."),
    explain: str | None = typer.Option(
        None, "--explain", "-e", help="Driver code to explain, e.g. VER."
    ),
    show_quali: bool = typer.Option(
        True, "--quali/--no-quali", help="Also show the predicted qualifying order."
    ),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write CSV or JSON."),
    simulations: int | None = typer.Option(
        None, "--simulations", "-s", help="Override the Monte Carlo sample count."
    ),
) -> None:
    """Predict a race: finishing order plus win, podium and points probabilities."""
    event = _resolve(year, round_num, gp, next_race, last_race)
    cfg = get_config()
    if simulations:
        from dataclasses import replace

        cfg = replace(cfg, simulation=replace(cfg.simulation, n_simulations=simulations))
        set_config(cfg)

    console.rule(f"[bold cyan]{event.name} {event.year} · Round {event.round}[/bold cyan]")

    pipeline = F1Pipeline(cfg)
    try:
        prediction = pipeline.predict_race(event.year, event.round, progress=_progress)
    except Exception as exc:
        _fail(str(exc), exc)
        return

    if show_quali and not prediction.quali_table.empty:
        render_quali(prediction.quali_table)
        console.print()

    render_race(prediction)

    if explain:
        console.print()
        try:
            contributions = pipeline.explain(prediction, explain.upper())
        except KeyError as exc:
            _fail(str(exc), exc)
            return
        render_explanation(contributions, explain.upper())

    _export(prediction.table, output)


# ── quali ─────────────────────────────────────────────────────────────────────

@app.command("quali")
def quali(
    year: int | None = typer.Option(None, "--year", "-y"),
    round_num: int | None = typer.Option(None, "--round", "-r"),
    gp: str | None = typer.Option(None, "--gp", "-g"),
    next_race: bool = typer.Option(False, "--next", "-n"),
    last_race: bool = typer.Option(False, "--last", "-l"),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Predict the qualifying order from Free Practice pace."""
    event = _resolve(year, round_num, gp, next_race, last_race)
    console.rule(
        f"[bold magenta]Qualifying · {event.name} {event.year} R{event.round}[/bold magenta]"
    )

    pipeline = F1Pipeline(get_config())
    try:
        result = pipeline.predict_quali(event.year, event.round, progress=_progress)
    except Exception as exc:
        _fail(str(exc), exc)
        return

    if result.empty:
        console.print(
            "[yellow]No qualifying prediction available — practice data is missing "
            "for this event.[/yellow]"
        )
        raise typer.Exit(0)

    render_quali(result)
    _export(result, output)


# ── train ─────────────────────────────────────────────────────────────────────

@app.command("train")
def train(
    seasons: list[int] | None = typer.Argument(
        None, help="Seasons to train on. Defaults to the config."
    ),
    refresh: bool = typer.Option(
        False, "--refresh", "-f", help="Rebuild cached features from scratch."
    ),
) -> None:
    """Train the race, qualifying and retirement models."""
    cfg = get_config()
    target = sorted(seasons) if seasons else cfg.training.seasons
    console.rule(f"[bold green]Training on {target}[/bold green]")

    pipeline = F1Pipeline(cfg)
    try:
        reports = pipeline.train(target, progress=_progress, refresh=refresh)
    except Exception as exc:
        _fail(str(exc), exc)
        return

    table = Table(title="Training summary", box=None, header_style="bold",
                  title_justify="left")
    table.add_column("Model", min_width=10)
    table.add_column("Result", min_width=54)
    for name, report in reports.items():
        table.add_row(name, report.summary() if report else "[dim]not trained[/dim]")
    console.print()
    console.print(table)

    race_report = reports.get("race")
    if race_report and race_report.feature_importance:
        console.print()
        importance = Table(title="Most influential race features", box=None,
                           header_style="bold", title_justify="left")
        importance.add_column("Feature", min_width=28)
        importance.add_column("Share", justify="right", width=8)
        importance.add_column("", width=22)
        for name, value in race_report.top_features(10):
            bar = "█" * max(1, round(value * 60))
            importance.add_row(name, f"{value:.1%}", f"[cyan]{bar}[/cyan]")
        console.print(importance)


# ── backtest ──────────────────────────────────────────────────────────────────

@app.command("backtest")
def backtest(
    year: int = typer.Argument(..., help="Season."),
    round_num: int | None = typer.Argument(
        None, help="Round. Omit to backtest the whole season."
    ),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Score predictions against races that have already happened."""
    pipeline = F1Pipeline(get_config())

    if round_num is not None:
        console.rule(f"[bold magenta]Backtest · {year} Round {round_num}[/bold magenta]")
        try:
            result = pipeline.backtest(year, round_num, progress=_progress)
        except Exception as exc:
            _fail(str(exc), exc)
            return
        render_backtest(result)
        _export(result.table, output)
        return

    console.rule(f"[bold magenta]Backtest · {year} season[/bold magenta]")
    try:
        overall, per_race = pipeline.backtest_season(year, progress=_progress)
    except Exception as exc:
        _fail(str(exc), exc)
        return

    console.print()
    summary = Table(title=f"{year} season accuracy over {len(per_race)} races",
                    box=None, header_style="bold", title_justify="left")
    summary.add_column("Metric", min_width=24)
    summary.add_column("Value", justify="right", width=10)
    for label, value, fmt in (
        ("Rank correlation (ρ)", overall.spearman, "{:+.3f}"),
        ("Mean position error", overall.mae, "{:.2f}"),
        ("Winner predicted", overall.top1, "{:.0%}"),
        ("Podium hit rate", overall.top3, "{:.0%}"),
        ("Top-10 hit rate", overall.top10, "{:.0%}"),
        ("Within 1 place", overall.within_1, "{:.0%}"),
        ("Within 3 places", overall.within_3, "{:.0%}"),
    ):
        summary.add_row(label, fmt.format(value))
    console.print(summary)

    console.print()
    detail = Table(title="Per race", box=None, header_style="bold", title_justify="left")
    detail.add_column("R", justify="right", width=3)
    detail.add_column("Grand Prix", min_width=24)
    detail.add_column("ρ", justify="right", width=7)
    detail.add_column("MAE", justify="right", width=6)
    detail.add_column("Winner", justify="right", width=7)
    detail.add_column("Podium", justify="right", width=7)
    for _, row in per_race.iterrows():
        detail.add_row(
            f"{row['round']:.0f}", str(row["race"]),
            f"{row['spearman']:+.3f}", f"{row['mae']:.2f}",
            "hit" if row["winner_hit"] >= 1 else "—",
            f"{row['podium_hit']:.0%}",
        )
    console.print(detail)
    _export(per_race, output)


# ── championship ──────────────────────────────────────────────────────────────

@app.command("championship")
def championship(
    year: int | None = typer.Argument(None, help="Season. Defaults to the current one."),
    constructors: bool = typer.Option(
        False, "--constructors", "-c", help="Constructors' championship instead."
    ),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Simulate the rest of the season to get title probabilities."""
    cfg = get_config()
    year = year or pd.Timestamp.now().year
    label = "Constructors" if constructors else "Drivers"
    console.rule(f"[bold cyan]{label} championship outlook · {year}[/bold cyan]")

    pipeline = F1Pipeline(cfg)
    try:
        outlook = pipeline.championship_outlook(
            year, progress=_progress, constructors=constructors
        )
    except Exception as exc:
        _fail(str(exc), exc)
        return

    console.print()
    render_championship(outlook, f"{label}' championship {year}")
    _export(outlook.table, output)


# ── schedule ──────────────────────────────────────────────────────────────────

@app.command("schedule")
def schedule(
    year: int | None = typer.Argument(None, help="Season. Defaults to the current one."),
) -> None:
    """Show a season calendar with a countdown to the next race."""
    year = year or pd.Timestamp.now().year
    events = season_events(year)
    if not events:
        _fail(f"No schedule available for {year}.")
        return

    console.rule(f"[bold cyan]{year} calendar[/bold cyan]")
    highlight = None
    with contextlib.suppress(LookupError):
        highlight = resolve_event(next_race=True).round
    render_schedule(events, highlight_round=highlight)


# ── standings ─────────────────────────────────────────────────────────────────

@app.command("standings")
def standings(
    year: int | None = typer.Argument(None, help="Season. Defaults to the current one."),
    constructors: bool = typer.Option(False, "--constructors", "-c"),
) -> None:
    """Show the current championship table."""
    cfg = get_config()
    year = year or pd.Timestamp.now().year
    table = (
        repo.constructor_standings(year, cfg=cfg) if constructors
        else repo.driver_standings(year, cfg=cfg)
    )
    label = "Constructors" if constructors else "Drivers"
    render_standings(table, f"{label}' championship {year}")


# ── info ──────────────────────────────────────────────────────────────────────

@app.command("info")
def info(
    as_json: bool = typer.Option(False, "--json", help="Emit machine-readable output."),
) -> None:
    """Show what models are trained and how much disk the caches use."""
    cfg = get_config()
    card = registry.describe(cfg)
    usage = C.usage(cfg)
    card["cache_bytes"] = usage

    if as_json:
        console.print_json(json.dumps(card, default=str))
        return

    console.rule("[bold cyan]Model card[/bold cyan]")
    console.print(f"Feature signature: [bold]{card['feature_signature']}[/bold]")
    console.print()

    for name, entry in card["models"].items():
        if entry.get("status") != "ready":
            console.print(f"[yellow]{name}: {entry.get('status')}[/yellow]")
            continue

        table = Table(title=name, box=None, header_style="bold", title_justify="left")
        table.add_column("Field", min_width=18)
        table.add_column("Value", min_width=40)
        table.add_row("Estimator", str(entry["kind"]))
        table.add_row("Trained", str(entry["trained_at"]))
        table.add_row("Seasons", ", ".join(str(s) for s in entry["seasons"]))
        table.add_row("Samples", str(entry["n_samples"]))
        table.add_row("Features", str(entry["n_features"]))
        if entry["cv_mae"] == entry["cv_mae"]:
            table.add_row("CV mean error", f"{entry['cv_mae']:.2f} positions")
        if entry["cv_spearman"] == entry["cv_spearman"]:
            table.add_row("CV rank correlation", f"{entry['cv_spearman']:+.3f}")
        if entry["top_features"]:
            table.add_row(
                "Top features",
                ", ".join(f"{n} ({v:.0%})" for n, v in entry["top_features"][:5]),
            )
        console.print(table)
        console.print()

    cache_table = Table(title="Cache usage", box=None, header_style="bold",
                        title_justify="left")
    cache_table.add_column("Section", min_width=12)
    cache_table.add_column("Size", justify="right", width=12)
    for section, size in usage.items():
        cache_table.add_row(section, _human_bytes(size))
    cache_table.add_row("[bold]total[/bold]", f"[bold]{_human_bytes(sum(usage.values()))}[/bold]")
    console.print(cache_table)


@app.command("clear-cache")
def clear_cache(
    section: str = typer.Argument(
        "all", help="One of: features, models, http, fastf1, all."
    ),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
) -> None:
    """Delete cached data so the next run starts fresh."""
    cfg = get_config()
    usage = C.usage(cfg)
    total = sum(usage.values()) if section == "all" else usage.get(section, 0)

    if not yes:
        confirmed = typer.confirm(
            f"Delete the '{section}' cache ({_human_bytes(total)})?", default=False
        )
        if not confirmed:
            console.print("Cancelled.")
            raise typer.Exit(0)

    try:
        removed = C.clear(cfg, section)
    except ValueError as exc:
        _fail(str(exc), exc)
        return
    for line in removed:
        console.print(f"[green]Cleared[/green] {line}")


@app.command("serve")
def serve(
    port: int = typer.Option(8501, "--port", "-p", help="Port for the web UI."),
) -> None:
    """Launch the Streamlit web interface."""
    import subprocess

    from f1predict.config import REPO_ROOT

    script = REPO_ROOT / "app" / "streamlit_app.py"
    if not script.exists():
        _fail(f"Web app not found at {script}")
        return

    console.print(f"[bold cyan]Starting the web UI on http://localhost:{port}[/bold cyan]")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(script),
         "--server.port", str(port)],
        check=False,
    )


def _human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


if __name__ == "__main__":
    app()
