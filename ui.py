import os
import sys

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm
from rich.theme import Theme

_theme = Theme(
    {
        "brand": "bold color(82)",
        "dim": "color(238)",
        "system": "bold green",
        "gpu": "bold green",
        "index": "green",
        "embed": "green",
        "vector": "green",
        "store": "green",
        "info": "cyan",
        "ok": "bold green",
        "warn": "bold yellow",
        "error": "bold red",
        "user": "bold green",
        "prompt": "bold green",
    }
)

console = Console(
    theme=_theme,
    no_color=bool(os.environ.get("NO_COLOR")),
    highlight=False,
)

_TAG_WIDTH = 7
_GREEN = "\033[97m"
_RESET = "\033[0m"


def _line(tag: str, style: str, message: str) -> None:
    console.print(f"[{style}][[{tag:^{_TAG_WIDTH}}]][/{style}]  {message}")


def system(msg: str) -> None:
    _line("SYSTEM", "system", msg)


def gpu(msg: str) -> None:
    _line("GPU", "gpu", msg)


def cpu(msg: str) -> None:
    _line("CPU", "gpu", msg)


def cuda(msg: str) -> None:
    _line("CUDA", "gpu", msg)


def pdf(msg: str) -> None:
    _line("PDF", "index", msg)


def pages(msg: str) -> None:
    _line("PAGES", "index", msg)


def index(msg: str) -> None:
    _line("INDEX", "index", msg)


def embed(msg: str) -> None:
    _line("EMBED", "embed", msg)


def vector(msg: str) -> None:
    _line("VECTOR", "vector", msg)


def store(msg: str) -> None:
    _line("STORE", "store", msg)


def llm(msg: str) -> None:
    _line("LLM", "brand", msg)


def info(msg: str) -> None:
    _line("INFO", "info", msg)


def ok(msg: str) -> None:
    _line("OK", "ok", msg)


def warn(msg: str) -> None:
    _line("WARN", "warn", msg)


def error(msg: str) -> None:
    _line("ERROR", "error", msg)


_SEP_W = 87
_SEP = "-" * _SEP_W

_LOGO_LINES = [
    "███████╗██╗███████╗████████╗",
    "██╔════╝██║██╔════╝╚══██╔══╝",
    "███████╗██║█████╗     ██║   ",
    "╚════██║██║██╔══╝     ██║   ",
    "███████║██║██║        ██║   ",
    "╚══════╝╚═╝╚═╝        ╚═╝   ",
]
_LOGO_PAD = " " * ((_SEP_W - 28) // 2)

_GOODBYE_LINES = [
    "██████╗ ██╗   ██╗███████╗██╗",
    "██╔══██╗╚██╗ ██╔╝██╔════╝██║",
    "██████╔╝ ╚████╔╝ █████╗  ██║",
    "██╔══██╗  ╚██╔╝  ██╔══╝  ╚═╝",
    "██████╔╝   ██║   ███████╗██╗",
    "╚═════╝    ╚═╝   ╚══════╝╚═╝",
]
_GOODBYE_PAD = " " * ((_SEP_W - 29) // 2)


def banner() -> None:
    console.print()
    console.print(f"[brand]{_SEP}[/brand]")
    console.print()
    for line in _LOGO_LINES:
        console.print(f"[brand]{_LOGO_PAD}{line}[/brand]")
    console.print()
    console.print(f"[brand]{_SEP}[/brand]")
    console.print()


def goodbye() -> None:
    console.print()
    console.print(f"[brand]{_SEP}[/brand]")
    console.print()
    for line in _GOODBYE_LINES:
        console.print(f"[brand]{_GOODBYE_PAD}{line}[/brand]")
    console.print()
    console.print(f"[dim]{"alr, I'm outta here.":^{_SEP_W}}[/dim]")
    console.print()
    console.print(f"[brand]{_SEP}[/brand]")
    console.print()


def confirm(question: str, default: bool = True) -> bool:
    console.print(f"[brand]>[/brand] [prompt]{question}[/prompt] ", end="")
    sys.stdout.write(_GREEN)
    sys.stdout.flush()
    result = Confirm.ask("", default=default, console=console)
    sys.stdout.write(_RESET)
    sys.stdout.flush()
    return result


def ask(question: str) -> str:
    """Prompt with green typed-input text."""
    console.print(
        f"[brand]SIFT[/brand][dim] -->[/dim] [prompt]{question}[/prompt]: ",
        end="",
    )
    sys.stdout.write(_GREEN)
    sys.stdout.flush()
    result = input()
    sys.stdout.write(_RESET)
    sys.stdout.flush()
    return result


def menu(title: str, options: dict) -> None:
    """options: {key: label} — printed in WIFI-CRACKER style."""
    console.print()
    console.print(f"[brand]{_SEP}[/brand]")
    console.print(f"[bold white]{title}[/bold white]")
    console.print(f"[brand]{_SEP}[/brand]")
    for key, label in options.items():
        console.print(f"[brand]({key})[/brand] {label}")
    console.print(f"[brand]{_SEP}[/brand]")
    console.print()


def answer_panel(text: str) -> None:
    """Output without a box — raw text between separator lines."""
    console.print(f"\n[brand]{_SEP}[/brand]")
    console.print()
    for line in text.splitlines():
        console.print(f"[white]{line}[/white]")
    console.print()
    console.print(f"[brand]{_SEP}[/brand]\n")


def progress_bar(label: str = "INDEX") -> Progress:
    return Progress(
        TextColumn(f"[embed][[{label:^{_TAG_WIDTH}}]][/embed]"),
        BarColumn(
            bar_width=32,
            complete_style="green",
            finished_style="bold green",
            pulse_style="color(22)",
        ),
        TaskProgressColumn(),
        TextColumn("[dim]{task.completed}/{task.total} pages[/dim]"),
        TimeElapsedColumn(),
        console=console,
    )


def spinner(message: str, tag: str = "VECTOR"):
    return console.status(
        f"[embed][[{tag:^{_TAG_WIDTH}}]][/embed]  [green]{message}[/green]",
        spinner="dots2",
        spinner_style="bold green",
    )
