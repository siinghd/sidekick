"""Progress bar utilities."""

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()


def create_progress(
    description: str = "Processing",
    transient: bool = False,
) -> Progress:
    """Create a rich progress bar.

    Args:
        description: Default task description
        transient: If True, progress bar disappears when complete

    Returns:
        Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=transient,
    )


def create_spinner(description: str = "Working") -> Progress:
    """Create a simple spinner for indeterminate progress.

    Args:
        description: Task description

    Returns:
        Progress instance configured as spinner
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )
