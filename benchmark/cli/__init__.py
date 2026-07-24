import click

from benchmark.cli.commands.evaluate import evaluate
from benchmark.cli.commands.generate import generate


@click.group()
def cli() -> None:
    """Command line interface for the benchmark module."""
    pass


cli.add_command(evaluate)
cli.add_command(generate)


def main() -> None:
    """Entry point for the benchmark CLI."""
    cli()
