import click

from spm.cli.commands.train import train
from spm.cli.commands.predict import predict


@click.group()
def cli():
    """Spatial Polygon Merging CLI Entrypoint"""
    pass


cli.add_command(train)
cli.add_command(predict)


def main():
    cli()
