import argparse
import shutil
from importlib.resources import files
from pathlib import Path
from typing import Any


def _copy_resource_tree(source: Any, destination: Path) -> None:
    """Copy an importlib resource tree from a directory or zipped wheel."""
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        child_destination = destination / child.name
        if child.is_dir():
            _copy_resource_tree(child, child_destination)
        elif child.name != "__pycache__" and not child.name.endswith(".pyc"):
            with child.open("rb") as source_file:
                child_destination.write_bytes(source_file.read())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Copy LIBERO training configs.")
    parser.add_argument("--destination", default="configs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    target = Path(args.destination).resolve()
    source = files("libero.configs")
    if target.exists():
        if not args.force:
            parser.error(
                f"destination already exists: {target}; pass --force to replace it"
            )
        shutil.rmtree(target)
    print(f"Copying configs to {target}")
    with source.joinpath("config.yaml").open("rb") as probe:
        probe.read(1)
    _copy_resource_tree(source, target)


if __name__ == "__main__":
    main()
