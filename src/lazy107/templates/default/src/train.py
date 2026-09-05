"""Training entry point.

Wires dataset loading, model construction and the training loop.
"""

from __future__ import annotations

from src.data import load_data
from src.model import build_model


def main() -> None:
    """Assemble the pipeline and run training."""
    data = load_data()
    model = build_model()

    # TODO: fill in the training loop (optimizer, criterion, epochs).
    print(f"Data: {data}")
    print(f"Model: {model}")


if __name__ == "__main__":
    main()
