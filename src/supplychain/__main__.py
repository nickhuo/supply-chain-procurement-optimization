"""`uv run supplychain`: solve all four problems and write outputs/ and figures/."""

import json

from . import report
from .data import load
from .pipeline import run


def main() -> None:
    ds = load()
    results = run(ds)
    report.write_tables(results, ds)
    report.write_figures(results, ds)
    print(json.dumps(report.summary(results, ds), indent=2))
    print(f"\nWrote {report.OUT_DIR} and {report.FIG_DIR}")


if __name__ == "__main__":
    main()
