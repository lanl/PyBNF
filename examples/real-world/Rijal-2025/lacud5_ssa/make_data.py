#!/usr/bin/env python
"""Regenerate lacud5.exp from the authors' released Jones-2014 data array."""

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "jones_fig3a_source.tsv"
OUTPUT = HERE / "lacud5.exp"


def main() -> None:
    raw = np.genfromtxt(
        SOURCE,
        comments="#",
        dtype=[("promoter", "U16"), ("mean", float), ("fano", float)],
    )
    # The Rijal notebooks use np.unique(mean, return_index=True), retaining the first Fano value
    # when two source points have the same digitized mean.
    _, first = np.unique(raw["mean"], return_index=True)
    selected = raw[np.sort(first)]
    sd = np.sqrt(selected["mean"] * selected["fano"])
    fit_data = np.column_stack((selected["mean"], selected["mean"], sd))
    np.savetxt(
        OUTPUT,
        fit_data,
        fmt="%.12g",
        header="mean_target Mean_mRNA mRNA_SD",
        comments="# ",
    )
    print(f"wrote {OUTPUT} ({len(fit_data)} rows)")


if __name__ == "__main__":
    main()
