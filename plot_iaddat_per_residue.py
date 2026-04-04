import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def load_and_aggregate(xlsx_path, chain_filter=None):
    """
    Read an IADDAT-table XLSX file and return a per-residue mean |IADDAT| Series.

    Parameters
    ----------
    xlsx_path : str
        Path to the XLSX file produced by new_iaddat_mtz.py.
    chain_filter : str or None
        If provided, restrict to this chain only.

    Returns
    -------
    pd.Series
        Index is tuples of (chain, residue_number, residue_name) where
        residue_number has been converted to numeric (NaN where conversion
        fails, e.g. insertion codes); values are mean absolute IADDAT.
    """
    df = pd.read_excel(xlsx_path)
    required = {"chain", "residue_number", "residue_name", "IADDAT"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "XLSX file '{}' is missing required columns: {}".format(xlsx_path, missing)
        )
    if chain_filter is not None:
        df = df[df["chain"] == chain_filter]
        if df.empty:
            raise ValueError(
                "No rows found for chain '{}' in '{}'".format(chain_filter, xlsx_path)
            )

    # Convert residue_number to numeric once, coercing non-numeric values to NaN
    df["residue_number"] = pd.to_numeric(df["residue_number"], errors="coerce")

    grouped = (
        df.groupby(["chain", "residue_number", "residue_name"])["IADDAT"]
        .apply(lambda x: x.abs().mean())
    )
    # Sort by chain then residue_number for natural ordering; NaN residue numbers
    # are sorted to the end via na_position='last'
    grouped = grouped.reset_index()
    grouped = grouped.sort_values(["chain", "residue_number"], na_position="last")
    grouped = grouped.set_index(["chain", "residue_number", "residue_name"])["IADDAT"]
    return grouped


def make_residue_labels(series):
    """Return a list of '{chain}_{residue_name}{residue_number}' strings for the index."""
    labels = []
    for chain, resnum, resname in series.index:
        labels.append("{}_{}{}".format(chain, resname, resnum))
    return labels


def _sort_key(t):
    """Sort key for (chain, residue_number, residue_name) tuples; NaN sorts last."""
    chain, resnum, resname = t
    # Use a large sentinel for NaN residue numbers so they sort after all others
    if pd.isna(resnum):
        return (chain, float("inf"), resname)
    return (chain, resnum, resname)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Plot mean absolute IADDAT values on a per-residue basis from one or more "
            "IADDAT-table XLSX files produced by new_iaddat_mtz.py."
        ),
        epilog="",
    )
    parser.add_argument(
        "xlsx_files",
        nargs="+",
        type=str,
        help="One or more IADDAT-table XLSX files produced by new_iaddat_mtz.py.",
    )
    parser.add_argument(
        "--chain",
        type=str,
        default=None,
        help="Filter to a specific chain (default: all chains).",
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=None,
        help="Only plot the top N residues by mean |IADDAT| (default: plot all).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for the saved figure (default: 300).",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Custom prefix for output files (default: derived from input filename(s)).",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load and aggregate each file
    # ------------------------------------------------------------------
    all_series = {}
    for path in args.xlsx_files:
        series = load_and_aggregate(path, chain_filter=args.chain)
        all_series[path] = series

    # ------------------------------------------------------------------
    # Determine output prefix
    # ------------------------------------------------------------------
    if args.output_prefix:
        prefix = args.output_prefix
    elif len(args.xlsx_files) == 1:
        basename = os.path.splitext(os.path.basename(args.xlsx_files[0]))[0]
        prefix = basename
    else:
        # Use the first file's basename as the prefix
        basename = os.path.splitext(os.path.basename(args.xlsx_files[0]))[0]
        prefix = basename + "_and_others"

    # ------------------------------------------------------------------
    # Determine the union of residues across all files (for multi-file
    # grouped bars) or just the residues in the single file.
    # ------------------------------------------------------------------
    if len(all_series) == 1:
        path, series = next(iter(all_series.items()))

        if args.top_n is not None:
            top_keys = set(series.nlargest(args.top_n).index)
            series = series[series.index.isin(top_keys)]
            # Re-sort naturally (load_and_aggregate already sorted; isin preserves order)
            series = series.reset_index()
            series = series.sort_values(["chain", "residue_number"], na_position="last")
            series = series.set_index(["chain", "residue_number", "residue_name"])[
                "IADDAT"
            ]

        labels = make_residue_labels(series)
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.35), 6))
        ax.bar(x, series.values, color="steelblue")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_xlabel("Residue")
        ax.set_ylabel("Mean |IADDAT|")
        title = "Per-residue Mean |IADDAT|\n{}".format(os.path.basename(path))
        ax.set_title(title, fontsize=9)
        plt.tight_layout()

        # Build summary table (from the full series, before top_n filtering, for the xlsx)
        summary_df = series.reset_index()
        summary_df.columns = ["chain", "residue_number", "residue_name", "mean_abs_IADDAT"]
        summary_df["residue_label"] = make_residue_labels(series)

        # Print top-10 from the full (unfiltered) series
        full_series = load_and_aggregate(path, chain_filter=args.chain)
        full_summary = full_series.reset_index()
        full_summary.columns = ["chain", "residue_number", "residue_name", "mean_abs_IADDAT"]
        full_summary["residue_label"] = make_residue_labels(full_series)
        top10 = full_summary.nlargest(10, "mean_abs_IADDAT")
        print("Top 10 residues by mean |IADDAT| (from '{}'):".format(path))
        print(top10[["residue_label", "mean_abs_IADDAT"]].to_string(index=False))
        print()

    else:
        # Multiple files: build the union of all residues, optionally filter to the
        # top N residues by their maximum mean |IADDAT| across all files, then plot.
        all_keys_set = set()
        for series in all_series.values():
            for key in series.index:
                all_keys_set.add(key)

        if args.top_n is not None:
            # Rank by maximum value across all files for each residue in the union
            max_across_files = {
                key: max(
                    (s.get(key, 0.0) for s in all_series.values()),
                    default=0.0,
                )
                for key in all_keys_set
            }
            top_keys = set(
                sorted(all_keys_set, key=lambda k: max_across_files[k], reverse=True)[
                    : args.top_n
                ]
            )
            all_keys_set = top_keys

        # Sort the (possibly filtered) union by chain then residue_number
        all_labels_sorted = sorted(all_keys_set, key=_sort_key)

        def _label(t):
            chain, resnum, resname = t
            return "{}_{}{}".format(chain, resname, resnum)

        labels = [_label(t) for t in all_labels_sorted]
        x = np.arange(len(labels))
        n_files = len(all_series)
        width = 0.8 / n_files
        offsets = np.linspace(-(n_files - 1) / 2, (n_files - 1) / 2, n_files) * width

        cmap = plt.get_cmap("tab10")
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.4), 6))

        summary_frames = []
        for i, (path, series) in enumerate(all_series.items()):
            values = [series.get(key, 0.0) for key in all_labels_sorted]
            ax.bar(x + offsets[i], values, width, label=os.path.basename(path), color=cmap(i))

            # Build summary df for this file
            file_summary = pd.DataFrame(
                {
                    "file": os.path.basename(path),
                    "residue_label": labels,
                    "chain": [t[0] for t in all_labels_sorted],
                    "residue_number": [t[1] for t in all_labels_sorted],
                    "residue_name": [t[2] for t in all_labels_sorted],
                    "mean_abs_IADDAT": values,
                }
            )
            summary_frames.append(file_summary)

            # Top-10 from the raw (full) series for this file, not the union values
            top10 = series.nlargest(10).reset_index()
            top10.columns = ["chain", "residue_number", "residue_name", "mean_abs_IADDAT"]
            top10["residue_label"] = make_residue_labels(
                series.nlargest(10)
            )
            print("Top 10 residues by mean |IADDAT| (from '{}'):".format(path))
            print(top10[["residue_label", "mean_abs_IADDAT"]].to_string(index=False))
            print()

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_xlabel("Residue")
        ax.set_ylabel("Mean |IADDAT|")
        title_files = ", ".join(os.path.basename(p) for p in args.xlsx_files)
        ax.set_title("Per-residue Mean |IADDAT|\n{}".format(title_files), fontsize=9)
        ax.legend(fontsize=7)
        plt.tight_layout()

        summary_df = pd.concat(summary_frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    png_path = prefix + "_per_residue_iaddat.png"
    xlsx_path = prefix + "_per_residue_summary.xlsx"

    fig.savefig(png_path, dpi=args.dpi)
    print("Plot saved to: {}".format(png_path))

    summary_df.to_excel(xlsx_path, index=False)
    print("Summary table saved to: {}".format(xlsx_path))


if __name__ == "__main__":
    main()
