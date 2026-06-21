import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "extra-exp-r.md"
DEFAULT_OUTPUT = SCRIPT_DIR / "figs" / "extra_exp_r_single_column.pdf"

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "red_soft": "#F2C6C3",
    "neutral_light": "#D8D8D8",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
}
PM = "\u00b1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a single-column publication figure for the r-ratio ablation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Markdown file to read. Default: {DEFAULT_INPUT}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output figure path. Default: {DEFAULT_OUTPUT}",
    )
    return parser.parse_args()


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "axes.linewidth": 0.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )


def save_publication_figure(
    fig: mpl.figure.Figure,
    out_path: Union[str, Path],
    formats: Tuple[str, ...] = ("svg", "pdf", "tiff", "png"),
    dpi: int = 600,
) -> List[Path]:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_path = out_path.with_suffix("") if out_path.suffix else out_path

    saved_paths: List[Path] = []
    for fmt in formats:
        path = base_path.with_suffix(f".{fmt}")
        save_kwargs = {"bbox_inches": "tight"}
        if fmt in {"png", "tif", "tiff"}:
            save_kwargs["dpi"] = dpi
        fig.savefig(path, **save_kwargs)
        saved_paths.append(path)
    return saved_paths


def display_dataset_name(name: str) -> str:
    return name.replace("_", " ")


def ratio_label(value: float) -> str:
    return f"{value:g}"


def parse_header_datasets(header_line: str) -> List[str]:
    parts = [part.strip() for part in header_line.strip().strip("|").split("|")]
    if len(parts) < 2:
        return []
    return [name.replace("\\_", "_") for name in parts[1:]]


def parse_table_row(
    line: str,
    dataset_names: List[str],
) -> Optional[Tuple[float, Dict[str, Tuple[float, float]]]]:
    parts = [part.strip() for part in line.strip().strip("|").split("|")]
    if len(parts) != len(dataset_names) + 1:
        return None

    ratio_values = re.findall(r"[0-9]+(?:\.[0-9]+)?", parts[0])
    if len(ratio_values) != 1:
        return None

    row_data: Dict[str, Tuple[float, float]] = {}
    for dataset_name, cell in zip(dataset_names, parts[1:]):
        values = re.findall(r"[0-9]+(?:\.[0-9]+)?", cell)
        if len(values) != 2:
            return None
        row_data[dataset_name] = (float(values[0]), float(values[1]))

    return float(ratio_values[0]), row_data


def parse_markdown(markdown_path: Union[str, Path]) -> Tuple[pd.DataFrame, List[float], List[str], Dict[str, float]]:
    markdown_path = Path(markdown_path)
    text = markdown_path.read_text(encoding="utf-8")
    lines = [line.rstrip() for line in text.splitlines()]
    blocks: List[Tuple[List[str], List[Tuple[float, Dict[str, Tuple[float, float]]]], Dict[str, float]]] = []

    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line.startswith("|"):
            idx += 1
            continue

        dataset_names = parse_header_datasets(line)
        if not dataset_names:
            idx += 1
            continue

        if idx + 1 >= len(lines) or "---" not in lines[idx + 1]:
            idx += 1
            continue

        idx += 2
        rows: List[Tuple[float, Dict[str, Tuple[float, float]]]] = []
        while idx < len(lines):
            row_line = lines[idx].strip()
            if not row_line.startswith("|"):
                break
            parsed_row = parse_table_row(row_line, dataset_names)
            if parsed_row is None:
                break
            rows.append(parsed_row)
            idx += 1

        main_settings = None
        search_idx = idx
        while search_idx < len(lines):
            candidate = lines[search_idx].strip()
            if candidate.startswith("|"):
                break
            if "$r$" in candidate:
                values = [float(token) for token in re.findall(r"[0-9]+(?:\.[0-9]+)?", candidate)]
                if len(values) == len(dataset_names):
                    main_settings = dict(zip(dataset_names, values))
                    break
            search_idx += 1

        if rows and main_settings is not None:
            blocks.append((dataset_names, rows, main_settings))

    if not blocks:
        raise ValueError(f"No valid experiment block found in {markdown_path}")

    dataset_names, rows, main_settings = blocks[-1]
    records = []
    for ratio, row_data in rows:
        for dataset_name in dataset_names:
            mean, std = row_data[dataset_name]
            records.append(
                {
                    "Dataset": dataset_name,
                    "ratio": ratio,
                    "ratio_label": ratio_label(ratio),
                    "AUC_Mean": mean,
                    "AUC_Std": std,
                    "selected_ratio": main_settings[dataset_name],
                }
            )

    df = pd.DataFrame.from_records(records)
    ratios = [ratio for ratio, _ in rows]
    ratio_labels = [ratio_label(ratio) for ratio in ratios]
    df["ratio_cat"] = pd.Categorical(df["ratio_label"], categories=ratio_labels, ordered=True)
    df["ratio_idx"] = df["ratio_cat"].cat.codes
    df["is_selected"] = np.isclose(df["ratio"], df["selected_ratio"])

    selected = (
        df[df["is_selected"]][["Dataset", "AUC_Mean"]]
        .rename(columns={"AUC_Mean": "selected_auc"})
        .copy()
    )
    if selected["Dataset"].duplicated().any() or len(selected) != len(dataset_names):
        raise ValueError("Each dataset must have exactly one selected main-experiment r")

    df = df.merge(selected, on="Dataset", how="left")
    df["delta_auc"] = df["AUC_Mean"] - df["selected_auc"]
    return df, ratios, dataset_names, main_settings


def summarize_r_sensitivity(df: pd.DataFrame, dataset_names: List[str]) -> pd.DataFrame:
    best = (
        df.sort_values(["Dataset", "AUC_Mean", "ratio_idx"], ascending=[True, False, True])
        .groupby("Dataset", sort=False)
        .head(1)
        .set_index("Dataset")
    )
    span = (
        df.groupby("Dataset")["AUC_Mean"]
        .agg(min_auc="min", max_auc="max")
        .assign(tuning_span=lambda x: x["max_auc"] - x["min_auc"])
    )
    selected = (
        df[df["is_selected"]][["Dataset", "ratio", "ratio_label", "selected_auc"]]
        .set_index("Dataset")
        .rename(columns={"ratio": "selected_ratio", "ratio_label": "selected_ratio_label"})
    )
    summary = selected.join(best[["ratio", "ratio_label", "AUC_Mean", "delta_auc"]]).join(span)
    summary = summary.loc[dataset_names].reset_index()
    summary = summary.rename(
        columns={
            "ratio": "best_ratio",
            "ratio_label": "best_ratio_label",
            "AUC_Mean": "best_auc",
            "delta_auc": "best_gain_vs_selected",
        }
    )
    return summary


def make_pivot(df: pd.DataFrame, dataset_names: List[str], ratio_labels: List[str], value: str) -> pd.DataFrame:
    return (
        df.pivot_table(index="Dataset", columns="ratio_cat", values=value, observed=False)
        .reindex(index=dataset_names, columns=ratio_labels)
    )


def plot_single_column(
    df: pd.DataFrame,
    ratios: List[float],
    dataset_names: List[str],
    out_path: Union[str, Path],
) -> None:
    ratio_labels = [ratio_label(ratio) for ratio in ratios]
    summary = summarize_r_sensitivity(df, dataset_names)
    delta_pivot = make_pivot(df, dataset_names, ratio_labels, "delta_auc")
    mean_pivot = make_pivot(df, dataset_names, ratio_labels, "AUC_Mean")
    std_pivot = make_pivot(df, dataset_names, ratio_labels, "AUC_Std")
    selected_lookup = dict(zip(summary["Dataset"], summary["selected_ratio_label"].astype(str)))
    best_lookup = dict(zip(summary["Dataset"], summary["best_ratio_label"].astype(str)))

    values = delta_pivot.to_numpy(dtype=float)
    max_abs = max(1.0, float(np.nanmax(np.abs(values))))
    color_limit = min(max_abs, 12.0)
    cmap = LinearSegmentedColormap.from_list(
        "r_delta_auc",
        [
            (0.0, PALETTE["red_soft"]),
            (0.5, "#FFFFFF"),
            (0.72, "#B4C0E4"),
            (1.0, PALETTE["blue_main"]),
        ],
    )
    norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0, vmax=color_limit)

    fig, ax = plt.subplots(figsize=(3.54, 2.28))
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(len(ratio_labels)))
    ax.set_xticklabels(ratio_labels)
    ax.set_yticks(range(len(dataset_names)))
    ax.set_yticklabels([display_dataset_name(x) for x in dataset_names])
    ax.set_xlabel(r"Downsampling ratio, $r$", labelpad=3)
    ax.set_title(r"$\Delta$AUC relative to selected $r$", loc="left", pad=5)

    ax.set_xticks(np.arange(-0.5, len(ratio_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(dataset_names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", colors=PALETTE["neutral_black"], pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row_i, dataset in enumerate(dataset_names):
        selected_col = ratio_labels.index(selected_lookup[dataset])
        best_col = ratio_labels.index(best_lookup[dataset])
        ax.add_patch(
            Rectangle(
                (selected_col - 0.5, row_i - 0.5),
                1,
                1,
                fill=False,
                edgecolor=PALETTE["neutral_black"],
                linewidth=0.9,
            )
        )
        for col_j, ratio_text in enumerate(ratio_labels):
            delta = float(delta_pivot.iloc[row_i, col_j])
            mean = float(mean_pivot.iloc[row_i, col_j])
            std = float(std_pivot.iloc[row_i, col_j])
            if col_j == selected_col:
                label = f"{mean:.1f}{PM}{std:.1f}\nref"
            else:
                label = f"{delta:+.1f}\n{mean:.1f}{PM}{std:.1f}"
            text_color = "white" if abs(norm(delta) - 0.5) > 0.34 else PALETTE["neutral_black"]
            ax.text(
                col_j,
                row_i,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=5.25,
                linespacing=0.9,
                fontweight="bold" if col_j == best_col else "normal",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.052, pad=0.025)
    cbar_label = r"$\Delta$AUC"
    if max_abs > color_limit:
        cbar_label = r"$\Delta$AUC (clipped)"
    cbar.set_label(cbar_label, rotation=270, labelpad=8, fontsize=6.5)
    cbar.outline.set_linewidth(0.55)
    cbar.ax.tick_params(labelsize=5.8, length=2.0, width=0.55)

    fig.subplots_adjust(left=0.23, right=0.88, top=0.85, bottom=0.2)
    save_publication_figure(fig, out_path)
    plt.close(fig)

    summary_out = summary.copy()
    numeric_cols = [
        "selected_ratio",
        "selected_auc",
        "best_ratio",
        "best_auc",
        "best_gain_vs_selected",
        "min_auc",
        "max_auc",
        "tuning_span",
    ]
    summary_out[numeric_cols] = summary_out[numeric_cols].round(1)
    summary_out.to_csv(Path(out_path).with_suffix("").with_name("extra_exp_r_summary.csv"), index=False)


def main() -> None:
    args = parse_args()
    apply_publication_style()
    df, ratios, dataset_names, _ = parse_markdown(args.input)
    plot_single_column(df, ratios, dataset_names, args.output)
    print(f"Saved figure bundle to {Path(args.output).with_suffix('')}")


if __name__ == "__main__":
    main()
