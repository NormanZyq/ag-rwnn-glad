from pathlib import Path
from typing import List, Tuple, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

SCRIPT_DIR = Path(__file__).resolve().parent
LAMBDA_VALUES = [0, 0.01, 0.1, 2, 10, 100, 9999]
LAMBDA_LABELS = ["0", "0.01", "0.1", "2", "10", "100", r"$\infty$"]
LAMBDA_LABEL_MAP = dict(zip(LAMBDA_VALUES, LAMBDA_LABELS))

PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "teal": "#42949E",
    "violet": "#9A4D8E",
    "red_strong": "#B64342",
    "red_soft": "#F2C6C3",
    "neutral_very_light": "#F4F4F4",
    "neutral_light": "#D8D8D8",
    "neutral_mid": "#8F8F8F",
    "neutral_dark": "#4D4D4D",
    "neutral_black": "#272727",
}

DATASET_COLORS = [
    PALETTE["blue_main"],
    PALETTE["teal"],
    PALETTE["violet"],
    PALETTE["red_strong"],
]


def apply_publication_style() -> None:
    """Apply compact, editable-text matplotlib defaults for paper figures."""
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
            "legend.fontsize": 7,
            "legend.frameon": False,
            "figure.dpi": 300,
            "savefig.dpi": 600,
        }
    )


def dataset_order(df: pd.DataFrame) -> List[str]:
    return list(dict.fromkeys(df["Dataset"].astype(str)))


def display_dataset_name(name: str) -> str:
    return name.replace("_", " ")


def save_publication_figure(
    fig: mpl.figure.Figure,
    out_path: Union[str, Path],
    formats: Tuple[str, ...] = ("svg", "pdf", "tiff", "png"),
    dpi: int = 600,
) -> List[Path]:
    """Save a figure as an editable vector bundle plus raster previews."""
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


def add_panel_label(ax: mpl.axes.Axes, label: str, x: float = -0.13, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        color=PALETTE["neutral_black"],
        ha="left",
        va="bottom",
    )


def load_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"Dataset", "AUC_Mean"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

    lambda_col = None
    for col in df.columns:
        if col.lower().strip("_\\ ") == "lambda":
            lambda_col = col
            break
    if lambda_col is None:
        raise ValueError("Could not find lambda column in CSV")

    df = df.rename(columns={lambda_col: "lambda"})
    df["lambda"] = pd.to_numeric(df["lambda"], errors="coerce")
    df["AUC_Mean"] = pd.to_numeric(df["AUC_Mean"], errors="coerce")
    if df[["lambda", "AUC_Mean"]].isna().any().any():
        raise ValueError("Found non-numeric values in lambda or AUC_Mean")

    df["Dataset"] = df["Dataset"].astype(str)
    df["lambda_label"] = df["lambda"].map(LAMBDA_LABEL_MAP)
    if df["lambda_label"].isna().any():
        unknown = sorted(df.loc[df["lambda_label"].isna(), "lambda"].unique())
        raise ValueError(f"Unexpected lambda value(s): {unknown}")

    df["lambda_cat"] = pd.Categorical(
        df["lambda_label"],
        categories=LAMBDA_LABELS,
        ordered=True,
    )
    df["lambda_idx"] = df["lambda_cat"].cat.codes
    return df


def summarize_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        df[df["lambda"] == 0][["Dataset", "AUC_Mean"]]
        .rename(columns={"AUC_Mean": "baseline_auc"})
        .copy()
    )
    if baseline["Dataset"].duplicated().any():
        raise ValueError("Each dataset should have exactly one lambda=0 baseline row")

    merged = df.merge(baseline, on="Dataset", how="left")
    if merged["baseline_auc"].isna().any():
        missing = sorted(merged.loc[merged["baseline_auc"].isna(), "Dataset"].unique())
        raise ValueError(f"Missing lambda=0 baseline for: {missing}")
    merged["delta_auc"] = merged["AUC_Mean"] - merged["baseline_auc"]

    best_rows = (
        merged.sort_values(["Dataset", "AUC_Mean", "lambda_idx"], ascending=[True, False, True])
        .groupby("Dataset", sort=False)
        .head(1)
        .set_index("Dataset")
    )
    span = (
        merged.groupby("Dataset")["AUC_Mean"]
        .agg(min_auc="min", max_auc="max")
        .assign(tuning_span=lambda x: x["max_auc"] - x["min_auc"])
    )
    summary = best_rows.join(span)
    summary = summary.loc[dataset_order(df)].reset_index()
    summary = summary.rename(
        columns={
            "lambda": "best_lambda",
            "lambda_cat": "best_lambda_label",
            "AUC_Mean": "best_auc",
            "delta_auc": "best_gain_vs_lambda0",
        }
    )
    return summary[
        [
            "Dataset",
            "baseline_auc",
            "best_lambda",
            "best_lambda_label",
            "best_auc",
            "best_gain_vs_lambda0",
            "min_auc",
            "max_auc",
            "tuning_span",
        ]
    ]


def make_delta_pivot(df: pd.DataFrame) -> pd.DataFrame:
    baseline = (
        df[df["lambda"] == 0][["Dataset", "AUC_Mean"]]
        .rename(columns={"AUC_Mean": "baseline_auc"})
        .copy()
    )
    plotted = df.merge(baseline, on="Dataset", how="left")
    plotted["delta_auc"] = plotted["AUC_Mean"] - plotted["baseline_auc"]
    return (
        plotted.pivot_table(
            index="Dataset",
            columns="lambda_cat",
            values="delta_auc",
            observed=False,
        )
        .reindex(index=dataset_order(df), columns=LAMBDA_LABELS)
    )


def make_auc_pivot(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.pivot_table(
            index="Dataset",
            columns="lambda_cat",
            values="AUC_Mean",
            observed=False,
        )
        .reindex(index=dataset_order(df), columns=LAMBDA_LABELS)
    )


def plot_delta_heatmap(
    ax: mpl.axes.Axes,
    delta_pivot: pd.DataFrame,
    auc_pivot: pd.DataFrame,
    summary: pd.DataFrame,
) -> mpl.image.AxesImage:
    values = delta_pivot.to_numpy(dtype=float)
    max_abs = max(1.0, float(np.nanmax(np.abs(values))))
    cmap = LinearSegmentedColormap.from_list(
        "delta_auc",
        [
            (0.0, PALETTE["red_soft"]),
            (0.5, "#FFFFFF"),
            (0.72, "#B4C0E4"),
            (1.0, PALETTE["blue_main"]),
        ],
    )
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(delta_pivot.shape[1]))
    ax.set_xticklabels(LAMBDA_LABELS)
    ax.set_yticks(range(delta_pivot.shape[0]))
    ax.set_yticklabels([display_dataset_name(str(x)) for x in delta_pivot.index])
    ax.set_xlabel(r"Sensitivity parameter, $\lambda$")
    ax.set_title(r"Change in AUC relative to $\lambda=0$", loc="left", pad=8)

    ax.set_xticks(np.arange(-0.5, delta_pivot.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, delta_pivot.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.85)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", colors=PALETTE["neutral_black"])
    for spine in ax.spines.values():
        spine.set_visible(False)

    best_lookup = dict(zip(summary["Dataset"], summary["best_lambda_label"].astype(str)))
    for row_i, dataset in enumerate(delta_pivot.index):
        best_col = LAMBDA_LABELS.index(best_lookup[dataset])
        ax.add_patch(
            Rectangle(
                (best_col - 0.5, row_i - 0.5),
                1,
                1,
                fill=False,
                edgecolor=PALETTE["neutral_black"],
                linewidth=1.0,
            )
        )
        for col_j in range(delta_pivot.shape[1]):
            delta = float(delta_pivot.iloc[row_i, col_j])
            auc = float(auc_pivot.iloc[row_i, col_j])
            if col_j == 0:
                label = f"{auc:.1f}\nref"
            else:
                label = f"{delta:+.1f}\n{auc:.1f}"
            text_color = "white" if abs(norm(delta) - 0.5) > 0.33 else PALETTE["neutral_black"]
            ax.text(
                col_j,
                row_i,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=6.0,
                linespacing=0.92,
                fontweight="bold" if col_j == best_col else "normal",
            )
    return image


def plot_best_gain(ax: mpl.axes.Axes, summary: pd.DataFrame) -> None:
    y = np.arange(len(summary))
    gains = summary["best_gain_vs_lambda0"].to_numpy(dtype=float)

    ax.axvline(0, color=PALETTE["neutral_dark"], linewidth=0.8)
    for i, (_, row) in enumerate(summary.iterrows()):
        color = DATASET_COLORS[i % len(DATASET_COLORS)]
        ax.hlines(i, 0, row["best_gain_vs_lambda0"], color=color, linewidth=2.2, alpha=0.85)
        ax.scatter(
            row["best_gain_vs_lambda0"],
            i,
            s=42,
            color=color,
            edgecolor=PALETTE["neutral_black"],
            linewidth=0.7,
            zorder=3,
        )
        ax.text(
            row["best_gain_vs_lambda0"] + 0.18,
            i,
            rf"$\lambda^*$={row['best_lambda_label']}, {row['best_auc']:.1f}%",
            va="center",
            ha="left",
            fontsize=6.5,
            color=PALETTE["neutral_black"],
        )

    ax.set_yticks(y)
    ax.set_yticklabels([display_dataset_name(x) for x in summary["Dataset"]])
    ax.invert_yaxis()
    ax.set_xlabel(r"Best gain over $\lambda=0$ (AUC points)")
    ax.set_title("Best setting", loc="left", pad=8)
    ax.set_xlim(min(-0.2, float(np.nanmin(gains)) - 0.4), float(np.nanmax(gains)) + 2.4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(True, axis="x", color=PALETTE["neutral_light"], linewidth=0.55)
    ax.tick_params(axis="both", colors=PALETTE["neutral_black"])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["neutral_black"])


def plot_tuning_span(ax: mpl.axes.Axes, summary: pd.DataFrame) -> None:
    y = np.arange(len(summary))
    spans = summary["tuning_span"].to_numpy(dtype=float)
    colors = [DATASET_COLORS[i % len(DATASET_COLORS)] for i in range(len(summary))]

    ax.barh(y, spans, color=colors, alpha=0.18, edgecolor=colors, linewidth=1.0)
    for i, span in enumerate(spans):
        ax.text(
            span + 0.12,
            i,
            f"{span:.1f}",
            va="center",
            ha="left",
            fontsize=6.5,
            color=PALETTE["neutral_black"],
        )

    ax.set_yticks(y)
    ax.set_yticklabels([display_dataset_name(x) for x in summary["Dataset"]])
    ax.invert_yaxis()
    ax.set_xlabel("Tuning span (max-min AUC)")
    ax.set_title("Sensitivity magnitude", loc="left", pad=8)
    ax.set_xlim(0, max(1.0, float(np.nanmax(spans)) + 1.0))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(True, axis="x", color=PALETTE["neutral_light"], linewidth=0.55)
    ax.tick_params(axis="both", colors=PALETTE["neutral_black"])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(PALETTE["neutral_black"])


def plot_sensitivity_summary(df: pd.DataFrame, out_path: Union[str, Path]) -> None:
    summary = summarize_sensitivity(df)
    delta_pivot = make_delta_pivot(df)
    auc_pivot = make_auc_pivot(df)

    fig = plt.figure(figsize=(7.2, 4.65))
    grid = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[2.25, 1.15],
        height_ratios=[1.0, 1.0],
        wspace=0.48,
        hspace=0.58,
    )
    ax_heatmap = fig.add_subplot(grid[:, 0])
    ax_gain = fig.add_subplot(grid[0, 1])
    ax_span = fig.add_subplot(grid[1, 1])

    image = plot_delta_heatmap(ax_heatmap, delta_pivot, auc_pivot, summary)
    plot_best_gain(ax_gain, summary)
    plot_tuning_span(ax_span, summary)

    add_panel_label(ax_heatmap, "a", x=-0.16, y=1.05)
    add_panel_label(ax_gain, "b", x=-0.18, y=1.06)
    add_panel_label(ax_span, "c", x=-0.18, y=1.06)

    cbar = fig.colorbar(image, ax=ax_heatmap, fraction=0.036, pad=0.025)
    cbar.set_label(r"$\Delta$AUC vs. $\lambda=0$", rotation=270, labelpad=12)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.tick_params(labelsize=6.5, length=2.5, width=0.6)

    fig.subplots_adjust(left=0.085, right=0.97, top=0.92, bottom=0.12)
    save_publication_figure(fig, out_path)
    plt.close(fig)

    summary_out = summary.copy()
    numeric_cols = [
        "baseline_auc",
        "best_lambda",
        "best_auc",
        "best_gain_vs_lambda0",
        "min_auc",
        "max_auc",
        "tuning_span",
    ]
    summary_out[numeric_cols] = summary_out[numeric_cols].round(1)
    summary_out.to_csv(Path(out_path).with_suffix("").with_name("best_lambda_summary.csv"), index=False)


def plot_sensitivity_single_column(df: pd.DataFrame, out_path: Union[str, Path]) -> None:
    summary = summarize_sensitivity(df)
    delta_pivot = make_delta_pivot(df)
    auc_pivot = make_auc_pivot(df)

    values = delta_pivot.to_numpy(dtype=float)
    max_abs = max(1.0, float(np.nanmax(np.abs(values))))
    cmap = LinearSegmentedColormap.from_list(
        "delta_auc_single",
        [
            (0.0, PALETTE["red_soft"]),
            (0.5, "#FFFFFF"),
            (0.72, "#B4C0E4"),
            (1.0, PALETTE["blue_main"]),
        ],
    )
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0, vmax=max_abs)

    fig, ax = plt.subplots(figsize=(3.54, 2.42))
    image = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(delta_pivot.shape[1]))
    ax.set_xticklabels(LAMBDA_LABELS)
    ax.set_yticks(range(delta_pivot.shape[0]))
    ax.set_yticklabels([display_dataset_name(str(x)) for x in delta_pivot.index])
    ax.set_xlabel(r"Sensitivity parameter, $\lambda$", labelpad=3)
    ax.set_title(r"$\Delta$AUC relative to $\lambda=0$", loc="left", pad=5)

    ax.set_xticks(np.arange(-0.5, delta_pivot.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, delta_pivot.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", colors=PALETTE["neutral_black"], pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)

    best_lookup = dict(zip(summary["Dataset"], summary["best_lambda_label"].astype(str)))
    for row_i, dataset in enumerate(delta_pivot.index):
        best_col = LAMBDA_LABELS.index(best_lookup[dataset])
        ax.add_patch(
            Rectangle(
                (best_col - 0.5, row_i - 0.5),
                1,
                1,
                fill=False,
                edgecolor=PALETTE["neutral_black"],
                linewidth=0.85,
            )
        )
        for col_j in range(delta_pivot.shape[1]):
            delta = float(delta_pivot.iloc[row_i, col_j])
            auc = float(auc_pivot.iloc[row_i, col_j])
            label = f"{auc:.1f}\nref" if col_j == 0 else f"{delta:+.1f}\n{auc:.1f}"
            text_color = "white" if abs(norm(delta) - 0.5) > 0.34 else PALETTE["neutral_black"]
            ax.text(
                col_j,
                row_i,
                label,
                ha="center",
                va="center",
                color=text_color,
                fontsize=5.2,
                linespacing=0.88,
                fontweight="bold" if col_j == best_col else "normal",
            )

    cbar = fig.colorbar(image, ax=ax, fraction=0.052, pad=0.025)
    cbar.set_label(r"$\Delta$AUC", rotation=270, labelpad=8, fontsize=6.5)
    cbar.outline.set_linewidth(0.55)
    cbar.ax.tick_params(labelsize=5.8, length=2.0, width=0.55)

    fig.subplots_adjust(left=0.22, right=0.88, top=0.86, bottom=0.2)
    save_publication_figure(fig, out_path)
    plt.close(fig)


def main() -> None:
    csv_path = SCRIPT_DIR / "sens0913.csv"
    out_dir = SCRIPT_DIR / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(csv_path)
    plot_sensitivity_summary(df, out_dir / "sensitivity_summary.pdf")
    plot_sensitivity_single_column(df, out_dir / "sensitivity_single_column.pdf")


if __name__ == "__main__":
    apply_publication_style()
    main()
