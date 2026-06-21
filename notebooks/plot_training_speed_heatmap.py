import csv
import math
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "notebooks" / "training-speed.csv"
DEFAULT_FIG_DIR = ROOT / "notebooks" / "figs"
DEFAULT_SVG = DEFAULT_FIG_DIR / "training_speed_heatmap.svg"
DEFAULT_PDF = DEFAULT_FIG_DIR / "training_speed_heatmap.pdf"

FONT_FAMILY = "'DejaVu Serif', serif"
AXES_LABEL_SIZE = 11.0
TICK_LABEL_SIZE = 10.0
LEGEND_FONT_SIZE = 9.0
ANNOTATION_FONT_SIZE = 8.5
CELL_VALUE_SIZE = 11
NA_FILL = "#F1F3F5"
GRID_COLOR = "#FFFFFF"
OURS_STROKE = "#C44E52"

# Clean blue sequential palette for paper figures.
COLOR_ANCHORS = [
    (0.00, "#F7FBFF"),
    (0.22, "#DCEAF7"),
    (0.45, "#A8D3E8"),
    (0.68, "#5FA8C9"),
    (0.85, "#2F7EA1"),
    (1.00, "#0B4F71"),
]


def load_speed_matrix(csv_path: Path) -> Tuple[List[str], List[str], List[List[Optional[float]]]]:
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    datasets = rows[0][1:]
    methods: List[str] = []
    values: List[List[Optional[float]]] = []

    for row in rows[1:]:
        methods.append(row[0])
        parsed_row: List[Optional[float]] = []
        for cell in row[1:]:
            parsed_row.append(None if cell in {"", "N/A"} else float(cell))
        values.append(parsed_row)

    return methods, datasets, values


def normalize_by_dataset(values: List[List[Optional[float]]]) -> List[List[Optional[float]]]:
    row_count = len(values)
    col_count = len(values[0]) if values else 0
    normalized: List[List[Optional[float]]] = [[None] * col_count for _ in range(row_count)]

    for col_idx in range(col_count):
        valid_values = [
            row[col_idx]
            for row in values
            if row[col_idx] is not None and row[col_idx] > 0
        ]
        if not valid_values:
            continue

        logged_values = [math.log10(value) for value in valid_values]
        min_log = min(logged_values)
        max_log = max(logged_values)

        for row_idx in range(row_count):
            value = values[row_idx][col_idx]
            if value is None or value <= 0:
                continue
            if math.isclose(min_log, max_log):
                normalized[row_idx][col_idx] = 0.5
            else:
                normalized[row_idx][col_idx] = (math.log10(value) - min_log) / (max_log - min_log)

    return normalized


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def text_color_for_fill(fill_color: str) -> str:
    red, green, blue = hex_to_rgb(fill_color)
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255.0
    return "#111111" if luminance >= 0.58 else "#FFFFFF"


def interpolate_color(value: float) -> str:
    value = max(0.0, min(1.0, value))
    for idx in range(len(COLOR_ANCHORS) - 1):
        left_pos, left_color = COLOR_ANCHORS[idx]
        right_pos, right_color = COLOR_ANCHORS[idx + 1]
        if value <= right_pos:
            span = right_pos - left_pos
            ratio = 0.0 if span == 0 else (value - left_pos) / span
            left_rgb = hex_to_rgb(left_color)
            right_rgb = hex_to_rgb(right_color)
            blended = tuple(
                round(left_rgb[channel] + ratio * (right_rgb[channel] - left_rgb[channel]))
                for channel in range(3)
            )
            return rgb_to_hex(blended)
    return COLOR_ANCHORS[-1][1]


def format_speed(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def build_linear_gradient() -> str:
    stops = []
    for offset, color in COLOR_ANCHORS:
        stops.append(f'<stop offset="{offset * 100:.1f}%" stop-color="{color}" />')
    return "\n".join(stops)


def add_text(
    elements: List[str],
    x: float,
    y: float,
    text: str,
    font_size: float,
    anchor: str = "start",
    fill: str = "#111111",
    weight: str = "normal",
    rotate: Optional[float] = None,
) -> None:
    attrs = [
        f'x="{x:.1f}"',
        f'y="{y:.1f}"',
        f'font-size="{font_size:.1f}"',
        f'font-family="{FONT_FAMILY}"',
        f'fill="{fill}"',
        f'text-anchor="{anchor}"',
        f'font-weight="{weight}"',
        'dominant-baseline="middle"',
    ]
    if rotate is not None:
        attrs.append(f'transform="rotate({rotate:.1f} {x:.1f} {y:.1f})"')
    elements.append(f"<text {' '.join(attrs)}>{escape(text)}</text>")


def plot_heatmap_svg(
    methods: List[str],
    datasets: List[str],
    raw_values: List[List[Optional[float]]],
    normalized_values: List[List[Optional[float]]],
    svg_path: Path,
) -> None:
    row_count = len(methods)
    col_count = len(datasets)

    left_margin = 198
    top_margin = 24
    cell_width = 94
    cell_height = 32
    heatmap_width = col_count * cell_width
    heatmap_height = row_count * cell_height
    colorbar_gap = 14
    colorbar_width = 20
    right_margin = 112
    bottom_margin = 70

    total_width = left_margin + heatmap_width + colorbar_gap + colorbar_width + right_margin
    total_height = top_margin + heatmap_height + bottom_margin

    elements: List[str] = []

    elements.append(
        f'<rect x="0" y="0" width="{total_width}" height="{total_height}" fill="white" />'
    )

    for row_idx, method in enumerate(methods):
        y = top_margin + row_idx * cell_height
        label_fill = "#8C2D2F" if (method == "AG-RWNN (ours)" or method == "AG-RWNN") else "#111111"
        label_weight = "bold" if (method == "AG-RWNN (ours)" or method == "AG-RWNN") else "normal"
        add_text(
            elements,
            left_margin - 10,
            y + cell_height / 2,
            method,
            font_size=TICK_LABEL_SIZE,
            anchor="end",
            fill=label_fill,
            weight=label_weight,
        )

    for col_idx, dataset in enumerate(datasets):
        x = left_margin + col_idx * cell_width + cell_width / 2
        add_text(
            elements,
            x,
            top_margin + heatmap_height + 24,
            dataset,
            font_size=TICK_LABEL_SIZE,
            anchor="middle",
            rotate=-25.0,
        )

    for row_idx in range(row_count):
        for col_idx in range(col_count):
            raw_value = raw_values[row_idx][col_idx]
            norm_value = normalized_values[row_idx][col_idx]
            x = left_margin + col_idx * cell_width
            y = top_margin + row_idx * cell_height

            fill = NA_FILL if norm_value is None else interpolate_color(norm_value)
            elements.append(
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{cell_width:.1f}" height="{cell_height:.1f}" '
                f'fill="{fill}" stroke="{GRID_COLOR}" stroke-width="1.2" />'
            )

            if raw_value is None:
                text_fill = "#666666"
            else:
                text_fill = text_color_for_fill(fill)
            add_text(
                elements,
                x + cell_width / 2,
                y + cell_height / 2,
                format_speed(raw_value),
                font_size=CELL_VALUE_SIZE,
                anchor="middle",
                fill=text_fill,
            )

    if "AG-RWNN (ours)" in methods:
        ours_idx = methods.index("AG-RWNN (ours)")
        y = top_margin + ours_idx * cell_height
        elements.append(
            f'<rect x="{left_margin - 1:.1f}" y="{y - 1:.1f}" width="{heatmap_width + 2:.1f}" '
            f'height="{cell_height + 2:.1f}" fill="none" stroke="{OURS_STROKE}" stroke-width="3" />'
        )

    outer_x = left_margin
    outer_y = top_margin
    elements.append(
        f'<rect x="{outer_x - 0.5:.1f}" y="{outer_y - 0.5:.1f}" width="{heatmap_width + 1:.1f}" '
        f'height="{heatmap_height + 1:.1f}" fill="none" stroke="#9A9A9A" stroke-width="1" />'
    )

    colorbar_x = left_margin + heatmap_width + colorbar_gap
    colorbar_y = top_margin
    elements.append(
        f'<rect x="{colorbar_x:.1f}" y="{colorbar_y:.1f}" width="{colorbar_width:.1f}" '
        f'height="{heatmap_height:.1f}" fill="url(#speedGradient)" stroke="#9A9A9A" stroke-width="1" />'
    )
    add_text(
        elements,
        colorbar_x + colorbar_width / 2,
        colorbar_y - 10,
        "Faster",
        font_size=LEGEND_FONT_SIZE,
        anchor="middle",
    )
    add_text(
        elements,
        colorbar_x + colorbar_width / 2,
        colorbar_y + heatmap_height + 12,
        "Slower",
        font_size=LEGEND_FONT_SIZE,
        anchor="middle",
    )
    add_text(
        elements,
        colorbar_x + colorbar_width + 30,
        colorbar_y + heatmap_height / 2,
        "Per-dataset relative speed (log-scaled)",
        font_size=AXES_LABEL_SIZE,
        anchor="middle",
        rotate=90.0,
    )
    na_legend_y = colorbar_y + heatmap_height + 28
    elements.append(
        f'<rect x="{colorbar_x - 1:.1f}" y="{na_legend_y - 8:.1f}" width="18" height="18" '
        f'fill="{NA_FILL}" stroke="#9A9A9A" stroke-width="1" />'
    )
    add_text(
        elements,
        colorbar_x + 24,
        na_legend_y + 1,
        "N/A",
        font_size=ANNOTATION_FONT_SIZE,
        anchor="start",
        fill="#4A4A4A",
    )

    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{total_height}" viewBox="0 0 {total_width} {total_height}">
  <defs>
    <linearGradient id="speedGradient" x1="0%" y1="100%" x2="0%" y2="0%">
{build_linear_gradient()}
    </linearGradient>
  </defs>
  {' '.join(elements)}
</svg>
"""

    svg_path.parent.mkdir(parents=True, exist_ok=True)
    svg_path.write_text(svg_content, encoding="utf-8")


def maybe_convert_to_pdf(svg_path: Path, pdf_path: Path) -> bool:
    if shutil.which("rsvg-convert"):
        subprocess.run(
            ["rsvg-convert", "-f", "pdf", "-o", str(pdf_path), str(svg_path)],
            check=True,
        )
        return True

    if shutil.which("inkscape"):
        subprocess.run(
            ["inkscape", str(svg_path), "--export-filename", str(pdf_path)],
            check=True,
        )
        return True

    return False


def main() -> None:
    methods, datasets, raw_values = load_speed_matrix(DEFAULT_INPUT)
    normalized_values = normalize_by_dataset(raw_values)
    plot_heatmap_svg(methods, datasets, raw_values, normalized_values, DEFAULT_SVG)
    maybe_convert_to_pdf(DEFAULT_SVG, DEFAULT_PDF)


if __name__ == "__main__":
    main()
