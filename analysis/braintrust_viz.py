"""Braintrust-style Matplotlib templates.

Portable helpers distilled from the provided reliability_analysis.ipynb and
cost_story_blog.ipynb notebooks, then generalized for common analytical charts.
No proprietary font files are bundled; if Braintrust Display V2 is installed
locally, register it with register_brand_fonts().
"""
from __future__ import annotations

import os
import glob
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

# Brand palette sampled/normalized from the source notebooks.
BRAND = {
    "indigo": "#2C1FEB",
    "indigo_dark": "#2518EA",
    "periwinkle": "#BCCFF9",
    "periwinkle_light": "#D4E0FB",
    "purple": "#956BDD",
    "pink": "#E6A4D3",
    "orange": "#F59E0B",
    "blue": "#2C1FEB",
    "violet": "#956BDD",
    "lavender": "#D4C7F3",
}
NEUTRAL = {
    "ink": "#18181B",
    "body": "#18181B",
    "subtle": "#6B6B76",
    "muted": "#9F9FA9",
    "grid": "#E8E8E8",
    "panel": "#F3F4F8",
    "thin": "#D8DEEE",
    "faint": "#B4B4BE",
    "white": "#FFFFFF",
}

# Pink -> purple -> indigo. Use for quality/rate encodings; avoids red/green.
BRAND_DIV = LinearSegmentedColormap.from_list(
    "braintrust_quality", ["#F2C6E6", BRAND["pink"], BRAND["purple"], "#533BE6", BRAND["indigo"]]
)
BRAND_SEQ = LinearSegmentedColormap.from_list(
    "braintrust_sequential", ["#F7F3FE", "#D4E0FB", "#BCCFF9", "#956BDD", "#533BE6", "#2C1FEB"]
)
BRAND_DIVERGING = LinearSegmentedColormap.from_list(
    "braintrust_diverging", ["#E6A4D3", "#F3F4F8", "#2C1FEB"]
)


def register_brand_fonts(font_dir: str | os.PathLike | None = None) -> None:
    """Register local Braintrust/OpenType fonts if available.

    Do not bundle font files in generated artifacts. Point this function at a local
    directory containing installed .otf/.ttf files when rendering on a machine that
    has the brand font.
    """
    if not font_dir:
        return
    for fp in glob.glob(str(Path(font_dir) / "*.otf")) + glob.glob(str(Path(font_dir) / "*.ttf")):
        try:
            fm.fontManager.addfont(fp)
        except Exception:
            pass


def _available_font(preferred=("Braintrust Display V2", "Helvetica Neue", "Arial", "Inter", "DejaVu Sans")) -> str:
    names = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in names:
            return name
    return "DejaVu Sans"


def use_braintrust_theme(font_dir: str | os.PathLike | None = None) -> None:
    """Apply the editorial Braintrust chart theme globally."""
    register_brand_fonts(font_dir)
    family = _available_font()
    plt.rcParams.update({
        "font.family": family,
        "figure.facecolor": NEUTRAL["white"],
        "axes.facecolor": NEUTRAL["white"],
        "savefig.facecolor": NEUTRAL["white"],
        "axes.edgecolor": NEUTRAL["grid"],
        "axes.linewidth": 1.2,
        "axes.grid": True,
        "grid.color": NEUTRAL["grid"],
        "grid.linewidth": 1.0,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": True,
        "text.color": NEUTRAL["ink"],
        "axes.labelcolor": NEUTRAL["subtle"],
        "axes.titlecolor": NEUTRAL["ink"],
        "xtick.color": NEUTRAL["subtle"],
        "ytick.color": NEUTRAL["subtle"],
        "axes.labelsize": 14,
        "axes.labelweight": "regular",
        "axes.labelpad": 10,
        "xtick.labelsize": 12.5,
        "ytick.labelsize": 12.5,
        "legend.fontsize": 12,
        "lines.linewidth": 2.2,
        "font.size": 14,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.35,
    })


def braintrust_style_path() -> Path:
    """Return the bundled .mplstyle path when this module sits inside the skill/template tree."""
    here = Path(__file__).resolve()
    candidates = [here.parent.parent / "assets" / "braintrust.mplstyle", here.parent / "braintrust.mplstyle"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("braintrust.mplstyle not found next to module or under ../assets")


def use_mplstyle() -> None:
    """Apply the bundled .mplstyle file."""
    plt.style.use(str(braintrust_style_path()))


def rate_cmap():
    """Return the brand quality colormap: pale pink/low -> indigo/high."""
    return BRAND_DIV


def sequential_cmap():
    """Return a sequential brand map for magnitude, density, and heatmap intensity."""
    return BRAND_SEQ


def diverging_cmap():
    """Return a diverging brand map for signed deltas: pink/negative -> neutral -> indigo/positive."""
    return BRAND_DIVERGING


def categorical(n: int, include_neutral: bool = False) -> list[str]:
    """Return a calm categorical palette for small-N comparisons."""
    base = [BRAND["indigo"], BRAND["purple"], BRAND["periwinkle"], BRAND["pink"], "#533BE6", "#7C3AED"]
    if include_neutral:
        base.append(NEUTRAL["muted"])
    return [base[i % len(base)] for i in range(n)]


def quality_colors(values: Sequence[float], pad_low: float = 0.04, pad_high: float = 0.02) -> list[str]:
    """Map rates/scores to brand quality colors."""
    vals = np.asarray(values, dtype=float)
    norm = Normalize(np.nanmin(vals) - pad_low, np.nanmax(vals) + pad_high)
    return [to_hex(BRAND_DIV(norm(v))) for v in vals]


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson confidence interval for a binomial rate."""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def compact_number(x: float, pos=None) -> str:
    """Compact tick formatter for counts/costs."""
    abs_x = abs(float(x))
    if abs_x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs_x >= 1_000:
        return f"{x/1_000:.0f}k"
    if abs_x >= 10:
        return f"{x:.0f}"
    if abs_x >= 1:
        return f"{x:.1f}"
    return f"{x:.2f}"


def currency_formatter(decimals: int = 2):
    return mticker.FuncFormatter(lambda x, pos: f"${x:,.{decimals}f}")


def percent_formatter(decimals: int = 0):
    return mticker.PercentFormatter(1.0, decimals=decimals)


def make_figure(width: float = 12, height: float = 7.2, *, top: float = 0.80, left: float = 0.09,
                right: float = 0.97, bottom: float = 0.12):
    """Create a single Braintrust-style figure with blog-friendly margins."""
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(top=top, left=left, right=right, bottom=bottom)
    return fig, ax


def header(ax, title: str, subtitle: str | None = None, key: str | None = None, tx: float = 0.0,
           title_size: float = 21, subtitle_size: float = 14.5, key_size: float = 11.5) -> None:
    """Left-aligned title/subtitle/key block above the axes."""
    ax.set_title("")
    ax.text(tx, 1.135, title, transform=ax.transAxes, fontsize=title_size, fontweight="bold",
            color=NEUTRAL["ink"], ha="left", va="bottom")
    if subtitle:
        ax.text(tx, 1.085, subtitle, transform=ax.transAxes, fontsize=subtitle_size,
                color=NEUTRAL["subtle"], ha="left", va="bottom")
    if key:
        ax.text(tx, 1.028, key, transform=ax.transAxes, fontsize=key_size,
                color=NEUTRAL["subtle"], ha="left", va="bottom")


def titled(ax, title: str, subtitle: str | None = None, pad: float = 20) -> None:
    """Reliability-notebook title helper using Matplotlib's left title slot."""
    ax.set_title(title, loc="left", pad=pad, fontsize=21, fontweight="bold", color=NEUTRAL["ink"])
    if subtitle:
        ax.text(0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=13.5, color=NEUTRAL["subtle"])


def clean_axes(ax, despine_left: bool = False, grid_axis: str = "y") -> None:
    """Apply the common minimal axes treatment."""
    ax.grid(axis=grid_axis, color=NEUTRAL["grid"], lw=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if despine_left:
        ax.spines["left"].set_visible(False)
        ax.tick_params(left=False)
    if sns is not None:
        sns.despine(ax=ax, left=despine_left)


def label_points(ax, data: pd.DataFrame, *, x: str, y: str, label: str, dx: float = 0,
                 dy: float = 0.012, fontsize: float = 11.5, bbox: bool = True) -> None:
    """Direct-label points with small white rounded labels."""
    box = dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.92) if bbox else None
    for _, r in data.iterrows():
        ax.text(r[x] + dx, r[y] + dy, str(r[label]), ha="center", va="bottom",
                fontsize=fontsize, fontweight="bold", color=NEUTRAL["ink"], bbox=box, zorder=5)


def value_label(ax, x: float, y: float, text: str, dx: float = 0.012, percent_axis: bool = True,
                bbox: bool = False, ha: str = "left", va: str = "center") -> None:
    """Place a bold value label near a bar/point."""
    box = dict(boxstyle="round,pad=0.16", fc="white", ec="none") if bbox else None
    ax.text(x + dx if percent_axis else x, y, text, ha=ha, va=va, fontsize=13.5,
            fontweight="bold", color=NEUTRAL["ink"], bbox=box)


def hbar(ax, labels: Sequence[str], values: Sequence[float], colors: Sequence[str] | None = None,
         value_fmt: Callable[[float], str] = lambda v: f"{v:.0%}", xpad_frac: float = 0.012,
         height: float = 0.68) -> None:
    """Horizontal bar chart with right-edge value labels."""
    y = np.arange(len(labels))
    if colors is None:
        colors = quality_colors(values)
    ax.barh(y, values, color=colors, edgecolor="white", linewidth=1.4, height=height, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    xmax = np.nanmax(values)
    for yi, v in zip(y, values):
        ax.text(v + xmax * xpad_frac, yi, value_fmt(v), va="center", ha="left",
                fontsize=13.5, fontweight="bold", color=NEUTRAL["ink"])
    clean_axes(ax, despine_left=True, grid_axis="x")


def vbar(ax, labels: Sequence[str], values: Sequence[float], colors: Sequence[str] | None = None,
         value_fmt: Callable[[float], str] | None = None, rotation: float = 0, width: float = 0.64) -> None:
    """Vertical bar chart for short categorical comparisons."""
    x = np.arange(len(labels))
    if colors is None:
        colors = categorical(len(labels))
    ax.bar(x, values, color=colors, edgecolor="white", linewidth=1.4, width=width, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotation, ha="right" if rotation else "center")
    if value_fmt is not None:
        ymax = np.nanmax(values)
        for xi, v in zip(x, values):
            ax.text(xi, v + ymax * 0.025, value_fmt(v), ha="center", va="bottom",
                    fontsize=13, fontweight="bold", color=NEUTRAL["ink"])
    clean_axes(ax, grid_axis="y")


def grouped_bar(ax, data: pd.DataFrame, *, category: str, series: Sequence[str],
                labels: Sequence[str] | None = None, value_fmt: Callable[[float], str] | None = None) -> None:
    """Grouped bars for a small number of series sharing units."""
    labels = list(labels or series)
    cats = data[category].astype(str).tolist()
    x = np.arange(len(cats))
    width = min(0.78 / max(1, len(series)), 0.28)
    colors = categorical(len(series))
    for i, col in enumerate(series):
        offset = (i - (len(series) - 1) / 2) * width
        ax.bar(x + offset, data[col], width=width, color=colors[i], edgecolor="white", linewidth=1.2,
               label=labels[i], zorder=3)
        if value_fmt is not None:
            ymax = np.nanmax(data[list(series)].values)
            for xi, v in zip(x + offset, data[col]):
                ax.text(xi, v + ymax * 0.018, value_fmt(v), ha="center", va="bottom", fontsize=10.5,
                        fontweight="bold", color=NEUTRAL["ink"])
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=len(series), handlelength=1.4)
    clean_axes(ax, grid_axis="y")


def line_plot(ax, data: pd.DataFrame, *, x: str, y: str | Sequence[str], labels: Sequence[str] | None = None,
              marker: bool = True, baseline: float | None = None) -> None:
    """Line/time-series template with restrained markers and direct legend."""
    ys = [y] if isinstance(y, str) else list(y)
    labels = list(labels or ys)
    colors = categorical(len(ys))
    for col, label, color in zip(ys, labels, colors):
        ax.plot(data[x], data[col], color=color, marker="o" if marker else None, markersize=6,
                markeredgecolor="white", markeredgewidth=1.4, label=label, zorder=3)
    if baseline is not None:
        ax.axhline(baseline, ls=(0, (4, 4)), color=NEUTRAL["subtle"], lw=1.6, alpha=0.7, zorder=0)
    if len(ys) > 1:
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.005), ncol=min(3, len(ys)))
    clean_axes(ax, grid_axis="y")


def scatter_plot(ax, data: pd.DataFrame, *, x: str, y: str, label: str | None = None,
                 color_by: str | None = None, size_by: str | None = None, size_range: tuple[float, float] = (90, 520),
                 trend: bool = False, annotate: bool = False) -> None:
    """General scatter/bubble chart for numeric relationships.

    Use color_by for quality/rate fields and size_by for volume/count fields. Keep bubbles calm:
    white edges, alpha under 1, and direct labels only for small-N or highlighted points.
    """
    colors = quality_colors(data[color_by]) if color_by else BRAND["indigo"]
    if size_by:
        vals = data[size_by].astype(float).values
        lo, hi = np.nanmin(vals), np.nanmax(vals)
        if hi == lo:
            sizes = np.full(len(vals), np.mean(size_range))
        else:
            sizes = size_range[0] + (vals - lo) / (hi - lo) * (size_range[1] - size_range[0])
    else:
        sizes = np.full(len(data), 180)
    ax.scatter(data[x], data[y], s=sizes, color=colors, edgecolor="white", linewidth=2.0, alpha=0.92, zorder=3)
    if trend and len(data) >= 3:
        mask = np.isfinite(data[x].astype(float)) & np.isfinite(data[y].astype(float))
        if mask.sum() >= 3:
            coef = np.polyfit(data.loc[mask, x].astype(float), data.loc[mask, y].astype(float), deg=1)
            xs = np.linspace(data.loc[mask, x].min(), data.loc[mask, x].max(), 100)
            ax.plot(xs, coef[0] * xs + coef[1], color=NEUTRAL["subtle"], lw=1.8, ls=(0, (4, 4)), zorder=2)
    if annotate and label:
        label_points(ax, data, x=x, y=y, label=label)
    clean_axes(ax, grid_axis="both")


def rate_bar_ci(ax, labels: Sequence[str], rates: Sequence[float], lo: Sequence[float], hi: Sequence[float],
                baseline: float | None = None, label_rotation: float = 16) -> None:
    """Vertical rate bars with Wilson-style confidence intervals."""
    rates = np.asarray(rates, dtype=float)
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    yerr = np.vstack([rates - lo, hi - rates])
    colors = quality_colors(rates)
    x = np.arange(len(labels))
    ax.bar(x, rates, color=colors, width=0.64, yerr=yerr, capsize=6, edgecolor="white", linewidth=1.4,
           error_kw={"elinewidth": 2.2, "ecolor": BRAND["indigo"], "capthick": 2.2}, zorder=3)
    for i, v in enumerate(rates):
        ax.text(i, hi[i] + 0.03, f"{v:.0%}", ha="center", fontsize=15, fontweight="bold",
                color=NEUTRAL["ink"], bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none"))
    if baseline is not None:
        ax.axhline(baseline, ls=(0, (4, 4)), color=NEUTRAL["subtle"], lw=1.8, alpha=0.7, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=label_rotation, ha="right")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    clean_axes(ax, grid_axis="y")


def cost_quality_quadrant(ax, data: pd.DataFrame, *, x: str, y: str, label: str,
                          x_baseline: float | None = None, y_baseline: float | None = None,
                          xlim: tuple[float, float] | None = None, ylim: tuple[float, float] | None = None) -> None:
    """Scatter quadrant for cost-vs-quality stories with baseline bands."""
    colors = quality_colors(data[y].values)
    if y_baseline is not None:
        top = ylim[1] if ylim else max(1.0, float(data[y].max()) * 1.05)
        ax.axhspan(y_baseline, top, color=BRAND["indigo"], alpha=0.06, zorder=0)
        ax.axhline(y_baseline, ls=(0, (4, 4)), lw=1.6, color=NEUTRAL["subtle"], alpha=0.6)
    if x_baseline is not None:
        ax.axvline(x_baseline, ls=(0, (4, 4)), lw=1.6, color=NEUTRAL["subtle"], alpha=0.6)
    ax.scatter(data[x], data[y], s=430, color=colors, edgecolor="white", linewidth=3, zorder=3)
    for _, r in data.iterrows():
        ax.text(r[x], r[y], str(r[label]), ha="center", va="center", fontsize=11.5,
                fontweight="bold", color=NEUTRAL["ink"])
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("$%.3f"))
    clean_axes(ax, grid_axis="both")


def matrix_heatmap(ax, table: pd.DataFrame, *, annot: bool = True, fmt: str | Callable[[float], str] = ".0%",
                   cmap=None, vmin: float | None = None, vmax: float | None = None,
                   cbar_label: str = "value", mask: pd.DataFrame | np.ndarray | None = None) -> None:
    """General heatmap for rates, deltas, counts, correlation, or confusion matrices.

    Falls back to Matplotlib imshow if seaborn is unavailable. Use rate_heatmap() for the
    common success-rate case.
    """
    cmap = cmap or sequential_cmap()
    plot_data = table.astype(float).copy()
    if mask is not None:
        plot_data = plot_data.mask(mask)
    if sns is not None:
        annot_data = None
        if annot:
            if callable(fmt):
                annot_data = plot_data.map(lambda v: "" if pd.isna(v) else fmt(v))
                fmt_arg = ""
            else:
                fmt_arg = fmt
        else:
            fmt_arg = fmt if isinstance(fmt, str) else ""
        sns.heatmap(plot_data, annot=annot_data if annot_data is not None else annot, fmt=fmt_arg, cmap=cmap,
                    vmin=vmin, vmax=vmax, linewidths=3, linecolor="white",
                    cbar_kws={"label": cbar_label, "shrink": .85}, ax=ax)
    else:
        im = ax.imshow(plot_data.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.figure.colorbar(im, ax=ax, shrink=.85, label=cbar_label)
        if annot:
            for i in range(plot_data.shape[0]):
                for j in range(plot_data.shape[1]):
                    val = plot_data.iloc[i, j]
                    if pd.isna(val):
                        continue
                    txt = fmt(val) if callable(fmt) else format(val, fmt)
                    ax.text(j, i, txt, ha="center", va="center", fontsize=11, fontweight="bold")
        ax.set_xticks(range(plot_data.shape[1]), table.columns)
        ax.set_yticks(range(plot_data.shape[0]), table.index)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(length=0)
    ax.grid(False)


def rate_heatmap(ax, table: pd.DataFrame, *, annot: bool = False, vmin: float = 0, vmax: float = 1,
                 cbar_label: str = "success rate") -> None:
    """Heatmap with Braintrust rate palette and white cell gutters."""
    matrix_heatmap(ax, table, annot=annot, fmt=".0%", cmap=rate_cmap(), vmin=vmin, vmax=vmax,
                   cbar_label=cbar_label)


def distribution(ax, values: Sequence[float], *, bins: int = 24, label: str | None = None,
                 reference: float | None = None, value_fmt: Callable[[float], str] | None = None) -> None:
    """Histogram template for latency, cost, score, or token distributions."""
    ax.hist(values, bins=bins, color=BRAND["periwinkle"], edgecolor="white", linewidth=1.2, zorder=3)
    if reference is not None:
        ax.axvline(reference, color=BRAND["indigo"], lw=2.2, ls=(0, (4, 4)), zorder=4)
        label_text = value_fmt(reference) if value_fmt else f"{reference:g}"
        ax.text(reference, ax.get_ylim()[1] * 0.92, label_text, ha="left", va="top",
                fontsize=12.5, fontweight="bold", color=NEUTRAL["ink"],
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none"))
    if label:
        ax.set_xlabel(label)
    ax.set_ylabel("Count")
    clean_axes(ax, grid_axis="y")


def small_multiples(data: pd.DataFrame, *, facet: str, x: str, y: str, plotter: Callable,
                    ncols: int = 3, width: float = 12, height_per_row: float = 3.2, sharey: bool = True,
                    title: str | None = None, subtitle: str | None = None):
    """Create small multiples with consistent styling.

    plotter receives (ax, subset_dataframe). Keep each panel simple; use the shared header
    for the overall story and small bold facet titles inside each panel.
    """
    facets = list(pd.unique(data[facet]))
    nrows = math.ceil(len(facets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, max(height_per_row * nrows, 3.2)), sharey=sharey)
    axes = np.atleast_1d(axes).ravel()
    for ax, fac in zip(axes, facets):
        subset = data[data[facet] == fac]
        plotter(ax, subset)
        ax.set_title(str(fac), loc="left", fontsize=13.5, fontweight="bold", color=NEUTRAL["ink"], pad=10)
        clean_axes(ax, grid_axis="y")
    for ax in axes[len(facets):]:
        ax.axis("off")
    fig.subplots_adjust(top=0.82 if title else 0.92, left=0.08, right=0.97, bottom=0.10, hspace=0.38, wspace=0.22)
    if title:
        axes[0].text(0, 1.35, title, transform=axes[0].transAxes, fontsize=21, fontweight="bold",
                     color=NEUTRAL["ink"], ha="left", va="bottom")
    if subtitle:
        axes[0].text(0, 1.24, subtitle, transform=axes[0].transAxes, fontsize=14.5,
                     color=NEUTRAL["subtle"], ha="left", va="bottom")
    return fig, axes


def legend_quality_ci(ax, baseline_label: str = "baseline") -> None:
    """Add a compact legend matching the CI bar pattern."""
    handles = [
        Patch(fc=BRAND["purple"], ec="none", label="rate (color = quality)"),
        Line2D([0], [0], color=BRAND["indigo"], lw=2.4, label="95% CI"),
        Line2D([0], [0], color=NEUTRAL["subtle"], lw=1.8, ls=(0, (4, 4)), label=baseline_label),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.005),
              ncol=3, frameon=False, handlelength=1.6, columnspacing=1.6, fontsize=12)


def savefig(fig, path: str | os.PathLike) -> None:
    """Save a chart with the source notebooks' export defaults."""
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.35, facecolor=NEUTRAL["white"])
