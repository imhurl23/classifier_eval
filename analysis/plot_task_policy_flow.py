"""Flow diagram: decision asked -> policy the model sees -> model interfaces.

Static layout drawn from the verified 2026-10-12 source run
(analysis/output/source-2026-10-12/quality_by_configuration.csv). No model calls.
Regenerate with `.venv/bin/python -m analysis.plot_task_policy_flow`.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath

from analysis.braintrust_viz import BRAND, NEUTRAL, savefig, use_braintrust_theme

OUT = Path(__file__).resolve().parent / "output/methodology-2026-10-19"

POLICY = {
    "study": BRAND["indigo"],  # our written policy is sent with the request
    "native": BRAND["pink"],  # model's own built-in taxonomy; our policy is not sent
    "none": NEUTRAL["muted"],  # refusal definition only; no harm policy involved
}

COLS = {"task": (0.00, 0.30), "policy": (0.37, 0.28), "model": (0.72, 0.28)}

TASKS = [
    ("Is the exchange unsafe?", "Aegis pairs and one-turn prompts; WildGuard pairs"),
    ("Is the user request harmful?", "WildGuard pairs; XSTest benign lookalikes"),
    ("Is the assistant answer harmful?", "WildGuard pairs"),
    ("Did the assistant refuse?", "WildGuard pairs"),
]
POLICIES = [
    (
        "Our written policy",
        "study",
        "Sent with every request. Unsafe = violates this text.",
    ),
    (
        "Model's own taxonomy",
        "native",
        "Our policy is not sent. Built-in categories decide.",
    ),
    ("No harm policy", "none", "Refusal vs compliance definition only."),
]
MODELS = [
    (
        "11 custom-policy interfaces",
        "study",
        "Decisions Luna · Luna structured · Claude Sonnet 5.5 · Jev Choice · Jev Noul · Clef · "
        "Clef Flash · Safeguard 120B · Tev1 · Qwen3.5-4B letters · Qwen3.5-4B JSON",
    ),
    (
        "2 guards fed our policy",
        "study",
        "Granite Guardian 4.1 (BYOC criterion, per turn) · PolicyLM 1.7B (user turn only)",
    ),
    (
        "2 fixed-taxonomy guards",
        "native",
        "Qwen3Guard-Gen 8B · Nemotron Safety Guard v3",
    ),
]
# task index -> policy index
EDGES_TP = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 2)]
# policy index -> model index
EDGES_PM = [(0, 0), (0, 1), (1, 2), (2, 0)]


def _stack(n, heights, top=0.97, bottom=0.03):
    gap = (top - bottom - sum(heights)) / max(n - 1, 1)
    pos, y = [], top
    for h in heights:
        pos.append((y - h, h))
        y -= h + gap
    return pos


def _box(ax, col, y0, h, title, body, accent, wrap):
    x0, w = COLS[col]
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0,rounding_size=0.012",
            fc=NEUTRAL["white"],
            ec=NEUTRAL["thin"],
            lw=1.2,
            zorder=3,
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            0.007,
            h,
            boxstyle="round,pad=0,rounding_size=0.003",
            fc=accent,
            ec="none",
            zorder=4,
        )
    )
    ax.text(
        x0 + 0.025,
        y0 + h - 0.02,
        title,
        fontsize=14,
        fontweight="bold",
        color=NEUTRAL["ink"],
        ha="left",
        va="top",
        zorder=5,
    )
    ax.text(
        x0 + 0.025,
        y0 + h - 0.068,
        textwrap.fill(body, wrap),
        fontsize=11.5,
        color=NEUTRAL["subtle"],
        ha="left",
        va="top",
        zorder=5,
        linespacing=1.3,
    )


def _edge(ax, a, b, color, lw=3.0):
    (x0, y0), (x1, y1) = a, b
    xm = (x0 + x1) / 2
    path = MPath(
        [(x0, y0), (xm, y0), (xm, y1), (x1, y1)],
        [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4],
    )
    ax.add_patch(
        PathPatch(
            path, fc="none", ec=color, lw=lw, alpha=0.85, zorder=2, capstyle="round"
        )
    )


def _port(col, pos, i, k, n, side):
    x0, w = COLS[col]
    y0, h = pos[i]
    return (x0 + (w if side == "r" else 0), y0 + h * (n - k) / (n + 1))


def _connect(ax, edges, src_col, src_pos, dst_col, dst_pos, color_of):
    out = {}
    inn = {}
    for a, b in edges:
        out[a] = out.get(a, 0) + 1
        inn[b] = inn.get(b, 0) + 1
    oi = {}
    ii = {}
    for a, b in edges:
        oi.setdefault(a, 0)
        ii.setdefault(b, 0)
        _edge(
            ax,
            _port(src_col, src_pos, a, oi[a], out[a], "r"),
            _port(dst_col, dst_pos, b, ii[b], inn[b], "l"),
            color_of(a, b),
        )
        oi[a] += 1
        ii[b] += 1


def main() -> None:
    use_braintrust_theme()
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 7.6))
    fig.subplots_adjust(top=0.80, bottom=0.08, left=0.03, right=0.985)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.grid(False)

    t_pos = _stack(4, [0.16] * 4)
    p_pos = _stack(3, [0.18] * 3)
    m_pos = _stack(3, [0.34, 0.20, 0.16])

    _connect(
        ax,
        EDGES_TP,
        "task",
        t_pos,
        "policy",
        p_pos,
        lambda a, b: POLICY[POLICIES[b][1]],
    )
    _connect(
        ax,
        EDGES_PM,
        "policy",
        p_pos,
        "model",
        m_pos,
        lambda a, b: POLICY[POLICIES[a][1]],
    )

    for i, (t, b) in enumerate(TASKS):
        _box(ax, "task", *t_pos[i], t, b, NEUTRAL["faint"], 44)
    for i, (t, k, b) in enumerate(POLICIES):
        _box(ax, "policy", *p_pos[i], t, b, POLICY[k], 40)
    for i, (t, k, b) in enumerate(MODELS):
        _box(ax, "model", *m_pos[i], t, b, POLICY[k], 40)

    for col, label in [
        ("task", "Decision asked"),
        ("policy", "Policy the model sees"),
        ("model", "Models"),
    ]:
        ax.text(
            COLS[col][0],
            0.995,
            label.upper(),
            fontsize=11,
            fontweight="bold",
            color=NEUTRAL["subtle"],
            ha="left",
            va="bottom",
        )

    fig.text(
        0.03,
        0.965,
        "Which policy each model is judged against",
        fontsize=22,
        fontweight="bold",
        color=NEUTRAL["ink"],
        ha="left",
        va="top",
    )
    fig.text(
        0.03,
        0.915,
        "Most models read our written policy. The two fixed-taxonomy guards never see it, "
        "so their scores measure a different boundary.",
        fontsize=13.5,
        color=NEUTRAL["subtle"],
        ha="left",
        va="top",
    )
    fig.text(
        0.03,
        0.03,
        "Jev Noul answers the exchange question only. Guards' exchange verdicts are the OR of "
        "their two turn verdicts. Verified 2026-10-12 source run.",
        fontsize=10,
        color=NEUTRAL["muted"],
        ha="left",
        va="bottom",
    )

    savefig(fig, OUT / "task_policy_flow.png")
    savefig(fig, OUT / "task_policy_flow.svg")
    print(OUT / "task_policy_flow.png")


if __name__ == "__main__":
    main()
