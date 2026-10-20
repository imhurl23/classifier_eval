"""Rebuild task-specific safety charts from the completed source evaluations.

Run: .venv/bin/python -m analysis.plot_source_agreement
The historical 01_source_agreement artifact now shows conversation-label
agreement only. Request screening, answer safety, and refusal detection have
separate figures and a common interpretation report. No inference is performed.
"""

from pathlib import Path
from shutil import copyfile

from analysis.plot_safety_f2 import main as plot_f2
from analysis.plot_safety_usecases import OUT as USECASE_OUT
from analysis.plot_safety_usecases import SOURCE
from analysis.plot_safety_usecases import main as plot_usecases
from analysis.safety_interpretation import build_report

ROOT = Path(__file__).resolve().parents[1]
TABLE = SOURCE / "quality_by_configuration.csv"
OUT = ROOT / "analysis/output/development-2026-10-07"


def plot_source_agreement(table=TABLE, out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # Keep the old chart entry point while making each decision explicit.
    plot_usecases(source=Path(table).parent, out=USECASE_OUT)
    plot_f2()
    build_report()
    archive = out / "superseded-mixed-tasks"
    archive.mkdir(exist_ok=True)
    for stem in ["01_source_agreement", "01_source_agreement_intervals"]:
        for suffix in ["png", "svg"]:
            prior = out / f"{stem}.{suffix}"
            saved = archive / prior.name
            if prior.exists() and not saved.exists():
                copyfile(prior, saved)
    for filename in ["source_agreement_ranks.csv", "source_agreement_by_dataset.csv"]:
        prior = out / filename
        if prior.exists() and not (archive / filename).exists():
            prior.rename(archive / filename)
    for suffix in ["png", "svg"]:
        copyfile(
            USECASE_OUT / f"03_conversation_safety.{suffix}",
            out / f"01_source_agreement.{suffix}",
        )
        copyfile(
            USECASE_OUT / f"03_conversation_safety.{suffix}",
            out / f"01_source_agreement_intervals.{suffix}",
        )


if __name__ == "__main__":
    plot_source_agreement()
