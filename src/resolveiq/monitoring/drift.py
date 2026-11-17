"""Compare recent requests with the training data and decide when to retrain.

Each monitored feature is tested in two ways:

- **Statistical test.** Kolmogorov-Smirnov for numeric features and a
  chi-square test of the category counts for categorical features.
- **Population stability index (PSI).** How much the distribution moved, on
  the bins of the reference data. Below 0.1 is usually read as stable, 0.1 to
  0.2 as a moderate shift and above 0.2 as a large one.

A feature counts as drifted only when both agree: the test is significant
*and* the PSI is at least the configured threshold. With hundreds of rows a
test alone flags shifts too small to matter, and PSI alone can move on tiny
samples.
"""

from __future__ import annotations

import base64
import html
import io
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

EPSILON = 1e-4  # floor for empty bins, keeps PSI finite


@dataclass(frozen=True)
class FeatureDrift:
    name: str
    kind: str  # "numeric" or "categorical"
    test: str
    statistic: float
    p_value: float
    psi: float
    drifted: bool
    reference_summary: str
    current_summary: str


@dataclass(frozen=True)
class DriftReport:
    features: list[FeatureDrift]
    n_reference: int
    n_current: int
    psi_threshold: float
    p_value: float

    @property
    def drifted(self) -> list[str]:
        return [f.name for f in self.features if f.drifted]

    @property
    def share_drifted(self) -> float:
        return len(self.drifted) / len(self.features) if self.features else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_reference": self.n_reference,
            "n_current": self.n_current,
            "psi_threshold": self.psi_threshold,
            "p_value": self.p_value,
            "share_drifted": round(self.share_drifted, 4),
            "drifted": self.drifted,
            "features": [asdict(f) for f in self.features],
        }


@dataclass(frozen=True)
class RetrainDecision:
    retrain: bool
    reasons: list[str] = field(default_factory=list)


def _psi(reference_share: np.ndarray, current_share: np.ndarray) -> float:
    ref = np.clip(reference_share, EPSILON, None)
    cur = np.clip(current_share, EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def numeric_psi(reference: Sequence[float], current: Sequence[float], bins: int = 10) -> float:
    """PSI on quantile bins of the reference data."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # constant reference: compare "equal" against "different"
        same = float(np.mean(cur == ref[0]))
        return _psi(np.array([1.0, 0.0]), np.array([same, 1 - same]))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_share = np.histogram(ref, edges)[0] / len(ref)
    cur_share = np.histogram(cur, edges)[0] / len(cur)
    return _psi(ref_share, cur_share)


def categorical_psi(reference: Sequence[str], current: Sequence[str]) -> float:
    ref_counts, cur_counts = Counter(reference), Counter(current)
    labels = sorted(set(ref_counts) | set(cur_counts))
    ref_share = np.array([ref_counts[label] for label in labels]) / len(reference)
    cur_share = np.array([cur_counts[label] for label in labels]) / len(current)
    return _psi(ref_share, cur_share)


def _numeric_drift(
    name: str, reference: Sequence[float], current: Sequence[float], psi_t: float, alpha: float
) -> FeatureDrift:
    from scipy.stats import ks_2samp

    result = ks_2samp(reference, current)
    psi = numeric_psi(reference, current)
    p_value = float(result.pvalue)
    return FeatureDrift(
        name=name,
        kind="numeric",
        test="Kolmogorov-Smirnov",
        statistic=round(float(result.statistic), 4),
        p_value=round(p_value, 6),
        psi=round(psi, 4),
        drifted=p_value < alpha and psi >= psi_t,
        reference_summary=_numeric_summary(reference),
        current_summary=_numeric_summary(current),
    )


def _categorical_drift(
    name: str, reference: Sequence[str], current: Sequence[str], psi_t: float, alpha: float
) -> FeatureDrift:
    from scipy.stats import chi2_contingency

    ref_counts, cur_counts = Counter(reference), Counter(current)
    labels = sorted(set(ref_counts) | set(cur_counts))
    table = np.array([[ref_counts[x] for x in labels], [cur_counts[x] for x in labels]])
    if len(labels) < 2:
        statistic, p_value = 0.0, 1.0
    else:
        result = chi2_contingency(table)
        statistic, p_value = float(result.statistic), float(result.pvalue)
    psi = categorical_psi(reference, current)
    return FeatureDrift(
        name=name,
        kind="categorical",
        test="Chi-square",
        statistic=round(statistic, 4),
        p_value=round(p_value, 6),
        psi=round(psi, 4),
        drifted=p_value < alpha and psi >= psi_t,
        reference_summary=_categorical_summary(reference),
        current_summary=_categorical_summary(current),
    )


def _numeric_summary(values: Sequence[float]) -> str:
    array = np.asarray(values, dtype=float)
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return f"median {median:.3g} (IQR {q1:.3g} to {q3:.3g})"


def _categorical_summary(values: Sequence[str], top: int = 3) -> str:
    counts = Counter(values)
    total = len(values)
    return ", ".join(f"{label} {n / total:.0%}" for label, n in counts.most_common(top))


def detect_drift(
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
    numeric: Sequence[str],
    categorical: Sequence[str],
    psi_threshold: float = 0.2,
    p_value: float = 0.05,
) -> DriftReport:
    """Test every listed feature of ``current`` against ``reference``."""
    n_reference = len(next(iter(reference.values()), []))
    n_current = len(next(iter(current.values()), []))
    if n_reference == 0 or n_current == 0:
        raise ValueError("Drift detection needs reference and current rows")
    missing = [n for n in [*numeric, *categorical] if n not in reference or n not in current]
    if missing:
        raise ValueError(f"Columns missing from the data: {missing}")
    features = [
        _numeric_drift(name, reference[name], current[name], psi_threshold, p_value)
        for name in numeric
    ] + [
        _categorical_drift(
            name,
            [str(v) for v in reference[name]],
            [str(v) for v in current[name]],
            psi_threshold,
            p_value,
        )
        for name in categorical
    ]
    return DriftReport(features, n_reference, n_current, psi_threshold, p_value)


def retrain_decision(
    report: DriftReport,
    feedback_total: int,
    feedback_down_rate: float,
    max_drift_share: float,
    max_down_rate: float,
    min_feedback: int,
) -> RetrainDecision:
    """Apply the retrain rule from docs/monitoring.md."""
    reasons = []
    if report.share_drifted >= max_drift_share:
        reasons.append(
            f"{len(report.drifted)} of {len(report.features)} monitored features drifted "
            f"({report.share_drifted:.0%}, limit {max_drift_share:.0%}): "
            + ", ".join(report.drifted)
        )
    if feedback_total >= min_feedback and feedback_down_rate > max_down_rate:
        reasons.append(
            f"Thumbs-down rate {feedback_down_rate:.0%} over {feedback_total} votes "
            f"is above {max_down_rate:.0%}"
        )
    return RetrainDecision(bool(reasons), reasons)


# ---------------------------------------------------------------------- report


def _histogram_png(name: str, reference: Sequence[Any], current: Sequence[Any], kind: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 2.6), dpi=110)
    if kind == "numeric":
        ref = np.asarray(reference, dtype=float)
        cur = np.asarray(current, dtype=float)
        edges = np.histogram_bin_edges(np.concatenate([ref, cur]), bins=20)
        ax.hist(ref, edges, density=True, alpha=0.55, label="reference", color="#2563eb")
        ax.hist(cur, edges, density=True, alpha=0.55, label="current", color="#ea580c")
    else:
        ref_counts, cur_counts = Counter(map(str, reference)), Counter(map(str, current))
        labels = [label for label, _ in (ref_counts + cur_counts).most_common(10)]
        positions = np.arange(len(labels))
        ax.barh(
            positions - 0.2,
            [ref_counts[x] / len(reference) for x in labels],
            0.4,
            label="reference",
            color="#2563eb",
        )
        ax.barh(
            positions + 0.2,
            [cur_counts[x] / len(current) for x in labels],
            0.4,
            label="current",
            color="#ea580c",
        )
        ax.set_yticks(positions, labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("share of rows")
    ax.set_title(name, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def render_html(
    report: DriftReport,
    decision: RetrainDecision,
    reference: Mapping[str, Sequence[Any]],
    current: Mapping[str, Sequence[Any]],
    title: str = "ResolveIQ drift report",
    notes: Sequence[str] = (),
) -> str:
    """A self-contained HTML report with one histogram per feature."""
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(f.name)}</td><td>{f.test}</td><td>{f.statistic:.4f}</td>"
        f"<td>{f.p_value:.4g}</td><td>{f.psi:.4f}</td>"
        f"<td class='{'bad' if f.drifted else 'good'}'>{'yes' if f.drifted else 'no'}</td>"
        f"<td>{html.escape(f.reference_summary)}</td><td>{html.escape(f.current_summary)}</td>"
        "</tr>"
        for f in report.features
    )
    charts = "".join(
        f"<figure><img alt='{html.escape(f.name)} distribution' src='data:image/png;base64,"
        f"{_histogram_png(f.name, reference[f.name], current[f.name], f.kind)}'/></figure>"
        for f in report.features
    )
    verdict = (
        "<p class='bad'><strong>Retrain recommended.</strong></p><ul>"
        + "".join(f"<li>{html.escape(r)}</li>" for r in decision.reasons)
        + "</ul>"
        if decision.retrain
        else "<p class='good'><strong>No retraining needed.</strong> No rule was triggered.</p>"
    )
    note_items = "".join(f"<li>{html.escape(n)}</li>" for n in notes)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
body {{font-family: system-ui, sans-serif; max-width: 1100px; margin: 24px auto; padding: 0 16px;
  color: #1f2937; background: #ffffff;}}
table {{border-collapse: collapse; width: 100%; font-size: 14px;}}
th, td {{border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left;}}
th {{background: #f3f4f6;}}
.bad {{color: #b91c1c;}} .good {{color: #15803d;}}
.charts {{display: flex; flex-wrap: wrap; gap: 12px;}}
figure {{margin: 0;}} img {{max-width: 100%;}}
</style></head><body>
<h1>{html.escape(title)}</h1>
<p>Reference rows: {report.n_reference}. Current rows: {report.n_current}.
A feature drifts when its test has p &lt; {report.p_value} and PSI &ge; {report.psi_threshold}.
Drifted features: {len(report.drifted)} of {len(report.features)}
({report.share_drifted:.0%}).</p>
{verdict}
<table><thead><tr><th>Feature</th><th>Test</th><th>Statistic</th><th>p-value</th><th>PSI</th>
<th>Drift</th><th>Reference</th><th>Current</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Distributions</h2><div class="charts">{charts}</div>
{f"<h2>Notes</h2><ul>{note_items}</ul>" if note_items else ""}
</body></html>
"""
