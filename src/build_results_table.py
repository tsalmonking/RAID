#!/usr/bin/env python3
"""Builds the paper's result tables from evaluate_detector.py output."""
import json
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Paper's row/column order (display name -> internal model name)
ORDER = [
    ("Ca24", "cavia2024"),
    ("Ch24C", "chen2024_clip"),
    ("Ch24CN", "chen2024_convnext"),
    ("Co23", "corvi2023"),
    ("K24", "koutlis2024"),
    ("O23", "ojha2023"),
    ("W20", "wang2020"),
]
MODELS = [m for _, m in ORDER]

# Order used by run_leave_one_out_attacks.sh's DETECTORS array when building
# the ensemble (and thus the folder name) -- must match exactly, independent
# of the paper's display order above.
ATTACK_ORDER = ["ojha2023", "corvi2023", "cavia2024", "chen2024_convnext",
                "chen2024_clip", "koutlis2024", "wang2020"]

BASE = os.path.join(SRC_DIR, "output/evaluate_detector/ADV/ELSA_TEST_1000")
CLEAN_BASE = os.environ.get(
    "RAID_CLEAN_BASE",
    os.path.join(SRC_DIR, "output/evaluate_detector/ELSA_TEST_1000"),
)


def model_list_repr(models):
    return "[" + ", ".join(f"'{m}'" for m in models) + "]"


def load_json_metrics(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["metrics"]["All"]


def load_metrics(source_models, eval_model, eps):
    """metrics['All'] for an attack source (models used to craft it)
    evaluated by eval_model, at the given epsilon."""
    src_repr = model_list_repr(source_models)
    return load_json_metrics(
        f"{BASE}/adv_raw_{src_repr}_{eps}_255_10_0.05/{eval_model}/results.json")


def load_clean(eval_model):
    return load_json_metrics(f"{CLEAN_BASE}/{eval_model}/results.json")


def cell_f1auc(adv, clean):
    if adv is None:
        return "---"
    return f"{adv['f1']:.2f}/{adv['auc']:.2f}"


def cell_asr(adv, clean):
    """ASR = (clean_AUC - adv_AUC) / clean_AUC * 100"""
    if adv is None or clean is None:
        return "---"
    return f"{(clean['auc'] - adv['auc']) / clean['auc'] * 100:.1f}"


def build_table(eps, cell_fn):
    rows = []

    # white-box rows: source = [row_model], evaluated against every column
    for disp, row_model in ORDER:
        row = [disp]
        for _, col_model in ORDER:
            row.append(cell_fn(load_metrics([row_model], col_model, eps),
                               load_clean(col_model)))
        rows.append(row)

    # N-model row: for each column, source = all models except that column,
    # evaluated by that same column model (the held-out one)
    row = ["N-model"]
    for _, col_model in ORDER:
        ensemble = [m for m in ATTACK_ORDER if m != col_model]
        row.append(cell_fn(load_metrics(ensemble, col_model, eps),
                           load_clean(col_model)))
    rows.append(row)

    return rows


def render(header, rows, title):
    widths = [max(len(str(r[i])) for r in ([header] + rows))
              for i in range(len(header))]

    def fmt_row(r):
        return "  ".join(str(c).ljust(w) for c, w in zip(r, widths))

    print(f"\n=== {title} ===")
    print(fmt_row(header))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))


def print_clean_table():
    rows = []
    for disp, model in ORDER:
        m = load_clean(model)
        if m is None:
            rows.append([disp, "---", "---", "---"])
        else:
            rows.append([disp, f"{m['f1']:.2f}", f"{m['acc']:.2f}",
                         f"{m['auc']:.2f}"])
    render(["Detector", "F1", "Acc", "AUROC"], rows,
           f"Table 1 -- clean performance  ({CLEAN_BASE})")


def print_merged_table(epsilons, cell_fn, title):
    """One table, budgets merged horizontally: each budget contributes a block
    of detector columns, with a spanning header row naming the budget."""
    labels = [disp for disp, _ in ORDER]
    n = len(ORDER)

    # rows: [label] + one cell per (eps, detector)
    rows = []
    for eps in epsilons:
        rows.append(build_table(eps, cell_fn))
    merged = []
    for i in range(len(rows[0])):
        row = [rows[0][i][0]]
        for block in rows:
            row.extend(block[i][1:])
        merged.append(row)

    header = ["Attack \\ Eval"] + labels * len(epsilons)
    widths = [max(len(str(r[i])) for r in ([header] + merged))
              for i in range(len(header))]

    def fmt_row(r):
        return "  ".join(str(c).ljust(w) for c, w in zip(r, widths))

    # spanning budget header, centred over each block of detector columns
    span = [" " * widths[0]]
    for b, eps in enumerate(epsilons):
        block = widths[1 + b * n: 1 + (b + 1) * n]
        w = sum(block) + 2 * (len(block) - 1)
        span.append(f"eps={eps}/255".center(w, "-"))

    print(f"\n=== {title} ===")
    print("  ".join(span))
    print(fmt_row(header))
    print("  ".join("-" * w for w in widths))
    for r in merged:
        print(fmt_row(r))


if __name__ == "__main__":
    epsilons = sys.argv[1:] or ["16", "32"]
    print_clean_table()

    # AUROC tables stay separate, one per budget
    header = ["Attack \\ Eval"] + [disp for disp, _ in ORDER]
    for eps in epsilons:
        render(header, build_table(eps, cell_f1auc),
               f"eps={eps}/255  (F1/AUROC, row=attacked model, col=evaluated model)")

    # ASR merged horizontally across budgets
    print_merged_table(epsilons, cell_asr,
                       "ASR %  (clean_AUC - adv_AUC) / clean_AUC * 100")
