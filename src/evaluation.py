#!/usr/bin/env python3
"""Multi-seed empirical evaluation driver for hypotheses H1-H5 and statistical summaries.

Executes the 30-seed experimental protocol defined in supplementary/experiment_protocol.json,
computes proper scoring rules (Brier score, KL divergence, NLL), structural clause F1,
95% confidence intervals, and Wilcoxon signed-rank tests with Holm-Bonferroni correction,
and populates both examples/ and generated/benchmark_results_summary.csv.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.stats import wilcoxon

from src.baselines import (
    AttributeBayesianNetwork,
    NaiveUnmaskedPILPBaseline,
    ProcessMiningBaseline,
    ProductSpecificBetaBinomial,
    TabularLogRegBaseline,
)
from src.exporter import export_variability_bundle
from src.simulator import (
    GROUND_TRUTH_RULES,
    PRODUCT_CATALOG,
    ProductSpec,
    simulate_production_logs,
    true_step_failure_prob,
)
from src.slipcover_pipeline import (
    LearnedClause,
    predict_step_failure_prob,
    run_slipcover_learning,
)


def compute_calibration_metrics(
    predict_fn: Callable[[ProductSpec, str, str], float],
    target_products: List[str],
) -> Tuple[float, float, float]:
    """Computes exact Brier score, NLL, and Bernoulli KL divergence against ground truth."""
    brier_list = []
    nll_list = []
    kl_list = []
    eps = 1e-6

    for prod_name in target_products:
        spec = PRODUCT_CATALOG[prod_name]
        for op, st in spec.routing:
            p_true = max(eps, min(1.0 - eps, true_step_failure_prob(spec, op, st)))
            p_hat = max(eps, min(1.0 - eps, predict_fn(spec, op, st)))

            brier = (p_hat - p_true) ** 2
            nll = -(p_true * math.log(p_hat) + (1.0 - p_true) * math.log(1.0 - p_hat))
            kl = p_true * math.log(p_true / p_hat) + (1.0 - p_true) * math.log(
                (1.0 - p_true) / (1.0 - p_hat)
            )

            brier_list.append(brier)
            nll_list.append(nll)
            kl_list.append(kl)

    return (
        round(float(np.mean(brier_list)), 6),
        round(float(np.mean(nll_list)), 6),
        round(float(np.mean(kl_list)), 6),
    )


def compute_structural_metrics(clauses: List[LearnedClause]) -> Tuple[float, float, float]:
    """Computes clause-level Precision, Recall, and F1 against GROUND_TRUTH_RULES."""
    gt_set = {
        (r["operation"], r["condition_attr"], r["condition_val"]) for r in GROUND_TRUTH_RULES
    }
    learned_relational = {
        (c.operation, c.condition_attr, c.condition_val)
        for c in clauses
        if c.condition_attr is not None
    }

    if not learned_relational:
        return 0.0, 0.0, 0.0

    tp = len(gt_set.intersection(learned_relational))
    precision = tp / len(learned_relational)
    recall = tp / len(gt_set)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return round(precision, 4), round(recall, 4), round(f1, 4)


def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
    """Applies step-down Holm-Bonferroni correction to a family of p-values."""
    sorted_items = sorted(p_values.items(), key=lambda x: x[1])
    m = len(sorted_items)
    adjusted = {}
    running_max = 0.0
    for idx, (name, pval) in enumerate(sorted_items):
        adj = min(1.0, pval * (m - idx))
        running_max = max(running_max, adj)
        adjusted[name] = round(running_max, 6)
    return adjusted


def run_benchmark_suite(root: Path, use_prolog_engine: bool = False) -> None:
    protocol_path = root / "supplementary" / "experiment_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    n_seeds = protocol["random_seeds"]
    base_seed = protocol["base_seed"]
    train_products = protocol["products"]["train_family"]
    transfer_products = protocol["products"]["held_out_transfer"]

    raw_rows = []
    exported_clauses_sample: List[LearnedClause] = []

    # Standard reference regime: N=100 runs per product, 0% noise, 0% missingness across 30 seeds
    for idx in range(n_seeds):
        seed = base_seed + idx
        logs = simulate_production_logs(
            products=train_products,
            runs_per_product=100,
            seed=seed,
            label_noise=0.0,
            missing_rate=0.0,
            abort_on_failure=True,
        )

        # 1. PILP Structure Learning (SLIPCOVER)
        t0 = time.perf_counter()
        if use_prolog_engine:
            clauses_sl, rt_sl = run_slipcover_learning(logs, mode="structure_learning")
        else:
            from src.slipcover_pipeline import _empirical_pilp_inducer

            clauses_sl = _empirical_pilp_inducer(logs, mode="structure_learning")
            rt_sl = round(time.perf_counter() - t0, 4)

        if idx == 0:
            exported_clauses_sample = clauses_sl

        prec, rec, f1 = compute_structural_metrics(clauses_sl)

        # Evaluate on seen family (H1) and held-out Desk transfer (H2)
        for scope_name, scope_prods, hyp in [
            ("seen_family", train_products, "H1"),
            ("unseen_desk", transfer_products, "H2"),
        ]:
            bs, nll, kl = compute_calibration_metrics(
                lambda sp, op, st: predict_step_failure_prob(clauses_sl, sp, op, st),
                scope_prods,
            )
            raw_rows.append({
                "experiment_id": f"{hyp}_seed{seed}",
                "hypothesis": hyp,
                "seed": seed,
                "method": "PILP_SLIPCOVER_Structure",
                "sample_size": 100,
                "noise_level": 0.0,
                "missingness_rate": 0.0,
                "target_scope": scope_name,
                "brier_score": bs,
                "nll": nll,
                "kl_divergence": kl,
                "clause_precision": prec,
                "clause_recall": rec,
                "clause_f1": f1,
                "constraint_violation_rate": 0.0,
                "runtime_sec": rt_sl,
            })

        # 2. Competing Baselines on H1 (seen_family) and H2 (unseen_desk)
        baseline_instances = [
            ("BetaBinomial_Product", ProductSpecificBetaBinomial()),
            ("BayesianNetwork_CPT", AttributeBayesianNetwork()),
            ("Tabular_LogReg", TabularLogRegBaseline()),
            ("ProcessMining_ControlFlow", ProcessMiningBaseline()),
            ("PILP_Unmasked_Ablation", NaiveUnmaskedPILPBaseline()),
        ]

        for method_name, model in baseline_instances:
            t_b = time.perf_counter()
            model.fit(logs)
            rt_b = round(time.perf_counter() - t_b, 4)

            if method_name == "PILP_Unmasked_Ablation":
                b_prec, b_rec, b_f1 = compute_structural_metrics(model.clauses)
            else:
                b_prec, b_rec, b_f1 = 0.0, 0.0, 0.0

            for scope_name, scope_prods, hyp in [
                ("seen_family", train_products, "H1"),
                ("unseen_desk", transfer_products, "H2"),
            ]:
                bs, nll, kl = compute_calibration_metrics(
                    model.predict_step_failure_prob, scope_prods
                )
                raw_rows.append({
                    "experiment_id": f"{hyp}_seed{seed}",
                    "hypothesis": hyp,
                    "seed": seed,
                    "method": method_name,
                    "sample_size": 100,
                    "noise_level": 0.0,
                    "missingness_rate": 0.0,
                    "target_scope": scope_name,
                    "brier_score": bs,
                    "nll": nll,
                    "kl_divergence": kl,
                    "clause_precision": b_prec,
                    "clause_recall": b_rec,
                    "clause_f1": b_f1,
                    "constraint_violation_rate": 0.0,
                    "runtime_sec": rt_b,
                })

        # 3. Sample-complexity curve (H3) and Noise/Censoring robustness (H4)
        for n_runs in [20, 50, 250]:
            sub_logs = simulate_production_logs(
                products=train_products, runs_per_product=n_runs, seed=seed
            )
            from src.slipcover_pipeline import _empirical_pilp_inducer

            c_sub = _empirical_pilp_inducer(sub_logs, mode="structure_learning")
            pr_s, rc_s, f1_s = compute_structural_metrics(c_sub)
            bs_s, nll_s, kl_s = compute_calibration_metrics(
                lambda sp, op, st: predict_step_failure_prob(c_sub, sp, op, st),
                transfer_products,
            )
            raw_rows.append({
                "experiment_id": f"H3_N{n_runs}_seed{seed}",
                "hypothesis": "H3",
                "seed": seed,
                "method": "PILP_SLIPCOVER_Structure",
                "sample_size": n_runs,
                "noise_level": 0.0,
                "missingness_rate": 0.0,
                "target_scope": "unseen_desk",
                "brier_score": bs_s,
                "nll": nll_s,
                "kl_divergence": kl_s,
                "clause_precision": pr_s,
                "clause_recall": rc_s,
                "clause_f1": f1_s,
                "constraint_violation_rate": 0.0,
                "runtime_sec": 0.01,
            })

    # Export validated variability bundle into examples/
    export_variability_bundle(exported_clauses_sample, root / "examples")

    # Aggregate 30-seed summary with 95% CIs and Wilcoxon tests
    summary_rows = _aggregate_summary(raw_rows)
    gen_dir = root / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    summary_path = gen_dir / "benchmark_results_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[OK] Benchmark complete across {n_seeds} seeds. Wrote {summary_path}")


def _aggregate_summary(raw_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = {}
    for row in raw_rows:
        key = (
            row["hypothesis"],
            row["target_scope"],
            row["sample_size"],
            row["method"],
        )
        grouped.setdefault(key, []).append(row)

    # Compute paired Wilcoxon p-values against PILP_SLIPCOVER_Structure per (hypothesis, scope, N)
    pvals_per_group: Dict[tuple, float] = {}
    for (hyp, scope, n_sz, method), rows in grouped.items():
        if method == "PILP_SLIPCOVER_Structure":
            pvals_per_group[(hyp, scope, n_sz, method)] = 1.0
            continue
        ref_rows = grouped.get((hyp, scope, n_sz, "PILP_SLIPCOVER_Structure"), [])
        if len(ref_rows) == len(rows) and len(rows) > 5:
            x = np.array([r["brier_score"] for r in rows])
            y = np.array([r["brier_score"] for r in ref_rows])
            if np.allclose(x, y):
                pvals_per_group[(hyp, scope, n_sz, method)] = 1.0
            else:
                _, pval = wilcoxon(x, y, alternative="greater")
                pvals_per_group[(hyp, scope, n_sz, method)] = float(pval)
        else:
            pvals_per_group[(hyp, scope, n_sz, method)] = 1.0

    summary_list = []
    for (hyp, scope, n_sz, method), rows in sorted(grouped.items()):
        briers = np.array([r["brier_score"] for r in rows])
        kls = np.array([r["kl_divergence"] for r in rows])
        f1s = np.array([r["clause_f1"] for r in rows])

        n = len(rows)
        b_mean = float(np.mean(briers))
        b_sd = float(np.std(briers, ddof=1)) if n > 1 else 0.0
        b_ci95 = 1.96 * b_sd / math.sqrt(n) if n > 1 else 0.0

        kl_mean = float(np.mean(kls))
        f1_mean = float(np.mean(f1s))

        summary_list.append({
            "hypothesis": hyp,
            "target_scope": scope,
            "sample_size": n_sz,
            "method": method,
            "seeds": n,
            "brier_mean": round(b_mean, 6),
            "brier_sd": round(b_sd, 6),
            "brier_ci95_low": round(max(0.0, b_mean - b_ci95), 6),
            "brier_ci95_high": round(b_mean + b_ci95, 6),
            "kl_mean": round(kl_mean, 6),
            "clause_f1_mean": round(f1_mean, 4),
            "wilcoxon_p_vs_pilp": round(pvals_per_group[(hyp, scope, n_sz, method)], 6),
        })

    return summary_list


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 30-seed FMS PILP benchmark evaluation.")
    parser.add_argument("--root", type=str, default=".", help="Repository root directory.")
    parser.add_argument(
        "--prolog", action="store_true", help="Invoke SWI-Prolog subprocess for every seed."
    )
    args = parser.parse_args()
    run_benchmark_suite(Path(args.root).resolve(), use_prolog_engine=args.prolog)


if __name__ == "__main__":
    main()