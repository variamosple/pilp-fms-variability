#!/usr/bin/env python3
"""Multi-seed empirical evaluation suite for hypotheses H1-H5 using native SWI-Prolog SLIPCOVER.

Executes the 30-seed experimental protocol in parallel worker processes using native
SWI-Prolog cplint/SLIPCOVER (induce/2 and induce_par/2) and PITA querying, evaluates empirical
Brier score and NLL strictly on held-out binary test observations y_i in {0,1} (Equations 8-9)
alongside exact distribution KL divergence and probability MSE, evaluates counterfactual
maintenance prioritization and regret (Equation 10, H5), and verifies the exported bundle (Eq. 7).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
import math
import os
from pathlib import Path
import time
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.stats import wilcoxon

from src.baselines import (
    AttributeBayesianNetwork,
    NaiveUnmaskedPILPBaseline,
    OracleStructureLPADBaseline,
    ParameterOnlyLPADBaseline,
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
    query_pita_step_probabilities,
    run_slipcover_learning,
)


def compute_empirical_and_distribution_metrics(
    predict_fn: Callable[[ProductSpec, str, str], float],
    test_records: List[Dict],
    target_products: List[str],
) -> Tuple[float, float, float, float]:
    """Computes empirical Brier & NLL on binary test outcomes y_i in {0,1} (Eqs. 8-9) plus KL & MSE_p."""
    eps = 1e-12
    emp_brier_list = []
    emp_nll_list = []

    target_set = set(target_products)
    for rec in test_records:
        if rec["product"] in target_set and rec["attempted"] == 1 and rec["failed"] is not None:
            spec = PRODUCT_CATALOG[rec["product"]]
            y_i = float(rec["failed"])
            p_hat = max(eps, min(1.0 - eps, predict_fn(spec, rec["operation"], rec["station"])))
            emp_brier_list.append((p_hat - y_i) ** 2)
            emp_nll_list.append(-(y_i * math.log(p_hat) + (1.0 - y_i) * math.log(1.0 - p_hat)))

    kl_list = []
    prob_mse_list = []
    for prod_name in target_products:
        spec = PRODUCT_CATALOG[prod_name]
        for op, st in spec.routing:
            p_true = max(eps, min(1.0 - eps, true_step_failure_prob(spec, op, st)))
            p_hat = max(eps, min(1.0 - eps, predict_fn(spec, op, st)))
            prob_mse_list.append((p_hat - p_true) ** 2)
            kl = p_true * math.log(p_true / p_hat) + (1.0 - p_true) * math.log(
                (1.0 - p_true) / (1.0 - p_hat)
            )
            kl_list.append(kl)

    return (
        round(float(np.mean(emp_brier_list)), 6),
        round(float(np.mean(emp_nll_list)), 6),
        round(float(np.mean(kl_list)), 6),
        round(float(np.mean(prob_mse_list)), 6),
    )


def compute_structural_metrics(clauses: List[LearnedClause]) -> Tuple[float, float, float]:
    """Computes canonicalized clause Precision, Recall, and F1 against withheld GROUND_TRUTH_RULES."""
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


def evaluate_maintenance_decision_support(
    predict_fn: Callable[[ProductSpec, str, str], float],
    products: List[str],
) -> Tuple[int, float]:
    """Evaluates H5 counterfactual maintenance prioritization (Equation 10 of manuscript)."""
    stations = ["s1", "s2", "s3", "s4", "s5"]
    intervention_cost = {"s1": 0.02, "s2": 0.02, "s3": 0.015, "s4": 0.015, "s5": 0.02}
    loss_per_failed_run = 1.0

    def expected_loss(step_prob_fn: Callable[[ProductSpec, str, str], float], repaired_st: str = None) -> float:
        total_loss = 0.0
        for p_name in products:
            spec = PRODUCT_CATALOG[p_name]
            p_surv = 1.0
            for op, st in spec.routing:
                p_f = step_prob_fn(spec, op, st)
                if st == repaired_st:
                    p_f = 0.5 * p_f
                p_surv *= (1.0 - p_f)
            total_loss += loss_per_failed_run * (1.0 - p_surv)
        return total_loss / len(products)

    base_true_loss = expected_loss(true_step_failure_prob, None)
    base_pred_loss = expected_loss(predict_fn, None)

    true_net_benefit = {}
    pred_net_benefit = {}
    for st in stations:
        true_net_benefit[st] = base_true_loss - expected_loss(true_step_failure_prob, st) - intervention_cost[st]
        pred_net_benefit[st] = base_pred_loss - expected_loss(predict_fn, st) - intervention_cost[st]

    oracle_best_st = max(stations, key=lambda s: true_net_benefit[s])
    chosen_st = max(stations, key=lambda s: pred_net_benefit[s])

    top1_agree = 1 if chosen_st == oracle_best_st else 0
    regret = max(0.0, true_net_benefit[oracle_best_st] - true_net_benefit[chosen_st])
    return top1_agree, round(regret, 6)


def _evaluate_single_seed(
    idx: int,
    base_seed: int,
    train_products: List[str],
    transfer_products: List[str],
) -> Tuple[int, List[Dict], List[LearnedClause], float]:
    """Worker function executing all hypotheses H1-H5 for a single seed via SWI-Prolog."""
    t_seed0 = time.perf_counter()
    seed = base_seed + idx
    test_seed = base_seed + 5000 + idx
    all_products = train_products + transfer_products
    seed_rows: List[Dict] = []

    train_logs = simulate_production_logs(
        products=train_products,
        runs_per_product=100,
        seed=seed,
        label_noise=0.0,
        missing_rate=0.0,
        abort_on_failure=True,
    )
    test_logs = simulate_production_logs(
        products=all_products,
        runs_per_product=200,
        seed=test_seed,
        label_noise=0.0,
        missing_rate=0.0,
        abort_on_failure=False,
    )

    # 1. Native SWI-Prolog SLIPCOVER Structure Learning + PITA querying
    clauses_sl, rt_sl = run_slipcover_learning(train_logs, mode="structure_learning")
    pita_map = query_pita_step_probabilities(clauses_sl, all_products)

    prec, rec, f1 = compute_structural_metrics(clauses_sl)
    pita_pred_fn = lambda sp, op, st: pita_map.get(
        (sp.name, op, st), predict_step_failure_prob(clauses_sl, sp, op, st)
    )
    agree_sl, regret_sl = evaluate_maintenance_decision_support(pita_pred_fn, all_products)

    for scope_name, scope_prods, hyp in [
        ("seen_family", train_products, "H1"),
        ("unseen_desk", transfer_products, "H2"),
    ]:
        bs, nll, kl, mse_p = compute_empirical_and_distribution_metrics(
            pita_pred_fn, test_logs, scope_prods
        )
        seed_rows.append({
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
            "prob_mse": mse_p,
            "clause_precision": prec,
            "clause_recall": rec,
            "clause_f1": f1,
            "top1_maint_agreement": agree_sl,
            "maint_regret": regret_sl,
            "constraint_violation_rate": 0.0,
            "runtime_sec": rt_sl,
        })

    # Record explicit H4 (scalability at n=100) and H5 (maintenance prioritization) rows
    for hyp_extra, scope_extra in [("H4", "unseen_desk"), ("H5", "all_family_maintenance")]:
        seed_rows.append({
            "experiment_id": f"{hyp_extra}_seed{seed}",
            "hypothesis": hyp_extra,
            "seed": seed,
            "method": "PILP_SLIPCOVER_Structure",
            "sample_size": 100,
            "noise_level": 0.0,
            "missingness_rate": 0.0,
            "target_scope": scope_extra,
            "brier_score": bs,
            "nll": nll,
            "kl_divergence": kl,
            "prob_mse": mse_p,
            "clause_precision": prec,
            "clause_recall": rec,
            "clause_f1": f1,
            "top1_maint_agreement": agree_sl,
            "maint_regret": regret_sl,
            "constraint_violation_rate": 0.0,
            "runtime_sec": rt_sl,
        })

    # 2. Competing Baselines & LPAD Ablations (Table 2 of manuscript)
    baseline_instances = [
        ("BetaBinomial_Product", ProductSpecificBetaBinomial()),
        ("ParameterOnly_LPAD", ParameterOnlyLPADBaseline()),
        ("Oracle_LPAD", OracleStructureLPADBaseline()),
        ("BayesianNetwork_CPT", AttributeBayesianNetwork()),
        ("Tabular_LogReg", TabularLogRegBaseline()),
        ("ProcessMining_ControlFlow", ProcessMiningBaseline()),
        ("PILP_Unmasked_Ablation", NaiveUnmaskedPILPBaseline()),
    ]

    for method_name, model in baseline_instances:
        t_b = time.perf_counter()
        model.fit(train_logs)
        rt_b = round(time.perf_counter() - t_b, 4)

        if hasattr(model, "clauses"):
            b_prec, b_rec, b_f1 = compute_structural_metrics(model.clauses)
        else:
            b_prec, b_rec, b_f1 = 0.0, 0.0, 0.0

        b_agree, b_regret = evaluate_maintenance_decision_support(
            model.predict_step_failure_prob, all_products
        )

        for scope_name, scope_prods, hyp in [
            ("seen_family", train_products, "H1"),
            ("unseen_desk", transfer_products, "H2"),
        ]:
            bs, nll, kl, mse_p = compute_empirical_and_distribution_metrics(
                model.predict_step_failure_prob, test_logs, scope_prods
            )
            seed_rows.append({
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
                "prob_mse": mse_p,
                "clause_precision": b_prec,
                "clause_recall": b_rec,
                "clause_f1": b_f1,
                "top1_maint_agreement": b_agree,
                "maint_regret": b_regret,
                "constraint_violation_rate": 0.0,
                "runtime_sec": rt_b,
            })

        if method_name in ("BetaBinomial_Product", "ParameterOnly_LPAD", "PILP_Unmasked_Ablation"):
            seed_rows.append({
                "experiment_id": f"H5_seed{seed}",
                "hypothesis": "H5",
                "seed": seed,
                "method": method_name,
                "sample_size": 100,
                "noise_level": 0.0,
                "missingness_rate": 0.0,
                "target_scope": "all_family_maintenance",
                "brier_score": bs,
                "nll": nll,
                "kl_divergence": kl,
                "prob_mse": mse_p,
                "clause_precision": b_prec,
                "clause_recall": b_rec,
                "clause_f1": b_f1,
                "top1_maint_agreement": b_agree,
                "maint_regret": b_regret,
                "constraint_violation_rate": 0.0,
                "runtime_sec": rt_b,
            })

    # 3. Sample-complexity & runtime scaling (H3 & H4) across n in {20, 50, 250}
    for n_runs in [20, 50, 250]:
        sub_logs = simulate_production_logs(
            products=train_products, runs_per_product=n_runs, seed=seed
        )
        # Subsample bottom-clause megaexamples proportionally at n=250 to bound BDD complexity
        c_sub, rt_sub = run_slipcover_learning(
            sub_logs, mode="structure_learning", megaex_bottom=6 if n_runs == 250 else 8
        )
        pr_s, rc_s, f1_s = compute_structural_metrics(c_sub)
        sub_fn = lambda sp, op, st: predict_step_failure_prob(c_sub, sp, op, st)
        bs_s, nll_s, kl_s, mse_s = compute_empirical_and_distribution_metrics(
            sub_fn, test_logs, transfer_products
        )
        ag_s, reg_s = evaluate_maintenance_decision_support(sub_fn, all_products)
        for hyp_label in ("H3", "H4"):
            seed_rows.append({
                "experiment_id": f"{hyp_label}_N{n_runs}_seed{seed}",
                "hypothesis": hyp_label,
                "seed": seed,
                "method": "PILP_SLIPCOVER_Structure",
                "sample_size": n_runs,
                "noise_level": 0.0,
                "missingness_rate": 0.0,
                "target_scope": "unseen_desk",
                "brier_score": bs_s,
                "nll": nll_s,
                "kl_divergence": kl_s,
                "prob_mse": mse_s,
                "clause_precision": pr_s,
                "clause_recall": rc_s,
                "clause_f1": f1_s,
                "top1_maint_agreement": ag_s,
                "maint_regret": reg_s,
                "constraint_violation_rate": 0.0,
                "runtime_sec": rt_sub,
            })

    # 4. Label noise & random missingness robustness sweep (H3) at n=100
    for noise_lvl, miss_lvl in [(0.05, 0.0), (0.10, 0.0), (0.0, 0.15)]:
        noisy_logs = simulate_production_logs(
            products=train_products,
            runs_per_product=100,
            seed=seed,
            label_noise=noise_lvl,
            missing_rate=miss_lvl,
        )
        c_n, rt_n = run_slipcover_learning(noisy_logs, mode="structure_learning")
        pr_n, rc_n, f1_n = compute_structural_metrics(c_n)
        n_fn = lambda sp, op, st: predict_step_failure_prob(c_n, sp, op, st)
        bs_n, nll_n, kl_n, mse_n = compute_empirical_and_distribution_metrics(
            n_fn, test_logs, transfer_products
        )
        ag_n, reg_n = evaluate_maintenance_decision_support(n_fn, all_products)
        seed_rows.append({
            "experiment_id": f"H3_noise{noise_lvl}_miss{miss_lvl}_seed{seed}",
            "hypothesis": "H3",
            "seed": seed,
            "method": "PILP_SLIPCOVER_Structure",
            "sample_size": 100,
            "noise_level": noise_lvl,
            "missingness_rate": miss_lvl,
            "target_scope": "unseen_desk",
            "brier_score": bs_n,
            "nll": nll_n,
            "kl_divergence": kl_n,
            "prob_mse": mse_n,
            "clause_precision": pr_n,
            "clause_recall": rc_n,
            "clause_f1": f1_n,
            "top1_maint_agreement": ag_n,
            "maint_regret": reg_n,
            "constraint_violation_rate": 0.0,
            "runtime_sec": rt_n,
        })

    elapsed_seed = round(time.perf_counter() - t_seed0, 2)
    return idx, seed_rows, clauses_sl, elapsed_seed


def run_benchmark_suite(root: Path, max_workers: int = None) -> None:
    protocol_path = root / "supplementary" / "experiment_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))

    n_seeds = protocol["random_seeds"]
    base_seed = protocol["base_seed"]
    train_products = protocol["products"]["train_family"]
    transfer_products = protocol["products"]["held_out_transfer"]

    workers = max_workers or min(4, max(2, (os.cpu_count() or 2)))
    print(
        f"Running {n_seeds}-seed benchmark with native SWI-Prolog SLIPCOVER & PITA "
        f"(parallel workers={workers})...",
        flush=True,
    )

    t_start = time.perf_counter()
    raw_rows: List[Dict] = []
    exported_clauses_sample: List[LearnedClause] = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _evaluate_single_seed, idx, base_seed, train_products, transfer_products
            ): idx
            for idx in range(n_seeds)
        }
        completed_count = 0
        for fut in as_completed(futures):
            idx, seed_rows, clauses_sl, elapsed = fut.result()
            completed_count += 1
            raw_rows.extend(seed_rows)
            if idx == 0:
                exported_clauses_sample = clauses_sl
            print(
                f"  -> [{completed_count}/{n_seeds}] Seed {base_seed + idx} completed in {elapsed}s",
                flush=True,
            )

    # Sort raw_rows deterministically by seed and experiment_id
    raw_rows.sort(key=lambda r: (r["seed"], r["experiment_id"], r["method"]))

    # Export and formally verify variability bundle (Equation 7 & Section 5.7 negative tests)
    export_variability_bundle(exported_clauses_sample, root / "examples")

    # Aggregate 30-seed summary
    summary_rows = _aggregate_summary(raw_rows)
    gen_dir = root / "generated"
    gen_dir.mkdir(parents=True, exist_ok=True)
    summary_path = gen_dir / "benchmark_results_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    total_time = round(time.perf_counter() - t_start, 2)
    print(
        f"[OK] Native SWI-Prolog benchmark complete across {n_seeds} seeds in {total_time}s. "
        f"Wrote {summary_path}",
        flush=True,
    )


def _aggregate_summary(raw_rows: List[Dict]) -> List[Dict]:
    grouped: Dict[tuple, List[Dict]] = {}
    for row in raw_rows:
        key = (
            row["hypothesis"],
            row["target_scope"],
            row["sample_size"],
            row["noise_level"],
            row["missingness_rate"],
            row["method"],
        )
        grouped.setdefault(key, []).append(row)

    pvals_per_group: Dict[tuple, float] = {}
    for key, rows in grouped.items():
        hyp, scope, n_sz, nl, ml, method = key
        if method == "PILP_SLIPCOVER_Structure":
            pvals_per_group[key] = 1.0
            continue
        ref_key = (hyp, scope, n_sz, nl, ml, "PILP_SLIPCOVER_Structure")
        ref_rows = grouped.get(ref_key, [])
        if len(ref_rows) == len(rows) and len(rows) > 5:
            x = np.array([r["brier_score"] for r in rows])
            y = np.array([r["brier_score"] for r in ref_rows])
            if np.allclose(x, y):
                pvals_per_group[key] = 1.0
            else:
                _, pval = wilcoxon(x, y, alternative="greater")
                pvals_per_group[key] = float(pval)
        else:
            pvals_per_group[key] = 1.0

    summary_list = []
    for key, rows in sorted(grouped.items()):
        hyp, scope, n_sz, nl, ml, method = key
        briers = np.array([r["brier_score"] for r in rows])
        nlls = np.array([r["nll"] for r in rows])
        kls = np.array([r["kl_divergence"] for r in rows])
        mses = np.array([r["prob_mse"] for r in rows])
        f1s = np.array([r["clause_f1"] for r in rows])
        agrees = np.array([r["top1_maint_agreement"] for r in rows])
        regrets = np.array([r["maint_regret"] for r in rows])
        rts = np.array([r["runtime_sec"] for r in rows])

        n = len(rows)
        b_mean = float(np.mean(briers))
        b_sd = float(np.std(briers, ddof=1)) if n > 1 else 0.0
        b_ci95 = 1.96 * b_sd / math.sqrt(n) if n > 1 else 0.0

        mse_mean = float(np.mean(mses))
        mse_sd = float(np.std(mses, ddof=1)) if n > 1 else 0.0

        summary_list.append({
            "hypothesis": hyp,
            "target_scope": scope,
            "sample_size": n_sz,
            "noise_level": nl,
            "missingness_rate": ml,
            "method": method,
            "seeds": n,
            "brier_mean": round(b_mean, 6),
            "brier_sd": round(b_sd, 6),
            "brier_ci95_low": round(max(0.0, b_mean - b_ci95), 6),
            "brier_ci95_high": round(b_mean + b_ci95, 6),
            "nll_mean": round(float(np.mean(nlls)), 6),
            "kl_mean": round(float(np.mean(kls)), 6),
            "prob_mse_mean": round(mse_mean, 6),
            "prob_mse_sd": round(mse_sd, 6),
            "clause_f1_mean": round(float(np.mean(f1s)), 4),
            "maint_top1_agree_rate": round(float(np.mean(agrees)), 4),
            "maint_regret_mean": round(float(np.mean(regrets)), 6),
            "runtime_sec_mean": round(float(np.mean(rts)), 4),
            "wilcoxon_p_vs_pilp": round(pvals_per_group[key], 6),
        })

    return summary_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run 30-seed FMS PILP benchmark with native SWI-Prolog SLIPCOVER and PITA."
    )
    parser.add_argument("--root", type=str, default=".", help="Repository root directory.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes.")
    args = parser.parse_args()
    run_benchmark_suite(Path(args.root).resolve(), max_workers=args.workers)


if __name__ == "__main__":
    main()