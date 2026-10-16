#!/usr/bin/env python3
"""Statistical, tabular ML, Bayesian network, and process-mining baselines.

Implements the comparative baselines specified in Section 4.1 of the manuscript:
1. ProductSpecificBetaBinomial: Smoothed Beta(1,1) estimator per (product, station).
2. AttributeBayesianNetwork: Conditional probability table with Dirichlet smoothing.
3. TabularLogRegBaseline: One-hot encoded L2-regularized logistic regression.
4. ProcessMiningBaseline: Station-transition frequency model (control-flow discovery).
"""

from collections import defaultdict
import math
from typing import Dict, List

from src.simulator import ProductSpec


class ProductSpecificBetaBinomial:
    """Smoothed Beta(1,1) tabular estimator conditioned on (product, operation, station).

    Cannot generalize across products via shared relational attributes on unseen
    product variants (H2), falling back to station-level marginals.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0) -> None:
        self.alpha = alpha
        self.beta = beta
        self.prod_station_counts: Dict[tuple, List[int]] = defaultdict(lambda: [0, 0])
        self.station_counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

    def fit(self, records: List[Dict]) -> None:
        for r in records:
            if r["attempted"] == 1 and r["censored"] == 0 and r["failed"] is not None:
                key = (r["product"], r["operation"], r["station"])
                self.prod_station_counts[key][0] += int(r["failed"])
                self.prod_station_counts[key][1] += 1
                self.station_counts[r["station"]][0] += int(r["failed"])
                self.station_counts[r["station"]][1] += 1

    def predict_step_failure_prob(self, spec: ProductSpec, operation: str, station: str) -> float:
        key = (spec.name, operation, station)
        if key in self.prod_station_counts and self.prod_station_counts[key][1] > 0:
            k, n = self.prod_station_counts[key]
            return round((k + self.alpha) / (n + self.alpha + self.beta), 6)
        # Unseen product fallback: station-level marginal
        k_st, n_st = self.station_counts.get(station, [0, 0])
        return round((k_st + self.alpha) / (n_st + self.alpha + self.beta), 6)


class AttributeBayesianNetwork:
    """Full joint attribute CPT estimator P(failed | op, station, material, geometry, finish).

    Suffers from combinatorial cell fragmentation in small-sample regimes (H3)
    because it does not perform selective clause structure induction.
    """

    def __init__(self, equivalent_sample_size: float = 2.0) -> None:
        self.ess = equivalent_sample_size
        self.cell_counts: Dict[tuple, List[int]] = defaultdict(lambda: [0, 0])
        self.station_prior: Dict[str, float] = {}

    def fit(self, records: List[Dict]) -> None:
        st_totals: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
        for r in records:
            if r["attempted"] == 1 and r["censored"] == 0 and r["failed"] is not None:
                cell = (
                    r["operation"],
                    r["station"],
                    r["material"],
                    r["geometry"],
                    r["finish_grade"],
                )
                self.cell_counts[cell][0] += int(r["failed"])
                self.cell_counts[cell][1] += 1
                st_totals[r["station"]][0] += int(r["failed"])
                st_totals[r["station"]][1] += 1

        for st, (k, n) in st_totals.items():
            self.station_prior[st] = (k + 1.0) / (n + 2.0)

    def predict_step_failure_prob(self, spec: ProductSpec, operation: str, station: str) -> float:
        cell = (operation, station, spec.material, spec.geometry, spec.finish_grade)
        prior = self.station_prior.get(station, 0.10)
        k, n = self.cell_counts.get(cell, [0, 0])
        return round((k + self.ess * prior) / (n + self.ess), 6)


class TabularLogRegBaseline:
    """L2-regularized logistic regression over one-hot encoded product/station features."""

    def __init__(self, l2_reg: float = 1.5, lr: float = 0.15, epochs: int = 120) -> None:
        self.l2_reg = l2_reg
        self.lr = lr
        self.epochs = epochs
        self.weights: Dict[str, float] = defaultdict(float)
        self.bias: float = -2.0

    @staticmethod
    def _extract_features(spec_or_rec: Dict) -> List[str]:
        return [
            f"st={spec_or_rec['station']}",
            f"op={spec_or_rec['operation']}",
            f"st_op={spec_or_rec['station']}_{spec_or_rec['operation']}",
            f"mat={spec_or_rec['material']}",
            f"geom={spec_or_rec['geometry']}",
            f"fin={spec_or_rec['finish_grade']}",
            f"op_mat={spec_or_rec['operation']}_{spec_or_rec['material']}",
            f"op_geom={spec_or_rec['operation']}_{spec_or_rec['geometry']}",
            f"op_fin={spec_or_rec['operation']}_{spec_or_rec['finish_grade']}",
        ]

    def fit(self, records: List[Dict]) -> None:
        valid = [
            r for r in records if r["attempted"] == 1 and r["censored"] == 0 and r["failed"] is not None
        ]
        if not valid:
            return
        n = len(valid)
        for _ in range(self.epochs):
            grad_w: Dict[str, float] = defaultdict(float)
            grad_b = 0.0
            for r in valid:
                feats = self._extract_features(r)
                z = self.bias + sum(self.weights[f] for f in feats)
                z = max(-20.0, min(20.0, z))
                p = 1.0 / (1.0 + math.exp(-z))
                err = p - float(r["failed"])
                grad_b += err
                for f in feats:
                    grad_w[f] += err

            self.bias -= self.lr * (grad_b / n)
            for f, gw in grad_w.items():
                self.weights[f] -= self.lr * ((gw / n) + (self.l2_reg / n) * self.weights[f])

    def predict_step_failure_prob(self, spec: ProductSpec, operation: str, station: str) -> float:
        rec = {
            "station": station,
            "operation": operation,
            "material": spec.material,
            "geometry": spec.geometry,
            "finish_grade": spec.finish_grade,
        }
        feats = self._extract_features(rec)
        z = self.bias + sum(self.weights[f] for f in feats)
        z = max(-20.0, min(20.0, z))
        return round(1.0 / (1.0 + math.exp(-z)), 6)


class ProcessMiningBaseline:
    """Control-flow process discovery baseline estimating station/operation transition risk.

    Captures routing precedence and station-level failure frequencies without
    conditioning on relational product attributes.
    """

    def __init__(self) -> None:
        self.op_station_counts: Dict[tuple, List[int]] = defaultdict(lambda: [0, 0])

    def fit(self, records: List[Dict]) -> None:
        for r in records:
            if r["attempted"] == 1 and r["censored"] == 0 and r["failed"] is not None:
                key = (r["operation"], r["station"])
                self.op_station_counts[key][0] += int(r["failed"])
                self.op_station_counts[key][1] += 1

    def predict_step_failure_prob(self, spec: ProductSpec, operation: str, station: str) -> float:
        k, n = self.op_station_counts.get((operation, station), [0, 0])
        return round((k + 1.0) / (n + 2.0), 6)


class NaiveUnmaskedPILPBaseline:
    """Ablation baseline (H5) treating unattempted steps (attempted=0) as negative failures."""

    def __init__(self) -> None:
        self.clauses = []

    def fit(self, records: List[Dict]) -> None:
        from src.slipcover_pipeline import _empirical_pilp_inducer

        corrupted = []
        for r in records:
            r_copy = dict(r)
            if r_copy["attempted"] == 0:
                r_copy["attempted"] = 1
                r_copy["censored"] = 0
                r_copy["failed"] = 0  # Conflates unattempted step with successful execution
            corrupted.append(r_copy)
        self.clauses = _empirical_pilp_inducer(corrupted, mode="structure_learning")

    def predict_step_failure_prob(self, spec: ProductSpec, operation: str, station: str) -> float:
        from src.slipcover_pipeline import predict_step_failure_prob

        return predict_step_failure_prob(self.clauses, spec, operation, station)