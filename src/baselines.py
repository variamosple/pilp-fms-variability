#!/usr/bin/env python3
"""Statistical, Bayesian network, tabular ML, process-mining, and LPAD ablation baselines.

Implements all comparative baselines specified in Table 2 (Section 5.4) of the manuscript:
1. ProductSpecificBetaBinomial: Smoothed Beta(1,1) estimator per (product, operation, station).
2. ParameterOnlyLPADBaseline: SWI-Prolog SLIPCOVER induce_par/2 on engineer-specified structure.
3. OracleStructureLPADBaseline: SWI-Prolog SLIPCOVER induce_par/2 on true generating structure.
4. AttributeBayesianNetwork: Joint conditional probability table with Dirichlet smoothing.
5. TabularLogRegBaseline: L2-regularized logistic regression over one-hot features.
6. ProcessMiningBaseline: Control-flow station transition failure frequency estimator.
7. NaiveUnmaskedPILPBaseline: SWI-Prolog SLIPCOVER ablation treating attempted=0 as failed=0.
"""

from collections import defaultdict
import math
from typing import Dict, List

from src.simulator import ProductSpec
from src.slipcover_pipeline import (
    LearnedClause,
    predict_step_failure_prob,
    run_slipcover_learning,
)


class ProductSpecificBetaBinomial:
    """Smoothed Beta(1,1) estimator conditioned on (product, operation, station) with pooled fallback."""

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
        k_st, n_st = self.station_counts.get(station, [0, 0])
        return round((k_st + self.alpha) / (n_st + self.alpha + self.beta), 6)


class ParameterOnlyLPADBaseline:
    """Engineer-structured LPAD fitted via native SWI-Prolog SLIPCOVER induce_par/2 (no clause search)."""

    def __init__(self) -> None:
        self.clauses: List[LearnedClause] = []

    def fit(self, records: List[Dict]) -> None:
        self.clauses, _ = run_slipcover_learning(records, mode="parameter_only")

    def predict_step_failure_prob(self, spec: ProductSpec, operation: str, station: str) -> float:
        return predict_step_failure_prob(self.clauses, spec, operation, station)


class OracleStructureLPADBaseline:
    """Privileged oracle-structure LPAD fitted via native SWI-Prolog SLIPCOVER induce_par/2."""

    def __init__(self) -> None:
        self.clauses: List[LearnedClause] = []

    def fit(self, records: List[Dict]) -> None:
        self.clauses, _ = run_slipcover_learning(records, mode="oracle_structure")

    def predict_step_failure_prob(self, spec: ProductSpec, operation: str, station: str) -> float:
        return predict_step_failure_prob(self.clauses, spec, operation, station)


class AttributeBayesianNetwork:
    """Full joint attribute CPT estimator P(failed | op, station, material, geometry, finish)."""

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
    """L2-regularized logistic regression over one-hot encoded product and station attributes."""

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
    """Control-flow process discovery baseline estimating station/operation transition risk."""

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
    """Observation-mask ablation baseline (H3/H5) running SWI-Prolog SLIPCOVER on unmasked logs."""

    def __init__(self) -> None:
        self.clauses: List[LearnedClause] = []

    def fit(self, records: List[Dict]) -> None:
        corrupted = []
        for r in records:
            r_copy = dict(r)
            if r_copy["attempted"] == 0 and r_copy["operation"] != "none":
                r_copy["attempted"] = 1
                r_copy["censored"] = 0
                r_copy["failed"] = 0  # Conflates unattempted downstream step with non-failure
            corrupted.append(r_copy)
        self.clauses, _ = run_slipcover_learning(corrupted, mode="structure_learning")

    def predict_step_failure_prob(self, spec: ProductSpec, operation: str, station: str) -> float:
        return predict_step_failure_prob(self.clauses, spec, operation, station)