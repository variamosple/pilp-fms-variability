#!/usr/bin/env python3
"""Parameterized discrete-event FMS production log simulator with known ground truth.

Generates event logs for the 6-product furniture family across 5 workstations,
incorporating explicit observation masks (attempted vs. unattempted downstream steps),
controlled sensor noise, and missing-event censoring.
"""

from dataclasses import dataclass, asdict
import random
from typing import Dict, List, Tuple


@dataclass
class ProductSpec:
    name: str
    material: str
    geometry: str
    finish_grade: str
    routing: List[Tuple[str, str]]  # List of (operation, station)


# Complete 6-product catalog aligned with Section 3.3 of the manuscript
PRODUCT_CATALOG: Dict[str, ProductSpec] = {
    "chair": ProductSpec(
        name="chair",
        material="light_wood",
        geometry="compact",
        finish_grade="standard",
        routing=[
            ("cutting", "s1"),
            ("leg_drilling", "s2"),
            ("seat_assembly", "s3"),
            ("sanding", "s4"),
            ("varnishing", "s5"),
        ],
    ),
    "stool": ProductSpec(
        name="stool",
        material="light_wood",
        geometry="compact",
        finish_grade="standard",
        routing=[
            ("cutting", "s1"),
            ("leg_drilling", "s2"),
            ("seat_assembly", "s3"),
            ("varnishing", "s5"),
        ],
    ),
    "table": ProductSpec(
        name="table",
        material="heavy_wood",
        geometry="wide",
        finish_grade="standard",
        routing=[
            ("cutting", "s1"),
            ("leg_drilling", "s2"),
            ("top_assembly", "s3"),
            ("sanding", "s4"),
            ("varnishing", "s5"),
        ],
    ),
    "bench": ProductSpec(
        name="bench",
        material="heavy_wood",
        geometry="wide",
        finish_grade="standard",
        routing=[
            ("cutting", "s1"),
            ("leg_drilling", "s2"),
            ("seat_assembly", "s3"),
            ("sanding", "s4"),
            ("varnishing", "s5"),
        ],
    ),
    "armchair": ProductSpec(
        name="armchair",
        material="composite",
        geometry="compact",
        finish_grade="precision",
        routing=[
            ("cutting", "s1"),
            ("leg_drilling", "s2"),
            ("seat_assembly", "s3"),
            ("sanding", "s4"),
            ("varnishing", "s5"),
        ],
    ),
    # Held-out product for zero-shot transfer evaluation (H2)
    "desk": ProductSpec(
        name="desk",
        material="heavy_wood",
        geometry="wide",
        finish_grade="precision",
        routing=[
            ("cutting", "s1"),
            ("leg_drilling", "s2"),
            ("top_assembly", "s3"),
            ("sanding", "s4"),
            ("varnishing", "s5"),
        ],
    ),
}

# Ground-truth relational failure mechanisms (known to benchmark evaluator, withheld from learners)
GROUND_TRUTH_RULES = [
    {
        "id": "R1_heavy_cutting",
        "station": "s1",
        "operation": "cutting",
        "condition_attr": "material",
        "condition_val": "heavy_wood",
        "failure_prob": 0.35,
    },
    {
        "id": "R2_wide_drilling",
        "station": "s2",
        "operation": "leg_drilling",
        "condition_attr": "geometry",
        "condition_val": "wide",
        "failure_prob": 0.25,
    },
    {
        "id": "R3_precision_varnishing",
        "station": "s5",
        "operation": "varnishing",
        "condition_attr": "finish_grade",
        "condition_val": "precision",
        "failure_prob": 0.30,
    },
]

BASE_STATION_NOISE = {
    "s1": 0.03,
    "s2": 0.02,
    "s3": 0.02,
    "s4": 0.02,
    "s5": 0.03,
}


def true_step_failure_prob(spec: ProductSpec, operation: str, station: str) -> float:
    """Computes exact noisy-OR ground-truth failure probability for an attempted step."""
    p_survival = 1.0 - BASE_STATION_NOISE.get(station, 0.02)
    for rule in GROUND_TRUTH_RULES:
        if rule["station"] == station and rule["operation"] == operation:
            attr_val = getattr(spec, rule["condition_attr"])
            if attr_val == rule["condition_val"]:
                p_survival *= (1.0 - rule["failure_prob"])
    return round(1.0 - p_survival, 6)


def true_product_completion_prob(product_name: str) -> float:
    """Computes exact sequential completion probability across all routed steps."""
    spec = PRODUCT_CATALOG[product_name]
    p_complete = 1.0
    for op, st in spec.routing:
        p_fail = true_step_failure_prob(spec, op, st)
        p_complete *= (1.0 - p_fail)
    return round(p_complete, 6)


def simulate_production_logs(
    products: List[str],
    runs_per_product: int,
    seed: int = 42,
    label_noise: float = 0.0,
    missing_rate: float = 0.0,
    abort_on_failure: bool = True,
) -> List[Dict]:
    """Simulates production runs with explicit observation masks (attempted=1 vs 0)."""
    rng = random.Random(seed)
    all_stations = ["s1", "s2", "s3", "s4", "s5"]
    records = []
    run_counter = 1

    for prod_name in products:
        spec = PRODUCT_CATALOG[prod_name]
        routed_map = {st: op for op, st in spec.routing}

        for _ in range(runs_per_product):
            run_id = f"r{run_counter}"
            run_counter += 1
            aborted = False

            for st in all_stations:
                if st not in routed_map or aborted:
                    # Unattempted operation: explicit observation mask attempted = 0
                    records.append({
                        "run_id": run_id,
                        "product": spec.name,
                        "material": spec.material,
                        "geometry": spec.geometry,
                        "finish_grade": spec.finish_grade,
                        "station": st,
                        "operation": routed_map.get(st, "none"),
                        "attempted": 0,
                        "failed": None,
                        "censored": 0,
                    })
                    continue

                op = routed_map[st]
                p_fail = true_step_failure_prob(spec, op, st)
                actual_fail = 1 if rng.random() < p_fail else 0

                # Apply controlled label noise if specified (H4 sensitivity)
                observed_fail = actual_fail
                if label_noise > 0.0 and rng.random() < label_noise:
                    observed_fail = 1 - actual_fail

                # Apply missing-event censoring if specified
                is_censored = 1 if (missing_rate > 0.0 and rng.random() < missing_rate) else 0

                records.append({
                    "run_id": run_id,
                    "product": spec.name,
                    "material": spec.material,
                    "geometry": spec.geometry,
                    "finish_grade": spec.finish_grade,
                    "station": st,
                    "operation": op,
                    "attempted": 1,
                    "failed": None if is_censored else observed_fail,
                    "censored": is_censored,
                })

                if actual_fail == 1 and abort_on_failure:
                    aborted = True

    return records