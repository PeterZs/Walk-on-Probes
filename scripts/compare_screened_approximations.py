#!/usr/bin/env python3
"""Compare the old 2D screened Green/Poisson approximations to exact series.

The disk radius is fixed to R=1. This loses no generality for relative error:
the screened problem depends on the dimensionless product kappa*R.

Reference formulas are Fourier-Bessel series evaluated with exponentially
scaled SciPy Bessel functions. The approximations reproduce the old WoS
implementation in WoS/core/include/wos/core/green_function.hpp.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import special


TWO_PI = 2.0 * math.pi


def _safe_i_ratio(order: float, x: float, y: float) -> float:
    """Stable I_order(x) / I_order(y), for 0 <= x <= y."""
    if x == y:
        return 1.0
    numerator = special.ive(order, x)
    denominator = special.ive(order, y)
    if np.isfinite(numerator) and np.isfinite(denominator) and denominator != 0.0:
        return float((numerator / denominator) * math.exp(x - y))
    if order == 0.0:
        return 1.0
    if x == 0.0:
        return 0.0
    return math.exp(order * math.log(x / y))


def _safe_k_ratio(order: float, x: float, y: float) -> float:
    """Stable K_order(x) / K_order(y), for x,y > 0."""
    if x == y:
        return 1.0
    numerator = special.kve(order, x)
    denominator = special.kve(order, y)
    if np.isfinite(numerator) and np.isfinite(denominator) and denominator != 0.0:
        return float((numerator / denominator) * math.exp(y - x))
    if order > 0.0:
        return math.exp(order * math.log(y / x))
    euler_gamma = 0.5772156649015329
    return (-math.log(0.5 * x) - euler_gamma) / (-math.log(0.5 * y) - euler_gamma)


def _safe_ik_product(order: float, x: float, y: float) -> float:
    """Stable I_order(x) K_order(y), for 0 <= x <= y."""
    i_scaled = special.ive(order, x)
    k_scaled = special.kve(order, y)
    with np.errstate(invalid="ignore", over="ignore", under="ignore"):
        product = i_scaled * k_scaled
    if np.isfinite(product):
        return float(product * math.exp(x - y))
    if order > 0.0:
        if x == 0.0:
            return 0.0
        return math.exp(order * math.log(x / y)) / (2.0 * order)
    return float(special.kv(0.0, y))


def exact_green_2d(
    x_radius: float,
    y_radius: float,
    theta: np.ndarray,
    kappa: float,
    *,
    max_terms: int,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    """Exact screened Green function in the unit disk."""
    r_minus = min(x_radius, y_radius)
    r_plus = max(x_radius, y_radius)
    kr_minus = kappa * r_minus
    kr_plus = kappa * r_plus
    k_radius = kappa

    result = np.zeros_like(theta)
    small_count = 0
    coefficient_scale = 0.0
    terms_used = max_terms

    for n in range(max_terms):
        order = float(n)
        ik_product = _safe_ik_product(order, kr_minus, kr_plus)
        boundary_ratio = _safe_k_ratio(order, k_radius, kr_plus) * _safe_i_ratio(
            order, kr_plus, k_radius
        )
        coefficient = (1.0 if n == 0 else 2.0) * ik_product * (1.0 - boundary_ratio)
        term = coefficient * np.cos(n * theta)
        result += term

        coefficient_scale = max(coefficient_scale, abs(coefficient))
        small_count = (
            small_count + 1
            if abs(coefficient) <= tolerance * max(coefficient_scale, 1e-300)
            else 0
        )
        if n >= 8 and small_count >= 4:
            terms_used = n + 1
            break

    return result / TWO_PI, terms_used


def old_green_approx_2d(
    x_radius: float, y_radius: float, theta: np.ndarray, kappa: float
) -> np.ndarray:
    """Old WoS mirror-distance approximation for the screened Green function."""
    dot_xy = x_radius * y_radius * np.cos(theta)
    distance = np.sqrt(
        np.maximum(x_radius * x_radius + y_radius * y_radius - 2.0 * dot_xy, 1e-24)
    )
    mirror_distance = np.maximum(1.0 - dot_xy, 1e-12)

    k_radius = kappa
    k0_radius = special.kv(0, k_radius)
    i0_radius = special.iv(0, k_radius)

    def centered_q(radius: np.ndarray) -> np.ndarray:
        kr = kappa * np.maximum(radius, 1e-12)
        return special.kv(0, kr) - special.iv(0, kr) * (k0_radius / i0_radius)

    return (centered_q(distance) - centered_q(mirror_distance)) / TWO_PI


def exact_poisson_2d(
    x_radius: float,
    theta: np.ndarray,
    kappa: float,
    *,
    max_terms: int,
    tolerance: float,
) -> tuple[np.ndarray, int]:
    """Exact screened Poisson kernel on the unit-circle boundary."""
    result = np.zeros_like(theta)
    small_count = 0
    coefficient_scale = 0.0
    terms_used = max_terms

    for n in range(max_terms):
        ratio = _safe_i_ratio(float(n), kappa * x_radius, kappa)
        coefficient = (1.0 if n == 0 else 2.0) * ratio
        term = coefficient * np.cos(n * theta)
        result += term

        coefficient_scale = max(coefficient_scale, abs(coefficient))
        small_count = (
            small_count + 1
            if abs(coefficient) <= tolerance * max(coefficient_scale, 1e-300)
            else 0
        )
        if n >= 8 and small_count >= 4:
            terms_used = n + 1
            break

    return result / TWO_PI, terms_used


def old_poisson_approx_2d(
    x_radius: float, theta: np.ndarray, kappa: float
) -> np.ndarray:
    """Old Sawhney-style screened Poisson-kernel approximation."""
    dot_xz = x_radius * np.cos(theta)
    distance = np.sqrt(np.maximum(1.0 + x_radius * x_radius - 2.0 * dot_xz, 1e-24))
    mirror_distance = np.maximum(1.0 - dot_xz, 1e-12)

    k_radius = kappa
    boundary_ratio = special.kv(0, k_radius) / special.iv(0, k_radius)

    def v_function(radius: np.ndarray) -> np.ndarray:
        kr = kappa * np.maximum(radius, 1e-12)
        return kappa * (special.kv(1, kr) + boundary_ratio * special.iv(1, kr))

    v_distance = v_function(distance)
    v_mirror = v_function(mirror_distance)
    geom = v_distance * (1.0 - dot_xz) / distance + v_mirror * dot_xz
    return geom / TWO_PI


@dataclass
class ErrorSummary:
    quantity: str
    kappa_r: float
    sample_count: int
    normalized_rmse: float
    median_relative: float
    p90_relative: float
    p99_relative: float
    max_relative: float
    negative_fraction: float
    max_reference_terms: int
    worst_x_radius: float
    worst_y_radius: float | None
    worst_theta: float
    worst_reference: float
    worst_approximation: float


def summarize(
    *,
    quantity: str,
    kappa: float,
    references: list[np.ndarray],
    approximations: list[np.ndarray],
    locations: list[tuple[float, float | None, np.ndarray]],
    term_counts: list[int],
) -> ErrorSummary:
    reference = np.concatenate(references)
    approximation = np.concatenate(approximations)
    error = approximation - reference

    # A floor avoids meaningless enormous pointwise relative errors where the
    # exact kernel is effectively zero. NRMSE remains floor-free.
    relative_floor = 1e-8 * max(float(np.max(np.abs(reference))), 1e-300)
    relative = np.abs(error) / np.maximum(np.abs(reference), relative_floor)
    worst_flat = int(np.argmax(relative))

    offset = 0
    worst_location = locations[0]
    worst_local = 0
    for arrays, location in zip(references, locations):
        if worst_flat < offset + arrays.size:
            worst_location = location
            worst_local = worst_flat - offset
            break
        offset += arrays.size

    x_radius, y_radius, theta = worst_location
    denominator = math.sqrt(float(np.mean(reference * reference)))
    normalized_rmse = math.sqrt(float(np.mean(error * error))) / max(denominator, 1e-300)

    return ErrorSummary(
        quantity=quantity,
        kappa_r=kappa,
        sample_count=int(reference.size),
        normalized_rmse=normalized_rmse,
        median_relative=float(np.percentile(relative, 50)),
        p90_relative=float(np.percentile(relative, 90)),
        p99_relative=float(np.percentile(relative, 99)),
        max_relative=float(relative[worst_flat]),
        negative_fraction=float(np.mean(approximation < 0.0)),
        max_reference_terms=max(term_counts),
        worst_x_radius=x_radius,
        worst_y_radius=y_radius,
        worst_theta=float(theta[worst_local]),
        worst_reference=float(reference[worst_flat]),
        worst_approximation=float(approximation[worst_flat]),
    )


def run_experiment(args: argparse.Namespace) -> list[ErrorSummary]:
    kappas = [float(value) for value in args.kappa_r]
    x_radii = np.asarray(args.x_radii, dtype=float)
    y_radii = np.asarray(args.y_radii, dtype=float)
    theta = np.linspace(0.0, math.pi, args.angles, endpoint=True)
    results: list[ErrorSummary] = []

    for kappa in kappas:
        green_refs: list[np.ndarray] = []
        green_approxs: list[np.ndarray] = []
        green_locations: list[tuple[float, float | None, np.ndarray]] = []
        green_terms: list[int] = []

        for x_radius in x_radii:
            for y_radius in y_radii:
                distance = np.sqrt(
                    np.maximum(
                        x_radius * x_radius
                        + y_radius * y_radius
                        - 2.0 * x_radius * y_radius * np.cos(theta),
                        0.0,
                    )
                )
                mask = distance > args.singularity_gap
                selected_theta = theta[mask]
                if selected_theta.size == 0:
                    continue
                reference, terms = exact_green_2d(
                    x_radius,
                    y_radius,
                    selected_theta,
                    kappa,
                    max_terms=args.max_terms,
                    tolerance=args.tolerance,
                )
                approximation = old_green_approx_2d(
                    x_radius, y_radius, selected_theta, kappa
                )
                green_refs.append(reference)
                green_approxs.append(approximation)
                green_locations.append((float(x_radius), float(y_radius), selected_theta))
                green_terms.append(terms)

        results.append(
            summarize(
                quantity="Green",
                kappa=kappa,
                references=green_refs,
                approximations=green_approxs,
                locations=green_locations,
                term_counts=green_terms,
            )
        )

        poisson_refs: list[np.ndarray] = []
        poisson_approxs: list[np.ndarray] = []
        poisson_locations: list[tuple[float, float | None, np.ndarray]] = []
        poisson_terms: list[int] = []

        for x_radius in x_radii:
            reference, terms = exact_poisson_2d(
                float(x_radius),
                theta,
                kappa,
                max_terms=args.max_terms,
                tolerance=args.tolerance,
            )
            approximation = old_poisson_approx_2d(float(x_radius), theta, kappa)
            poisson_refs.append(reference)
            poisson_approxs.append(approximation)
            poisson_locations.append((float(x_radius), None, theta))
            poisson_terms.append(terms)

        results.append(
            summarize(
                quantity="Poisson",
                kappa=kappa,
                references=poisson_refs,
                approximations=poisson_approxs,
                locations=poisson_locations,
                term_counts=poisson_terms,
            )
        )

    return results


def print_results(results: list[ErrorSummary]) -> None:
    header = (
        f"{'quantity':<8} {'kR':>5} {'samples':>8} {'NRMSE':>11} "
        f"{'rel50':>11} {'rel90':>11} {'rel99':>11} {'relmax':>11} {'neg%':>8}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.quantity:<8} {result.kappa_r:5.1f} {result.sample_count:8d} "
            f"{result.normalized_rmse:11.3e} {result.median_relative:11.3e} "
            f"{result.p90_relative:11.3e} {result.p99_relative:11.3e} "
            f"{result.max_relative:11.3e} {100.0 * result.negative_fraction:8.3f}"
        )

    print("\nWorst locations (angles are radians):")
    for result in results:
        y_text = "-" if result.worst_y_radius is None else f"{result.worst_y_radius:.3f}"
        print(
            f"{result.quantity:<8} kR={result.kappa_r:4.1f}: "
            f"x={result.worst_x_radius:.3f}, y={y_text}, theta={result.worst_theta:.6f}, "
            f"exact={result.worst_reference:.6e}, approx={result.worst_approximation:.6e}, "
            f"terms<={result.max_reference_terms}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kappa-r",
        nargs="+",
        type=float,
        default=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        help="Dimensionless kappa*R values (R is fixed to one).",
    )
    parser.add_argument(
        "--x-radii",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.5, 0.6],
    )
    parser.add_argument(
        "--y-radii",
        nargs="+",
        type=float,
        default=[0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
    )
    parser.add_argument("--angles", type=int, default=181)
    parser.add_argument("--max-terms", type=int, default=600)
    parser.add_argument("--tolerance", type=float, default=1e-13)
    parser.add_argument(
        "--singularity-gap",
        type=float,
        default=0.02,
        help="Skip Green samples closer than this to x=y.",
    )
    parser.add_argument("--json", type=Path, help="Optional path for machine-readable results.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_experiment(args)
    print_results(results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps([asdict(result) for result in results], indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
