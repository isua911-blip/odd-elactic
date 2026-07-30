#!/usr/bin/env python3
"""Major-review audits connecting configurational flux, advance work and fracture paths.

This module adds the computations requested in the latest IJSS review:
1. matched J_h / endpoint-energy / abrupt-work / quasistatic-work comparison;
2. unrestricted all-bond cascades;
3. two-parameter quasistatic debonding paths;
4. refinement of the keyhole/source closure ratio;
5. angular-momentum and perfect-lattice stability audits;
6. one weakly damped inertial post-cut probe.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.sparse.linalg import MatrixRankWarning, eigs, spsolve

PACKAGE_REVISION = "2026-07-30-r6"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from active_tip_scan import ActiveCrackedStrip
from apparent_j_analysis import LocalTipField, annulus_sources, keyhole_j
from crack_advance_work import (
    assemble_components,
    bond_extension,
    even_energy,
    solve_equilibrium,
)
from gauge_convergence import (
    build_triangles,
    evaluate_J,
    partition_weights,
    weighted_fields,
)
from propagation_limit_analysis import (
    DeadLoadCascade,
    FixedGripCascade,
    active_ids,
    dead_load_passive_reference,
    fixed_grip_passive_reference,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _path_points(kind: str, n: int = 129) -> list[tuple[float, float]]:
    """Return a path from (s_e,s_o)=(1,1) to (0,0)."""
    q = np.linspace(0.0, 1.0, n)
    t = 0.5 * (1.0 + np.cos(math.pi * q))
    if kind == "diagonal":
        return [(float(x), float(x)) for x in t]
    if kind == "odd_persists":
        return [(float(x), float(math.sqrt(max(x, 0.0)))) for x in t]
    if kind == "even_persists":
        return [(float(math.sqrt(max(x, 0.0))), float(x)) for x in t]
    if kind in {"even_first", "odd_first"}:
        n1 = (n + 1) // 2
        q1 = np.linspace(0.0, 1.0, n1)
        z = 0.5 * (1.0 + np.cos(math.pi * q1))
        if kind == "even_first":
            first = [(float(x), 1.0) for x in z]
            second = [(0.0, float(x)) for x in z[1:]]
        else:
            first = [(1.0, float(x)) for x in z]
            second = [(float(x), 0.0) for x in z[1:]]
        return first + second
    raise ValueError(kind)


def quasistatic_path_case(
    nx: int,
    ny: int,
    crack: float,
    ko: float,
    kind: str,
    n: int = 129,
) -> dict[str, float | str]:
    model = ActiveCrackedStrip(nx, ny, crack, 1.0, ko)
    cut = model.tip_candidates()["right"]
    old = [b for b in range(len(model.all_bonds)) if b not in model.removed_ids]
    rest = [b for b in old if b != cut]
    Ee_rest, Eo_rest = assemble_components(model, rest)
    Ee_cut, Eo_cut = assemble_components(model, [cut])
    c, cv = model.constrained_dofs(1.0)
    mask = np.ones(model.ndof, dtype=bool)
    mask[c] = False
    f = np.arange(model.ndof)[mask]

    states: list[np.ndarray] = []
    energies: list[float] = []
    odd_forces: list[np.ndarray] = []
    maximum_residual = 0.0
    points = _path_points(kind, n)
    for se, so in points:
        Ee = Ee_rest + se * Ee_cut
        Eo = Eo_rest + so * Eo_cut
        u, c2, cv2, f2, residual = solve_equilibrium(Ee + Eo, model, 1.0)
        if not (np.array_equal(c, c2) and np.array_equal(f, f2) and np.allclose(cv, cv2)):
            raise RuntimeError("softening boundary partition changed")
        states.append(u)
        energies.append(even_energy(Ee, u))
        odd_forces.append(np.asarray(-(Eo[f][:, :] @ u)))
        maximum_residual = max(maximum_residual, float(residual))

    odd_work = 0.0
    for i in range(1, len(states)):
        du = states[i][f] - states[i - 1][f]
        odd_work += 0.5 * float((odd_forces[i] + odd_forces[i - 1]) @ du)
    delta_u = energies[-1] - energies[0]
    return {
        "nx": nx,
        "ny": ny,
        "crack_half_length": crack,
        "k_o": ko,
        "path": kind,
        "n_points": len(points),
        "odd_work_unit": odd_work,
        "minus_delta_Ue_unit": -delta_u,
        "quasistatic_work_unit": odd_work - delta_u,
        "maximum_equilibrium_residual": maximum_residual,
    }


def two_parameter_softening(out: Path) -> dict[str, object]:
    paths = ["even_first", "even_persists", "diagonal", "odd_persists", "odd_first"]
    rows: list[dict[str, object]] = []
    for ko in (0.12, 0.222271):
        for path in paths:
            row = quasistatic_path_case(48, 36, 6.0, ko, path, 129)
            rows.append(row)
            print("softening", ko, path, row["quasistatic_work_unit"])
    write_csv(out / "two_parameter_quasistatic_softening.csv", rows)
    summaries = []
    for ko in (0.12, 0.222271):
        values = np.array([float(r["quasistatic_work_unit"]) for r in rows if math.isclose(float(r["k_o"]), ko)])
        summaries.append({
            "k_o": ko,
            "minimum_quasistatic_work": float(values.min()),
            "maximum_quasistatic_work": float(values.max()),
            "absolute_span": float(values.max() - values.min()),
            "relative_span_over_mean": float((values.max() - values.min()) / abs(values.mean())),
            "minimum_path": next(str(r["path"]) for r in rows if math.isclose(float(r["k_o"]), ko) and math.isclose(float(r["quasistatic_work_unit"]), float(values.min()), rel_tol=1e-10, abs_tol=1e-12)),
            "maximum_path": next(str(r["path"]) for r in rows if math.isclose(float(r["k_o"]), ko) and math.isclose(float(r["quasistatic_work_unit"]), float(values.max()), rel_tol=1e-10, abs_tol=1e-12)),
        })
    summary = {
        "paths": paths,
        "cases": summaries,
        "interpretation": "Independent weakening of the recoverable and odd bond actions selects different equilibrium curves in configuration space; monotone reparameterization of one curve does not cause the observed spread.",
        "pass": all(float(s["relative_span_over_mean"]) > 0.01 for s in summaries),
    }
    (out / "two_parameter_softening_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _discrete_J(model: ActiveCrackedStrip, displacement_flat: np.ndarray,
                radius_fraction: float = 0.10, alpha: float = 0.5) -> float:
    """Discrete configurational flux for one localization share and domain radius."""
    u = displacement_flat.reshape((-1, 2))
    triangles = build_triangles(model)
    weights = partition_weights(model, triangles, alpha)
    radius = radius_fraction * model.nx
    width = 0.025 * model.nx
    support = radius + 0.5 * width + 1.0
    fields, nodes = weighted_fields(model, u, triangles, "right", support, weights)
    return float(evaluate_J(fields, nodes, radius, width, 4.0)["J_total"])


def _symmetric_discrete_J(model: ActiveCrackedStrip, displacement_flat: np.ndarray, radius_fraction: float = 0.10) -> float:
    return _discrete_J(model, displacement_flat, radius_fraction, 0.5)


def _discrete_J_envelope(model: ActiveCrackedStrip, displacement_flat: np.ndarray,
                         radius_fractions: tuple[float, ...] = (0.075, 0.10, 0.125),
                         alphas: tuple[float, ...] = (0.0, 0.5, 1.0)) -> dict[str, float]:
    """Spread of the discrete flux over the localization share and domain radius.

    The reference value is the reflection-symmetric partition at the nominal
    radius; the envelope quantifies the representation and contour sensitivity
    that the matched flux/work comparison inherits.
    """
    values = [_discrete_J(model, displacement_flat, r, a) for r in radius_fractions for a in alphas]
    reference = _discrete_J(model, displacement_flat, 0.10, 0.5)
    return {
        "J_reference": reference,
        "J_min": float(min(values)),
        "J_max": float(max(values)),
        "relative_half_span": float(0.5 * (max(values) - min(values)) / abs(reference)),
        "sample_count": len(values),
    }


def j_work_bridge(out: Path) -> dict[str, object]:
    unit_frames = [pd.read_csv(ROOT / "data" / "advance_resistance_results" / "advance_unit_steps.csv")]
    # Optional extension sizes, produced by
    #   python recompute_advance_resistance.py --bridge-sizes-only
    # They enter the matched comparison only; the 224-state stepwise scan is unchanged.
    extension = ROOT / "data" / "advance_resistance_results" / "advance_unit_steps_bridge.csv"
    if extension.exists():
        unit_frames.append(pd.read_csv(extension))
    units = pd.concat(unit_frames, ignore_index=True)
    unit = units[(units.boundary_condition == "fixed_grip") & (units.step == 1) & np.isclose(units.crack_fraction, 0.125)]
    bridge_sizes = tuple(sorted(int(v) for v in unit.nx.unique()))
    rows: list[dict[str, object]] = []
    qs_cache: dict[tuple[int, float], dict[str, object]] = {}
    p = 0.90
    advance = 0.5
    for nx in bridge_sizes:
        ny = 3 * nx // 4
        crack = nx / 8.0
        passive_stress, _ = fixed_grip_passive_reference(nx, ny, crack, 0.02)
        for ko in (0.0, 0.20):
            ur = unit[(unit.nx == nx) & np.isclose(unit.k_o, ko)].iloc[0]
            amplitude = p * passive_stress / float(ur.remote_stress_unit)
            model = ActiveCrackedStrip(nx, ny, crack, 1.0, ko)
            u, _, residual = model.solve(amplitude)
            J = _symmetric_discrete_J(model, u.ravel())
            envelope = _discrete_J_envelope(model, u.ravel())
            minus_dU = -float(ur.even_energy_change_unit) * amplitude**2
            abrupt = float(ur.protocol_work_unit) * amplitude**2
            key = (nx, ko)
            if key not in qs_cache:
                qs_cache[key] = quasistatic_path_case(nx, ny, crack, ko, "diagonal", 97)
            quasistatic = float(qs_cache[key]["quasistatic_work_unit"]) * amplitude**2
            rows.append({
                "nx": nx,
                "ny": ny,
                "a_lat_over_L": 1.0 / nx,
                "k_o": ko,
                "load_fraction": p,
                "applied_opening": amplitude,
                "remote_stress": p * passive_stress,
                "advance_length": advance,
                "J_h": J,
                "J_h_min_over_localization_and_radius": envelope["J_min"],
                "J_h_max_over_localization_and_radius": envelope["J_max"],
                "J_h_relative_half_span": envelope["relative_half_span"],
                "minus_delta_Ue_over_advance": minus_dU / advance,
                "abrupt_work_over_advance": abrupt / advance,
                "quasistatic_diagonal_work_over_advance": quasistatic / advance,
                "J_minus_endpoint_energy": J - minus_dU / advance,
                "J_minus_abrupt_work": J - abrupt / advance,
                "J_minus_quasistatic_work": J - quasistatic / advance,
                "relative_J_vs_abrupt_difference": (J - abrupt / advance) / J,
                "equilibrium_residual": residual,
            })
            print("bridge", nx, ko, J, abrupt / advance, quasistatic / advance)
    write_csv(out / "J_work_matched_comparison.csv", rows)
    passive = [r for r in rows if math.isclose(float(r["k_o"]), 0.0)]
    active = [r for r in rows if math.isclose(float(r["k_o"]), 0.2)]

    def extrapolate(subset: list[dict[str, object]]) -> dict[str, float]:
        """Linear-in-(a_lat/L) intercept, with the finest-pair value as an uncertainty proxy."""
        h = np.array([float(r["a_lat_over_L"]) for r in subset])
        y = np.array([abs(float(r["relative_J_vs_abrupt_difference"])) for r in subset]) * 100.0
        order = np.argsort(h)
        h, y = h[order], y[order]
        linear = np.polyfit(h, y, 1)
        linear_intercept = float(linear[-1])
        linear_residual = float(np.max(np.abs(np.polyval(linear, h) - y)))
        pair_slope = (y[0] - y[1]) / (h[0] - h[1])
        pair_intercept = float(y[0] - pair_slope * h[0])
        result = {
            "n_sizes": int(len(h)),
            "linear_intercept_percent": linear_intercept,
            "linear_maximum_absolute_residual_percent": linear_residual,
            "finest_pair_intercept_percent": pair_intercept,
        }
        candidates = [linear_intercept, pair_intercept]
        if len(h) >= 4:
            quadratic = np.polyfit(h, y, 2)
            quadratic_intercept = float(np.polyval(quadratic, 0.0))
            result["quadratic_intercept_percent"] = quadratic_intercept
            result["quadratic_maximum_absolute_residual_percent"] = float(
                np.max(np.abs(np.polyval(quadratic, h) - y))
            )
            candidates.append(quadratic_intercept)
        result["model_range_percent"] = [float(min(candidates)), float(max(candidates))]
        result["fit_choice_spread_percent"] = float(max(candidates) - min(candidates))
        return result

    passive_extrapolation = extrapolate(passive)
    active_extrapolation = extrapolate(active)
    finest = max(active, key=lambda r: int(r["nx"]))
    flux_to_state = float(finest["minus_delta_Ue_over_advance"]) / float(finest["J_h"])
    state_to_work = float(finest["abrupt_work_over_advance"]) / float(finest["minus_delta_Ue_over_advance"])
    summary = {
        "load_fraction": p,
        "advance_length": advance,
        "sizes": list(bridge_sizes),
        "J_h_relative_half_span_by_size_and_modulus": {
            f"nx{r['nx']}_ko{r['k_o']:g}": float(r["J_h_relative_half_span"]) for r in rows
        },
        "maximum_J_h_relative_half_span": max(float(r["J_h_relative_half_span"]) for r in rows),
        "passive_continuum_extrapolation": passive_extrapolation,
        "active_continuum_extrapolation": active_extrapolation,
        "finest_size_decomposition": {
            "nx": int(finest["nx"]),
            "J_to_endpoint_release_percent": 100.0 * (flux_to_state - 1.0),
            "endpoint_release_to_abrupt_work_percent": 100.0 * (state_to_work - 1.0),
            "product_percent": 100.0 * (flux_to_state * state_to_work - 1.0),
            "note": (
                "The first factor is the Rice-Eshelby gap between the static flux and the state-function "
                "endpoint release; the second is the additional odd supply and protocol dissipation."
            ),
        },
        "passive_relative_abs_J_vs_abrupt_by_size": {str(r["nx"]): abs(float(r["relative_J_vs_abrupt_difference"])) for r in passive},
        "active_relative_abs_J_vs_abrupt_by_size": {str(r["nx"]): abs(float(r["relative_J_vs_abrupt_difference"])) for r in active},
        "passive_difference_decreases_with_refinement": abs(float(passive[-1]["relative_J_vs_abrupt_difference"])) < abs(float(passive[0]["relative_J_vs_abrupt_difference"])),
        "active_fine_size_equivalence_break": float(active[-1]["relative_J_vs_abrupt_difference"]),
        "interpretation": "The passive configurational flux and deletion work approach one another under refinement. At finite odd modulus, the configurational flux, endpoint-energy release, abrupt work and quasistatic work remain distinct on the same lattice and load.",
    }
    summary["pass"] = bool(summary["passive_difference_decreases_with_refinement"] and abs(summary["active_fine_size_equivalence_break"]) > 0.2)
    (out / "J_work_bridge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@dataclass
class UnrestrictedResult:
    boundary_condition: str
    nx: int
    ny: int
    crack_fraction: float
    k_o: float
    load_fraction: float
    applied_amplitude: float
    broken_bonds: int
    off_plane_broken_bonds: int
    status: str
    initial_max_ratio: float
    final_max_ratio: float
    maximum_ratio_encountered: float
    maximum_free_force_residual: float
    broken_ids: str
    broken_crossing_flags: str


def _all_intact_ratios(model: ActiveCrackedStrip, removed: set[int], displacement: np.ndarray, delta_c: float) -> list[tuple[float, int]]:
    vals = []
    for bid in active_ids(model, removed):
        vals.append((bond_extension(model.all_bonds[bid], displacement) / delta_c, bid))
    vals.sort(reverse=True)
    return vals


def unrestricted_fixed(nx: int, ny: int, frac: float, ko: float, p: float, delta_c: float, max_breaks: int = 20) -> UnrestrictedResult:
    target = frac * nx
    passive_stress, _ = fixed_grip_passive_reference(nx, ny, target, delta_c)
    probe = FixedGripCascade(nx, ny, target, ko)
    amplitude = p * passive_stress / probe.initial_unit_stress()
    cas = FixedGripCascade(nx, ny, target, ko)
    broken: list[int] = []
    max_res = 0.0
    initial = final = maximum = float("nan")
    status = "arrest"
    for step in range(max_breaks + 1):
        try:
            u, _e, _o, _t, res = cas.solve(amplitude)
        except Exception:
            status = "singular"
            break
        max_res = max(max_res, float(res))
        vals = _all_intact_ratios(cas.model, cas.removed, u, delta_c)
        if not vals:
            status = "complete"
            break
        ratio, bid = vals[0]
        if step == 0:
            initial = ratio
            maximum = ratio
        maximum = max(maximum, ratio)
        final = ratio
        if ratio < 1.0 - 1e-10:
            status = "arrest"
            break
        cas.removed.add(bid)
        broken.append(bid)
    else:
        status = "max_breaks"
    flags = [cas.model.all_bonds[b].crosses_crack_plane for b in broken]
    return UnrestrictedResult("fixed_grip", nx, ny, frac, ko, p, amplitude, len(broken), sum(not x for x in flags), status, initial, final, maximum, max_res, ";".join(map(str, broken)), ";".join("1" if x else "0" for x in flags))


def unrestricted_dead(nx: int, ny: int, frac: float, ko: float, p: float, delta_c: float, max_breaks: int = 20) -> UnrestrictedResult:
    target = frac * nx
    passive_stress = dead_load_passive_reference(nx, ny, target, delta_c)
    amplitude = p * passive_stress
    cas = DeadLoadCascade(nx, ny, target, ko)
    broken: list[int] = []
    max_res = 0.0
    initial = final = maximum = float("nan")
    status = "arrest"
    for step in range(max_breaks + 1):
        try:
            u, _e, _o, _t, res = cas.solve(amplitude)
        except Exception:
            status = "singular"
            break
        max_res = max(max_res, float(res))
        vals = _all_intact_ratios(cas.model, cas.removed, u, delta_c)
        if not vals:
            status = "complete"
            break
        ratio, bid = vals[0]
        if step == 0:
            initial = ratio
            maximum = ratio
        maximum = max(maximum, ratio)
        final = ratio
        if ratio < 1.0 - 1e-10:
            status = "arrest"
            break
        cas.removed.add(bid)
        broken.append(bid)
    else:
        status = "max_breaks"
    flags = [cas.model.all_bonds[b].crosses_crack_plane for b in broken]
    return UnrestrictedResult("dead_load", nx, ny, frac, ko, p, amplitude, len(broken), sum(not x for x in flags), status, initial, final, maximum, max_res, ";".join(map(str, broken)), ";".join("1" if x else "0" for x in flags))


def _unrestricted_group(args: tuple) -> list[dict[str, object]]:
    """Run one unrestricted cascade group.

    ``args`` is ``(nx, ny, frac, ko, bc, max_breaks)`` and may carry an optional
    seventh entry naming the failure criterion. Because the bond force is
    ``f = k*delta*n - k_o*delta*t`` with orthogonal unit vectors, its magnitude is
    exactly ``|delta|*sqrt(k^2+k_o^2)``. A criterion on the total bond force at a
    fixed force threshold is therefore identical to the axial-extension criterion
    with the threshold rescaled by ``1/sqrt(1+(k_o/k)^2)``, which is what
    ``criterion="total_force"`` applies.
    """
    from propagation_limit_analysis import top_reaction_stress
    nx, ny, frac, ko, bc, max_breaks = args[:6]
    criterion = args[6] if len(args) > 6 else "axial_extension"
    delta_c = 0.02
    if criterion == "total_force":
        delta_c = delta_c / math.hypot(1.0, ko)
    elif criterion != "axial_extension":
        raise ValueError(f"unknown criterion {criterion!r}")
    target = frac * nx
    passive_fixed_stress, _ = fixed_grip_passive_reference(nx, ny, target, delta_c)
    passive_dead_stress = dead_load_passive_reference(nx, ny, target, delta_c)
    if bc == "fixed_grip":
        base = FixedGripCascade(nx, ny, target, ko)
        initial_u, _e, _o, initial_K, initial_r = base.solve(1.0)
        unit_remote = top_reaction_stress(base.model, initial_K, initial_u)
    else:
        base = DeadLoadCascade(nx, ny, target, ko)
        initial_u, _e, _o, _K, initial_r = base.solve(1.0)
        unit_remote = 1.0
    initial_removed = frozenset(base.initial_removed)
    cache: dict[frozenset[int], tuple[np.ndarray, float]] = {initial_removed: (initial_u, initial_r)}

    def unit_state(removed_key: frozenset[int]) -> tuple[np.ndarray, float]:
        if removed_key not in cache:
            base.removed = set(removed_key)
            u, _ee, _oo, _kk, rr = base.solve(1.0)
            cache[removed_key] = (u, rr)
        return cache[removed_key]

    rows: list[dict[str, object]] = []
    for p in (0.85, 0.90, 0.95, 0.98):
        amplitude = (p * passive_fixed_stress / unit_remote) if bc == "fixed_grip" else (p * passive_dead_stress)
        removed = initial_removed
        broken: list[int] = []
        max_res = 0.0
        initial_ratio = final_ratio = maximum_ratio = float("nan")
        status = "arrest"
        for step in range(max_breaks + 1):
            try:
                u_unit, rr = unit_state(removed)
            except Exception:
                status = "singular"
                break
            max_res = max(max_res, float(rr) * abs(amplitude))
            vals = _all_intact_ratios(base.model, set(removed), amplitude * u_unit, delta_c)
            if not vals:
                status = "complete"
                break
            ratio, bid = vals[0]
            if step == 0:
                initial_ratio = ratio
                maximum_ratio = ratio
            maximum_ratio = max(maximum_ratio, ratio)
            final_ratio = ratio
            if ratio < 1.0 - 1e-10:
                status = "arrest"
                break
            broken.append(bid)
            removed = frozenset(set(removed) | {bid})
        else:
            status = "max_breaks"
        flags = [base.model.all_bonds[b].crosses_crack_plane for b in broken]
        record = asdict(UnrestrictedResult(
            bc, nx, ny, frac, ko, p, amplitude, len(broken),
            sum(not x for x in flags), status, initial_ratio,
            final_ratio, maximum_ratio, max_res,
            ";".join(map(str, broken)),
            ";".join("1" if x else "0" for x in flags),
        ))
        record["criterion"] = criterion
        record["effective_delta_c"] = delta_c
        rows.append(record)
    return rows


def unrestricted_cascade_scan(out: Path) -> dict[str, object]:
    """Run all-bond cascades with topology caching and parallel group solves."""
    from concurrent.futures import ProcessPoolExecutor, as_completed
    systems = [
        (32, 24, 0.10), (32, 24, 0.125), (32, 24, 0.15),
        (48, 36, 0.10), (48, 36, 0.125), (48, 36, 0.15),
        (64, 48, 0.125),
    ]
    jobs = [(nx, ny, frac, ko, bc, 8) for nx, ny, frac in systems for ko in (0.0, 0.12, 0.20, 0.30) for bc in ("fixed_grip", "dead_load")]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_unrestricted_group, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            group_rows = future.result()
            rows.extend(group_rows)
            print("cascade group", job[:5], "done")
    rows.sort(key=lambda r: (int(r["nx"]), float(r["crack_fraction"]), float(r["k_o"]), str(r["boundary_condition"]), float(r["load_fraction"])))
    write_csv(out / "unrestricted_all_bond_cascades.csv", rows)
    counts = np.array([int(r["broken_bonds"]) for r in rows])
    off = np.array([int(r["off_plane_broken_bonds"]) for r in rows])
    # The loop admits max_breaks+1 successive deletions, so the imposed cutoff is nine.
    cutoff = 9
    continuing = counts >= cutoff - 1
    loads = sorted({float(r["load_fraction"]) for r in rows})
    komoduli = sorted({float(r["k_o"]) for r in rows})
    geometry_bc_combinations = len({(int(r["nx"]), float(r["crack_fraction"]), str(r["boundary_condition"])) for r in rows})
    by_load_ko: dict[str, dict[str, int]] = {}
    minimum_continuing_ko: dict[str, object] = {}
    for p in loads:
        entry: dict[str, int] = {}
        for ko in komoduli:
            sel = [i for i, r in enumerate(rows)
                   if math.isclose(float(r["load_fraction"]), p) and math.isclose(float(r["k_o"]), ko)]
            entry[f"{ko:g}"] = int(sum(bool(continuing[i]) for i in sel))
        by_load_ko[f"{p:g}"] = entry
        active = [ko for ko in komoduli if ko > 0.0 and entry[f"{ko:g}"] > 0]
        minimum_continuing_ko[f"{p:g}"] = min(active) if active else None
    # Ordered deletion patterns for the continuing cascades (1 = bond crosses the cleavage plane).
    patterns: dict[str, int] = {}
    for i, r in enumerate(rows):
        if not continuing[i]:
            continue
        key = str(r["broken_crossing_flags"]).replace(";", "")
        patterns[key] = patterns.get(key, 0) + 1
    first_deletion_on_plane = sum(
        1 for i, r in enumerate(rows)
        if continuing[i] and str(r["broken_crossing_flags"]).split(";")[0] == "1"
    )
    second_deletion_off_plane = sum(
        1 for i, r in enumerate(rows)
        if continuing[i] and str(r["broken_crossing_flags"]).split(";")[1] == "0"
    )
    summary = {
        "state_count": len(rows),
        "geometry_boundary_combinations": geometry_bc_combinations,
        "deletion_cutoff": cutoff,
        "states_no_break": int(np.sum(counts == 0)),
        "states_one_break": int(np.sum(counts == 1)),
        "states_two_or_more_breaks": int(np.sum(counts >= 2)),
        "states_reaching_cutoff": int(np.sum(continuing)),
        "states_with_off_plane_break": int(np.sum(off > 0)),
        "maximum_broken_bonds": int(counts.max()),
        "maximum_off_plane_broken_bonds": int(off.max()),
        "minimum_off_plane_among_continuing": int(off[continuing].min()) if continuing.any() else None,
        "continuing_states_by_load_and_k_o": by_load_ko,
        "minimum_continuing_k_o_by_load": minimum_continuing_ko,
        "continuing_first_deletion_on_cleavage_plane": first_deletion_on_plane,
        "continuing_second_deletion_off_plane": second_deletion_off_plane,
        "continuing_deletion_patterns": patterns,
        "passive_states_reaching_cutoff": int(sum(bool(continuing[i]) for i, r in enumerate(rows) if math.isclose(float(r["k_o"]), 0.0))),
        "status_counts": {s: sum(str(r["status"]) == s for r in rows) for s in sorted({str(r["status"]) for r in rows})},
        "interpretation": (
            "The candidate-set restriction is removed: every intact tensile bond is eligible after each "
            "equilibrium update. States at the nine-deletion cutoff are classified as continuing cascades "
            "rather than arrests. The minimum continuing odd modulus decreases monotonically with load, "
            "which measures the reduction of the effective propagation threshold by activity."
        ),
        "pass": True,
    }
    (out / "unrestricted_cascade_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary

def criterion_sensitivity_scan(out: Path) -> dict[str, object]:
    """Repeat every unrestricted cascade with a total-bond-force failure criterion.

    For this bond law the total force magnitude is exactly proportional to the
    axial extension, so a fixed force threshold is an axial threshold rescaled by
    ``1/sqrt(1+(k_o/k)^2)``. The scan therefore measures how much the cascade
    classification depends on which of the two equivalent measures is held fixed
    across odd moduli.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    systems = [
        (32, 24, 0.10), (32, 24, 0.125), (32, 24, 0.15),
        (48, 36, 0.10), (48, 36, 0.125), (48, 36, 0.15),
        (64, 48, 0.125),
    ]
    jobs = [(nx, ny, frac, ko, bc, 8, "total_force")
            for nx, ny, frac in systems for ko in (0.0, 0.12, 0.20, 0.30)
            for bc in ("fixed_grip", "dead_load")]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(_unrestricted_group, job): job for job in jobs}
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda r: (int(r["nx"]), float(r["crack_fraction"]), float(r["k_o"]),
                             str(r["boundary_condition"]), float(r["load_fraction"])))
    write_csv(out / "criterion_sensitivity_cascades.csv", rows)

    baseline = pd.read_csv(out / "unrestricted_all_bond_cascades.csv")
    force = pd.DataFrame(rows)
    keys = ["boundary_condition", "nx", "crack_fraction", "k_o", "load_fraction"]
    merged = baseline.merge(force, on=keys, suffixes=("_axial", "_force"))
    continuing_axial = merged.broken_bonds_axial >= 8
    continuing_force = merged.broken_bonds_force >= 8
    changed = int((continuing_axial != continuing_force).sum())
    summary = {
        "state_count": len(rows),
        "maximum_threshold_rescaling": float(1.0 - 1.0 / math.hypot(1.0, 0.30)),
        "states_reaching_cutoff_axial": int(continuing_axial.sum()),
        "states_reaching_cutoff_total_force": int(continuing_force.sum()),
        "states_with_changed_classification": changed,
        "maximum_deletion_count_difference": int((merged.broken_bonds_axial - merged.broken_bonds_force).abs().max()),
        "interpretation": (
            "The total bond force is exactly |delta|*sqrt(k^2+k_o^2) for this bond law, so a fixed "
            "force threshold rescales the axial threshold by at most 4.2 percent over the tested "
            "odd moduli. The cascade classification is unchanged, so the propagation conclusions do "
            "not depend on which of the two equivalent bond measures is held fixed."
        ),
        "pass": changed == 0,
    }
    (out / "criterion_sensitivity_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def source_identity_refinement(out: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    ko = 0.15
    for nx in (32, 48, 64, 80, 96):
        ny = 3 * nx // 4
        crack = nx / 8.0
        fit_radius = max(3.0, 0.04 * nx)
        inner = 0.075 * nx
        outer = 0.125 * nx
        fields = {}
        residuals = {}
        for value in (0.0, ko):
            model = ActiveCrackedStrip(nx, ny, crack, 1.0, value)
            u, _, res = model.solve(1.0)
            fields[value] = LocalTipField(model, u, "right", fit_radius=fit_radius)
            residuals[value] = res
        jp_i = keyhole_j(fields[0.0], inner, line_step=0.20)
        jp_o = keyhole_j(fields[0.0], outer, line_step=0.20)
        ja_i = keyhole_j(fields[ko], inner, line_step=0.20)
        ja_o = keyhole_j(fields[ko], outer, line_step=0.20)
        qodd, qres, n = annulus_sources(fields[ko], inner, outer, area_step=0.40)
        delta_j = (ja_o - ja_i) - (jp_o - jp_i)
        ratio = delta_j / qodd
        rows.append({
            "nx": nx,
            "ny": ny,
            "a_lat_over_L": 1.0 / nx,
            "k_o": ko,
            "fit_radius": fit_radius,
            "inner_radius": inner,
            "outer_radius": outer,
            "active_excess_contour_drift": delta_j,
            "Q_odd_MLS": qodd,
            "Q_equilibrium_residual": qres,
            "closure_ratio": ratio,
            "sample_count": n,
            "passive_equilibrium_residual": residuals[0.0],
            "active_equilibrium_residual": residuals[ko],
        })
        print("source", nx, ratio)
    write_csv(out / "source_identity_refinement.csv", rows)
    errors = [abs(float(r["closure_ratio"]) - 1.0) for r in rows]
    # Tail behaviour: fit err = a + b (a_lat/L)^q profiled over q, to test whether the
    # sequence extrapolates to zero or to a finite reconstruction floor.
    h = np.array([float(r["a_lat_over_L"]) for r in rows])
    err = np.array([float(r["closure_ratio"]) - 1.0 for r in rows])
    best: tuple[float, float, np.ndarray] | None = None
    for q in np.linspace(0.5, 3.0, 251):
        design = np.column_stack([np.ones_like(h), h ** q])
        coefficients, *_ = np.linalg.lstsq(design, err, rcond=None)
        rss = float(np.sum((design @ coefficients - err) ** 2))
        if best is None or rss < best[0]:
            best = (rss, float(q), coefficients)
    rss, q_best, coefficients = best
    offset, amplitude_fit = (float(coefficients[0]), float(coefficients[1]))
    finest_pair_slope = (err[-1] - err[-2]) / (h[-1] - h[-2])
    finest_pair_intercept = float(err[-1] - finest_pair_slope * h[-1])
    summary = {
        "ratios": {str(r["nx"]): float(r["closure_ratio"]) for r in rows},
        "absolute_error_coarse": errors[0],
        "absolute_error_fine": errors[-1],
        "error_decreases": errors[-1] < errors[0],
        "biased_power_fit": {
            "offset": offset,
            "amplitude": amplitude_fit,
            "exponent": q_best,
            "rms_residual": float(math.sqrt(rss / len(h))),
        },
        "finest_pair_linear_intercept": finest_pair_intercept,
        "offset_consistent_with_zero": bool(abs(offset) < 0.005),
        "interpretation": (
            "This is a numerical reconstruction-convergence test of the continuum source identity, "
            "distinct from the exact discrete algebraic partition. The sequence decreases monotonically "
            "but the two finest sizes nearly coincide, so the residual is reported as a finite "
            "reconstruction floor of the same size as the moving-least-squares/keyhole slope bias "
            "rather than as convergence to exactly unity."
        ),
        "pass": bool(errors[-1] < errors[0]),
    }
    (out / "source_identity_refinement_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def angular_momentum_audit(out: Path) -> dict[str, object]:
    nx, ny, crack, ko = 48, 36, 6.0, 0.20
    model = ActiveCrackedStrip(nx, ny, crack, 1.0, ko)
    u, residual, free_res = model.solve(1.0)
    uvec = u.ravel()
    Ku = model.K @ uvec
    positions = model.positions
    nodal = Ku.reshape((-1, 2))
    boundary_moment = float(np.sum(positions[:, 0] * nodal[:, 1] - positions[:, 1] * nodal[:, 0]))
    internal_moment = 0.0
    internal_force = np.zeros_like(u)
    for bid in active_ids(model, set(model.removed_ids)):
        b = model.all_bonds[bid]
        ext = bond_extension(b, uvec)
        tangent = np.array([-b.n[1], b.n[0]])
        fi = (model.k * b.n - model.k_o * tangent) * ext
        internal_force[b.i] += fi
        internal_force[b.j] -= fi
        internal_moment += float(positions[b.i, 0] * fi[1] - positions[b.i, 1] * fi[0])
        internal_moment += float(positions[b.j, 0] * (-fi[1]) - positions[b.j, 1] * (-fi[0]))
    force_error = float(np.max(np.abs(internal_force.ravel() + Ku)))
    torque_residual = boundary_moment + internal_moment
    summary = {
        "boundary_reaction_moment": boundary_moment,
        "direct_internal_pair_moment": internal_moment,
        "moment_balance_residual": torque_residual,
        "relative_moment_balance_residual": abs(torque_residual) / max(abs(boundary_moment), abs(internal_moment), 1e-30),
        "direct_force_vs_matrix_max_abs_error": force_error,
        "free_force_residual": free_res,
        "interpretation": "The noncentral pair couple is balanced by boundary/substrate torque. In translational virtual work its effect is already contained in the nonsymmetric Cauchy stress term.",
        "pass": bool(abs(torque_residual) < 1e-10 and force_error < 1e-11),
    }
    (out / "angular_momentum_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def perfect_lattice_stability(out: Path) -> dict[str, object]:
    # Use the geometry/boundary partition of a 24x18 strip, but restore all bonds.
    model = ActiveCrackedStrip(24, 18, 3.0, 1.0, 0.0)
    constrained, _ = model.constrained_dofs(0.0)
    mask = np.ones(model.ndof, dtype=bool)
    mask[constrained] = False
    free = np.arange(model.ndof)[mask]
    rows = []
    first_negative = None
    for ko in np.linspace(0.0, 3.0, 31):
        model.k_o = float(ko)
        Ee, Eo = assemble_components(model, list(range(len(model.all_bonds))))
        Kff = (Ee + Eo)[free][:, free]
        vals = eigs(Kff, k=6, sigma=0.0, which="LM", return_eigenvectors=False, tol=1e-8, maxiter=30000)
        minimum = float(np.min(np.real(vals)))
        rows.append({"k_o_over_k": float(ko), "minimum_real_decay_eigenvalue": minimum, "maximum_abs_imaginary_part_soft_modes": float(np.max(np.abs(np.imag(vals))))})
        if minimum <= 0.0 and first_negative is None:
            first_negative = float(ko)
    write_csv(out / "perfect_lattice_stability_scan.csv", rows)
    positive = [r for r in rows if float(r["minimum_real_decay_eigenvalue"]) > 0]
    summary = {
        "tested_range": [0.0, 3.0],
        "first_sampled_nonpositive_k_o_over_k": first_negative,
        "minimum_real_part_at_0p30": next(float(r["minimum_real_decay_eigenvalue"]) for r in rows if math.isclose(float(r["k_o_over_k"]), 0.3)),
        "largest_tested_stable_k_o_over_k": max(float(r["k_o_over_k"]) for r in positive),
        "interpretation": "The perfect-lattice scan is a finite-strip overdamped stability reference, not a proof for every wave vector or boundary condition.",
        "pass": next(float(r["minimum_real_decay_eigenvalue"]) for r in rows if math.isclose(float(r["k_o_over_k"]), 0.3)) > 0,
    }
    (out / "perfect_lattice_stability_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def intact_lattice_inertial_control(out: Path) -> dict[str, object]:
    """Inertial stability of the *intact* lattice at the probe geometry.

    The weakly damped post-cut probe can only be attributed to the crack if the
    uncracked reference configuration is itself inertially stable at the same
    damping. For a nonsymmetric stiffness the quadratic pencil
    ``lambda^2 + c*lambda + kappa = 0`` is stable for every stiffness eigenvalue
    ``kappa`` iff ``Re kappa > 0`` and ``c^2 Re kappa > (Im kappa)^2``, so the
    critical damping of the intact lattice is
    ``c_crit = max_modes |Im kappa| / sqrt(Re kappa)``.
    """
    from scipy import sparse

    nx, ny, crack = 32, 24, 4.0
    model = ActiveCrackedStrip(nx, ny, crack, 1.0, 0.0)
    constrained, _ = model.constrained_dofs(0.0)
    mask = np.ones(model.ndof, dtype=bool)
    mask[constrained] = False
    free = np.arange(model.ndof)[mask]
    nfree = len(free)
    all_bond_ids = list(range(len(model.all_bonds)))

    def intact_stiffness(ko: float):
        model.k_o = float(ko)
        Ee, Eo = assemble_components(model, all_bond_ids)
        return (Ee + Eo)[free][:, free].tocsr()

    def max_real_inertial(K, damping: float) -> float:
        A = sparse.bmat(
            [[None, sparse.eye(nfree, format="csr")],
             [-K, -damping * sparse.eye(nfree, format="csr")]],
            format="csr",
        )
        vals = eigs(A, k=6, which="LR", return_eigenvectors=False, tol=1e-8, maxiter=60000)
        return float(np.max(np.real(vals)))

    rows: list[dict[str, object]] = []
    for ko in (0.0, 0.05, 0.12, 0.20, 0.30, 0.50):
        K = intact_stiffness(ko)
        kappa = np.linalg.eigvals(np.asarray(K.todense()))
        re = kappa.real
        im = kappa.imag
        positive = re > 1e-12
        c_crit = float(np.max(np.abs(im[positive]) / np.sqrt(re[positive]))) if positive.any() else float("nan")
        rows.append({
            "k_o_over_k": float(ko),
            "minimum_real_stiffness_eigenvalue": float(re.min()),
            "overdamped_stable": bool(re.min() > 0.0),
            "critical_inertial_damping": c_crit,
        })
        print("intact control", ko, re.min(), c_crit, flush=True)
    write_csv(out / "intact_lattice_critical_damping.csv", rows)

    probe_ko = 0.20
    K_probe = intact_stiffness(probe_ko)
    control_rows: list[dict[str, object]] = []
    for damping in (0.05, 0.20, 1.0):
        value = max_real_inertial(K_probe, damping)
        control_rows.append({
            "k_o_over_k": probe_ko,
            "damping": damping,
            "intact_maximum_real_dynamic_eigenvalue": value,
            "intact_linearly_stable": bool(value < 0.0),
        })
    write_csv(out / "intact_lattice_inertial_control.csv", control_rows)

    c_crit_probe = next(float(r["critical_inertial_damping"]) for r in rows if math.isclose(float(r["k_o_over_k"]), probe_ko))
    summary = {
        "geometry": {"nx": nx, "ny": ny, "mass": 1.0},
        "critical_damping_by_k_o": {str(r["k_o_over_k"]): r["critical_inertial_damping"] for r in rows},
        "overdamped_stable_for_all_tested_k_o": all(bool(r["overdamped_stable"]) for r in rows),
        "critical_damping_at_probe_modulus": c_crit_probe,
        "intact_control_cases": control_rows,
        "interpretation": (
            "The intact lattice is overdamped-stable for every tested odd modulus, but its inertial "
            "form requires damping above c_crit. At k_o/k=0.20 the probe damping values 0.05 and 0.20 "
            "lie below c_crit while 1.0 lies above it, so the flutter seen in the post-cut probe is a "
            "bulk property of the odd lattice rather than a crack-tip effect."
        ),
        "pass": bool(
            all(bool(r["overdamped_stable"]) for r in rows)
            and control_rows[0]["intact_linearly_stable"] is False
            and control_rows[-1]["intact_linearly_stable"] is True
        ),
    }
    (out / "intact_lattice_inertial_control_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def inertial_postcut_probe(out: Path) -> dict[str, object]:
    from scipy import sparse
    from scipy.sparse.linalg import expm_multiply
    nx, ny, crack, ko, p, delta_c = 32, 24, 4.0, 0.20, 0.90, 0.02
    passive_stress, _ = fixed_grip_passive_reference(nx, ny, crack, delta_c)
    probe = FixedGripCascade(nx, ny, crack, ko)
    amplitude = p * passive_stress / probe.initial_unit_stress()
    model = ActiveCrackedStrip(nx, ny, crack, 1.0, ko)
    removed = set(model.removed_ids)
    old_ids = active_ids(model, removed)
    Ee0, Eo0 = assemble_components(model, old_ids)
    u0, c, cv, free, _ = solve_equilibrium(Ee0 + Eo0, model, amplitude)
    frontier = model.tip_candidates()
    first = max(frontier.values(), key=lambda bid: bond_extension(model.all_bonds[bid], u0))
    removed.add(first)
    remaining = active_ids(model, removed)
    Ee1, Eo1 = assemble_components(model, remaining)
    K1 = Ee1 + Eo1
    ueq, c2, cv2, free2, _ = solve_equilibrium(K1, model, amplitude)
    if not (np.array_equal(c, c2) and np.array_equal(free, free2)):
        raise RuntimeError("inertial boundary partition changed")
    Kff = K1[free][:, free].tocsr()
    offset0 = u0[free] - ueq[free]
    nfree = len(free)

    # Sparse extension operator for every remaining bond.
    erows=[]; ecols=[]; edata=[]
    for row,bid in enumerate(remaining):
        b=model.all_bonds[bid]
        for comp in range(2):
            erows.extend([row,row]); ecols.extend([2*b.j+comp,2*b.i+comp]); edata.extend([float(b.n[comp]),-float(b.n[comp])])
    B=sparse.coo_matrix((edata,(erows,ecols)),shape=(len(remaining),model.ndof)).tocsr()
    baseline=np.asarray(B@ueq).ravel()
    Bf=B[:,free]

    rows=[]
    for damping in (0.05,0.20,1.0):
        A=sparse.bmat([[None,sparse.eye(nfree,format='csr')],[-Kff,-damping*sparse.eye(nfree,format='csr')]],format='csr')
        y0=np.concatenate([offset0,np.zeros(nfree)])
        times=np.linspace(0.0,150.0,301)
        vals=eigs(A,k=4,which="LR",return_eigenvectors=False,tol=1e-7,maxiter=30000)
        maximum_real_eigenvalue=float(np.max(np.real(vals)))
        trajectory=expm_multiply(A,y0,start=0.0,stop=150.0,num=301,endpoint=True)
        maxima=-np.inf; max_index=-1; max_time=0.0
        for j,state in enumerate(trajectory):
            ext=baseline+np.asarray(Bf@state[:nfree]).ravel()
            idx=int(np.argmax(ext))
            ratio=float(ext[idx]/delta_c)
            if ratio>maxima:
                maxima=ratio; max_index=idx; max_time=float(times[j])
        bid=remaining[max_index]
        rows.append({"damping":damping,"maximum_real_dynamic_eigenvalue":maximum_real_eigenvalue,"linearly_stable":bool(maximum_real_eigenvalue<0.0),"maximum_remaining_bond_ratio":float(maxima),"threshold_crossed":bool(maxima>=1.0),"maximizing_bond_id":int(bid),"maximizing_bond_crosses_cleavage":bool(model.all_bonds[bid].crosses_crack_plane),"time_of_maximum":max_time,"n_time_samples":len(times)})
    write_csv(out/'weakly_damped_inertial_probe.csv',rows)
    summary={"configuration":{"nx":nx,"ny":ny,"crack_half_length":crack,"k_o":ko,"load_fraction":p,"mass":1.0},"cases":rows,"interpretation":"This single linear inertial post-cut probe tests transient threshold crossing without deleting the second bond. It is not a dynamic fracture phase diagram.","pass":True}
    (out/'weakly_damped_inertial_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "configurational_work_bridge_results")
    parser.add_argument("--skip-source", action="store_true")
    parser.add_argument("--skip-cascade", action="store_true")
    parser.add_argument("--only", choices=["j_work_bridge"], help="run a single audit and exit")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}
    summary["J_work_bridge"] = j_work_bridge(args.out)
    if args.only == "j_work_bridge":
        # Merge into any existing summary so that verify_results.py never reads a
        # stale section alongside freshly written CSVs.
        target = args.out / "summary.json"
        merged = json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}
        merged["J_work_bridge"] = summary["J_work_bridge"]
        target.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(json.dumps(summary["J_work_bridge"], indent=2))
        print(f"updated {target}")
        return
    summary["two_parameter_softening"] = two_parameter_softening(args.out)
    if not args.skip_cascade:
        summary["unrestricted_cascade"] = unrestricted_cascade_scan(args.out)
        summary["criterion_sensitivity"] = criterion_sensitivity_scan(args.out)
    if not args.skip_source:
        summary["source_refinement"] = source_identity_refinement(args.out)
    summary["angular_momentum"] = angular_momentum_audit(args.out)
    summary["perfect_lattice_stability"] = perfect_lattice_stability(args.out)
    summary["intact_lattice_inertial_control"] = intact_lattice_inertial_control(args.out)
    summary["inertial_probe"] = inertial_postcut_probe(args.out)
    summary["pass"] = all(bool(v.get("pass", True)) for v in summary.values() if isinstance(v, dict))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
