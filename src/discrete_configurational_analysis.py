#!/usr/bin/env python3
""" bond-exact discrete configurational domain functional.

This script avoids MLS stress/Hessian reconstruction.  The triangular lattice is
partitioned into elementary triangles.  Each active bond contributes one half of
its even energy and stress to each adjacent triangle.  The nodal displacement and
a smooth material-translation weight q are interpolated affinely on each
triangle.  The discrete domain functional

    J_h[q] = - sum_T A_T P_{1j,T} q_{,j,T}

is therefore evaluated exactly from bond extensions and nodal values.  It admits
the exact algebraic decomposition

    Delta J_h = Delta J_h^even + J_h^odd,

where Delta denotes active-minus-passive and the two terms use the even and odd
parts of the bond stress.  No continuum stress reconstruction or second spatial
derivative is used.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lattice_baselines import A1, A2, R90, SQRT3, wrap_centered
from active_tip_scan import ActiveCrackedStrip

AREA_TRI = SQRT3 / 4.0


@dataclass(frozen=True)
class Triangle:
    nodes: tuple[int, int, int]
    coords: np.ndarray  # unwrapped global reference coordinates, shape (3,2)
    bond_ids: tuple[int, int, int]


@dataclass(frozen=True)
class TriangleField:
    centroid_local: np.ndarray
    grad_u: np.ndarray
    stress_even: np.ndarray
    stress_odd: np.ndarray
    energy_even: float
    area: float


def _pair_key(i: int, j: int) -> tuple[int, int]:
    return (i, j) if i < j else (j, i)


def build_triangles(model: ActiveCrackedStrip) -> list[Triangle]:
    """Construct the two elementary triangles in every rhombic lattice cell."""
    bond_map: dict[tuple[int, int], int] = {}
    for bid, bond in enumerate(model.all_bonds):
        key = _pair_key(bond.i, bond.j)
        if key in bond_map:
            raise RuntimeError(f"Duplicate bond pair {key}")
        bond_map[key] = bid

    triangles: list[Triangle] = []
    for j in range(model.ny - 1):
        for i in range(model.nx):
            n00 = model.node_id(i, j)
            n10 = model.node_id(i + 1, j)
            n01 = model.node_id(i, j + 1)
            n11 = model.node_id(i + 1, j + 1)

            x00 = i * A1 + j * A2
            x10 = (i + 1) * A1 + j * A2
            x01 = i * A1 + (j + 1) * A2
            x11 = (i + 1) * A1 + (j + 1) * A2

            for nodes, coords in (
                ((n00, n10, n01), np.vstack([x00, x10, x01])),
                ((n10, n11, n01), np.vstack([x10, x11, x01])),
            ):
                edges = ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0]))
                try:
                    bids = tuple(bond_map[_pair_key(a, b)] for a, b in edges)
                except KeyError as exc:
                    raise RuntimeError(f"Triangle edge is not a lattice bond: {nodes}") from exc
                triangles.append(Triangle(nodes=nodes, coords=coords, bond_ids=bids))
    return triangles


def affine_gradient(coords: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Return grad(values) for affine interpolation on a triangle.

    values has shape (3,m); output has shape (m,2), with columns x,y.
    """
    B = np.column_stack([np.ones(3), coords[:, 0], coords[:, 1]])
    coeff = np.linalg.solve(B, values)
    return coeff[1:, :].T


def tip_geometry(model: ActiveCrackedStrip, tip: str) -> tuple[float, float, float]:
    if tip not in {"right", "left"}:
        raise ValueError("tip must be right or left")
    direction = 1.0 if tip == "right" else -1.0
    x_tip = 0.5 * model.period + direction * model.a_eff
    y_crack = 0.5 * (
        model.positions[model.node_id(0, model.j_lower), 1]
        + model.positions[model.node_id(0, model.j_lower + 1), 1]
    )
    return x_tip, y_crack, direction


def localize_coords(coords: np.ndarray, model: ActiveCrackedStrip, tip: str) -> np.ndarray:
    x_tip, y_crack, direction = tip_geometry(model, tip)
    out = coords.copy()
    # Shift the entire triangle by one period so its centroid is closest to the tip.
    c = float(np.mean(out[:, 0]))
    shift = round((x_tip - c) / model.period) * model.period
    out[:, 0] += shift
    out[:, 0] = direction * (out[:, 0] - x_tip)
    out[:, 1] -= y_crack
    return out


def triangle_fields(
    model: ActiveCrackedStrip,
    u: np.ndarray,
    triangles: Iterable[Triangle],
    tip: str,
    support_radius: float,
) -> list[TriangleField]:
    active = set(range(len(model.all_bonds))) - set(model.removed_ids)
    _, _, direction = tip_geometry(model, tip)
    S = np.diag([direction, 1.0])
    fields: list[TriangleField] = []
    for tri in triangles:
        xloc = localize_coords(tri.coords, model, tip)
        centroid = np.mean(xloc, axis=0)
        if np.max(np.abs(centroid)) > support_radius + 2.0:
            continue

        uloc = (S @ u[np.asarray(tri.nodes)].T).T
        grad_u = affine_gradient(xloc, uloc)
        se = np.zeros((2, 2), dtype=float)
        so = np.zeros((2, 2), dtype=float)
        W = 0.0
        for bid in tri.bond_ids:
            if bid not in active:
                continue
            bond = model.all_bonds[bid]
            n = S @ bond.n
            t = R90 @ n
            # Use the stored oriented bond difference, transformed to local coordinates.
            du = S @ (u[bond.j] - u[bond.i])
            ext = float(du @ n)
            # Half of each bond belongs to each adjacent elementary triangle.
            fac = 0.5 / AREA_TRI
            se += fac * model.k * ext * np.outer(n, n)
            so += fac * (-direction * model.k_o) * ext * np.outer(t, n)
            W += fac * 0.5 * model.k * ext * ext

        fields.append(
            TriangleField(
                centroid_local=centroid,
                grad_u=grad_u,
                stress_even=se,
                stress_odd=so,
                energy_even=float(W),
                area=AREA_TRI,
            )
        )
    return fields


def lp_radius_and_grad(x: np.ndarray, p: float) -> tuple[float, np.ndarray]:
    ax = np.abs(x)
    r = float((ax[0] ** p + ax[1] ** p) ** (1.0 / p))
    if r < 1e-14:
        return r, np.zeros(2)
    grad = np.sign(x) * ax ** (p - 1.0) / (r ** (p - 1.0))
    return r, grad


def q_value(x: np.ndarray, radius: float, width: float, p: float) -> float:
    r, _ = lp_radius_and_grad(x, p)
    a = radius - 0.5 * width
    b = radius + 0.5 * width
    if r <= a:
        return 1.0
    if r >= b:
        return 0.0
    s = (r - a) / width
    return float(0.5 * (1.0 + math.cos(math.pi * s)))


def discrete_J(
    fields: list[TriangleField],
    triangles_local_nodes: list[np.ndarray],
    radius: float,
    width: float = 1.5,
    p: float = 4.0,
    shift: tuple[float, float] = (0.0, 0.0),
) -> dict[str, float]:
    """Evaluate the piecewise-affine discrete domain functional."""
    total = even = odd = 0.0
    shiftv = np.asarray(shift, dtype=float)
    for fld, xnodes in zip(fields, triangles_local_nodes):
        qnod = np.array([q_value(x - shiftv, radius, width, p) for x in xnodes])[:, None]
        grad_q = affine_gradient(xnodes, qnod)[0]
        ux = fld.grad_u[:, 0]
        P_even = np.array([fld.energy_even, 0.0]) - fld.stress_even.T @ ux
        P_odd = -fld.stress_odd.T @ ux
        je = -fld.area * float(P_even @ grad_q)
        jo = -fld.area * float(P_odd @ grad_q)
        even += je
        odd += jo
        total += je + jo
    return {"J_total": total, "J_even": even, "J_odd": odd}


def prepare_model_fields(
    nx: int,
    ny: int,
    a: float,
    ko: float,
    tip: str,
    support_radius: float,
) -> tuple[ActiveCrackedStrip, np.ndarray, list[TriangleField], list[np.ndarray], float]:
    model = ActiveCrackedStrip(nx=nx, ny=ny, crack_half_length=a, k=1.0, k_o=ko)
    u, _, residual = model.solve(delta=1.0)
    triangles = build_triangles(model)
    fields = triangle_fields(model, u, triangles, tip, support_radius)
    local_nodes: list[np.ndarray] = []
    # Keep the same support filtering/order used in triangle_fields.
    for tri in triangles:
        xloc = localize_coords(tri.coords, model, tip)
        centroid = np.mean(xloc, axis=0)
        if np.max(np.abs(centroid)) <= support_radius + 2.0:
            local_nodes.append(xloc)
    if len(fields) != len(local_nodes):
        raise RuntimeError("Triangle field/local-coordinate mismatch")
    return model, u, fields, local_nodes, residual


def fit_origin(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    slope = float(x @ y / (x @ x))
    res = y - slope * x
    r2 = 1.0 - float(res @ res) / float(y @ y)
    return slope, r2


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("discrete_configurational_results"))
    ap.add_argument("--nx", type=int, default=80)
    ap.add_argument("--ny", type=int, default=56)
    ap.add_argument("--a", type=float, default=10.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    nx, ny, a = args.nx, args.ny, args.a
    radii = (4.0, 5.0, 6.0, 7.0, 8.0)
    ko_values = (0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20)
    support = 9.5

    _, _, pf, px, pres = prepare_model_fields(nx, ny, a, 0.0, "right", support)
    passive_by_R = {R: discrete_J(pf, px, R) for R in radii}

    radius_rows: list[dict[str, float]] = []
    for R in radii:
        radius_rows.append({"k_o": 0.0, "R": R, **passive_by_R[R]})

    balance_rows: list[dict[str, float]] = []
    for ko in ko_values:
        _, _, af, ax, ares = prepare_model_fields(nx, ny, a, ko, "right", support)
        vals = {R: discrete_J(af, ax, R) for R in radii}
        for R in radii:
            radius_rows.append({"k_o": ko, "R": R, **vals[R]})

        def shell(comp: str) -> float:
            return vals[8.0][comp] - vals[4.0][comp]

        def pshell(comp: str) -> float:
            return passive_by_R[8.0][comp] - passive_by_R[4.0][comp]

        excess_total = shell("J_total") - pshell("J_total")
        excess_even = shell("J_even") - pshell("J_even")
        odd_direct = shell("J_odd")
        closure = excess_total - excess_even - odd_direct
        balance_rows.append(
            {
                "k_o": ko,
                "excess_total": excess_total,
                "active_change_even": excess_even,
                "direct_odd_term": odd_direct,
                "algebraic_closure": closure,
                "odd_fraction": odd_direct / excess_total,
                "even_redistribution_fraction": excess_even / excess_total,
                "free_residual_inf": ares,
            }
        )

    write_csv(args.out / "discrete_domain_radius_scan.csv", radius_rows)
    write_csv(args.out / "discrete_configurational_balance.csv", balance_rows)

    passive_vals = np.array([passive_by_R[R]["J_total"] for R in radii])
    passive_spread = float((passive_vals.max() - passive_vals.min()) / abs(passive_vals.mean()))
    total = np.array([r["excess_total"] for r in balance_rows])
    odd = np.array([r["direct_odd_term"] for r in balance_rows])
    even = np.array([r["active_change_even"] for r in balance_rows])
    slope_odd, r2_odd = fit_origin(odd, total)
    slope_sum, r2_sum = fit_origin(odd + even, total)

    # Weight/center robustness at ko=0.15.
    _, _, af15, ax15, _ = prepare_model_fields(nx, ny, a, 0.15, "right", support)
    robustness: list[dict[str, float]] = []
    for p in (2.0, 4.0, 8.0, 16.0):
        for width in (0.75, 1.0, 1.5, 2.0, 2.5):
            pa = {R: discrete_J(pf, px, R, width=width, p=p) for R in (4.0, 8.0)}
            aa = {R: discrete_J(af15, ax15, R, width=width, p=p) for R in (4.0, 8.0)}
            et = (aa[8.0]["J_total"]-aa[4.0]["J_total"]) - (pa[8.0]["J_total"]-pa[4.0]["J_total"])
            ee = (aa[8.0]["J_even"]-aa[4.0]["J_even"]) - (pa[8.0]["J_even"]-pa[4.0]["J_even"])
            oo = aa[8.0]["J_odd"]-aa[4.0]["J_odd"]
            robustness.append({"scan":"shape_width","p":p,"width":width,"shift_x":0.0,"shift_y":0.0,"excess_total":et,"even":ee,"odd":oo,"closure":et-ee-oo,"odd_fraction":oo/et})
    for sx in (-0.3,-0.15,0.0,0.15,0.3):
        for sy in (-0.3,-0.15,0.0,0.15,0.3):
            pa = {R: discrete_J(pf, px, R, shift=(sx,sy)) for R in (4.0,8.0)}
            aa = {R: discrete_J(af15, ax15, R, shift=(sx,sy)) for R in (4.0,8.0)}
            et=(aa[8.0]["J_total"]-aa[4.0]["J_total"])-(pa[8.0]["J_total"]-pa[4.0]["J_total"])
            ee=(aa[8.0]["J_even"]-aa[4.0]["J_even"])-(pa[8.0]["J_even"]-pa[4.0]["J_even"])
            oo=aa[8.0]["J_odd"]-aa[4.0]["J_odd"]
            robustness.append({"scan":"center_shift","p":4.0,"width":1.5,"shift_x":sx,"shift_y":sy,"excess_total":et,"even":ee,"odd":oo,"closure":et-ee-oo,"odd_fraction":oo/et})
    write_csv(args.out / "discrete_weight_robustness.csv", robustness)

    # Chirality mirror.
    chirality_rows=[]
    for ko in (0.05,0.10,0.15,0.20):
        _,_,rf,rx,_=prepare_model_fields(nx,ny,a,ko,"right",support)
        _,_,lf,lx,_=prepare_model_fields(nx,ny,a,-ko,"left",support)
        for R in radii:
            jr=discrete_J(rf,rx,R)
            jl=discrete_J(lf,lx,R)
            chirality_rows.append({"abs_k_o":ko,"R":R,"right_total":jr["J_total"],"left_total":jl["J_total"],"abs_error":abs(jr["J_total"]-jl["J_total"])})
    write_csv(args.out / "discrete_chirality_mirror.csv",chirality_rows)

    fig, axp = plt.subplots(figsize=(6.4,4.6))
    for ko in (0.0,0.05,0.10,0.15,0.20):
        rr=[r for r in radius_rows if abs(r["k_o"]-ko)<1e-12]
        axp.plot([r["R"] for r in rr],[r["J_total"] for r in rr],"o-",label=fr"$k_o={ko:.2f}$")
    axp.set_xlabel("discrete domain radius $R$")
    axp.set_ylabel(r"$J_h[q_R]$")
    axp.grid(alpha=.25); axp.legend()
    fig.tight_layout();fig.savefig(args.out/"discrete_J_vs_radius.pdf");plt.close(fig)

    fig, axp = plt.subplots(figsize=(6.4,4.8))
    axp.plot(odd,total,"o",label="direct odd bond term")
    axp.plot(odd+even,total,"s",label="odd + even redistribution")
    lim=1.08*max(np.max(np.abs(total)),np.max(np.abs(odd+even)))
    xx=np.linspace(-lim,lim,200);axp.plot(xx,xx,"--",label="unit slope")
    axp.set_xlabel("discrete source contribution")
    axp.set_ylabel("active excess discrete domain drift")
    axp.grid(alpha=.25);axp.legend()
    fig.tight_layout();fig.savefig(args.out/"discrete_source_decomposition.pdf");plt.close(fig)

    summary={
        "grid":{"nx":nx,"ny":ny,"a":a},
        "passive_relative_spread":passive_spread,
        "odd_only_fit":{"slope":slope_odd,"r2_origin":r2_odd},
        "exact_decomposition_fit":{"slope":slope_sum,"r2_origin":r2_sum},
        "max_algebraic_closure_abs":max(abs(r["algebraic_closure"]) for r in balance_rows),
        "odd_fraction_range":[min(r["odd_fraction"] for r in balance_rows),max(r["odd_fraction"] for r in balance_rows)],
        "even_redistribution_fraction_range":[min(r["even_redistribution_fraction"] for r in balance_rows),max(r["even_redistribution_fraction"] for r in balance_rows)],
        "robustness_odd_fraction_range":[min(r["odd_fraction"] for r in robustness),max(r["odd_fraction"] for r in robustness)],
        "robustness_max_closure_abs":max(abs(r["closure"]) for r in robustness),
        "chirality_max_abs_error":max(r["abs_error"] for r in chirality_rows),
        "passive_free_residual_inf":pres,
    }
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
