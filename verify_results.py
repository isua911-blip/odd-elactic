#!/usr/bin/env python3
"""Verify the numerical claims that carry the manuscript's central conclusions."""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
# Evaluate conditions.
#
# A rule is one of
#   "<x", ">x", "<=x", ">=x", "==x"   numeric comparison against a threshold
#   "a to b"                          closed numeric interval
#   "True" / "False"                  boolean assertion
# Anything else is a specification error and is reported as such rather than
# silently skipped.
_NUMERIC_OPS = {
    "<=": lambda v, r: v <= r,
    ">=": lambda v, r: v >= r,
    "==": lambda v, r: v == r,
    "<": lambda v, r: v < r,
    ">": lambda v, r: v > r,
}


def passes(value, rule: str) -> bool:
    text = rule.strip()
    if text in ("True", "False"):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"boolean rule {rule!r} applied to non-boolean value {value!r}")
        return bool(value) is (text == "True")
    if " to " in text:
        low, high = (float(x) for x in text.split(" to ", 1))
        return low <= float(value) <= high
    for token, test in _NUMERIC_OPS.items():
        if text.startswith(token):
            return bool(test(float(value), float(text[len(token):].strip())))
    raise ValueError(f"unparsable verification rule {rule!r}")


def _format(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    return f"{float(value):.12g}"


def _self_test() -> None:
    """Unit test for the rule grammar; run with `python verify_results.py --self-test`."""
    positive = [(0.5, "< 1"), (2.0, "> 1"), (1.0, ">= 1"), (1.0, "<= 1"), (0.0, "== 0"),
                (True, "True"), (False, "False"), (0.41, "0.40 to 0.42")]
    negative = [(2.0, "< 1"), (0.5, "> 1"), (0.9, ">= 1"), (1.0, "== 0"),
                (False, "True"), (True, "False"), (0.5, "0.40 to 0.42")]
    for value, rule in positive:
        assert passes(value, rule) is True, (value, rule)
    for value, rule in negative:
        assert passes(value, rule) is False, (value, rule)
    for bad in ("approximately 1", "~1", "", "between 1 and 2"):
        try:
            passes(1.0, bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"rule {bad!r} should have been rejected")
    try:
        passes(0.5, "True")
    except ValueError:
        pass
    else:
        raise AssertionError("boolean rule on a float should have been rejected")
    # Statically confirm that every rule literal in this file is parsable,
    # without needing the datasets to be present.
    source = Path(__file__).read_text(encoding="utf-8")
    # The rule is the last quoted string on the line; a non-greedy match would
    # instead capture an inner argument such as a summary key.
    literals = sorted(set(re.findall(r'^\s*checks\.append\(\(.*,\s*"([^"]+)"\)\)\s*$',
                                     source, re.MULTILINE)))
    for rule in literals:
        passes(True if rule in ("True", "False") else 1.0, rule)
    print(f"PASS  rule grammar self-test ({len(literals)} distinct rule literals parsed)")


if "--self-test" in sys.argv:
    _self_test()
    raise SystemExit(0)

PACKAGE_REVISION = "2026-07-30-r6"

# Catch partially updated working copies: the verification suite and the module
# that produces the summaries must come from the same package revision.
sys.path.insert(0, str(ROOT / "src"))
try:
    from configurational_work_bridge import PACKAGE_REVISION as _MODULE_REVISION
except ImportError as _error:
    if "PACKAGE_REVISION" in str(_error) or "cannot import name" in str(_error):
        raise SystemExit(
            f"package revision mismatch: verify_results.py is {PACKAGE_REVISION} but "
            "src/configurational_work_bridge.py predates the revision stamp. "
            "Copy the whole package, not individual files, and re-run."
        ) from None
    raise SystemExit(
        f"could not import src/configurational_work_bridge.py: {_error}. "
        "This is an environment or installation problem, not a stale-file problem."
    ) from None
if _MODULE_REVISION != PACKAGE_REVISION:
    raise SystemExit(
        f"package revision mismatch: verify_results.py is {PACKAGE_REVISION} but "
        f"src/configurational_work_bridge.py is {_MODULE_REVISION}. "
        "Copy the whole package, not individual files, and re-run."
    )


def bridge_field(section, *keys):
    """Read a summary field, naming the required rebuild if the schema is older."""
    node = bridge_bridge_validation[section]
    for key in keys:
        try:
            node = node[key]
        except (KeyError, IndexError):
            path = ".".join(map(str, keys))
            raise SystemExit(
                f"summary field {section}.{path} is absent; the summary predates this "
                "package revision. Re-run: "
                "python recompute_configurational_work_bridge.py --only j_work_bridge"
            ) from None
    return node


D = ROOT / "data"
checks: list[tuple[str, float, str]] = []

# Continuum theory and crack-tip asymptotics.
continuum = json.loads((D / "continuum_theory_results" / "continuum_balance_verification.json").read_text())
checks.append(("continuum material-balance residual", float(continuum["max_local_balance_abs_residual"]), "< 1e-13"))

crack_tip = json.loads((D / "crack_tip_asymptotics_results" / "summary.json").read_text())
checks.append(("crack-tip leading exponent error", abs(float(crack_tip["leading_nonrigid_singular_exponent"]) - 0.5), "< 1e-12"))
checks.append(("passive K-J recovery error", float(crack_tip["passive_J_matrix_max_error"]), "< 1e-10"))
checks.append(("lambda-half traction-face residual", float(crack_tip["lambda_half_face_map_max_singular_value"]), "< 1e-10"))
checks.append(("closed-form propagator cross-check", float(crack_tip["max_closed_form_propagator_error_vs_expm"]), "< 1e-10"))
checks.append(("analytic first-order J derivative certificate", float(crack_tip["analytic_first_order_derivative_max_error"]), "< 1e-6"))
checks.append(("finite K-only stress invariance", float(crack_tip["max_finite_K_only_stress_invariance_error"]), "< 1e-10"))

lattice_fit = json.loads((D / "crack_tip_lattice_fit_results" / "summary.json").read_text())
checks.append(("active odd-basis fit residual", float(lattice_fit["ko_0p15_matched_relative_residual"]), "< 0.10"))
checks.append(("odd-basis relative residual reduction", float(lattice_fit["ko_0p15_relative_residual_reduction"]), "> 0.25"))
checks.append(("lattice-fit annulus coefficient variation", float(lattice_fit["ko_0p15_KI_annulus_coefficient_of_variation"]), "< 0.03"))
checks.append(("lattice-fit chirality mirror coefficient error", float(lattice_fit["mirror_max_abs_coefficient_difference"]), "< 1e-10"))

same_endpoint_scaling = json.loads((D / "same_endpoint_scaling_results" / "summary.json").read_text())
checks.append(("same-endpoint r=1 angle invariance", float(same_endpoint_scaling["r1_angle_invariance_abs"]), "< 1e-12"))
checks.append(("same-endpoint reciprocal rotation symmetry", float(same_endpoint_scaling["reciprocal_rotation_symmetry_abs"]), "< 1e-12"))
checks.append(("same-endpoint small-ko first-order relative error", float(same_endpoint_scaling["max_small_ko_prediction_relative_error"]), "< 0.08"))
checks.append(("same-endpoint scaling final offset", float(same_endpoint_scaling["max_final_relative_norm"]), "< 2e-6"))

# Localization refinement and size-scaled same-endpoint coefficient.
gauge_convergence = json.loads((D / "gauge_convergence_results" / "summary.json").read_text())
checks.append(("gauge-convergence target exponent lower bound", float(gauge_convergence["target_active_relative_fit_exponent"]), "> 0.8"))
checks.append(("gauge-convergence target exponent upper bound", float(gauge_convergence["target_active_relative_fit_exponent"]), "< 1.1"))
checks.append(("gauge-convergence log-fit quality", float(gauge_convergence["target_active_relative_fit_r2_log"]), "> 0.99"))
checks.append(("gauge-convergence nodal-force reproduction", float(gauge_convergence["max_force_reproduction_error"]), "< 1e-11"))

size_scaling = json.loads((D / "same_endpoint_size_scaling_results" / "summary.json").read_text())
checks.append(("size-scaled path coefficient lower bound", float(size_scaling["linear_in_a_lat_over_L_continuum_intercept"]), "> 0.65"))
checks.append(("size-scaled path coefficient upper bound", float(size_scaling["linear_in_a_lat_over_L_continuum_intercept"]), "< 0.75"))
checks.append(("size-scaled first-order relative error", float(size_scaling["max_small_ko_prediction_relative_error"]), "< 0.05"))
checks.append(("size-scaled path final offset", float(size_scaling["max_final_relative_norm"]), "< 2e-6"))


# Reviewer-requested convention, refinement, stability, trapping and uncertainty checks.
refinement_validation = json.loads((D / "refinement_validation_results" / "summary.json").read_text())
if not bool(refinement_validation["pass"]):
    raise SystemExit("FAIL  reviewer-requested revision checks did not pass")
conv = refinement_validation["convention_audit"]
checks.append(("irreducible stress-pairing audit", float(conv["max_pairing_error"]), "< 1e-12"))
checks.append(("corrected d-c PDE Fourier residual", float(conv["max_fourier_pde_error"]), "< 1e-12"))
refine = refinement_validation["crack_tip_refinement"]
checks.append(("refined active matched residual", float(refine["active_matched_residual_96"]), "< 0.08"))
checks.append(("minimum wrong-basis residual penalty", float(refine["minimum_wrong_basis_penalty"]), "> 0.03"))
stability = refinement_validation["mobility_stability"]
checks.append(("minimum tested mobility decay rate", float(stability["minimum_soft_mode_real_part"]), "> 0"))
trap = refinement_validation["passive_step_comparison"]
checks.append(("active-to-passive step oscillation amplification", float(trap["active_to_passive_extension_span_ratio"]), "> 10"))
unc = refinement_validation["gauge_exponent_uncertainty"]
checks.append(("gauge exponent confidence lower bound", float(unc["confidence_interval"][0]), "< 1"))
checks.append(("gauge exponent confidence upper bound", float(unc["confidence_interval"][1]), "> 1"))

# Second-round structural-theory and protocol audits.
continuum_validation = json.loads((D / "continuum_lattice_validation_results" / "summary.json").read_text())
if not bool(continuum_validation["pass"]):
    raise SystemExit("FAIL  second-round revision checks did not pass")
energy = continuum_validation["energetic_split"]
checks.append(("microenergetic Ao derivative error", abs(float(energy["micro_hessian_dG12_dAo"]) - float(energy["analytic_micro_hessian_dG12_dAo"])), "< 1e-7"))
checks.append(("major-projection first-order Ao coefficient", abs(float(energy["major_projection_dG12_dAo"])), "< 1e-7"))
glide = continuum_validation["glide_reflection"]
checks.append(("glide even-operator conjugacy", float(glide["operator_even_conjugacy_inf"]), "< 1e-12"))
checks.append(("glide odd-operator sign reversal", float(glide["operator_odd_sign_conjugacy_inf"]), "< 1e-12"))
checks.append(("glide protocol-work identity", float(glide["exact_glide_protocol_work_abs_error"]), "< 1e-12"))
soft = continuum_validation["quasistatic_softening"]
checks.append(("quasistatic softening interval inclusion", float(soft["maximum_interval_violation"]), "< 1e-12"))
bridge = continuum_validation["continuum_lattice_flux_bridge"]
# The second-round absolute normalization is retained only as an archival check;
# the third-round audit below uses active increments as the informative metric.
checks.append(("archival absolute continuum-lattice bridge", float(bridge["micro_hessian_max_abs_relative_error"]), "< 0.05"))
stats2 = continuum_validation["statistics"]
checks.append(("four-size path extrapolation residual", float(stats2["path_four_size_linear_fit"]["maximum_abs_residual"]), "< 1e-4"))
checks.append(("N80 path coefficient final offset", float(stats2["path_nx80"]["maximum_final_relative_norm"]), "< 1e-6"))
tail2 = continuum_validation["relaxation_tail"]
checks.append(("mobility span tail convergence", float(tail2["maximum_relative_span_change"]), "< 5e-4"))
ind = continuum_validation["independent_bond_sum"]
checks.append(("independent bond-force implementation", max(float(ind["even_force_max_abs_error"]), float(ind["odd_force_max_abs_error"]), float(ind["odd_virtual_power_abs_error"])), "< 1e-11"))

# Third-round gauge cancellation, increment bridge and theorem-scope audits.
representation_validation = json.loads((D / "representation_symmetry_results" / "summary.json").read_text())
if not bool(representation_validation["pass"]):
    raise SystemExit("FAIL  third-round revision checks did not pass")
gauge3 = representation_validation["closed_form_energy_gauge"]
checks.append(("closed-form major-symmetric gauge cancellation", abs(float(gauge3["net_major_symmetric_first_order_coefficient"])), "< 1e-12"))
bridge3 = representation_validation["increment_bridge"]
checks.append(("single passive bridge calibration factor lower bound", float(bridge3["passive_calibration_scalar"]), "> 1.2"))
checks.append(("microenergetic active-increment overprediction", float(bridge3["microenergetic_increment_relative_error_ko0p2"]), "> 0.3"))
checks.append(("KII annulus uncertainty ratio", float(bridge3["KII_ko0p2_sd_over_abs_mean"]), "> 0.5"))
glide3 = representation_validation["glide_even_part"]
checks.append(("maximum complete-period ratio at p=0.90 after theorem-scope audit", float(glide3["maximum_observed_period_ratio_p0p90"]), "< 1"))
checks.append(("period-envelope quadratic slope", float(glide3["envelope_slope_vs_ko_squared"]), "< 0"))
dir3 = representation_validation["directional_convergence"]
if bool(dir3["convergence_claim_supported"]):
    raise SystemExit("FAIL  directional chi_A was incorrectly marked converged")
state3 = representation_validation["constructive_state_criterion"]
checks.append(("state-function criterion static residual", float(state3["maximum_static_force_residual"]), "< 1e-12"))


# Fourth-round exact finite-K_o, full-bridge, all-bond and period-fit audits.
finite_modulus_validation = json.loads((D / "finite_modulus_validation_results" / "summary.json").read_text())
if not bool(finite_modulus_validation["pass"]):
    raise SystemExit("FAIL  fourth-round revision checks did not pass")
ko4 = finite_modulus_validation["exact_finite_Ko_flux"]
checks.append(("exact finite-K_o flux formula", float(ko4["maximum_flux_relative_error"]), "< 1e-12"))
bridge4 = finite_modulus_validation["finite_modulus_bridge"]
checks.append(("finite-modulus bridge first-order truncation difference in percentage points", abs(float(bridge4["first_order_minus_full_microenergetic_percentage_points"])), "< 0.5"))
checks.append(("full finite-modulus gauge separation in percentage points", float(bridge4["full_gauge_separation_percentage_points"]), "> 3"))
allbond4 = finite_modulus_validation["all_bond_arrest"]
checks.append(("all-bond audit maximum off-plane tensile ratio", float(allbond4["maximum_nonfrontier_bond_ratio"]), "> 1"))
checks.append(("all-bond audit threshold-exceeding states", float(allbond4["states_with_nonfrontier_bond_at_or_above_threshold"]), "> 0"))
period4 = finite_modulus_validation["period_even_component"]
checks.append(("period-ratio positive quadratic coefficients", float(period4["number_of_positive_b2"]), "<= 0"))
checks.append(("period-ratio fit residual", float(period4["maximum_fit_residual"]), "< 0.001"))

# Major-review matched flux/work, debonding, unrestricted branching, source and moment audits.
bridge_validation = json.loads((D / "configurational_work_bridge_results" / "summary.json").read_text())
if not bool(bridge_validation["pass"]):
    raise SystemExit("FAIL  major-review checks did not pass")
bridge_bridge_validation = bridge_validation["J_work_bridge"]
# Guard against a summary section that predates the CSV it describes.
_bridge_csv = pd.read_csv(D / "configurational_work_bridge_results" / "J_work_matched_comparison.csv")
_csv_sizes = sorted(int(v) for v in _bridge_csv.nx.unique())
_summary_sizes = sorted(int(v) for v in bridge_bridge_validation.get("sizes", []))
if _csv_sizes != _summary_sizes:
    raise SystemExit(
        "stale bridge summary: summary.json lists sizes "
        f"{_summary_sizes} but J_work_matched_comparison.csv contains {_csv_sizes}. "
        "Re-run: python recompute_configurational_work_bridge.py --only j_work_bridge"
    )
checks.append(("bridge sizes agree between summary and CSV", _csv_sizes == _summary_sizes, "True"))
checks.append(("number of bridge sizes", float(len(_csv_sizes)), ">= 3"))

_finest = str(max(_csv_sizes))
checks.append((f"passive J/work mismatch at finest tested size (Nx={_finest})", float(bridge_field("passive_relative_abs_J_vs_abrupt_by_size", _finest)), "< 0.017"))
checks.append((f"active J/work equivalence break at finest tested size (Nx={_finest})", float(bridge_field("active_relative_abs_J_vs_abrupt_by_size", _finest)), "> 0.40"))
checks.append(("passive J/work mismatch continuum intercept (percent)", float(bridge_field("passive_continuum_extrapolation", "model_range_percent", 1)), "< 0.30"))
checks.append(("active J/work mismatch continuum intercept (percent)", float(bridge_field("active_continuum_extrapolation", "model_range_percent", 0)), "> 38"))
checks.append(("Rice-Eshelby gap against endpoint release (percent)", float(bridge_field("finest_size_decomposition", "J_to_endpoint_release_percent")), "10 to 20"))
soft_bridge_validation = bridge_validation["two_parameter_softening"]
checks.append(("minimum quasistatic debonding-path spread", min(float(x["relative_span_over_mean"]) for x in soft_bridge_validation["cases"]), "> 0.05"))
cascade_bridge_validation = bridge_validation["unrestricted_cascade"]
checks.append(("unrestricted continuing cascades", float(cascade_bridge_validation["states_reaching_cutoff"]), "> 90"))
checks.append(("passive states reaching the cascade cutoff", float(cascade_bridge_validation["passive_states_reaching_cutoff"]), "== 0"))
checks.append(("continuing cascades leaving the cleavage plane at deletion two", float(cascade_bridge_validation["continuing_second_deletion_off_plane"]), "> 90"))
minimum_continuing = cascade_bridge_validation["minimum_continuing_k_o_by_load"]
checks.append(("smallest continuing odd modulus is monotone in load", float(minimum_continuing["0.85"]) >= float(minimum_continuing["0.9"]) >= float(minimum_continuing["0.95"]), "True"))
intact_control = bridge_validation["intact_lattice_inertial_control"]
checks.append(("intact lattice overdamped-stable for every tested odd modulus", bool(intact_control["overdamped_stable_for_all_tested_k_o"]), "True"))
checks.append(("intact-lattice critical inertial damping at the probe modulus", float(intact_control["critical_damping_at_probe_modulus"]), "0.40 to 0.42"))
checks.append(("unrestricted continuing cascades with off-plane breaks", float(cascade_bridge_validation["states_with_off_plane_break"]), "> 90"))
criterion_bridge_validation = bridge_validation["criterion_sensitivity"]
checks.append(("cascade classification changes under a total-force criterion", float(criterion_bridge_validation["states_with_changed_classification"]), "== 0"))
checks.append(("maximum failure-threshold rescaling across odd moduli", float(criterion_bridge_validation["maximum_threshold_rescaling"]), "< 0.05"))
checks.append(("localization/contour half-span of the bridge flux", float(bridge_bridge_validation["maximum_J_h_relative_half_span"]), "< 0.02"))
source_bridge_validation = bridge_validation["source_refinement"]
checks.append(("source closure error at finest size", float(source_bridge_validation["absolute_error_fine"]), "< 0.035"))
moment_bridge_validation = bridge_validation["angular_momentum"]
checks.append(("finite-strip angular momentum residual", abs(float(moment_bridge_validation["relative_moment_balance_residual"])), "< 1e-12"))
stability_bridge_validation = bridge_validation["perfect_lattice_stability"]
checks.append(("intact finite-strip stability at ko/k=0.30", float(stability_bridge_validation["minimum_real_part_at_0p30"]), "> 0"))
inertial_bridge_validation = bridge_validation["inertial_probe"]
if not any((not bool(x["linearly_stable"])) and bool(x["threshold_crossed"]) for x in inertial_bridge_validation["cases"]):
    raise SystemExit("FAIL  inertial sensitivity probe did not detect the archived weak-damping instability")

# Cycle-work agreement.
cycle = pd.read_csv(D / "baseline_results" / "cycle_scan_ko.csv")
cycle_rel = float(np.max(np.abs(cycle.work_total_density - cycle.analytic_density)) / np.max(np.abs(cycle.analytic_density)))
checks.append(("cycle analytic relative error", cycle_rel, "< 1e-6"))

# Passive contour and discrete gauge metrics from archived summaries.
s1 = json.loads((D / "baseline_results" / "summary.json").read_text())
checks.append(("passive force residual", float(s1["passive_max_free_residual_inf"]), "< 1e-12"))

s9 = json.loads((D / "discrete_configurational_results" / "summary.json").read_text())
checks.append(("discrete nodal-force reconstruction", float(s9["interior_nodal_force_reproduction_max_abs_error"]), "< 1e-12"))
checks.append(("discrete algebraic closure", float(s9["max_algebraic_closure_abs"]), "< 1e-12"))

# Same-endpoint path dependence.
m = pd.read_csv(D / "protocol_family_results" / "same_endpoint_mobility_protocol_normalized.csv")
for ko, target in [(0.0, 1e-8), (0.12, 0.05), (0.222271, 0.09)]:
    d = m[np.isclose(m.k_o, ko)]
    span = float(d.protocol_work_over_isotropic.max() - d.protocol_work_over_isotropic.min())
    if ko == 0.0:
        checks.append(("passive same-endpoint work span", span, "< 1e-8"))
    else:
        checks.append((f"active same-endpoint work span at ko={ko}", span, f"> {target}"))

# Protocol tracking at independently determined threshold.
t = pd.read_csv(D / "protocol_family_results" / "protocol_family_thresholds.csv")
max_fixed = float(np.max(np.abs(t.fixed_grip_work_ratio_at_bond_threshold - 1.0)))
max_traction = float(np.max(np.abs(t.matched_traction_work_ratio_at_bond_threshold - 1.0)))
checks.append(("max fixed-grip work deviation at bond threshold", max_fixed, "< 0.034"))
checks.append(("max matched-traction work deviation at bond threshold", max_traction, "< 0.038"))

# Stepwise resistance, two-period alternation, and arrest robustness.
advance = json.loads((D / "advance_resistance_results" / "summary.json").read_text())
checks.append(("maximum cleavage-constrained below-initiation broken bonds over 224 states", float(advance["maximum_broken_bonds_subgriffith"]), "<= 1"))
checks.append(("number of sustained cleavage-constrained below-initiation states", float(advance["number_sustained_subgriffith_states_ge_2"]), "<= 0"))
checks.append(("maximum complete-period work ratio at p=0.90", float(advance["maximum_period_work_ratio_at_p0p90"]), "< 1"))
checks.append(("maximum effective resistance increase", float(advance["maximum_effective_resistance_increase_fraction"]), "< 0.008"))
checks.append(("maximum broad-scan work-balance residual", float(advance["maximum_abs_scaled_balance_residual"]), "< 2.5e-7"))
checks.append(("maximum baseline-to-fine protocol change", float(advance["maximum_baseline_to_fine_protocol_work_relative_change"]), "< 5e-5"))
checks.append(("passive dead-load positive-control breaks", float(advance["positive_control_dead_load_minimum_breaks"]), ">= 9"))
nominal = advance["nominal_fixed_p0p90_ko0p20"]
checks.append(("nominal first-step work ratio", float(nominal[0]["work_ratio"]), "> 1"))
checks.append(("nominal second-step work ratio", float(nominal[1]["work_ratio"]), "< 0.5"))
if not bool(advance["odd_work_sign_alternation_all_positive_ko"]):
    raise SystemExit("FAIL  odd-work sign alternation was not satisfied")

# Virtual-direction comparison and localization-robust directional excess.
directional = json.loads((D / "directional_driving_results" / "summary.json").read_text())
checks.append(("directional maximum work-balance residual", float(directional["maximum_abs_work_balance_residual"]), "< 4e-7"))
checks.append(("directional final offset", float(directional["maximum_final_state_relative_norm"]), "< 3e-4"))
checks.append(("passive lattice-registry directional bias", float(directional["maximum_passive_registry_kink_bias_over_straight"]), "< 0.03"))
checks.append(("odd directional reversal relative error", float(directional["maximum_odd_excess_work_reversal_relative_error"]), "< 0.03"))
checks.append(("right-left directional reflection error", float(directional["maximum_tip_reflection_relative_error"]), "< 1e-10"))
checks.append(("work/configurational directional sign agreement", float(directional["work_and_configurational_bias_sign_agreement_fraction"]), ">= 1"))
size_bias = directional["right_tip_ko0p20_normalized_odd_excess_bias_by_size"]
checks.append(("minimum three-size normalized directional excess", min(float(v) for v in size_bias.values()), "> 0.44"))
if bool(directional["short_kink_gate_triggered"]):
    raise SystemExit("FAIL  short-kink gate was unexpectedly triggered")
if directional["total_preferred_work_directions_active"] != [0]:
    raise SystemExit("FAIL  straight direction was not preferred in every active case")

ok = True
for name, value, rule in checks:
    try:
        good = passes(value, rule)
    except ValueError as error:
        ok = False
        print(f"ERROR {name}: {error}")
        continue
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name}: {_format(value)}  ({rule})")

if not ok:
    raise SystemExit(1)
