"""Optional in-process BNGL -> .net simulation using bngsim.

Facade for the simulator-interface package (the CQ-1b split of the former
2.7k-line bngsim_model.py god-file, #408). The implementation lives in the
submodules -- parsing, expressions, scan, output (all bngsim-free and
CI-testable), classification, net_model (BngsimModel), and nf_model
(BngsimNfModel) -- and this module only re-exports the public surface so
``pybnf.bngsim_model.<name>`` keeps resolving for ``pybnf.algorithms.base`` and
the tests. The bngsim capability flags are read through the ``_runtime`` seam
(ADR-0018); tests patch ``pybnf.bngsim_model._runtime``.
"""


from . import _runtime as _runtime
# Re-export the bngsim capability flags so ``pybnf.bngsim_model.<flag>`` keeps
# resolving for value-importers (``pybnf.algorithms.base``) and read-only test
# access. Production code in this package reads them as ``_runtime.<flag>`` so a
# patch on the _runtime seam (ADR-0018) bites every reader; these bindings are
# snapshots for re-export only, never read by this package's own logic.
from ._runtime import (
    bngsim as bngsim,
    BNGSIM_AVAILABLE as BNGSIM_AVAILABLE,
    BNGSIM_ERROR as BNGSIM_ERROR,
    BNGSIM_HAS_NFSIM as BNGSIM_HAS_NFSIM,
    BNGSIM_HAS_RULEMONKEY as BNGSIM_HAS_RULEMONKEY,
    BNGSIM_VERSION as BNGSIM_VERSION,
)
# Pure BNGL action-line parsing lives in parsing.py (CI-testable, bngsim-free).
# Re-exported here so the model classes / classification resolve the bare names
# and the package facade (and tests) keep resolving pybnf.bngsim_model.<name>.
from .parsing import (
    _collapse_action_line_continuations as _collapse_action_line_continuations,
    _extract_action_body as _extract_action_body,
    _split_top_level_commas as _split_top_level_commas,
    _parse_action_value as _parse_action_value,
    _parse_action_dict as _parse_action_dict,
    _parse_simulate_action as _parse_simulate_action,
    _parse_parameter_scan_action as _parse_parameter_scan_action,
    _parse_bifurcate_action as _parse_bifurcate_action,
    _parse_set_parameter as _parse_set_parameter,
    _parse_set_concentration as _parse_set_concentration,
    _parse_set_concentration_expr as _parse_set_concentration_expr,
    _parse_set_concentration_nf as _parse_set_concentration_nf,
    _parse_add_concentration as _parse_add_concentration,
    _is_reset_concentrations as _is_reset_concentrations,
    _is_reset_parameters as _is_reset_parameters,
    _is_save_concentrations as _is_save_concentrations,
    _is_save_parameters as _is_save_parameters,
)
# Pure expression / parameter evaluation lives in expressions.py (CI-testable,
# bngsim-free). Re-exported here so the model classes resolve the bare names and
# the facade (and tests) keep resolving pybnf.bngsim_model.<name>.
from .expressions import (
    _build_safe_eval_namespace as _build_safe_eval_namespace,
    _eval_numeric as _eval_numeric,
    _eval_model_expression as _eval_model_expression,
    _model_param_values as _model_param_values,
    _build_mutant_param_set as _build_mutant_param_set,
    _parse_bngl_param_block as _parse_bngl_param_block,
    _evaluate_bngl_params as _evaluate_bngl_params,
    _parse_net_species_initializers as _parse_net_species_initializers,
)
# Scan-point/sample-time resolution + steady-state constants live in scan.py;
# on-disk action-output writing in output.py (both numpy-only, bngsim-free).
# Re-exported so the model classes / classification resolve the bare names.
from .scan import (
    _with_sim_timeout as _with_sim_timeout,
    _resolve_scan_points as _resolve_scan_points,
    _resolve_sample_times as _resolve_sample_times,
    _SS_METHOD_PARITY as _SS_METHOD_PARITY,
    _SS_METHOD_NEWTON as _SS_METHOD_NEWTON,
    _SS_SCAN_N_POINTS as _SS_SCAN_N_POINTS,
)
from .output import (
    _ext_for_simtype as _ext_for_simtype,
    _write_saved_action_outputs as _write_saved_action_outputs,
)
# Backend classification, method/timeout normalization, and NF session
# management live in classification.py (reads the _runtime flag seam).
# Re-exported so the model classes resolve the bare names and the facade keeps
# resolving pybnf.bngsim_model.<name> for algorithms.base and the tests.
from .classification import (
    BNGSIM_BACKEND_NET as BNGSIM_BACKEND_NET,
    BNGSIM_BACKEND_NF as BNGSIM_BACKEND_NF,
    BNGSIM_BACKEND_HYBRID as BNGSIM_BACKEND_HYBRID,
    BNGSIM_NF_BACKEND_NFSIM as BNGSIM_NF_BACKEND_NFSIM,
    BNGSIM_NF_BACKEND_RULEMONKEY as BNGSIM_NF_BACKEND_RULEMONKEY,
    _BNGSIM_ACTION_BACKENDS as _BNGSIM_ACTION_BACKENDS,
    _BNGSIM_NF_CANONICAL_METHODS as _BNGSIM_NF_CANONICAL_METHODS,
    _normalize_ss_method as _normalize_ss_method,
    _coerce_positive_timeout as _coerce_positive_timeout,
    _normalize_sim_timeout as _normalize_sim_timeout,
    _normalize_session_timeout as _normalize_session_timeout,
    _normalize_action_method as _normalize_action_method,
    _bngsim_normalize_method as _bngsim_normalize_method,
    _normalize_nf_action_method as _normalize_nf_action_method,
    _nf_session_backend_for_method as _nf_session_backend_for_method,
    _bngsim_has_nf_session_backend as _bngsim_has_nf_session_backend,
    _nf_session_backend_label as _nf_session_backend_label,
    _nf_method_from_action_params as _nf_method_from_action_params,
    _required_nf_session_backends as _required_nf_session_backends,
    _first_nf_action_method as _first_nf_action_method,
    missing_bngsim_nf_action_support as missing_bngsim_nf_action_support,
    _get_nf_session_class as _get_nf_session_class,
    _create_nf_session as _create_nf_session,
    _destroy_nf_session as _destroy_nf_session,
    _classify_action_method_backend as _classify_action_method_backend,
    _allowed_bngsim_backends_for_action as _allowed_bngsim_backends_for_action,
    classify_actions_for_bngsim as classify_actions_for_bngsim,
    actions_compatible_with_bngsim as actions_compatible_with_bngsim,
)
# The net-backend model class lives in net_model.py. Re-exported so
# pybnf.algorithms.base, the tests, and BngsimNfModel (which delegates its
# result conversion to BngsimModel) keep resolving pybnf.bngsim_model.<name>.
from .net_model import (
    BngsimModel as BngsimModel,
    _try_prepare_codegen as _try_prepare_codegen,
)
# The network-free model class lives in nf_model.py. Re-exported so
# pybnf.algorithms.base and the tests keep resolving pybnf.bngsim_model.<name>.
from .nf_model import BngsimNfModel as BngsimNfModel