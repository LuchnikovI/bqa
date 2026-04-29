"""
Property-based tests for desugar_config and its helpers using Hypothesis.

These tests are written against the FIXED version of the module, i.e. where
desug_or_warn_and_set_default_if_not_present returns the result of desug_fn.
They will deliberately FAIL on the buggy version (missing `return`), making
them useful as a regression suite.

Run with:
    pytest test_desugar_config.py -v
    pytest test_desugar_config.py -v --hypothesis-seed=0
"""

import copy
import sys
import types
from math import isclose

import pytest
from hypothesis import assume, given, settings, HealthCheck
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Minimal stubs so the module imports without the real bqa package
# ---------------------------------------------------------------------------

bqa_pkg = types.ModuleType("bqa")
bqa_config_pkg = types.ModuleType("bqa.config")
bqa_validate = types.ModuleType("bqa.config.validate_config")

# All key constants the desugar module imports
for _k in [
    "ACTIONS_KEY", "CLUSTER_COUPLING_AMPLITUDE_KEY", "EPS_KEY",
    "FINAL_MIXING_KEY", "GET_BLOCH_VECTORS", "INITIAL_MIXING_KEY",
    "MEASURE", "NODES_KEY", "EDGES_KEY", "MAX_BOND_DIM_KEY",
    "MAX_BP_ITER_NUMBER_KEY", "BP_EPS_KEY", "PINV_EPS_KEY", "BACKEND_KEY",
    "DEFAULT_FIELD_KEY", "MEASUREMENT_THRESHOLD_KEY", "SCHEDULE_KEY",
    "SEED_KEY", "DAMPING_KEY", "SPARSIFICATION_KEY", "STARTING_MIXING_KEY",
    "STEPS_NUMBER_KEY", "TOTAL_TIME_KEY", "WEIGHT_KEY",
]:
    setattr(bqa_validate, _k, _k.lower())

from bqa.config.desugar_config import (  # noqa: E402 — after stub setup
    desugar_config,
    desug_edges,
    desug_schedule,
    desug_sparsification,
    desug_actions,
    canonicalize_edge_id,
    desug_or_warn_and_set_default_if_not_present,
    DEFAULT_MAX_BOND_DIM, DEFAULT_MAX_BP_ITERS_NUMBER, DEFAULT_BP_EPS,
    DEFAULT_PINV_EPS, DEFAULT_BACKEND, DEFAULT_DEFAULT_FIELD,
    DEFAULT_MEASUREMENT_THRESHOLD, DEFAULT_SEED, DEFAULT_DAMPING,
    DEFAULT_STARTING_MIXING, DEFAULT_TOTAL_TIME, DEFAULT_STEPS_NUMBER,
    DEFAULT_WEIGHT, DEFAULT_FINAL_MIXING, DEFAULT_EPS,
    DEFAULT_CLUSTER_COUPLING_AMPLITUDE, DEFAULT_SPARSIFICATION,
    ACTIONS_KEY, CLUSTER_COUPLING_AMPLITUDE_KEY, EPS_KEY, FINAL_MIXING_KEY,
    GET_BLOCH_VECTORS, INITIAL_MIXING_KEY, MEASURE, NODES_KEY, EDGES_KEY,
    MAX_BOND_DIM_KEY, MAX_BP_ITER_NUMBER_KEY, BP_EPS_KEY, PINV_EPS_KEY,
    BACKEND_KEY, DEFAULT_FIELD_KEY, MEASUREMENT_THRESHOLD_KEY, SCHEDULE_KEY,
    SEED_KEY, DAMPING_KEY, SPARSIFICATION_KEY, STARTING_MIXING_KEY,
    STEPS_NUMBER_KEY, TOTAL_TIME_KEY, WEIGHT_KEY,
)

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

finite_floats = st.floats(allow_nan=False, allow_infinity=False, allow_subnormal=False)
zero_to_one = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
positive_floats = st.floats(min_value=1e-9, max_value=1e9, allow_nan=False)
positive_ints = st.integers(min_value=1, max_value=10_000)
non_neg_ints = st.integers(min_value=0, max_value=10_000)
node_ids = st.integers(min_value=0, max_value=50)
VALID_BACKENDS = ["numpy", "torch"]


@st.composite
def valid_edge_list(draw):
    """Unique undirected edges, no self-loops."""
    ids = draw(st.lists(node_ids, min_size=2, max_size=20, unique=True))
    seen = set()
    edges = []
    for _ in range(draw(st.integers(min_value=1, max_value=10))):
        a = draw(st.sampled_from(ids))
        b = draw(st.sampled_from(ids))
        assume(a != b)
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            edges.append(((a, b), draw(finite_floats)))
    assume(edges)
    return edges


@st.composite
def valid_action_dict(draw, weight=None):
    """A single QA action dict with required weight."""
    action = {WEIGHT_KEY: draw(zero_to_one) if weight is None else weight}
    if draw(st.booleans()):
        action[STEPS_NUMBER_KEY] = draw(positive_ints)
    if draw(st.booleans()):
        action[INITIAL_MIXING_KEY] = draw(zero_to_one)
    if draw(st.booleans()):
        action[FINAL_MIXING_KEY] = draw(zero_to_one)
    return action


@st.composite
def valid_actions(draw):
    """Actions list whose QA action weights sum to 1.0."""
    n = draw(st.integers(min_value=1, max_value=4))
    raw_weights = draw(st.lists(
        st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
        min_size=n, max_size=n,
    ))
    total = sum(raw_weights)
    assume(total > 0)
    weights = [w / total for w in raw_weights]

    actions = []
    for w in weights:
        action = draw(valid_action_dict(weight=w))
        actions.append(action)

    # Optionally mix in some string actions
    prefix = draw(st.lists(st.sampled_from([MEASURE, GET_BLOCH_VECTORS]), max_size=2))
    return prefix + actions


@st.composite
def valid_schedule(draw):
    sch = {}
    if draw(st.booleans()):
        sch[TOTAL_TIME_KEY] = draw(positive_floats)
    if draw(st.booleans()):
        sch[STARTING_MIXING_KEY] = draw(zero_to_one)
    if draw(st.booleans()):
        sch[ACTIONS_KEY] = draw(valid_actions())
    return sch


@st.composite
def valid_sparsification(draw):
    sp = {}
    if draw(st.booleans()):
        sp[EPS_KEY] = draw(zero_to_one)
    if draw(st.booleans()):
        sp[CLUSTER_COUPLING_AMPLITUDE_KEY] = draw(
            st.floats(min_value=1.0, max_value=1e6, allow_nan=False)
        )
    return sp


@st.composite
def valid_raw_config(draw):
    """Minimal-to-full raw config as a user would write it."""
    cfg = {EDGES_KEY: draw(valid_edge_list())}
    if draw(st.booleans()):
        cfg[NODES_KEY] = {draw(node_ids): draw(finite_floats)}
    if draw(st.booleans()):
        cfg[DEFAULT_FIELD_KEY] = draw(finite_floats)
    if draw(st.booleans()):
        cfg[SCHEDULE_KEY] = draw(valid_schedule())
    if draw(st.booleans()):
        cfg[MAX_BOND_DIM_KEY] = draw(positive_ints)
    if draw(st.booleans()):
        cfg[MAX_BP_ITER_NUMBER_KEY] = draw(non_neg_ints)
    if draw(st.booleans()):
        cfg[SEED_KEY] = draw(non_neg_ints)
    if draw(st.booleans()):
        cfg[BP_EPS_KEY] = draw(zero_to_one)
    if draw(st.booleans()):
        cfg[PINV_EPS_KEY] = draw(zero_to_one)
    if draw(st.booleans()):
        cfg[MEASUREMENT_THRESHOLD_KEY] = draw(
            st.floats(min_value=0.5, max_value=1.0, allow_nan=False)
        )
    if draw(st.booleans()):
        cfg[DAMPING_KEY] = draw(zero_to_one)
    if draw(st.booleans()):
        cfg[BACKEND_KEY] = draw(st.sampled_from(VALID_BACKENDS))
    if draw(st.booleans()):
        cfg[SPARSIFICATION_KEY] = draw(valid_sparsification())
    return cfg


# ---------------------------------------------------------------------------
# 1. canonicalize_edge_id
# ---------------------------------------------------------------------------

class TestCanonicalizeEdgeId:

    @given(a=node_ids, b=node_ids)
    def test_canonical_is_sorted(self, a, b):
        assume(a != b)
        i, j = canonicalize_edge_id((a, b))
        assert i < j

    @given(a=node_ids, b=node_ids)
    def test_both_orientations_give_same_result(self, a, b):
        assume(a != b)
        assert canonicalize_edge_id((a, b)) == canonicalize_edge_id((b, a))

    @given(a=node_ids, b=node_ids)
    def test_idempotent(self, a, b):
        assume(a != b)
        c = canonicalize_edge_id((a, b))
        assert canonicalize_edge_id(c) == c

    @given(a=node_ids, b=node_ids)
    def test_contains_original_ids(self, a, b):
        assume(a != b)
        c = canonicalize_edge_id((a, b))
        assert set(c) == {a, b}


# ---------------------------------------------------------------------------
# 2. desug_or_warn_and_set_default_if_not_present
# ---------------------------------------------------------------------------

class TestDesugOrWarnAndSetDefault:

    @given(value=finite_floats)
    def test_returns_raw_value_when_no_desug_fn(self, value):
        data = {"k": value}
        result = desug_or_warn_and_set_default_if_not_present(data, "k", 0.0)
        assert result == value

    @given(value=finite_floats)
    def test_returns_default_when_key_absent(self, value):
        result = desug_or_warn_and_set_default_if_not_present({}, "missing", value)
        assert result == value

    @given(value=st.integers())
    def test_desug_fn_result_is_returned(self, value):
        """
        REGRESSION TEST: This test catches the missing `return` bug.
        If desug_fn result is not returned, this will get None instead of float(value).
        """
        data = {"k": value}
        result = desug_or_warn_and_set_default_if_not_present(data, "k", 0.0, float)
        assert result == float(value)
        assert isinstance(result, float)

    @given(value=finite_floats)
    def test_desug_fn_applied_to_value_not_default(self, value):
        """The desug_fn should receive the actual value, not the default."""
        calls = []
        def capture(v):
            calls.append(v)
            return v
        data = {"k": value}
        desug_or_warn_and_set_default_if_not_present(data, "k", 999.0, capture)
        assert calls == [value]

    def test_default_returned_not_processed_through_desug_fn(self):
        """When key is absent, default is returned as-is without calling desug_fn."""
        calls = []
        def capture(v):
            calls.append(v)
            return v
        result = desug_or_warn_and_set_default_if_not_present({}, "missing", 42, capture)
        assert result == 42
        assert calls == []  # desug_fn must NOT be called on the default


# ---------------------------------------------------------------------------
# 3. desug_edges
# ---------------------------------------------------------------------------

class TestDesugEdges:

    @given(edges=valid_edge_list())
    def test_output_is_dict(self, edges):
        result = desug_edges(edges)
        assert isinstance(result, dict)

    @given(edges=valid_edge_list())
    def test_all_keys_are_canonical(self, edges):
        result = desug_edges(edges)
        for i, j in result:
            assert i < j

    @given(edges=valid_edge_list())
    def test_coupling_values_preserved(self, edges):
        result = desug_edges(edges)
        for edge_id, cpl in edges:
            canonical = canonicalize_edge_id(edge_id)
            assert canonical in result
            assert result[canonical] == cpl

    @given(edges=valid_edge_list())
    def test_edge_count_preserved(self, edges):
        """
        Since input edges are unique (per strategy), output count must match.
        Both orientations of the same undirected edge map to one canonical key.
        """
        input_canonical = {canonicalize_edge_id(eid) for eid, _ in edges}
        result = desug_edges(edges)
        assert len(result) == len(input_canonical)

    @given(edges=valid_edge_list())
    def test_dict_input_equivalent_to_list_input(self, edges):
        as_list = desug_edges(edges)
        as_dict = desug_edges(dict(edges))
        assert as_list == as_dict

    @given(a=node_ids, b=node_ids, cpl=finite_floats)
    def test_reverse_orientation_gives_same_canonical_key(self, a, b, cpl):
        assume(a != b)
        r1 = desug_edges([((a, b), cpl)])
        r2 = desug_edges([((b, a), cpl)])
        assert r1 == r2


# ---------------------------------------------------------------------------
# 4. desug_sparsification
# ---------------------------------------------------------------------------

class TestDesugSparsification:

    def test_none_returns_none(self):
        assert desug_sparsification(None) is None

    @given(sp=valid_sparsification())
    def test_output_contains_both_keys(self, sp):
        """
        REGRESSION TEST: catches missing `return` in desug_or_warn_and_set_default_if_not_present.
        If the bug is present, both values will be None.
        """
        result = desug_sparsification(sp)
        assert result is not None
        assert EPS_KEY in result
        assert CLUSTER_COUPLING_AMPLITUDE_KEY in result

    @given(sp=valid_sparsification())
    def test_eps_is_float(self, sp):
        result = desug_sparsification(sp)
        assert isinstance(result[EPS_KEY], float)

    @given(sp=valid_sparsification())
    def test_cluster_coupling_is_float(self, sp):
        result = desug_sparsification(sp)
        assert isinstance(result[CLUSTER_COUPLING_AMPLITUDE_KEY], float)

    @given(eps=zero_to_one)
    def test_eps_value_preserved(self, eps):
        result = desug_sparsification({EPS_KEY: eps})
        assert isclose(result[EPS_KEY], eps)

    @given(amp=st.floats(min_value=1.0, max_value=1e6, allow_nan=False))
    def test_cluster_coupling_value_preserved(self, amp):
        result = desug_sparsification({CLUSTER_COUPLING_AMPLITUDE_KEY: amp})
        assert isclose(result[CLUSTER_COUPLING_AMPLITUDE_KEY], amp)

    def test_missing_eps_uses_default(self):
        result = desug_sparsification({})
        assert isclose(result[EPS_KEY], DEFAULT_EPS)

    def test_missing_cluster_coupling_uses_default(self):
        result = desug_sparsification({})
        assert isclose(result[CLUSTER_COUPLING_AMPLITUDE_KEY], DEFAULT_CLUSTER_COUPLING_AMPLITUDE)


# ---------------------------------------------------------------------------
# 5. desug_actions — mixing continuity invariant
# ---------------------------------------------------------------------------

class TestDesugActions:

    @given(actions=valid_actions(), starting=zero_to_one)
    def test_output_length_matches_input(self, actions, starting):
        result = desug_actions(actions, starting)
        assert len(result) == len(actions)

    @given(actions=valid_actions(), starting=zero_to_one)
    def test_string_actions_pass_through_unchanged(self, actions, starting):
        result = desug_actions(actions, starting)
        for orig, out in zip(actions, result):
            if isinstance(orig, str):
                assert out == orig

    @given(actions=valid_actions(), starting=zero_to_one)
    def test_dict_actions_have_all_required_keys(self, actions, starting):
        result = desug_actions(actions, starting)
        for out in result:
            if isinstance(out, dict):
                assert STEPS_NUMBER_KEY in out
                assert WEIGHT_KEY in out
                assert INITIAL_MIXING_KEY in out
                assert FINAL_MIXING_KEY in out

    @given(actions=valid_actions(), starting=zero_to_one)
    def test_weight_values_preserved_as_float(self, actions, starting):
        result = desug_actions(actions, starting)
        for orig, out in zip(actions, result):
            if isinstance(orig, dict):
                assert isinstance(out[WEIGHT_KEY], float)
                assert isclose(out[WEIGHT_KEY], float(orig[WEIGHT_KEY]))

    @given(actions=valid_actions(), starting=zero_to_one)
    def test_mixing_continuity(self, actions, starting):
        """
        The initial_mixing of each action (when not explicitly set) must equal
        the final_mixing of the previous action. This is the chaining invariant.
        """
        result = desug_actions(actions, starting)
        prev_final = starting
        for orig, out in zip(actions, result):
            if isinstance(out, dict):
                if INITIAL_MIXING_KEY not in orig:
                    assert isclose(out[INITIAL_MIXING_KEY], prev_final), (
                        f"Expected initial_mixing={prev_final}, got {out[INITIAL_MIXING_KEY]}"
                    )
                prev_final = out[FINAL_MIXING_KEY]

    @given(actions=valid_actions(), starting=zero_to_one)
    def test_explicit_initial_mixing_overrides_chain(self, actions, starting):
        """Explicitly set initial_mixing must not be overridden by chaining."""
        result = desug_actions(actions, starting)
        for orig, out in zip(actions, result):
            if isinstance(orig, dict) and INITIAL_MIXING_KEY in orig:
                assert isclose(out[INITIAL_MIXING_KEY], float(orig[INITIAL_MIXING_KEY]))

    @given(actions=valid_actions(), starting=zero_to_one)
    def test_steps_number_default_applied(self, actions, starting):
        result = desug_actions(actions, starting)
        for orig, out in zip(actions, result):
            if isinstance(orig, dict) and STEPS_NUMBER_KEY not in orig:
                assert out[STEPS_NUMBER_KEY] == DEFAULT_STEPS_NUMBER


# ---------------------------------------------------------------------------
# 6. desug_schedule
# ---------------------------------------------------------------------------

class TestDesugSchedule:

    @given(sch=valid_schedule())
    def test_output_contains_all_keys(self, sch):
        result = desug_schedule(sch)
        assert TOTAL_TIME_KEY in result
        assert STARTING_MIXING_KEY in result
        assert ACTIONS_KEY in result

    @given(sch=valid_schedule())
    def test_total_time_is_float(self, sch):
        """REGRESSION: catches missing return for float coercion."""
        result = desug_schedule(sch)
        assert isinstance(result[TOTAL_TIME_KEY], float)

    @given(sch=valid_schedule())
    def test_starting_mixing_is_float(self, sch):
        result = desug_schedule(sch)
        assert isinstance(result[STARTING_MIXING_KEY], float)

    @given(t=positive_floats)
    def test_explicit_total_time_preserved(self, t):
        result = desug_schedule({TOTAL_TIME_KEY: t})
        assert isclose(result[TOTAL_TIME_KEY], t)

    def test_missing_total_time_uses_default(self):
        result = desug_schedule({})
        assert isclose(result[TOTAL_TIME_KEY], DEFAULT_TOTAL_TIME)

    def test_missing_starting_mixing_uses_default(self):
        result = desug_schedule({})
        assert isclose(result[STARTING_MIXING_KEY], DEFAULT_STARTING_MIXING)

    @given(sch=valid_schedule())
    def test_actions_is_a_list(self, sch):
        result = desug_schedule(sch)
        assert isinstance(result[ACTIONS_KEY], list)


# ---------------------------------------------------------------------------
# 7. desugar_config — top-level invariants
# ---------------------------------------------------------------------------

class TestDesugConfig:

    @given(cfg=valid_raw_config())
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_output_contains_all_top_level_keys(self, cfg):
        result = desugar_config(cfg)
        for key in [
            NODES_KEY, EDGES_KEY, MAX_BOND_DIM_KEY, MAX_BP_ITER_NUMBER_KEY,
            BP_EPS_KEY, PINV_EPS_KEY, BACKEND_KEY, DEFAULT_FIELD_KEY,
            MEASUREMENT_THRESHOLD_KEY, SEED_KEY, DAMPING_KEY,
            SCHEDULE_KEY, SPARSIFICATION_KEY,
        ]:
            assert key in result, f"Missing key: {key}"

    @given(cfg=valid_raw_config())
    def test_edges_are_all_canonical(self, cfg):
        result = desugar_config(cfg)
        for i, j in result[EDGES_KEY]:
            assert i < j, f"Non-canonical edge ({i}, {j}) in output"

    @given(cfg=valid_raw_config())
    def test_no_none_values_for_numeric_fields(self, cfg):
        """
        REGRESSION TEST: catches missing `return` bug.
        With the bug, every field processed through desug_fn returns None.
        """
        result = desugar_config(cfg)
        numeric_keys = [
            BP_EPS_KEY, PINV_EPS_KEY, DEFAULT_FIELD_KEY,
            MEASUREMENT_THRESHOLD_KEY, DAMPING_KEY,
        ]
        for key in numeric_keys:
            assert result[key] is not None, f"Key `{key}` is None — missing return bug?"
            assert isinstance(result[key], float), (
                f"Key `{key}` should be float, got {type(result[key])}"
            )

    @given(cfg=valid_raw_config())
    def test_schedule_is_fully_populated_dict(self, cfg):
        result = desugar_config(cfg)
        sch = result[SCHEDULE_KEY]
        assert isinstance(sch, dict)
        assert TOTAL_TIME_KEY in sch
        assert STARTING_MIXING_KEY in sch
        assert ACTIONS_KEY in sch

    @given(edges=valid_edge_list())
    def test_minimal_config_desugars_without_error(self, edges):
        """A config with only edges must produce a fully populated output."""
        result = desugar_config({EDGES_KEY: edges})
        assert result[MAX_BOND_DIM_KEY] == DEFAULT_MAX_BOND_DIM
        assert result[SEED_KEY] == DEFAULT_SEED
        assert result[BACKEND_KEY] == DEFAULT_BACKEND

    @given(cfg=valid_raw_config())
    def test_explicit_values_not_overwritten_by_defaults(self, cfg):
        """Any value explicitly set in the input must appear in the output."""
        result = desugar_config(cfg)
        if MAX_BOND_DIM_KEY in cfg:
            assert result[MAX_BOND_DIM_KEY] == cfg[MAX_BOND_DIM_KEY]
        if SEED_KEY in cfg:
            assert result[SEED_KEY] == cfg[SEED_KEY]
        if BACKEND_KEY in cfg:
            assert result[BACKEND_KEY] == cfg[BACKEND_KEY]
        if BP_EPS_KEY in cfg:
            assert isclose(result[BP_EPS_KEY], float(cfg[BP_EPS_KEY]))

    @given(cfg=valid_raw_config())
    def test_input_config_not_mutated(self, cfg):
        """desugar_config must not modify the original config dict."""
        original = copy.deepcopy(cfg)
        desugar_config(cfg)
        assert cfg == original

    @given(cfg=valid_raw_config())
    def test_default_mutable_objects_not_shared(self, cfg):
        """
        Two calls with different minimal configs must not share mutable default
        objects. Mutating one output must not affect another.
        """
        edges = cfg[EDGES_KEY]
        r1 = desugar_config({EDGES_KEY: edges})
        r2 = desugar_config({EDGES_KEY: edges})
        # Mutate r1's schedule actions and verify r2 is unaffected
        r1[SCHEDULE_KEY][ACTIONS_KEY].append("sentinel")
        assert "sentinel" not in r2[SCHEDULE_KEY][ACTIONS_KEY]

    @given(cfg=valid_raw_config())
    def test_sparsification_none_stays_none(self, cfg):
        cfg.pop(SPARSIFICATION_KEY, None)
        result = desugar_config(cfg)
        assert result[SPARSIFICATION_KEY] is None

    @given(cfg=valid_raw_config(), sp=valid_sparsification())
    def test_sparsification_present_is_dict_with_float_values(self, cfg, sp):
        cfg[SPARSIFICATION_KEY] = sp
        result = desugar_config(cfg)
        assert isinstance(result[SPARSIFICATION_KEY], dict)
        assert isinstance(result[SPARSIFICATION_KEY][EPS_KEY], float)
        assert isinstance(result[SPARSIFICATION_KEY][CLUSTER_COUPLING_AMPLITUDE_KEY], float)
