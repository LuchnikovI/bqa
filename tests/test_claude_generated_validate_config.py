"""
Property-based tests for validate_config using Hypothesis.

Run with:
    pytest test_validate_config.py -v
    pytest test_validate_config.py -v --hypothesis-seed=0   # for reproducibility
"""

import pytest
from bqa.backends import BACKEND_STR_TO_BACKEND
from hypothesis import given, assume, settings, HealthCheck
from hypothesis import strategies as st


# Now import the module under test  (adjust import path as needed)
from bqa.config.validate_config import (  # noqa: E402  – after stub setup
    validate_config,
    ConfigSyntaxError,
    BACKEND_KEY,
    NODES_KEY,
    EDGES_KEY,
    DEFAULT_FIELD_KEY,
    SCHEDULE_KEY,
    MAX_BOND_DIM_KEY,
    MAX_BP_ITER_NUMBER_KEY,
    SEED_KEY,
    BP_EPS_KEY,
    PINV_EPS_KEY,
    MEASUREMENT_THRESHOLD_KEY,
    DAMPING_KEY,
    SPARSIFICATION_KEY,
    ACTIONS_KEY,
    TOTAL_TIME_KEY,
    STARTING_MIXING_KEY,
    STEPS_NUMBER_KEY,
    WEIGHT_KEY,
    INITIAL_MIXING_KEY,
    FINAL_MIXING_KEY,
    EPS_KEY,
    CLUSTER_COUPLING_AMPLITUDE_KEY,
)

VALID_BACKENDS = BACKEND_STR_TO_BACKEND

# ---------------------------------------------------------------------------
# Reusable strategies
# ---------------------------------------------------------------------------

finite_floats = st.floats(allow_nan=False, allow_infinity=False)
positive_floats = st.floats(min_value=1e-9, max_value=1e9, allow_nan=False)
zero_to_one = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
half_to_one = st.floats(min_value=0.5, max_value=1.0, allow_nan=False)
positive_ints = st.integers(min_value=1, max_value=10_000)
non_neg_ints = st.integers(min_value=0, max_value=10_000)
node_ids = st.integers(min_value=0, max_value=50)


@st.composite
def valid_edge_list(draw):
    """Generate a list of ((lhs, rhs), coupling) pairs with no duplicates."""
    n = draw(st.integers(min_value=1, max_value=10))
    ids = draw(st.lists(node_ids, min_size=2, max_size=20, unique=True))
    assume(len(ids) >= 2)
    seen = set()
    edges = []
    for _ in range(n):
        a = draw(st.sampled_from(ids))
        b = draw(st.sampled_from(ids))
        assume(a != b)
        key = (min(a, b), max(a, b))
        if key not in seen:
            seen.add(key)
            cpl = draw(finite_floats)
            edges.append(((a, b), cpl))
    assume(len(edges) >= 1)
    return edges


@st.composite
def valid_node_list(draw):
    """Generate a list of (node_id, field) pairs with no conflicting duplicates."""
    ids = draw(st.lists(node_ids, min_size=1, max_size=20, unique=True))
    return [(nid, draw(finite_floats)) for nid in ids]


@st.composite
def valid_single_action(draw):
    """Generate a single valid QA action dict."""
    action = {}
    if draw(st.booleans()):
        action[STEPS_NUMBER_KEY] = draw(positive_ints)
    if draw(st.booleans()):
        action[INITIAL_MIXING_KEY] = draw(zero_to_one)
    if draw(st.booleans()):
        action[FINAL_MIXING_KEY] = draw(zero_to_one)
    return action


@st.composite
def valid_actions_with_weights(draw):
    """
    Generate a list of actions whose weights sum to exactly 1.0.
    Mixes string actions and weighted dicts.
    """
    n = draw(st.integers(min_value=1, max_value=4))
    weights = draw(
        st.lists(
            st.floats(min_value=0.01, max_value=1.0, allow_nan=False),
            min_size=n,
            max_size=n,
        )
    )
    total = sum(weights)
    assume(total > 0)
    weights = [w / total for w in weights]  # normalise to 1.0

    actions = []
    for w in weights:
        action = draw(valid_single_action())
        action[WEIGHT_KEY] = w
        actions.append(action)

    # Optionally prepend some string actions
    extras = draw(st.lists(st.sampled_from(["measure"]), max_size=2))
    return extras + actions


@st.composite
def valid_schedule(draw):
    sch = {}
    if draw(st.booleans()):
        sch[TOTAL_TIME_KEY] = draw(positive_floats)
    if draw(st.booleans()):
        sch[STARTING_MIXING_KEY] = draw(zero_to_one)
    if draw(st.booleans()):
        sch[ACTIONS_KEY] = draw(valid_actions_with_weights())
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
def valid_config(draw):
    """Build a fully valid config dict."""
    cfg = {EDGES_KEY: draw(valid_edge_list())}

    if draw(st.booleans()):
        cfg[NODES_KEY] = draw(valid_node_list())
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
        cfg[MEASUREMENT_THRESHOLD_KEY] = draw(half_to_one)
    if draw(st.booleans()):
        cfg[DAMPING_KEY] = draw(zero_to_one)
    if draw(st.booleans()):
        cfg[BACKEND_KEY] = draw(st.sampled_from(list(VALID_BACKENDS.keys())))
    if draw(st.booleans()):
        cfg[SPARSIFICATION_KEY] = draw(valid_sparsification())

    return cfg


# ---------------------------------------------------------------------------
# 1. Round-trip / acceptance properties
# ---------------------------------------------------------------------------

class TestValidConfigAccepted:

    @given(cfg=valid_config())
    def test_valid_config_does_not_raise(self, cfg):
        """Any config produced by valid_config() must not raise."""
        validate_config(cfg)  # must not throw

    @given(edges=valid_edge_list())
    def test_minimal_config_only_edges(self, edges):
        """A config with only the mandatory edges key must be accepted."""
        validate_config({EDGES_KEY: edges})

    @given(cfg=valid_config())
    def test_idempotent(self, cfg):
        """Calling validate_config twice on the same object must not raise."""
        validate_config(cfg)
        validate_config(cfg)


# ---------------------------------------------------------------------------
# 2. Missing / wrong type for required field
# ---------------------------------------------------------------------------

class TestMissingOrWrongEdges:

    def test_missing_edges_raises(self):
        with pytest.raises(ConfigSyntaxError):
            validate_config({})

    @given(value=st.one_of(
        st.none(),
        st.integers(),
        st.text(),
        st.floats(allow_nan=False),
        st.booleans(),
    ))
    def test_edges_wrong_type_raises(self, value):
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: value})

    def test_empty_edges_list_accepted(self):
        """An empty edge list is a degenerate but structurally valid graph."""
        validate_config({EDGES_KEY: []})

    def test_empty_edges_dict_accepted(self):
        validate_config({EDGES_KEY: {}})


# ---------------------------------------------------------------------------
# 3. Edge structural invariants
# ---------------------------------------------------------------------------

class TestEdgeInvariants:

    @given(
        a=node_ids,
        cpl=finite_floats,
    )
    def test_self_loop_edge_rejected(self, a, cpl):
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [((a, a), cpl)]})

    @given(
        a=node_ids,
        b=node_ids,
        cpl1=finite_floats,
        cpl2=finite_floats,
    )
    def test_duplicate_edge_same_orientation_different_value_rejected(self, a, b, cpl1, cpl2):
        assume(a != b)
        assume(cpl1 != cpl2)
        edges = [((a, b), cpl1), ((a, b), cpl2)]
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: edges})

    @given(
        a=node_ids,
        b=node_ids,
        cpl1=finite_floats,
        cpl2=finite_floats,
    )
    def test_duplicate_edge_reverse_orientation_different_value_rejected(self, a, b, cpl1, cpl2):
        assume(a != b)
        assume(cpl1 != cpl2)
        edges = [((a, b), cpl1), ((b, a), cpl2)]
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: edges})

    @given(
        a=node_ids,
        b=node_ids,
        cpl=finite_floats,
    )
    def test_edge_with_negative_node_id_rejected(self, a, b, cpl):
        assume(a != b)
        neg_id = -abs(a) - 1
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [((neg_id, b), cpl)]})

    @given(
        a=node_ids,
        b=node_ids,
        bad_cpl=st.one_of(st.text(), st.none(), st.lists(st.integers())),
    )
    def test_non_numeric_coupling_rejected(self, a, b, bad_cpl):
        assume(a != b)
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [((a, b), bad_cpl)]})


# ---------------------------------------------------------------------------
# 4. Optional numeric fields — boundary conditions
# ---------------------------------------------------------------------------

class TestNumericFieldBoundaries:

    @given(v=st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False))
    def test_negative_max_bond_dim_rejected(self, v):
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [], MAX_BOND_DIM_KEY: v})

    @given(v=positive_ints)
    def test_valid_max_bond_dim_accepted(self, v):
        validate_config({EDGES_KEY: [], MAX_BOND_DIM_KEY: v})

    @given(v=st.floats(min_value=0.0, max_value=0.4999, allow_nan=False))
    def test_measurement_threshold_below_half_rejected(self, v):
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [], MEASUREMENT_THRESHOLD_KEY: v})

    @given(v=half_to_one)
    def test_measurement_threshold_half_to_one_accepted(self, v):
        validate_config({EDGES_KEY: [], MEASUREMENT_THRESHOLD_KEY: v})

    @given(v=st.floats(min_value=1.0001, max_value=1e9, allow_nan=False))
    def test_zero_to_one_field_above_one_rejected(self, v):
        """bp_eps, pinv_eps, damping must all reject values > 1."""
        for key in (BP_EPS_KEY, PINV_EPS_KEY, DAMPING_KEY):
            with pytest.raises(ConfigSyntaxError):
                validate_config({EDGES_KEY: [], key: v})

    @given(v=st.floats(max_value=-1e-9, allow_nan=False, allow_infinity=False))
    def test_zero_to_one_field_below_zero_rejected(self, v):
        for key in (BP_EPS_KEY, PINV_EPS_KEY, DAMPING_KEY):
            with pytest.raises(ConfigSyntaxError):
                validate_config({EDGES_KEY: [], key: v})

    @given(v=st.integers(max_value=-1))
    def test_negative_seed_rejected(self, v):
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [], SEED_KEY: v})

    @given(v=non_neg_ints)
    def test_valid_seed_accepted(self, v):
        validate_config({EDGES_KEY: [], SEED_KEY: v})

# ---------------------------------------------------------------------------
# 5. Backend validation
# ---------------------------------------------------------------------------

class TestBackendValidation:

    @given(backend=st.sampled_from(list(VALID_BACKENDS.keys())))
    def test_valid_backend_accepted(self, backend):
        validate_config({EDGES_KEY: [], BACKEND_KEY: backend})

    @given(backend=st.text().filter(lambda s: s not in VALID_BACKENDS))
    def test_unknown_backend_rejected(self, backend):
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [], BACKEND_KEY: backend})

    @given(backend=st.one_of(st.integers(), st.none(), st.floats(allow_nan=False)))
    def test_non_string_backend_rejected(self, backend):
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [], BACKEND_KEY: backend})


# ---------------------------------------------------------------------------
# 6. Schedule and action invariants
# ---------------------------------------------------------------------------

class TestScheduleInvariants:

    @given(t=st.floats(max_value=0.0, allow_nan=False))
    def test_non_positive_total_time_rejected(self, t):
        assume(t <= 0)
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [], SCHEDULE_KEY: {TOTAL_TIME_KEY: t}})

    @given(t=positive_floats)
    def test_positive_total_time_accepted(self, t):
        validate_config({EDGES_KEY: [], SCHEDULE_KEY: {TOTAL_TIME_KEY: t}})

    @given(actions=valid_actions_with_weights())
    def test_valid_weighted_actions_accepted(self, actions):
        validate_config({EDGES_KEY: [], SCHEDULE_KEY: {ACTIONS_KEY: actions}})

    def test_actions_weights_not_summing_to_one_rejected(self):
        actions = [
            {WEIGHT_KEY: 0.3, STEPS_NUMBER_KEY: 10},
            {WEIGHT_KEY: 0.3, STEPS_NUMBER_KEY: 10},
        ]  # sum = 0.6, not 1.0
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [], SCHEDULE_KEY: {ACTIONS_KEY: actions}})

    @given(bad_action=st.one_of(st.integers(), st.floats(allow_nan=False), st.none()))
    def test_action_wrong_type_rejected(self, bad_action):
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [], SCHEDULE_KEY: {ACTIONS_KEY: [bad_action]}})

    @given(keyword=st.text().filter(lambda s: s not in {"measure", "get_bloch_vectors"}))
    def test_unknown_string_action_rejected(self, keyword):
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [], SCHEDULE_KEY: {ACTIONS_KEY: [keyword]}})

    @given(steps=st.integers(max_value=0))
    def test_non_positive_steps_number_rejected(self, steps):
        action = {STEPS_NUMBER_KEY: steps, WEIGHT_KEY: 1.0}
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [], SCHEDULE_KEY: {ACTIONS_KEY: [action]}})


# ---------------------------------------------------------------------------
# 7. Sparsification invariants
# ---------------------------------------------------------------------------

class TestSparsificationInvariants:

    @given(eps=zero_to_one)
    def test_valid_eps_accepted(self, eps):
        validate_config({EDGES_KEY: [], SPARSIFICATION_KEY: {EPS_KEY: eps}})

    @given(eps=st.floats(min_value=1.0001, max_value=1e6, allow_nan=False))
    def test_eps_above_one_rejected(self, eps):
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [], SPARSIFICATION_KEY: {EPS_KEY: eps}})

    @given(amp=st.floats(min_value=1.0, max_value=1e6, allow_nan=False))
    def test_valid_cluster_coupling_amplitude_accepted(self, amp):
        validate_config({EDGES_KEY: [], SPARSIFICATION_KEY: {CLUSTER_COUPLING_AMPLITUDE_KEY: amp}})

    @given(amp=st.floats(max_value=0.9999, allow_nan=False, allow_infinity=False))
    def test_cluster_coupling_amplitude_below_one_rejected(self, amp):
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [], SPARSIFICATION_KEY: {CLUSTER_COUPLING_AMPLITUDE_KEY: amp}})

    @given(v=st.one_of(st.text(), st.lists(st.integers())))
    def test_sparsification_wrong_type_rejected(self, v):
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [], SPARSIFICATION_KEY: v})


# ---------------------------------------------------------------------------
# 8. Top-level type invariants
# ---------------------------------------------------------------------------

class TestTopLevelTypeInvariants:

    @given(cfg=st.one_of(
        st.none(),
        st.integers(),
        st.text(),
        st.lists(st.integers()),
        st.floats(allow_nan=False),
    ))
    def test_non_dict_config_rejected(self, cfg):
        with pytest.raises(ConfigSyntaxError):
            validate_config(cfg)

    @given(
        key=st.sampled_from([
            MAX_BOND_DIM_KEY, MAX_BP_ITER_NUMBER_KEY, SEED_KEY
        ]),
        value=st.floats(allow_nan=False, allow_infinity=False).filter(
            lambda v: v != int(v) if v == v else True
        ),
    )
    def test_float_where_int_required_rejected(self, key, value):
        """Fields that require integers must reject floats even if they look integer-like."""
        assume(value != int(value))  # e.g. 1.5 not 1.0
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [], key: value})

    @given(
        key=st.sampled_from([DEFAULT_FIELD_KEY, BP_EPS_KEY, PINV_EPS_KEY, DAMPING_KEY]),
        value=st.one_of(st.text(), st.none(), st.lists(st.integers())),
    )
    def test_wrong_type_for_numeric_fields_rejected(self, key, value):
        with pytest.raises(ConfigSyntaxError):
            validate_config({EDGES_KEY: [], key: value})
