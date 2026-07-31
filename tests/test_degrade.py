import pytest

from memgw.capabilities import Capabilities
from memgw.degrade import assert_delete_supported, assert_scope_supported, resolve_mode
from memgw.errors import UnsupportedCapability
from memgw.types import Scope


def caps(**over) -> Capabilities:
    base = dict(
        supports_ingest=True,
        supports_upsert=True,
        supports_update=True,
        supports_delete=True,
        supports_delete_by_scope=True,
        search_modes=["semantic"],
        supports_score=True,
        max_limit=100,
        scope_dims=["subject", "agent", "session"],
        supports_labels=True,
        memory_model="flat_facts",
        dedup="none",
        supports_export=True,
        supports_import=True,
        consistency="read_after_write",
        metered_externally=False,
    )
    base.update(over)
    return Capabilities(**base)


class TestResolveMode:
    def test_supported_mode_is_served_undegraded(self):
        r = resolve_mode("semantic", caps(), "reject")
        assert (r.served, r.degraded, r.lost) == ("semantic", False, [])

    def test_supported_mode_is_undegraded_even_when_degrade_is_allowed(self):
        r = resolve_mode("graph", caps(search_modes=["semantic", "graph"]), "degrade")
        assert r.degraded is False
        assert r.served == "graph"

    def test_graph_on_a_flat_provider_is_rejected_by_default(self):
        with pytest.raises(UnsupportedCapability):
            resolve_mode("graph", caps(), "reject")

    @pytest.mark.parametrize(
        ("requested", "lost"),
        [
            ("graph", ["graph_traversal"]),
            ("temporal", ["fact_invalidation"]),
            ("hybrid", ["keyword_match"]),
        ],
    )
    def test_permitted_substitutions_say_what_was_lost(self, requested, lost):
        r = resolve_mode(requested, caps(), "degrade")
        assert r.served == "semantic"
        assert r.degraded is True
        assert r.lost == lost

    def test_a_mode_with_no_defined_substitution_is_rejected_under_both(self):
        # keyword has no documented fallback: pretending semantic search is keyword
        # search would be a quiet lie about what was run.
        for on_unsupported in ("reject", "degrade"):
            with pytest.raises(UnsupportedCapability):
                resolve_mode("keyword", caps(), on_unsupported)

    def test_substitution_target_must_itself_be_supported(self):
        with pytest.raises(UnsupportedCapability):
            resolve_mode("graph", caps(search_modes=["keyword"]), "degrade")


class TestScopeIsNeverDegraded:
    def test_missing_scope_dimension_is_rejected_under_both(self):
        subject_only = caps(scope_dims=["subject"])
        scope = Scope(subject="u_1", agent="lugo")
        for on_unsupported in ("reject", "degrade"):
            with pytest.raises(UnsupportedCapability) as exc:
                assert_scope_supported(scope, subject_only, on_unsupported)
            assert exc.value.details["missing_scope_dims"] == ["agent"]

    def test_scope_within_the_declared_dimensions_passes(self):
        assert_scope_supported(Scope(subject="u_1"), caps(scope_dims=["subject"]), "degrade")

    def test_labels_are_rejected_when_unsupported(self):
        scope = Scope(subject="u_1", labels={"tier": "pro"})
        with pytest.raises(UnsupportedCapability):
            assert_scope_supported(scope, caps(supports_labels=False), "degrade")


class TestDeleteIsNeverDegraded:
    def test_delete_shortfall_is_rejected_under_both(self):
        for on_unsupported in ("reject", "degrade"):
            with pytest.raises(UnsupportedCapability):
                assert_delete_supported(caps(supports_delete=False), on_unsupported)

    def test_delete_by_scope_shortfall_is_rejected_under_both(self):
        for on_unsupported in ("reject", "degrade"):
            with pytest.raises(UnsupportedCapability):
                assert_delete_supported(
                    caps(supports_delete_by_scope=False), on_unsupported, by_scope=True
                )
