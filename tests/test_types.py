import pytest
from pydantic import ValidationError

from memgw.errors import GatewayError, ProviderMismatch, UnsupportedCapability
from memgw.types import Episode, Message, Scope, SearchQuery


class TestScope:
    def test_subject_is_required(self):
        with pytest.raises(ValidationError):
            Scope()

    @pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
    def test_blank_subject_is_rejected(self, bad):
        # "" is the platform's shared-device bucket. It is deliberately NOT one here:
        # in a multi-tenant product an empty subject is a collision bucket.
        with pytest.raises(ValidationError):
            Scope(subject=bad)

    def test_blank_agent_and_session_normalise_to_none(self):
        scope = Scope(subject="u_1", agent="  ", session="")
        assert scope.agent is None
        assert scope.session is None

    def test_dims_reports_only_what_is_set(self):
        assert Scope(subject="u_1").dims() == {"subject": "u_1"}
        assert Scope(subject="u_1", agent="lugo").dims() == {"subject": "u_1", "agent": "lugo"}
        assert Scope(subject="u_1", agent="lugo", session="s_9").dims() == {
            "subject": "u_1",
            "agent": "lugo",
            "session": "s_9",
        }


class TestEpisode:
    def test_both_messages_and_text_is_rejected(self):
        with pytest.raises(ValidationError):
            Episode(messages=[Message(role="user", content="hi")], text="hi")

    def test_neither_messages_nor_text_is_rejected(self):
        with pytest.raises(ValidationError):
            Episode()

    def test_empty_messages_is_rejected(self):
        with pytest.raises(ValidationError):
            Episode(messages=[])

    def test_blank_text_is_rejected(self):
        with pytest.raises(ValidationError):
            Episode(text="   ")

    def test_as_text_flattens_messages(self):
        ep = Episode(
            messages=[
                Message(role="user", content="I drink black coffee"),
                Message(role="assistant", content="Noted"),
            ]
        )
        assert ep.as_text() == "user: I drink black coffee\nassistant: Noted"

    def test_as_text_passes_plain_text_through(self):
        assert Episode(text="a note").as_text() == "a note"


class TestSearchQuery:
    def test_defaults_are_the_conservative_ones(self):
        q = SearchQuery(query="coffee")
        assert q.mode == "semantic"
        assert q.on_unsupported == "reject"
        assert q.fail_open is False
        assert q.limit == 10

    def test_limit_must_be_positive(self):
        with pytest.raises(ValidationError):
            SearchQuery(query="coffee", limit=0)


class TestErrors:
    def test_status_and_code_are_carried_per_class(self):
        assert UnsupportedCapability().status == 422
        assert UnsupportedCapability().code == "unsupported_capability"
        assert ProviderMismatch().status == 409

    def test_body_shape_is_the_documented_one(self):
        err = ProviderMismatch("nope", details={"asserted": "pgvector", "bound": "mem0"})
        assert err.to_body() == {
            "error": {
                "code": "provider_mismatch",
                "message": "nope",
                "details": {"asserted": "pgvector", "bound": "mem0"},
            }
        }

    def test_code_can_be_narrowed_without_changing_status(self):
        from memgw.errors import InvalidRequest

        err = InvalidRequest("no provider", code="no_provider_resolved")
        assert err.code == "no_provider_resolved"
        assert err.status == 400
        assert isinstance(err, GatewayError)
