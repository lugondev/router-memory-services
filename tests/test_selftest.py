"""``self_test``: does this provider actually work, end to end?

``health()`` answers "is it reachable". That is not the question Zep failed. Zep
was reachable the whole time: it accepted every write with a ``200`` and an id,
and then never built the graph, so every read came back empty with no error
anywhere. A reachability check passes that provider all day.

So an adapter may also answer a harder question -- write something, read it back,
clean up -- and ``memgw doctor`` asks it.
"""

from __future__ import annotations

import pytest

from memgw.types import HealthStatus


class TestAdaptersDeclareWhetherTheyAreProven:
    def test_zep_admits_it_is_experimental(self):
        # It has never passed the live conformance suite. A provider you can select
        # by name, next to two that are proven, has to say which it is.
        from tests.conformance.test_zep import adapter

        assert adapter().capabilities().experimental is True

    def test_the_proven_adapters_do_not_claim_to_be_experimental(self):
        from tests.fake import default_caps

        assert default_caps().experimental is False


class TestSelfTestSeparatesReachableFromWorking:
    async def test_a_provider_that_never_processes_fails_its_self_test(self):
        from tests.conformance.test_zep import FakeZep, adapter_with

        # Accepts the write, returns an id, and never builds an edge -- exactly what
        # the real Zep did for two hours.
        broken = FakeZep(processes=False)
        result = await adapter_with(broken).self_test(timeout=0.3, interval=0.1)
        assert result.ok is False
        assert "processed" in (result.detail or "").lower()

    async def test_a_working_provider_passes_and_cleans_up_after_itself(self):
        from tests.conformance.test_zep import FakeZep, adapter_with

        working = FakeZep()
        result = await adapter_with(working).self_test(timeout=2.0, interval=0.05)
        assert result.ok is True
        assert working.episodes == [], "the probe left its own data behind"
        assert working.deleted_users, "the probe left a user behind"

    async def test_the_probe_never_touches_a_real_subject(self):
        from tests.conformance.test_zep import FakeZep, adapter_with

        client = FakeZep()
        await adapter_with(client).self_test(timeout=2.0, interval=0.05)
        assert all("__memgw" in user for user in client.added_users), client.added_users


class TestDoctorAsksTheHarderQuestion:
    async def test_the_probe_is_off_unless_asked_for(self):
        # doctor runs on every `serve`, and a boot that writes to a customer's Zep
        # every time is not a health check, it is a side effect.
        from memgw.cli import doctor

        report = await doctor(_env(), probe=False)
        assert not any(c.name.endswith("_pipeline") for c in report.checks)

    async def test_a_provider_that_cannot_process_is_named_before_it_is_deployed(self):
        from memgw.cli import doctor

        report = await doctor(_env(), probe=True, adapters={"zep": _Broken()})
        check = next(c for c in report.checks if c.name == "zep_pipeline")
        assert check.ok is False
        assert "empty" in check.detail.lower()
        assert report.ok is False

    async def test_a_working_provider_passes_the_probe(self):
        from memgw.cli import doctor

        report = await doctor(_env(), probe=True, adapters={"zep": _Working()})
        assert next(c for c in report.checks if c.name == "zep_pipeline").ok is True

    async def test_an_adapter_with_no_self_test_is_not_held_against_it(self):
        # Most adapters will not implement one. Absence is not failure.
        from memgw.cli import doctor

        report = await doctor(_env(), probe=True, adapters={"zep": object()})
        assert not any(c.name == "zep_pipeline" for c in report.checks)


class TestServeDoesNotBlockOnSomebodyElsesOutage:
    async def test_preflight_checks_configuration_and_not_the_provider(self):
        """A provider outage must not stop the gateway from booting.

        The other providers still work, `fail_open` exists for exactly this, and a
        gateway that refuses to start because a third party is down converts their
        outage into yours.
        """
        from memgw.cli import preflight

        report = await preflight(_env())
        assert not any(c.name.endswith("_pipeline") for c in report.checks)
        assert report.ok is True


class _Working:
    async def self_test(self, **kw):
        del kw
        return HealthStatus(ok=True, detail="wrote and read back a probe fact")


class _Broken:
    async def self_test(self, **kw):
        del kw
        return HealthStatus(
            ok=False,
            detail="the write was accepted but never processed; reads return empty with no error",
        )


def _env() -> dict[str, str]:
    return {
        "OPENAI_API_KEY": "sk-test",
        "ZEP_API_KEY": "z_test",
        "MEMGW_PROVIDERS": "zep",
        "MEMGW_API_KEYS": "k1:tenant-a",
        "MEMGW_CATALOG_URL": "sqlite+aiosqlite:///:memory:",
    }


pytest.register_assert_rewrite("tests.conformance.test_zep")
