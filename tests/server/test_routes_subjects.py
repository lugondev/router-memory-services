from tests.server.conftest import auth

SCOPE = {"subject": "u_1"}


class TestBinding:
    async def test_an_unbound_subject_reports_none(self, gateway):
        client, _, _ = gateway
        response = await client.get("/v1/subjects/u_1", headers=auth())
        assert response.status_code == 200
        assert response.json() == {"subject": "u_1", "provider": None}

    async def test_the_first_write_binds(self, gateway):
        client, _, _ = gateway
        await client.post(
            "/v1/memories:ingest", json={"scope": SCOPE, "text": "coffee"}, headers=auth()
        )
        response = await client.get("/v1/subjects/u_1", headers=auth())
        assert response.json()["provider"] == "fake"


class TestRebind:
    async def test_fresh_start_moves_the_binding_and_names_the_casualties(self, gateway):
        client, _, catalog = gateway
        await client.post(
            "/v1/memories:ingest", json={"scope": SCOPE, "text": "coffee one"}, headers=auth()
        )
        await client.post(
            "/v1/memories:ingest", json={"scope": SCOPE, "text": "coffee two"}, headers=auth()
        )

        response = await client.post(
            "/v1/subjects/u_1:rebind",
            json={"provider": "flat", "strategy": "fresh_start"},
            headers=auth(),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "flat"
        assert body["orphaned_at"] == "fake"
        assert body["orphaned_count"] == 2
        assert "nothing was deleted" in body["note"]

        # Stranded, not destroyed.
        assert await catalog.live_count("t1", "u_1", "fake") == 2

    async def test_migrate_is_published_and_returns_501(self, gateway):
        client, _, _ = gateway
        response = await client.post(
            "/v1/subjects/u_1:rebind",
            json={"provider": "flat", "strategy": "migrate"},
            headers=auth(),
        )
        assert response.status_code == 501
        assert response.json()["error"]["code"] == "not_implemented"

    async def test_rebinding_to_an_unconfigured_provider_is_refused(self, gateway):
        client, _, _ = gateway
        response = await client.post(
            "/v1/subjects/u_1:rebind", json={"provider": "zep"}, headers=auth()
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "unknown_provider"
