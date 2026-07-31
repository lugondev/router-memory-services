from tests.server.conftest import KEY, OTHER_KEY, auth

BODY = {"scope": {"subject": "u_1"}, "text": "black coffee"}


class TestAuthentication:
    async def test_no_credential_is_a_401(self, gateway):
        client, _, _ = gateway
        response = await client.post("/v1/memories:ingest", json=BODY)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "unauthenticated"

    async def test_a_wrong_scheme_is_a_401(self, gateway):
        client, _, _ = gateway
        response = await client.post(
            "/v1/memories:ingest", json=BODY, headers={"Authorization": f"Basic {KEY}"}
        )
        assert response.status_code == 401

    async def test_an_unknown_key_is_a_401(self, gateway):
        client, _, _ = gateway
        response = await client.post("/v1/memories:ingest", json=BODY, headers=auth("nope"))
        assert response.status_code == 401


class TestTenantCannotBeWidened:
    async def test_a_payload_asserting_another_tenant_is_a_403(self, gateway):
        client, _, _ = gateway
        response = await client.post(
            "/v1/memories:ingest", json={**BODY, "tenant": "t2"}, headers=auth()
        )
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "tenant_mismatch"

    async def test_a_payload_asserting_its_own_tenant_is_fine(self, gateway):
        client, _, _ = gateway
        response = await client.post(
            "/v1/memories:ingest", json={**BODY, "tenant": "t1"}, headers=auth()
        )
        assert response.status_code == 200


class TestTenantIsolation:
    async def test_another_tenants_memory_id_is_a_404_not_a_403(self, gateway):
        # A 403 would confirm the id exists — an existence oracle across tenants.
        client, _, _ = gateway
        written = await client.post("/v1/memories:ingest", json=BODY, headers=auth())
        gateway_id = written.json()["results"][0]["id"]

        response = await client.get(f"/v1/memories/{gateway_id}", headers=auth(OTHER_KEY))
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "memory_not_found"

    async def test_two_tenants_do_not_share_a_binding(self, gateway):
        client, _, _ = gateway
        await client.post("/v1/memories:ingest", json=BODY, headers=auth())

        mine = await client.get("/v1/subjects/u_1", headers=auth())
        theirs = await client.get("/v1/subjects/u_1", headers=auth(OTHER_KEY))
        assert mine.json()["provider"] == "fake"
        assert theirs.json()["provider"] is None
