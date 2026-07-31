from tests.server.conftest import auth


class TestHealth:
    async def test_healthz_needs_no_credential(self, gateway):
        client, _, _ = gateway
        response = await client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestRequestId:
    async def test_every_response_carries_one(self, gateway):
        client, _, _ = gateway
        response = await client.get("/healthz")
        assert response.headers["x-request-id"].startswith("req_")

    async def test_a_callers_own_id_is_kept_rather_than_replaced(self, gateway):
        # So a trace that starts upstream does not break at the gateway boundary.
        client, _, _ = gateway
        response = await client.get("/healthz", headers={"x-request-id": "trace-abc"})
        assert response.headers["x-request-id"] == "trace-abc"


class TestProviders:
    async def test_it_reports_health_and_capabilities_per_provider(self, gateway):
        client, _, _ = gateway
        response = await client.get("/v1/providers", headers=auth())
        assert response.status_code == 200
        by_name = {entry["name"]: entry for entry in response.json()}

        assert by_name["fake"]["healthy"] is True
        assert by_name["down"]["healthy"] is False
        assert by_name["fake"]["capabilities"]["memory_model"] == "flat_facts"

    async def test_it_needs_a_credential(self, gateway):
        client, _, _ = gateway
        assert (await client.get("/v1/providers")).status_code == 401


class TestCapabilities:
    async def test_it_reports_the_configured_instance_not_a_class_constant(self, gateway):
        client, _, _ = gateway
        default = await client.get("/v1/capabilities", headers=auth())
        flat = await client.get("/v1/capabilities?provider=flat", headers=auth())

        assert default.json()["scope_dims"] == ["tenant", "subject", "agent", "session"]
        assert flat.json()["scope_dims"] == ["subject"]

    async def test_an_unknown_provider_is_a_400_naming_the_real_ones(self, gateway):
        client, _, _ = gateway
        response = await client.get("/v1/capabilities?provider=zep", headers=auth())
        assert response.status_code == 400
        assert "fake" in response.json()["error"]["details"]["available"]


class TestOpenApi:
    async def test_the_schema_publishes_every_v1_route(self, gateway):
        client, _, _ = gateway
        paths = (await client.get("/openapi.json")).json()["paths"]
        for path in (
            "/v1/memories:ingest",
            "/v1/memories:upsert",
            "/v1/memories:search",
            "/v1/memories:delete",
            "/v1/memories:list",
            "/v1/memories/{gateway_id}",
            "/v1/subjects/{subject}",
            "/v1/subjects/{subject}:rebind",
            "/v1/providers",
            "/v1/capabilities",
        ):
            assert path in paths, f"{path} missing from the published schema"
