from tests.server.conftest import auth

SCOPE = {"subject": "u_1", "agent": "lugo", "session": "s_9"}


async def ingest(client, text="black coffee", **extra):
    return await client.post(
        "/v1/memories:ingest", json={"scope": SCOPE, "text": text, **extra}, headers=auth()
    )


class TestIngest:
    async def test_it_returns_records_with_gateway_ids(self, gateway):
        client, _, _ = gateway
        response = await ingest(client)
        assert response.status_code == 200
        body = response.json()
        assert body["provider"] == "fake"
        assert body["results"][0]["id"].startswith("mg_")
        assert body["results"][0]["content"] == "black coffee"

    async def test_a_body_with_neither_text_nor_messages_is_a_400(self, gateway):
        # 422 is reserved for unsupported_capability; a malformed body is a 400.
        client, _, _ = gateway
        response = await client.post(
            "/v1/memories:ingest", json={"scope": SCOPE}, headers=auth()
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "invalid_request"

    async def test_an_empty_subject_is_a_400(self, gateway):
        client, _, _ = gateway
        response = await client.post(
            "/v1/memories:ingest", json={"scope": {"subject": ""}, "text": "x"}, headers=auth()
        )
        assert response.status_code == 400


class TestProviderAssertion:
    async def test_a_disagreeing_assertion_is_a_409_and_never_reaches_the_adapter(self, gateway):
        client, _, _ = gateway
        await ingest(client)  # binds u_1 to "fake"

        response = await client.post(
            "/v1/memories:search",
            json={"scope": SCOPE, "query": "coffee", "provider": "down"},
            headers=auth(),
        )
        # "down" would answer 503 if it had been called at all.
        assert response.status_code == 409
        body = response.json()["error"]
        assert body["code"] == "provider_mismatch"
        assert body["details"] == {"asserted": "down", "bound": "fake"}


class TestSearch:
    async def test_it_finds_what_was_ingested(self, gateway):
        client, _, _ = gateway
        await ingest(client)
        response = await client.post(
            "/v1/memories:search", json={"scope": SCOPE, "query": "coffee"}, headers=auth()
        )
        assert response.status_code == 200
        body = response.json()
        assert [r["content"] for r in body["results"]] == ["black coffee"]
        assert body["degraded"] is False

    async def test_recall_across_sessions_needs_no_session(self, gateway):
        client, _, _ = gateway
        await ingest(client, "coffee one")
        await client.post(
            "/v1/memories:ingest",
            json={"scope": {"subject": "u_1", "agent": "lugo", "session": "s_2"}, "text": "coffee two"},
            headers=auth(),
        )
        response = await client.post(
            "/v1/memories:search",
            json={"scope": {"subject": "u_1", "agent": "lugo"}, "query": "coffee"},
            headers=auth(),
        )
        assert {r["content"] for r in response.json()["results"]} == {"coffee one", "coffee two"}

    async def test_graph_on_a_flat_provider_is_a_422(self, gateway):
        client, _, _ = gateway
        await ingest(client)
        response = await client.post(
            "/v1/memories:search",
            json={"scope": SCOPE, "query": "coffee", "mode": "graph"},
            headers=auth(),
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "unsupported_capability"

    async def test_degrade_says_what_it_gave_up(self, gateway):
        client, _, _ = gateway
        await ingest(client)
        response = await client.post(
            "/v1/memories:search",
            json={
                "scope": SCOPE,
                "query": "coffee",
                "mode": "graph",
                "on_unsupported": "degrade",
            },
            headers=auth(),
        )
        body = response.json()
        assert response.status_code == 200
        assert body["degraded"] is True
        assert (body["requested"], body["served"]) == ("graph", "semantic")
        assert body["lost"] == ["graph_traversal"]

    async def test_a_dead_provider_is_a_503(self, gateway):
        client, _, catalog = gateway
        await catalog.bind("t1", "u_9", "down")
        response = await client.post(
            "/v1/memories:search",
            json={"scope": {"subject": "u_9"}, "query": "coffee"},
            headers=auth(),
        )
        assert response.status_code == 503

    async def test_fail_open_is_a_200_that_admits_the_outage(self, gateway):
        client, _, catalog = gateway
        await catalog.bind("t1", "u_9", "down")
        response = await client.post(
            "/v1/memories:search",
            json={"scope": {"subject": "u_9"}, "query": "coffee", "fail_open": True},
            headers=auth(),
        )
        assert response.status_code == 200
        body = response.json()
        assert body["results"] == []
        assert body["provider_unavailable"] is True

    async def test_a_scope_the_provider_cannot_honour_is_a_422_even_under_degrade(self, gateway):
        client, _, catalog = gateway
        await catalog.bind("t1", "u_9", "flat")
        response = await client.post(
            "/v1/memories:search",
            json={
                "scope": {"subject": "u_9", "agent": "lugo"},
                "query": "coffee",
                "on_unsupported": "degrade",
            },
            headers=auth(),
        )
        assert response.status_code == 422
        assert response.json()["error"]["details"]["missing_scope_dims"] == ["agent"]


class TestGetUpdateDelete:
    async def test_get_update_and_delete_round_trip(self, gateway):
        client, _, _ = gateway
        gateway_id = (await ingest(client)).json()["results"][0]["id"]

        got = await client.get(f"/v1/memories/{gateway_id}", headers=auth())
        assert got.status_code == 200
        assert got.json()["content"] == "black coffee"

        patched = await client.patch(
            f"/v1/memories/{gateway_id}", json={"content": "green tea"}, headers=auth()
        )
        assert patched.status_code == 200
        assert patched.json()["content"] == "green tea"

        removed = await client.delete(f"/v1/memories/{gateway_id}", headers=auth())
        assert removed.status_code == 204

        gone = await client.get(f"/v1/memories/{gateway_id}", headers=auth())
        assert gone.status_code == 404

    async def test_delete_by_scope_reports_the_count(self, gateway):
        client, _, _ = gateway
        await ingest(client, "coffee one")
        await ingest(client, "coffee two")

        response = await client.post(
            "/v1/memories:delete", json={"scope": SCOPE}, headers=auth()
        )
        assert response.status_code == 200
        assert response.json()["deleted"] == 2


class TestReservedVerbs:
    async def test_upsert_is_published_and_returns_501(self, gateway):
        client, _, _ = gateway
        response = await client.post(
            "/v1/memories:upsert",
            json={"scope": SCOPE, "facts": ["prefers black coffee"]},
            headers=auth(),
        )
        assert response.status_code == 501
        assert response.json()["error"]["code"] == "not_implemented"
