from fastapi.testclient import TestClient

from lemonade.api import app, sessions


def setup_function() -> None:
    sessions.clear()


def test_human_game_contract_and_terminal_rejection() -> None:
    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}
    created = client.post("/sessions?seed=42")
    assert created.status_code == 201
    session_id = created.json()["session_id"]
    assert created.json()["state"]["weather_forecast"] == "downpour"
    for day in range(10):
        response = client.post(
            f"/sessions/{session_id}/actions", json={"price": 20, "production": 100}
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["result"]["day"] == day + 1
        assert payload["result"]["profit"] == (
            payload["result"]["revenue"] - payload["result"]["production_cost"]
        )
    terminal = client.post(
        f"/sessions/{session_id}/actions", json={"price": 20, "production": 100}
    )
    assert terminal.status_code == 409
    history = client.get(f"/sessions/{session_id}").json()
    assert history["state"]["terminated"] is True
    assert len(history["history"]) == 10


def test_api_rejects_unknown_session_and_invalid_action() -> None:
    client = TestClient(app)
    assert client.get("/sessions/nope").status_code == 404
    session_id = client.post("/sessions").json()["session_id"]
    response = client.post(f"/sessions/{session_id}/actions", json={"price": 100, "production": 1})
    assert response.status_code == 422
