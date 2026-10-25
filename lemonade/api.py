"""Local, in-memory HTTP API for human-play sessions."""

from __future__ import annotations

from dataclasses import asdict
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from lemonade.core import DayResult, LemonadeGame


class ActionRequest(BaseModel):
    price: int = Field(ge=0, le=99)
    production: int = Field(ge=0, le=200)


class GameSession:
    def __init__(self, seed: int | None) -> None:
        self.game = LemonadeGame()
        self.game.reset(seed=seed)


sessions: dict[str, GameSession] = {}
app = FastAPI(title="Lemonade RL Lab API", version="0.1.0")


def _result_payload(result: DayResult) -> dict[str, object]:
    payload = asdict(result)
    payload["weather"] = result.weather.name.lower()
    return payload


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/sessions", status_code=status.HTTP_201_CREATED)
def create_session(seed: int | None = None) -> dict[str, object]:
    session_id = str(uuid4())
    session = GameSession(seed)
    sessions[session_id] = session
    return {"session_id": session_id, "state": session.game.public_state()}


def _session(session_id: str) -> GameSession:
    session = sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session not found")
    return session


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict[str, object]:
    game = _session(session_id).game
    return {
        "session_id": session_id,
        "state": game.public_state(),
        "history": [_result_payload(result) for result in game.history],
    }


@app.post("/sessions/{session_id}/actions")
def take_action(session_id: str, action: ActionRequest) -> dict[str, object]:
    game = _session(session_id).game
    if game.terminated:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="episode has terminated")
    result = game.step(action.price, action.production)
    return {
        "session_id": session_id,
        "result": _result_payload(result),
        "state": game.public_state(),
    }
