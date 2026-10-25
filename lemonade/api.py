"""Local, in-memory HTTP API for human-play sessions."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from lemonade.core import DayResult, LemonadeGame
from lemonade.env import LemonadeEnv


class ActionRequest(BaseModel):
    price: int = Field(ge=0, le=99)
    production: int = Field(ge=0, le=200)


class ReplayRequest(BaseModel):
    model_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    seed: int = Field(ge=0)


class GameSession:
    def __init__(self, seed: int | None) -> None:
        self.game = LemonadeGame()
        self.game.reset(seed=seed)


class ReplaySession:
    def __init__(self, model_id: str, seed: int) -> None:
        from stable_baselines3 import PPO

        model_path = ARTIFACTS_DIR / model_id / "model.zip"
        if not model_path.is_file():
            raise FileNotFoundError(model_id)
        self.env = LemonadeEnv()
        self.model = PPO.load(model_path, device="cpu")
        self.observation, _ = self.env.reset(seed=seed)
        self.step_number = 0


sessions: dict[str, GameSession] = {}
replays: dict[str, ReplaySession] = {}
ARTIFACTS_DIR = Path("artifacts").resolve()
app = FastAPI(title="Lemonade RL Lab API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


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


@app.get("/replay-models")
def replay_models() -> dict[str, list[str]]:
    if not ARTIFACTS_DIR.is_dir():
        return {"models": []}
    models = sorted(path.name for path in ARTIFACTS_DIR.iterdir() if (path / "model.zip").is_file())
    return {"models": models}


@app.post("/replays", status_code=status.HTTP_201_CREATED)
def create_replay(request: ReplayRequest) -> dict[str, object]:
    try:
        replay = ReplaySession(request.model_id, request.seed)
    except ImportError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="RL dependency unavailable"
        ) from error
    except FileNotFoundError as error:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="model not found"
        ) from error
    replay_id = str(uuid4())
    replays[replay_id] = replay
    return {"replay_id": replay_id, "seed": request.seed, "terminated": False}


@app.post("/replays/{replay_id}/next")
def next_replay_step(replay_id: str) -> dict[str, object]:
    replay = replays.get(replay_id)
    if replay is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="replay not found")
    if replay.env.game.terminated:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="replay has terminated")
    observation = replay.observation.astype(float).tolist()
    action, _ = replay.model.predict(replay.observation, deterministic=True)
    replay.observation, _, terminated, truncated, info = replay.env.step(action)
    assert not truncated
    replay.step_number += 1
    result = info["day_result"]
    assert isinstance(result, DayResult)
    return {
        "step": replay.step_number,
        "observation": observation,
        "action": {"price": int(action[0]), "production": int(action[1])},
        "result": _result_payload(result),
        "terminated": terminated,
    }
