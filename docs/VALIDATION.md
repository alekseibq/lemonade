# Validation evidence

Executed on 2026-10-25 in the local Python 3.12 virtual environment.

| Check | Result |
|---|---|
| `ruff check .` | passed |
| `pytest` | 9 passed (property invariants and Gym checker included) |
| `coverage report --include='lemonade/core.py'` | 90% lines |
| API contract tests | passed (2 tests, including terminal and validation responses) |
| `python scripts/benchmark.py --episodes 100000` | 18,320 episodes/s |
| held-out baseline seeds 1000–1199 | random mean profit −1,952.35 ₽; rule mean 8,408.25 ₽ |

The benchmark used the pure core and fixed action `(price=20, production=100)`;
it measures simulation throughput, not PPO speed. The requested 10,000 eps/s
target is met on this host.

PPO experiment `lemonade-ppo-10000` was trained for 10,000 CPU timesteps with
seed 123 and evaluated on held-out seeds 10,000–10,199. It achieved mean profit
3,259.60 ₽ (median 3,268 ₽; stddev 850.88; 100% profitable) versus random
−1,976.48 ₽ (median −1,998 ₽; stddev 166.97; 0% profitable): an improvement of
5,236.08 ₽ in mean profit. Rule-based remains stronger at 8,304.85 ₽ mean.
SB3 2.9.0 checker and training smoke passed. The PyTorch build is CUDA-capable,
but CUDA was unavailable and PPO was explicitly configured with `device="cpu"`.

The API test run emits one upstream Starlette `TestClient` deprecation warning;
it does not affect the contract assertions and is tracked for dependency upgrade.
