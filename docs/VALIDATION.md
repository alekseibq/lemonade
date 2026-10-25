# Validation evidence

Executed on 2026-10-25 in the local Python 3.12 virtual environment.

| Check | Result |
|---|---|
| `ruff check .` | passed |
| `pytest` | 9 passed (property invariants and Gym checker included) |
| `coverage report --include='lemonade/core.py'` | 90% lines |
| `python scripts/benchmark.py --episodes 100000` | 18,320 episodes/s |
| held-out baseline seeds 1000–1199 | random mean profit −1,952.35 ₽; rule mean 8,408.25 ₽ |

The benchmark used the pure core and fixed action `(price=20, production=100)`;
it measures simulation throughput, not PPO speed. The requested 10,000 eps/s
target is met on this host. SB3 compatibility and PPO smoke are **not run**:
the supplied Python 3.12 package resolver selected a multi-gigabyte CUDA PyTorch
distribution and did not complete installation. This is an environment/dependency
limitation, not evidence of compatibility.
