# Development

Fast check: `ruff check . && pytest`.

Install with `pip install -e '.[dev]'`. Run an episode with
`lemonade --policy rule --seed 42`. The test suite includes economic
property tests and the Gymnasium environment checker. SB3, API and browser
checks are intentionally future backlog items, not claimed as run.

Run the pure-core benchmark with `.venv/bin/python scripts/benchmark.py`.
It should be run on the target host before performance claims are made.
