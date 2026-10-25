# Working agreement

- Preserve the pure `lemonade.core` boundary; do not introduce HTTP/UI imports there.
- Change game rules only with a rules-version update and tests.
- Preserve reset-seed reproducibility and pre-generated exogenous scenarios.
- Run `ruff check .` and `pytest` after Python changes; document checks not run.
- Keep a focused Conventional Commit. Never push without explicit user approval.
