# Development

Fast check: `ruff check . && pytest`.

Install with `pip install -e '.[dev]'`. Run an episode with
`lemonade --policy rule --seed 42`. The test suite includes economic
property tests and the Gymnasium environment checker. SB3, API and browser
checks are intentionally future backlog items, not claimed as run.

Run the pure-core benchmark with `.venv/bin/python scripts/benchmark.py`.
It should be run on the target host before performance claims are made.

With the optional RL dependencies installed, run a bounded training experiment:
`python scripts/train_ppo.py --output artifacts/ppo-smoke --timesteps 10000`.
The output contains `model.zip` and `report.json`; the latter captures rules,
configuration, package versions, held-out seeds and comparisons with baselines.
Re-evaluate a saved policy with
`python scripts/evaluate_model.py artifacts/ppo-smoke/model.zip`.

Install the HTTP optional dependency with `pip install -e '.[dev,api]'`, then
run `uvicorn lemonade.api:app --reload`. The API is local and holds sessions in
memory; restarting the process drops them.

In another terminal, run `cd web && npm install && npm run dev`. Vite listens
only on `127.0.0.1:5605`; the Vue UI uses the local API at `http://localhost:8000`.
Start the API first.

For browser access from another machine, proxy that loopback port through nginx
on the Ubuntu host. The UI uses relative `/api` requests, which Vite then
proxies to the local FastAPI service, so the remote browser does not need access
to port 8000:

```nginx
location / {
    proxy_pass http://127.0.0.1:5605;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

Place this in a dedicated server block or virtual host, then reload nginx after
validating its full configuration with `nginx -t`. Restart the Vite process after
changing its configuration; it does not switch ports while running.
