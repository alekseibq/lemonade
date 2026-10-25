# RL experiments

Train and evaluation use disjoint fixed lists of scenario seeds. Scenario weather
and demand noise are generated at reset, so comparison does not depend on a
policy's random-number consumption. Reports will record rules version, config,
algorithm, model/training seed, package versions, budget and mean/median/stddev
profit plus profitable share. A training run is successful only if held-out
evaluation improves over random baseline.

`scripts/train_ppo.py` uses PPO on CPU and requires an explicit timestep budget.
Its evaluation scenarios start at seed 10,000 and are disjoint from the training
seed. The report is evidence of a particular run; it does not establish success
until its held-out trained mean profit exceeds the random baseline.

Use `scripts/compare_ppo_seeds.py --output artifacts/ppo-comparison` for repeated
training. It trains seeds 101, 202 and 303 by default and evaluates every run on
the same held-out seeds 20,000–20,199, writing per-run artifacts and one
`comparison.json` aggregate report.
