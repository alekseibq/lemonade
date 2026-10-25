# RL experiments

Train and evaluation use disjoint fixed lists of scenario seeds. Scenario weather
and demand noise are generated at reset, so comparison does not depend on a
policy's random-number consumption. Reports will record rules version, config,
algorithm, model/training seed, package versions, budget and mean/median/stddev
profit plus profitable share. A training run is successful only if held-out
evaluation improves over random baseline.
