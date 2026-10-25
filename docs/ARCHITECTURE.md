# Architecture

Решение — modular monolith на Python. `lemonade.core` — чистая модель правил,
не зависящая от Gym, HTTP и UI. `lemonade.env` адаптирует её к Gymnasium.
`policies` и `evaluation` исполняют baseline и сравнение на одних reset-seed.
CLI запускает headless эпизод. Будущие FastAPI и Vue будут вызывать core через
application layer; обучение останется in-process через Gym, не через HTTP.

Observation: день, капитал, текущий точный прогноз погоды. Скрыты будущие дни,
шум и спрос. Action: дискретные цена и план выпуска. Reward: дневная прибыль.
Окончание 10-го дня — `terminated=True`; `truncated` не используется средой.

Performance target для v1: не менее 10 000 простых эпизодов/с на CPU. Current
pure-core measurement is 18,320 episodes/s; method and limitations are in
`VALIDATION.md`.
