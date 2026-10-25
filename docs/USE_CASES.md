# Use cases

| ID | Минимальная реализация | Критерий приёмки | Проверка |
|---|---|---|---|
| UC-04 | `LemonadeGame` | policy завершает 10 шагов | unit tests |
| UC-05 | `RandomPolicy`, `evaluate` | общие scenario-seed для серий | integration test |
| UC-06 | fixed и weather-aware rule | policy возвращает допустимые действия | integration test |
| UC-07 | `LemonadeEnv` | Gym API и natural termination | Gym checker |
| UC-01–03 | запланировано | человек играет в UI | browser E2E |
| UC-08–11 | запланировано | модель/эксперимент можно оценить | RL/API tests |
