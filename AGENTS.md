# Repository Guidelines

## Project Structure & Module Organization
Core simulator types and event plumbing live in `simulator/core.py`, while `simulator/behavior.py`, `simulator/cost.py`, `simulator/models/`, and `simulator/schedulers/` host pluggable user, workload, environment, and scheduler implementations. The CLI entry point is `simulate.py` for Matplotlib dashboards that write JSON logs into `logs/`.

## Build, Test, and Development Commands
- `python simulate.py --priority new-first --days 90 --no-log` plots workload, retention, and projected retrievability without touching disk.
- After modifying code, run `uv run ruff format` to format the codebase.

## Coding Style & Naming Conventions
Target Python 3.10+, keep modules dependency-light, and favor dataclasses with `slots=True` plus explicit type hints (see `simulator/core.py`). Use four-space indentation, Black/PEP 8 defaults for formatting, and concise docstrings describing expectations between environment, behavior, and scheduler layers. Modules and functions stay `snake_case`, classes use `PascalCase`, and tests mirror the production filename (`schedulers/fsrs6_scheduler.py` → `tests/test_fsrs6_scheduler.py`). Prefer pure functions, pass deterministic RNG callables, and route logging through `SimulationStats` rather than ad-hoc prints.

## Testing Guidelines
The suite uses the stdlib `unittest` runner. Place new specs in `tests/test_<area>.py`, group behavior inside `unittest.TestCase` subclasses, and stub RNGs so tests remain reproducible. When validating schedulers or behaviors, cover both action selection and ready-queue ordering, and capture deferrals or skipped days. There is no formal coverage gate, but every feature change should add or adapt tests plus a note describing intentional metric shifts.

## Commit & Pull Request Guidelines
History favors concise prefixes such as `Visualizer/core: add projected retention stats to logs` or `Core: richer event serialization`; follow `Area: summary` with ≤72 characters. Each commit must keep `uv run simulate.py` runnable and include related tests/docs. Pull requests should describe the scenario being simulated, list the commands you ran, link the driving issue, and attach updated plots or log snippets whenever visualization or logging output changes. Call out scheduler/API adjustments that downstream agents must adopt.

## Security & Configuration Tips
No secrets are needed; configuration is driven via CLI flags. Always pass explicit `--seed`/RNG seeds when sharing repro steps, keep large log dumps in `logs/` but out of git, and guard optional visualization dependencies with clear import errors. Document any new third-party packages in `README.md` to preserve the dependency-light promise.
