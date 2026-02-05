# AGENTS.md

Project rules:

1. Use `uv` to manage the project environment.
2. When code changes affect README examples, update `README.md`.
3. Before each commit, run `uv run ruff format`.
4. Commit messages must follow Conventional Commits.
5. Do not push code to remotes.
6. For performance-related changes, run a baseline performance test first and report results by engine (event vs vectorized) affected by the change.
7. Use `uv run pyright` for static type checking.

## Conventional Commits

The commit message should be structured as follows:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

The commit contains the following structural elements, to communicate intent to the consumers of your library:

1. **fix:** a commit of the *type* `fix` patches a bug in your codebase (this correlates with [`PATCH`](http://semver.org/#summary) in Semantic Versioning).
2. **feat:** a commit of the *type* `feat` introduces a new feature to the codebase (this correlates with [`MINOR`](http://semver.org/#summary) in Semantic Versioning).
3. **BREAKING CHANGE:** a commit that has a footer `BREAKING CHANGE:`, or appends a `!` after the type/scope, introduces a breaking API change (correlating with [`MAJOR`](http://semver.org/#summary) in Semantic Versioning). A BREAKING CHANGE can be part of commits of any *type*.
4. *types* other than `fix:` and `feat:` are allowed, for example [@commitlint/config-conventional](https://github.com/conventional-changelog/commitlint/tree/master/@commitlint/config-conventional) (based on the [Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines)) recommends `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, and others.
5. *footers* other than `BREAKING CHANGE: <description>` may be provided and follow a convention similar to [git trailer format](https://git-scm.com/docs/git-interpret-trailers).

Additional types are not mandated by the Conventional Commits specification, and have no implicit effect in Semantic Versioning (unless they include a BREAKING CHANGE). A scope may be provided to a commit’s type, to provide additional contextual information and is contained within parenthesis, e.g., `feat(parser): add ability to parse arrays`.

## Project Structure & Module Organization
Core simulator types and event plumbing live in `simulator/core.py`, while `simulator/behavior.py`, `simulator/cost.py`, `simulator/models/`, and `simulator/schedulers/` host pluggable user, workload, environment, and scheduler implementations. The CLI entry point is `simulate.py` for Matplotlib dashboards that write JSON logs into `logs/`.

## Architecture Style & Responsibilities
Overall style: layered, plug-in architecture with explicit separation of concerns and dual simulation engines (event-driven + vectorized).

### Responsibilities (by area)
- **Core abstractions & event engine** (`simulator/core.py`):
  - Defines `Card`, `CardView`, `SimulationStats`, and abstract interfaces (`MemoryModel`, `Scheduler`, `BehaviorModel`, `CostModel`).
  - Implements the event-driven `SimulationEngine` and `simulate`.
- **Environment / Memory models** (`simulator/models/`):
  - FSRS3/FSRS6/LSTM/HLR/DASH models implement `MemoryModel`.
  - Each model may host its own vectorized env ops (e.g., `FSRS6VectorizedEnvOps`, `LSTMVectorizedEnvOps`).
- **Schedulers** (`simulator/schedulers/`):
  - Scheduler implementations (FSRS3/FSRS6/HLR/DASH/Fixed/AnkiSM2/Memrise/SSPMMC).
  - Each scheduler may host vectorized scheduler ops (e.g., `FSRS6VectorizedSchedulerOps`).
- **Vectorized engine package** (`simulator/vectorized/`):
  - `engine.py`: unified vectorized loop (`simulate`).
  - `types.py`: Protocols + `VectorizedConfig`.
  - `registry.py`: resolves env/scheduler ops for a given instance.
  - `math.py`: shared vectorized math utilities (FSRS/SSPMMC helpers).
- **CLI + experiments**:
  - `simulate.py` is the CLI entry point; `experiments/` hosts sweep and plotting scripts.

### Architecture Notes / Guardrails
- **Dual engines**: event-driven engine is the reference; vectorized engine trades per-event logging for speed.
- **Explicit separation**: environments update memory state; schedulers update scheduling state; behavior/cost are independent.
- **Vectorized integration**: new models/schedulers should provide corresponding vectorized ops to hook into the shared vectorized engine. Avoid hardcoding logic in `simulator/vectorized/engine.py`.
- **Priority plumbing**: scheduler priority hints live in `Card.metadata["scheduler_priority"]` and are consumed by behavior priority rules (review-first/new-first).

## Security & Configuration Tips
No secrets are needed; configuration is driven via CLI flags. Always pass explicit `--seed`/RNG seeds when sharing repro steps, keep large log dumps in `logs/` but out of git, and guard optional visualization dependencies with clear import errors. Document any new third-party packages in `README.md` to preserve the dependency-light promise.
