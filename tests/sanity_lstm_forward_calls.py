from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.behavior import StochasticBehavior
from simulator.benchmark_loader import load_benchmark_weights
from simulator.cost import StatefulCostModel
from simulator.models.lstm import LSTMModel
from simulator.schedulers import FSRS6Scheduler
from simulator.vectorized import simulate as simulate_vectorized

DEFAULT_DAYS = 365
DEFAULT_DECK_SIZE = 1000
DEFAULT_DEVICE = "cuda"


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    weights = load_benchmark_weights(
        repo_root=repo_root,
        benchmark_root=None,
        environment="fsrs6",
        user_id=1,
        partition_key="0",
        overrides=None,
    )

    scheduler = FSRS6Scheduler(
        weights=weights, desired_retention=0.9, priority_mode="low_retrievability"
    )
    env = LSTMModel(user_id=1, benchmark_root=None, device=DEFAULT_DEVICE)
    behavior = StochasticBehavior()
    cost_model = StatefulCostModel()

    env.reset_forward_calls()
    stats = simulate_vectorized(
        days=DEFAULT_DAYS,
        deck_size=DEFAULT_DECK_SIZE,
        environment=env,
        scheduler=scheduler,
        behavior=behavior,
        cost_model=cost_model,
        seed=0,
        device=DEFAULT_DEVICE,
        progress=False,
    )

    print("days", DEFAULT_DAYS)
    print("deck_size", DEFAULT_DECK_SIZE)
    print("total_reviews", stats.total_reviews)
    print("forward_calls", env.forward_calls)
    print("forward_calls_per_day", env.forward_calls / DEFAULT_DAYS)


if __name__ == "__main__":
    main()
