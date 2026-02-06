from __future__ import annotations

from pathlib import Path


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def resolve_sspmmc_policy_paths(
    *,
    repo_root: Path,
    user_id: int,
    run_sspmmc: bool,
    sspmmc_policies: str | None,
    sspmmc_policy: Path | None,
    sspmmc_policy_dir: Path | None,
    sspmmc_policy_glob: str,
    sspmmc_max: int | None,
) -> list[Path]:
    """Resolve SSP-MMC policy metadata JSON paths.

    Kept consistent across run_sweep.py and run_sweep_users.py. The default
    directory is discovered only when SSP-MMC is part of the sweep.
    """
    if sspmmc_policies:
        paths = [Path(path) for path in _parse_csv(sspmmc_policies)]
        return [path.resolve() for path in paths]

    if sspmmc_policy:
        return [sspmmc_policy.resolve()]

    policy_dir = sspmmc_policy_dir
    if policy_dir is None and run_sspmmc:
        candidate = (
            repo_root.parent
            / "SSP-MMC-FSRS"
            / "outputs"
            / "policies"
            / f"user_{user_id}"
        )
        if candidate.exists():
            policy_dir = candidate

    if policy_dir is None:
        return []

    policy_dir = policy_dir.resolve()
    paths = sorted(policy_dir.glob(sspmmc_policy_glob))
    if sspmmc_max is not None:
        paths = paths[:sspmmc_max]
    return paths
