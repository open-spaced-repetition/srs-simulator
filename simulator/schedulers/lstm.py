from __future__ import annotations

from dataclasses import dataclass
import math
import os
import sys
import torch
from pathlib import Path
from typing import Sequence

from simulator.core import CardView, Scheduler
from simulator.models.lstm import EPS, LSTMModel
from simulator.models.lstm_batch import PackedLSTMWeights, forward_step


def _resolve_max_batch_size(value: int | None) -> int | None:
    if value is not None:
        if value < 1:
            raise ValueError("max_batch_size must be >= 1 when set.")
        return value
    raw = os.getenv("SRS_LSTM_MAX_BATCH", "").strip().lower()
    if not raw or raw in {"0", "none", "off"}:
        return None
    parsed = int(raw)
    if parsed < 1:
        raise ValueError("SRS_LSTM_MAX_BATCH must be >= 1 when set.")
    return parsed


@dataclass
class LSTMSchedulerState:
    curves: dict[str, list[float]]


class LSTMScheduler(Scheduler):
    """
    Scheduler that targets desired retention using the LSTM forgetting curves.

    It rebuilds LSTM features from the review history (including the latest rating)
    and searches for the interval whose predicted retention matches desired_retention.
    """

    def __init__(
        self,
        *,
        weights_path: str | Path | None = None,
        user_id: int | None = None,
        benchmark_root: str | Path | None = None,
        short_term: bool = False,
        desired_retention: float = 0.9,
        use_duration_feature: bool = False,
        default_duration_ms: float = 2500.0,
        min_interval: float = 1.0,
        max_interval: float = 3650.0,
        search_steps: int = 24,
        interval_mode: str = "integer",
        device: str | None = None,
    ) -> None:
        if desired_retention <= 0.0 or desired_retention >= 1.0:
            raise ValueError("desired_retention must be in (0, 1).")
        self.desired_retention = float(desired_retention)
        if interval_mode not in {"integer", "float"}:
            raise ValueError("interval_mode must be 'integer' or 'float'.")
        self.interval_mode = interval_mode
        self.min_interval = float(min_interval)
        self.max_interval = float(max_interval)
        self.search_steps = int(search_steps)
        if self.interval_mode == "integer":
            if self.min_interval <= 0.0:
                raise ValueError("min_interval must be > 0.")
        else:
            if self.min_interval < 0.0:
                raise ValueError("min_interval must be >= 0.")
        if self.max_interval < self.min_interval:
            raise ValueError("max_interval must be >= min_interval.")

        self.model = LSTMModel(
            weights_path=weights_path,
            user_id=user_id,
            benchmark_root=benchmark_root,
            short_term=short_term,
            use_duration_feature=use_duration_feature,
            default_duration_ms=default_duration_ms,
            device=device,
        )
        self._debug_interval = bool(os.getenv("SRS_DEBUG_LSTM_INTERVAL"))
        self._debug_interval_limit = int(
            os.getenv("SRS_DEBUG_LSTM_INTERVAL_LIMIT", "50")
        )
        self._debug_interval_calls = 0

    def init_card(self, card_view: CardView, rating: int, day: float):
        curves = self._curves_with_event(card_view, rating, elapsed=0.0)
        interval = self._target_interval(curves, card_view)
        return interval, LSTMSchedulerState(curves=curves)

    def schedule(self, card_view: CardView, rating: int, elapsed: float, day: float):
        curves = self._curves_with_event(card_view, rating, elapsed)
        interval = self._target_interval(curves, card_view)
        return interval, LSTMSchedulerState(curves=curves)

    def _curves_with_event(
        self, view: CardView, rating: int, elapsed: float
    ) -> dict[str, list[float]]:
        events = [(log.elapsed, log.rating) for log in view.history]
        events.append((elapsed, int(rating)))
        return self.model.curves_from_events(events)

    def _target_interval(self, curves: dict[str, list[float]], view: CardView) -> float:
        if not curves:
            return self.min_interval
        target = self.desired_retention
        if self.interval_mode != "integer":
            return self._target_interval_float(curves, view, target)
        min_int = max(1, int(math.ceil(self.min_interval)))
        max_int = max(min_int, int(math.floor(self.max_interval)))
        low = min_int
        high = max(low, int(math.ceil(float(view.interval or low))))

        r_low = self.model.predict_retention_from_curves(curves, float(low))
        if r_low <= target:
            return float(low)

        t0 = self._effective_initial_guess(curves, target)
        if t0 is not None and math.isfinite(t0):
            high = max(high, int(math.ceil(t0)))
        high = min(max_int, high)

        while (
            high < max_int
            and self.model.predict_retention_from_curves(curves, float(high)) > target
        ):
            high = min(max_int, high * 2)

        for _ in range(self.search_steps):
            if low >= high:
                break
            mid = (low + high) // 2
            pred = self.model.predict_retention_from_curves(curves, float(mid))
            if pred > target:
                low = mid + 1
            else:
                high = mid
        result = float(max(min_int, min(high, max_int)))
        if (
            self._debug_interval
            and self._debug_interval_calls < self._debug_interval_limit
        ):
            self._debug_interval_calls += 1
            pred = self.model.predict_retention_from_curves(curves, result)
            sys.stderr.write(
                "[lstm interval] mode=integer card_id="
                f"{view.id} reps={view.reps} target={target:.6f} "
                f"min={min_int} max={max_int} low={low} high={high} "
                f"result={result:.6f} pred={pred:.6f}\n"
            )
        return result

    def _target_interval_float(
        self,
        curves: dict[str, list[float]],
        view: CardView,
        target: float,
    ) -> float:
        low = max(0.0, self.min_interval)
        high = max(low, float(view.interval or low))
        if high <= 0.0:
            high = max(self.max_interval, 1e-6) if self.max_interval > 0 else 1e-6

        r_low = self.model.predict_retention_from_curves(curves, low)
        if r_low <= target:
            return low

        t0 = self._effective_initial_guess(curves, target)
        if t0 is not None and math.isfinite(t0):
            high = max(high, t0)
        high = min(self.max_interval, high)

        r_high = self.model.predict_retention_from_curves(curves, high)
        while high < self.max_interval and r_high > target:
            high = min(self.max_interval, max(high * 2.0, high + 1e-6))
            r_high = self.model.predict_retention_from_curves(curves, high)

        if r_high > target:
            result = self.max_interval
        else:
            f_low = r_low - target
            f_high = r_high - target
            result = None
            for _ in range(self.search_steps):
                if low >= high:
                    break
                denom = f_high - f_low
                if abs(denom) < 1e-12 or not math.isfinite(denom):
                    guess = (low + high) / 2.0
                else:
                    guess = high - f_high * (high - low) / denom
                if not (low < guess < high) or not math.isfinite(guess):
                    guess = (low + high) / 2.0
                pred = self.model.predict_retention_from_curves(curves, guess)
                f_guess = pred - target
                if abs(f_guess) <= 0.001:
                    result = guess
                    break
                if f_guess > 0.0:
                    low = guess
                    f_low = f_guess
                    f_high *= 0.5
                else:
                    high = guess
                    f_high = f_guess
                    f_low *= 0.5

            if result is None:
                result = high

        result = max(self.min_interval, min(result, self.max_interval))
        if (
            self._debug_interval
            and self._debug_interval_calls < self._debug_interval_limit
        ):
            self._debug_interval_calls += 1
            pred = self.model.predict_retention_from_curves(curves, result)
            sys.stderr.write(
                "[lstm interval] mode=float card_id="
                f"{view.id} reps={view.reps} target={target:.6f} "
                f"min={self.min_interval:.6f} max={self.max_interval:.6f} "
                f"low={low:.6f} high={high:.6f} result={result:.6f} "
                f"pred={pred:.6f}\n"
            )
        return result

    def _retention_and_derivative(
        self, curves: dict[str, list[float]], days: float
    ) -> tuple[float, float]:
        weights = curves.get("w") or []
        stabilities = curves.get("s") or []
        decays = curves.get("d") or []
        if not (weights and stabilities and decays):
            return self.model.default_retention, 0.0
        total = 0.0
        deriv = 0.0
        for w, s, d in zip(weights, stabilities, decays):
            denom = s + EPS
            x = 1.0 + days / denom
            x_pow = x ** (-d)
            total += w * x_pow
            deriv += w * (-d / denom) * x_pow / x
        total = max(0.0, min(1.0, total))
        return (1.0 - EPS) * total, (1.0 - EPS) * deriv

    def _effective_initial_guess(
        self, curves: dict[str, list[float]], target: float
    ) -> float | None:
        weights = curves.get("w") or []
        stabilities = curves.get("s") or []
        decays = curves.get("d") or []
        if not (weights and stabilities and decays):
            return None
        a = 0.0
        b = 0.0
        for w, s, d in zip(weights, stabilities, decays):
            denom = s + EPS
            a += w * d / denom
            b += w * d * (d + 1.0) / (denom * denom)
        if a <= 0.0:
            return None
        ratio = b / (a * a)
        ratio = max(ratio, 1.0 + 1e-6)
        d_eff = 1.0 / (ratio - 1.0)
        s_eff = d_eff / a
        base = target ** (-1.0 / d_eff) - 1.0
        t0 = s_eff * base
        if not math.isfinite(t0):
            return None
        return max(self.min_interval, min(self.max_interval, t0))

    def review_priority(self, card_view: CardView, day: float) -> Sequence[float]:
        state = card_view.scheduler_state
        if isinstance(state, LSTMSchedulerState):
            elapsed = max(0.0, float(day) - card_view.last_review)
            r = self.model.predict_retention_from_curves(state.curves, elapsed)
            return (r, card_view.due, card_view.id)
        return super().review_priority(card_view, day)


@dataclass
class LSTMVectorizedSchedulerState:
    lstm_h: "torch.Tensor"
    lstm_c: "torch.Tensor"
    mem_w: "torch.Tensor"
    mem_s: "torch.Tensor"
    mem_d: "torch.Tensor"
    has_curves: "torch.Tensor"


class LSTMVectorizedSchedulerOps:
    def __init__(
        self,
        scheduler: LSTMScheduler,
        *,
        device: "torch.device",
        dtype: "torch.dtype",
        max_batch_size: int | None = None,
    ) -> None:
        import torch

        if scheduler.interval_mode not in {"integer", "float"}:
            raise ValueError("interval_mode must be 'integer' or 'float'.")

        self._torch = torch
        self.device = device
        self.dtype = dtype
        self.scheduler = scheduler
        self.interval_mode = scheduler.interval_mode
        self.model = scheduler.model
        self.desired_retention = float(scheduler.desired_retention)
        self.min_interval = float(scheduler.min_interval)
        self.max_interval = float(scheduler.max_interval)
        self.search_steps = int(scheduler.search_steps)
        self.max_batch_size = _resolve_max_batch_size(max_batch_size)

        self.model.device = device
        self.model.dtype = dtype
        self.model.network.to(device=device, dtype=dtype)

        self.n_rnns = int(self.model.network.n_rnns)
        self.n_hidden = int(self.model.network.n_hidden)
        self.n_curves = int(self.model.network.n_curves)
        self.default_retention = float(self.model.default_retention)
        self.use_duration_feature = self.model.use_duration_feature
        self.duration_value = None
        if self.use_duration_feature:
            self.duration_value = torch.tensor(
                self.model.default_duration_ms, device=device, dtype=dtype
            )
        self._target = torch.tensor(self.desired_retention, device=device, dtype=dtype)
        self._min_interval = torch.tensor(self.min_interval, device=device, dtype=dtype)
        self._max_interval = torch.tensor(self.max_interval, device=device, dtype=dtype)

    def init_state(self, deck_size: int) -> LSTMVectorizedSchedulerState:
        lstm_h = self._torch.zeros(
            (self.n_rnns, deck_size, self.n_hidden),
            dtype=self.dtype,
            device=self.device,
        )
        lstm_c = self._torch.zeros_like(lstm_h)
        mem_w = self._torch.zeros(
            (deck_size, self.n_curves), dtype=self.dtype, device=self.device
        )
        mem_s = self._torch.zeros(
            (deck_size, self.n_curves), dtype=self.dtype, device=self.device
        )
        mem_d = self._torch.zeros(
            (deck_size, self.n_curves), dtype=self.dtype, device=self.device
        )
        has_curves = self._torch.zeros(
            deck_size, dtype=self._torch.bool, device=self.device
        )
        return LSTMVectorizedSchedulerState(
            lstm_h=lstm_h,
            lstm_c=lstm_c,
            mem_w=mem_w,
            mem_s=mem_s,
            mem_d=mem_d,
            has_curves=has_curves,
        )

    def review_priority(
        self,
        state: LSTMVectorizedSchedulerState,
        idx: "torch.Tensor",
        elapsed: "torch.Tensor",
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        elapsed_clamped = self._torch.clamp(elapsed, min=0.0)
        scores = self._torch.full(
            elapsed_clamped.shape,
            self.default_retention,
            dtype=self.dtype,
            device=self.device,
        )
        curves_mask = state.has_curves[idx]
        if curves_mask.any():
            curves_idx = idx[curves_mask]
            scores[curves_mask] = self._lstm_retention(
                elapsed_clamped[curves_mask],
                state.mem_w[curves_idx],
                state.mem_s[curves_idx],
                state.mem_d[curves_idx],
            )
        return scores

    def update_review(
        self,
        state: LSTMVectorizedSchedulerState,
        idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        self._update_curves(state, idx, elapsed, rating)
        return self._target_interval(state, idx, prev_interval)

    def update_learn(
        self,
        state: LSTMVectorizedSchedulerState,
        idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        if idx.numel() == 0:
            return self._torch.zeros(0, device=self.device, dtype=self.dtype)
        elapsed = self._torch.zeros(idx.shape[0], device=self.device, dtype=self.dtype)
        self._update_curves(state, idx, elapsed, rating)
        return self._target_interval(state, idx, None)

    def _update_curves(
        self,
        state: LSTMVectorizedSchedulerState,
        idx: "torch.Tensor",
        delays: "torch.Tensor",
        ratings: "torch.Tensor",
    ) -> None:
        def _update_chunk(
            idx_chunk: "torch.Tensor",
            delays_chunk: "torch.Tensor",
            ratings_chunk: "torch.Tensor",
        ) -> None:
            delay_clamped = self._torch.clamp(delays_chunk, min=0.0)
            rating_clamped = self._torch.clamp(ratings_chunk, min=1, max=4).to(
                self.dtype
            )
            delay_feature = delay_clamped.unsqueeze(-1)
            rating_feature = rating_clamped.unsqueeze(-1)
            if self.use_duration_feature and self.duration_value is not None:
                duration_feature = self.duration_value.expand_as(delay_feature)
                step = self._torch.cat(
                    [delay_feature, duration_feature, rating_feature], dim=-1
                )
            else:
                step = self._torch.cat([delay_feature, rating_feature], dim=-1)

            h = state.lstm_h[:, idx_chunk, :]
            c = state.lstm_c[:, idx_chunk, :]
            w_last, s_last, d_last, (h_new, c_new) = self.model.network.forward_step(
                step, (h, c)
            )
            state.lstm_h[:, idx_chunk, :] = h_new
            state.lstm_c[:, idx_chunk, :] = c_new
            state.mem_w[idx_chunk] = w_last
            state.mem_s[idx_chunk] = s_last
            state.mem_d[idx_chunk] = d_last
            state.has_curves[idx_chunk] = True

        total = int(idx.numel())
        if self.max_batch_size is None or total <= self.max_batch_size:
            _update_chunk(idx, delays, ratings)
            return
        for start in range(0, total, self.max_batch_size):
            end = start + self.max_batch_size
            _update_chunk(idx[start:end], delays[start:end], ratings[start:end])

    def _target_interval(
        self,
        state: LSTMVectorizedSchedulerState,
        idx: "torch.Tensor",
        prev_interval: "torch.Tensor | None",
    ) -> "torch.Tensor":
        if self.interval_mode == "float":
            return self._target_interval_float(state, idx, prev_interval)
        return self._target_interval_integer(state, idx, prev_interval)

    def _target_interval_integer(
        self,
        state: LSTMVectorizedSchedulerState,
        idx: "torch.Tensor",
        prev_interval: "torch.Tensor | None",
    ) -> "torch.Tensor":
        weights = state.mem_w[idx]
        stabilities = state.mem_s[idx]
        decays = state.mem_d[idx]
        min_int = max(1, int(math.ceil(self.min_interval)))
        max_int = max(min_int, int(math.floor(self.max_interval)))
        low = self._torch.full(
            (idx.numel(),), min_int, dtype=self._torch.int64, device=self.device
        )
        if prev_interval is None:
            high = low.clone()
        else:
            high = self._torch.maximum(
                low,
                self._torch.ceil(prev_interval.to(self.dtype)).to(self._torch.int64),
            )
        t0 = self._effective_initial_guess(weights, stabilities, decays)
        high = self._torch.maximum(high, self._torch.ceil(t0).to(self._torch.int64))
        high = self._torch.clamp(high, max=max_int)

        pred_low = self._retention_at(low.to(self.dtype), weights, stabilities, decays)
        active = pred_low > self._target
        if active.any():
            max_tensor = self._torch.full_like(high, max_int)
            while True:
                pred_high = self._retention_at(
                    high.to(self.dtype), weights, stabilities, decays
                )
                still = (pred_high > self._target) & (high < max_tensor) & active
                if not still.any():
                    break
                high = self._torch.where(
                    still, self._torch.minimum(high * 2, max_tensor), high
                )

            for _ in range(self.search_steps):
                if not (active & (low < high)).any():
                    break
                mid = (low + high) // 2
                pred_mid = self._retention_at(
                    mid.to(self.dtype), weights, stabilities, decays
                )
                go_high = pred_mid <= self._target
                high = self._torch.where(active & go_high, mid, high)
                low = self._torch.where(active & ~go_high, mid + 1, low)

        result = self._torch.where(active, high, low).to(self.dtype)
        return self._torch.clamp(result, min=self.min_interval, max=self.max_interval)

    def _target_interval_float(
        self,
        state: LSTMVectorizedSchedulerState,
        idx: "torch.Tensor",
        prev_interval: "torch.Tensor | None",
    ) -> "torch.Tensor":
        weights = state.mem_w[idx]
        stabilities = state.mem_s[idx]
        decays = state.mem_d[idx]
        low = self._torch.full(
            (idx.numel(),),
            max(0.0, self.min_interval),
            dtype=self.dtype,
            device=self.device,
        )
        if prev_interval is None:
            high = low.clone()
        else:
            high = self._torch.maximum(low, prev_interval.to(self.dtype))
        high = self._torch.where(
            high <= 0.0,
            self._torch.full_like(
                high, max(self.max_interval, 1e-6), dtype=self.dtype, device=self.device
            ),
            high,
        )
        t0 = self._effective_initial_guess(weights, stabilities, decays)
        high = self._torch.maximum(high, t0)
        high = self._torch.clamp(high, max=self.max_interval)

        r_low = self._retention_at(low, weights, stabilities, decays)
        result = self._torch.full_like(low, float("nan"))
        done_low = r_low <= self._target
        result = self._torch.where(done_low, low, result)

        r_high = self._retention_at(high, weights, stabilities, decays)
        active = (~done_low) & (r_high > self._target) & (high < self._max_interval)
        while active.any():
            high = self._torch.where(
                active,
                self._torch.minimum(
                    self._max_interval, self._torch.maximum(high * 2.0, high + 1e-6)
                ),
                high,
            )
            r_high = self._torch.where(
                active, self._retention_at(high, weights, stabilities, decays), r_high
            )
            active = (~done_low) & (r_high > self._target) & (high < self._max_interval)

        done_high = (~done_low) & (r_high > self._target)
        result = self._torch.where(done_high, self._max_interval, result)

        active = (~done_low) & ~done_high
        f_low = r_low - self._target
        f_high = r_high - self._target
        tol = 1e-3
        for _ in range(self.search_steps):
            active_mask = active & (low < high)
            if not active_mask.any():
                break
            denom = f_high - f_low
            guess = high - f_high * (high - low) / denom
            invalid = (
                ~self._torch.isfinite(guess)
                | (guess <= low)
                | (guess >= high)
                | (self._torch.abs(denom) < 1e-12)
            )
            guess = self._torch.where(invalid, 0.5 * (low + high), guess)
            pred = self._retention_at(guess, weights, stabilities, decays)
            f_guess = pred - self._target
            done = active_mask & (self._torch.abs(f_guess) <= tol)
            result = self._torch.where(done, guess, result)

            upper_mask = active_mask & (f_guess <= 0.0)
            lower_mask = active_mask & (f_guess > 0.0)
            high = self._torch.where(upper_mask, guess, high)
            f_high = self._torch.where(upper_mask, f_guess, f_high)
            f_low = self._torch.where(upper_mask, f_low * 0.5, f_low)

            low = self._torch.where(lower_mask, guess, low)
            f_low = self._torch.where(lower_mask, f_guess, f_low)
            f_high = self._torch.where(lower_mask, f_high * 0.5, f_high)

            active = active & ~done

        result = self._torch.where(self._torch.isnan(result), high, result)
        return self._torch.clamp(result, min=self.min_interval, max=self.max_interval)

    def _retention_at(
        self,
        days: "torch.Tensor",
        weights: "torch.Tensor",
        stabilities: "torch.Tensor",
        decays: "torch.Tensor",
    ) -> "torch.Tensor":
        elapsed_clamped = self._torch.clamp(days, min=0.0)
        return self._lstm_retention(elapsed_clamped, weights, stabilities, decays)

    def _retention_and_derivative(
        self,
        days: "torch.Tensor",
        weights: "torch.Tensor",
        stabilities: "torch.Tensor",
        decays: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        elapsed_clamped = self._torch.clamp(days, min=0.0)
        denom = stabilities + EPS
        x = 1.0 + elapsed_clamped.unsqueeze(-1) / denom
        x_pow = x ** (-decays)
        total = self._torch.sum(weights * x_pow, dim=-1)
        deriv = self._torch.sum(weights * (-decays / denom) * x_pow / x, dim=-1)
        total = self._torch.clamp(total, min=0.0, max=1.0)
        return (1.0 - EPS) * total, (1.0 - EPS) * deriv

    def _effective_initial_guess(
        self,
        weights: "torch.Tensor",
        stabilities: "torch.Tensor",
        decays: "torch.Tensor",
    ) -> "torch.Tensor":
        denom = stabilities + EPS
        a = self._torch.sum(weights * decays / denom, dim=-1)
        b = self._torch.sum(weights * decays * (decays + 1.0) / (denom * denom), dim=-1)
        a = self._torch.clamp(a, min=EPS)
        ratio = b / (a * a)
        ratio = self._torch.clamp(ratio, min=1.0 + 1e-6)
        d_eff = 1.0 / (ratio - 1.0)
        s_eff = d_eff / a
        base = self._torch.pow(self._target, -1.0 / d_eff) - 1.0
        t0 = s_eff * base
        return self._torch.clamp(t0, min=self.min_interval, max=self.max_interval)

    @staticmethod
    def _lstm_retention(
        elapsed_scaled: "torch.Tensor",
        weights: "torch.Tensor",
        stabilities: "torch.Tensor",
        decays: "torch.Tensor",
    ) -> "torch.Tensor":
        denom = stabilities + EPS
        total = weights * (1.0 + elapsed_scaled.unsqueeze(-1) / denom) ** (-decays)
        total = total.sum(dim=-1)
        total = total.clamp(min=0.0, max=1.0)
        return (1.0 - EPS) * total


@dataclass
class LSTMBatchSchedulerState:
    lstm_h: "torch.Tensor"
    lstm_c: "torch.Tensor"
    mem_w: "torch.Tensor"
    mem_s: "torch.Tensor"
    mem_d: "torch.Tensor"
    has_curves: "torch.Tensor"


class LSTMBatchSchedulerOps:
    def __init__(
        self,
        weights: PackedLSTMWeights,
        *,
        desired_retention: float,
        min_interval: float = 1.0,
        max_interval: float = 3650.0,
        search_steps: int = 24,
        default_retention: float = 0.85,
        default_duration_ms: float = 2500.0,
        interval_mode: str = "integer",
        max_batch_size: int | None = None,
        device: "torch.device",
        dtype: "torch.dtype",
    ) -> None:
        if interval_mode not in {"integer", "float"}:
            raise ValueError("interval_mode must be 'integer' or 'float'.")
        self.weights = weights
        self.device = device
        self.dtype = dtype
        self.desired_retention = float(desired_retention)
        self.min_interval = float(min_interval)
        self.max_interval = float(max_interval)
        self.search_steps = int(search_steps)
        self.default_retention = float(default_retention)
        self.interval_mode = interval_mode
        self.max_batch_size = _resolve_max_batch_size(max_batch_size)
        self.use_duration_feature = bool(weights.use_duration_feature)
        self.duration_value = None
        if self.use_duration_feature:
            self.duration_value = torch.tensor(
                default_duration_ms, device=device, dtype=dtype
            )
        self._target = torch.tensor(self.desired_retention, device=device, dtype=dtype)
        self._min_interval = torch.tensor(self.min_interval, device=device, dtype=dtype)
        self._max_interval = torch.tensor(self.max_interval, device=device, dtype=dtype)

    def init_state(self, user_count: int, deck_size: int) -> LSTMBatchSchedulerState:
        lstm_h = torch.zeros(
            (self.weights.n_rnns, user_count, deck_size, self.weights.n_hidden),
            dtype=self.dtype,
            device=self.device,
        )
        lstm_c = torch.zeros_like(lstm_h)
        mem_w = torch.zeros(
            (user_count, deck_size, self.weights.n_curves),
            dtype=self.dtype,
            device=self.device,
        )
        mem_s = torch.zeros_like(mem_w)
        mem_d = torch.zeros_like(mem_w)
        has_curves = torch.zeros(
            (user_count, deck_size), dtype=torch.bool, device=self.device
        )
        return LSTMBatchSchedulerState(
            lstm_h=lstm_h,
            lstm_c=lstm_c,
            mem_w=mem_w,
            mem_s=mem_s,
            mem_d=mem_d,
            has_curves=has_curves,
        )

    def review_priority(
        self, state: LSTMBatchSchedulerState, elapsed: "torch.Tensor"
    ) -> "torch.Tensor":
        elapsed_clamped = torch.clamp(elapsed, min=0.0)
        scores = torch.full(
            elapsed_clamped.shape,
            self.default_retention,
            dtype=self.dtype,
            device=self.device,
        )
        curves_mask = state.has_curves
        if curves_mask.any():
            scores[curves_mask] = self._lstm_retention(
                elapsed_clamped[curves_mask],
                state.mem_w[curves_mask],
                state.mem_s[curves_mask],
                state.mem_d[curves_mask],
            )
        return scores

    def update_review(
        self,
        state: LSTMBatchSchedulerState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
        elapsed: "torch.Tensor",
        rating: "torch.Tensor",
        prev_interval: "torch.Tensor",
    ) -> "torch.Tensor":
        if user_idx.numel() == 0:
            return torch.zeros(0, device=self.device, dtype=self.dtype)
        self._update_curves(state, user_idx, card_idx, elapsed, rating)
        return self._target_interval(state, user_idx, card_idx, prev_interval)

    def update_learn(
        self,
        state: LSTMBatchSchedulerState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
        rating: "torch.Tensor",
    ) -> "torch.Tensor":
        if user_idx.numel() == 0:
            return torch.zeros(0, device=self.device, dtype=self.dtype)
        elapsed = torch.zeros(rating.shape[0], device=self.device, dtype=self.dtype)
        self._update_curves(state, user_idx, card_idx, elapsed, rating)
        return self._target_interval(state, user_idx, card_idx, None)

    def _update_curves(
        self,
        state: LSTMBatchSchedulerState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
        delays: "torch.Tensor",
        ratings: "torch.Tensor",
    ) -> None:
        def _update_chunk(
            chunk_user: "torch.Tensor",
            chunk_card: "torch.Tensor",
            chunk_delays: "torch.Tensor",
            chunk_ratings: "torch.Tensor",
        ) -> None:
            delay_clamped = torch.clamp(chunk_delays, min=0.0)
            rating_clamped = torch.clamp(chunk_ratings, min=1, max=4).to(self.dtype)
            delay_feature = delay_clamped.unsqueeze(-1)
            rating_feature = rating_clamped.unsqueeze(-1)
            if self.use_duration_feature and self.duration_value is not None:
                duration_feature = self.duration_value.expand_as(delay_feature)
                step = torch.cat(
                    [delay_feature, duration_feature, rating_feature], dim=-1
                )
            else:
                step = torch.cat([delay_feature, rating_feature], dim=-1)

            h = state.lstm_h[:, chunk_user, chunk_card, :]
            c = state.lstm_c[:, chunk_user, chunk_card, :]
            w_last, s_last, d_last, (h_new, c_new) = forward_step(
                self.weights, step, (h, c), chunk_user
            )
            state.lstm_h[:, chunk_user, chunk_card, :] = h_new
            state.lstm_c[:, chunk_user, chunk_card, :] = c_new
            state.mem_w[chunk_user, chunk_card] = w_last
            state.mem_s[chunk_user, chunk_card] = s_last
            state.mem_d[chunk_user, chunk_card] = d_last
            state.has_curves[chunk_user, chunk_card] = True

        total = int(user_idx.numel())
        if self.max_batch_size is None or total <= self.max_batch_size:
            _update_chunk(user_idx, card_idx, delays, ratings)
            return
        for start in range(0, total, self.max_batch_size):
            end = start + self.max_batch_size
            _update_chunk(
                user_idx[start:end],
                card_idx[start:end],
                delays[start:end],
                ratings[start:end],
            )

    def _target_interval(
        self,
        state: LSTMBatchSchedulerState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
        prev_interval: "torch.Tensor | None",
    ) -> "torch.Tensor":
        if self.interval_mode == "float":
            return self._target_interval_float(state, user_idx, card_idx, prev_interval)
        weights = state.mem_w[user_idx, card_idx]
        stabilities = state.mem_s[user_idx, card_idx]
        decays = state.mem_d[user_idx, card_idx]
        min_int = max(1, int(math.ceil(self.min_interval)))
        max_int = max(min_int, int(math.floor(self.max_interval)))
        low = torch.full(
            (user_idx.numel(),),
            min_int,
            dtype=torch.int64,
            device=self.device,
        )
        if prev_interval is None:
            high = low.clone()
        else:
            high = torch.maximum(
                low, torch.ceil(prev_interval.to(self.dtype)).to(torch.int64)
            )
        t0 = self._effective_initial_guess(weights, stabilities, decays)
        high = torch.maximum(high, torch.ceil(t0).to(torch.int64))
        high = torch.clamp(high, max=max_int)

        pred_low = self._retention_at(low.to(self.dtype), weights, stabilities, decays)
        active = pred_low > self._target
        if active.any():
            max_tensor = torch.full_like(high, max_int)
            while True:
                pred_high = self._retention_at(
                    high.to(self.dtype), weights, stabilities, decays
                )
                still = (pred_high > self._target) & (high < max_tensor) & active
                if not still.any():
                    break
                high = torch.where(still, torch.minimum(high * 2, max_tensor), high)

            for _ in range(self.search_steps):
                if not (active & (low < high)).any():
                    break
                mid = (low + high) // 2
                pred_mid = self._retention_at(
                    mid.to(self.dtype), weights, stabilities, decays
                )
                go_high = pred_mid <= self._target
                high = torch.where(active & go_high, mid, high)
                low = torch.where(active & ~go_high, mid + 1, low)

        result = torch.where(active, high, low).to(self.dtype)
        return torch.clamp(result, min=self.min_interval, max=self.max_interval)

    def _target_interval_float(
        self,
        state: LSTMBatchSchedulerState,
        user_idx: "torch.Tensor",
        card_idx: "torch.Tensor",
        prev_interval: "torch.Tensor | None",
    ) -> "torch.Tensor":
        weights = state.mem_w[user_idx, card_idx]
        stabilities = state.mem_s[user_idx, card_idx]
        decays = state.mem_d[user_idx, card_idx]
        low = torch.full(
            (user_idx.numel(),),
            max(0.0, self.min_interval),
            dtype=self.dtype,
            device=self.device,
        )
        if prev_interval is None:
            high = low.clone()
        else:
            high = torch.maximum(low, prev_interval.to(self.dtype))
        high = torch.where(
            high <= 0.0,
            torch.full_like(
                high,
                max(self.max_interval, 1e-6),
                dtype=self.dtype,
                device=self.device,
            ),
            high,
        )
        t0 = self._effective_initial_guess(weights, stabilities, decays)
        high = torch.maximum(high, t0)
        high = torch.clamp(high, max=self.max_interval)

        r_low = self._retention_at(low, weights, stabilities, decays)
        result = torch.full_like(low, float("nan"))
        done_low = r_low <= self._target
        result = torch.where(done_low, low, result)

        r_high = self._retention_at(high, weights, stabilities, decays)
        active = (~done_low) & (r_high > self._target) & (high < self._max_interval)
        while active.any():
            high = torch.where(
                active,
                torch.minimum(
                    self._max_interval, torch.maximum(high * 2.0, high + 1e-6)
                ),
                high,
            )
            r_high = torch.where(
                active, self._retention_at(high, weights, stabilities, decays), r_high
            )
            active = (~done_low) & (r_high > self._target) & (high < self._max_interval)

        done_high = (~done_low) & (r_high > self._target)
        result = torch.where(done_high, self._max_interval, result)

        active = (~done_low) & ~done_high
        f_low = r_low - self._target
        f_high = r_high - self._target
        tol = 1e-3
        for _ in range(self.search_steps):
            active_mask = active & (low < high)
            if not active_mask.any():
                break
            denom = f_high - f_low
            guess = high - f_high * (high - low) / denom
            invalid = (
                ~torch.isfinite(guess)
                | (guess <= low)
                | (guess >= high)
                | (torch.abs(denom) < 1e-12)
            )
            guess = torch.where(invalid, 0.5 * (low + high), guess)
            pred = self._retention_at(guess, weights, stabilities, decays)
            f_guess = pred - self._target
            done = active_mask & (torch.abs(f_guess) <= tol)
            result = torch.where(done, guess, result)

            upper_mask = active_mask & (f_guess <= 0.0)
            lower_mask = active_mask & (f_guess > 0.0)
            high = torch.where(upper_mask, guess, high)
            f_high = torch.where(upper_mask, f_guess, f_high)
            f_low = torch.where(upper_mask, f_low * 0.5, f_low)

            low = torch.where(lower_mask, guess, low)
            f_low = torch.where(lower_mask, f_guess, f_low)
            f_high = torch.where(lower_mask, f_high * 0.5, f_high)

            active = active & ~done

        result = torch.where(torch.isnan(result), high, result)
        return torch.clamp(result, min=self.min_interval, max=self.max_interval)

    def _retention_at(
        self,
        days: "torch.Tensor",
        weights: "torch.Tensor",
        stabilities: "torch.Tensor",
        decays: "torch.Tensor",
    ) -> "torch.Tensor":
        elapsed_clamped = torch.clamp(days, min=0.0)
        return self._lstm_retention(elapsed_clamped, weights, stabilities, decays)

    def _retention_and_derivative(
        self,
        days: "torch.Tensor",
        weights: "torch.Tensor",
        stabilities: "torch.Tensor",
        decays: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        elapsed_clamped = torch.clamp(days, min=0.0)
        denom = stabilities + EPS
        x = 1.0 + elapsed_clamped.unsqueeze(-1) / denom
        x_pow = x ** (-decays)
        total = torch.sum(weights * x_pow, dim=-1)
        deriv = torch.sum(weights * (-decays / denom) * x_pow / x, dim=-1)
        total = torch.clamp(total, min=0.0, max=1.0)
        return (1.0 - EPS) * total, (1.0 - EPS) * deriv

    def _effective_initial_guess(
        self,
        weights: "torch.Tensor",
        stabilities: "torch.Tensor",
        decays: "torch.Tensor",
    ) -> "torch.Tensor":
        denom = stabilities + EPS
        a = torch.sum(weights * decays / denom, dim=-1)
        b = torch.sum(weights * decays * (decays + 1.0) / (denom * denom), dim=-1)
        a = torch.clamp(a, min=EPS)
        ratio = b / (a * a)
        ratio = torch.clamp(ratio, min=1.0 + 1e-6)
        d_eff = 1.0 / (ratio - 1.0)
        s_eff = d_eff / a
        base = torch.pow(self._target, -1.0 / d_eff) - 1.0
        t0 = s_eff * base
        return torch.clamp(t0, min=self.min_interval, max=self.max_interval)

    @staticmethod
    def _lstm_retention(
        elapsed_scaled: "torch.Tensor",
        weights: "torch.Tensor",
        stabilities: "torch.Tensor",
        decays: "torch.Tensor",
    ) -> "torch.Tensor":
        denom = stabilities + EPS
        total = weights * torch.pow(1.0 + elapsed_scaled.unsqueeze(-1) / denom, -decays)
        total = total.sum(dim=-1)
        total = total.clamp(min=0.0, max=1.0)
        return (1.0 - EPS) * total
