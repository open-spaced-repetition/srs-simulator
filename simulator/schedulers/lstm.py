from __future__ import annotations

from dataclasses import dataclass
import math
import torch
from pathlib import Path
from typing import Sequence

from simulator.core import CardView, Scheduler
from simulator.models.lstm import EPS, LSTMModel


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
        desired_retention: float = 0.9,
        use_duration_feature: bool = False,
        default_duration_ms: float = 2500.0,
        interval_scale: float = 1.0,
        min_interval: float = 1.0,
        max_interval: float = 3650.0,
        search_steps: int = 24,
        device: str | None = None,
    ) -> None:
        if desired_retention <= 0.0 or desired_retention >= 1.0:
            raise ValueError("desired_retention must be in (0, 1).")
        self.desired_retention = float(desired_retention)
        self.min_interval = float(min_interval)
        self.max_interval = float(max_interval)
        self.search_steps = int(search_steps)
        if self.min_interval <= 0.0:
            raise ValueError("min_interval must be > 0.")
        if self.max_interval < self.min_interval:
            raise ValueError("max_interval must be >= min_interval.")

        self.model = LSTMModel(
            weights_path=weights_path,
            user_id=user_id,
            benchmark_root=benchmark_root,
            use_duration_feature=use_duration_feature,
            default_duration_ms=default_duration_ms,
            interval_scale=interval_scale,
            device=device,
        )

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
        low = self.min_interval
        high = max(low, float(view.interval or low))

        r_low = self.model.predict_retention_from_curves(curves, low)
        if r_low <= target:
            return max(low, self.min_interval)

        t0 = self._effective_initial_guess(curves, target)
        if t0 is not None and t0 > high:
            high = min(self.max_interval, t0)

        while (
            high < self.max_interval
            and self.model.predict_retention_from_curves(curves, high) > target
        ):
            high = min(self.max_interval, high * 2.0)

        t = 0.5 * (low + high)
        if t0 is not None and math.isfinite(t0):
            t = min(high, max(low, t0))
        tol = 1e-5
        newton_steps = min(self.search_steps, 6)
        remaining_steps = self.search_steps - newton_steps
        for _ in range(newton_steps):
            pred, deriv = self._retention_and_derivative(curves, t)
            diff = pred - target
            if abs(diff) < tol:
                remaining_steps = 0
                break
            if diff > 0.0:
                low = t
            else:
                high = t
            if deriv == 0.0:
                t_new = 0.5 * (low + high)
            else:
                t_new = t - diff / deriv
            if not math.isfinite(t_new) or t_new < low or t_new > high:
                t = 0.5 * (low + high)
            else:
                t = t_new
        for _ in range(remaining_steps):
            mid = 0.5 * (low + high)
            pred = self.model.predict_retention_from_curves(curves, mid)
            if pred > target:
                low = mid
            else:
                high = mid
        if remaining_steps > 0:
            t = high
        return max(self.min_interval, min(t, self.max_interval))

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
    ) -> None:
        import torch

        self._torch = torch
        self.device = device
        self.dtype = dtype
        self.scheduler = scheduler
        self.model = scheduler.model
        self.desired_retention = float(scheduler.desired_retention)
        self.min_interval = float(scheduler.min_interval)
        self.max_interval = float(scheduler.max_interval)
        self.search_steps = int(scheduler.search_steps)

        self.model.device = device
        self.model.dtype = dtype
        self.model.network.to(device=device, dtype=dtype)

        self.n_rnns = int(self.model.network.n_rnns)
        self.n_hidden = int(self.model.network.n_hidden)
        self.n_curves = int(self.model.network.n_curves)
        self.interval_scale = float(self.model.interval_scale)
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
        elapsed_scaled = self._torch.clamp(elapsed, min=0.0) * self.interval_scale
        scores = self._torch.full(
            elapsed_scaled.shape,
            self.default_retention,
            dtype=self.dtype,
            device=self.device,
        )
        curves_mask = state.has_curves[idx]
        if curves_mask.any():
            curves_idx = idx[curves_mask]
            scores[curves_mask] = self._lstm_retention(
                elapsed_scaled[curves_mask],
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
        delay_scaled = self._torch.clamp(delays, min=0.0) * self.interval_scale
        rating_clamped = self._torch.clamp(ratings, min=1, max=4).to(self.dtype)
        delay_feature = delay_scaled.unsqueeze(-1)
        rating_feature = rating_clamped.unsqueeze(-1)
        if self.use_duration_feature and self.duration_value is not None:
            duration_feature = self.duration_value.expand_as(delay_feature)
            step = self._torch.cat(
                [delay_feature, duration_feature, rating_feature], dim=-1
            )
        else:
            step = self._torch.cat([delay_feature, rating_feature], dim=-1)

        h = state.lstm_h[:, idx, :]
        c = state.lstm_c[:, idx, :]
        w_last, s_last, d_last, (h_new, c_new) = self.model.network.forward_step(
            step, (h, c)
        )
        state.lstm_h[:, idx, :] = h_new
        state.lstm_c[:, idx, :] = c_new
        state.mem_w[idx] = w_last
        state.mem_s[idx] = s_last
        state.mem_d[idx] = d_last
        state.has_curves[idx] = True

    def _target_interval(
        self,
        state: LSTMVectorizedSchedulerState,
        idx: "torch.Tensor",
        prev_interval: "torch.Tensor | None",
    ) -> "torch.Tensor":
        weights = state.mem_w[idx]
        stabilities = state.mem_s[idx]
        decays = state.mem_d[idx]
        low = self._torch.full(
            (idx.numel(),), self.min_interval, dtype=self.dtype, device=self.device
        )
        t0 = self._effective_initial_guess(weights, stabilities, decays)
        if prev_interval is None:
            high = low.clone()
        else:
            high = self._torch.maximum(low, prev_interval.to(self.dtype))
        high = self._torch.maximum(high, t0)

        pred_low = self._retention_at(low, weights, stabilities, decays)
        high = self._torch.where(pred_low <= self._target, low, high)
        needs_expand = pred_low > self._target
        if needs_expand.any():
            while True:
                pred_high = self._retention_at(high, weights, stabilities, decays)
                still = (pred_high > self._target) & (high < self._max_interval)
                if not still.any():
                    break
                high = self._torch.where(
                    still, self._torch.minimum(high * 2.0, self._max_interval), high
                )
                if not (high < self._max_interval).any():
                    break

        t = self._torch.clamp(t0, min=low, max=high)
        tol = 1e-5
        newton_steps = min(self.search_steps, 6)
        remaining_steps = self.search_steps - newton_steps
        for _ in range(newton_steps):
            pred, deriv = self._retention_and_derivative(
                t, weights, stabilities, decays
            )
            diff = pred - self._target
            if self._torch.max(self._torch.abs(diff)).item() < tol:
                remaining_steps = 0
                break
            go_low = diff > 0.0
            low = self._torch.where(go_low, t, low)
            high = self._torch.where(go_low, high, t)
            mid = 0.5 * (low + high)
            step = diff / deriv
            t_new = t - step
            invalid = (
                (deriv == 0.0)
                | ~self._torch.isfinite(t_new)
                | (t_new < low)
                | (t_new > high)
            )
            t = self._torch.where(invalid, mid, t_new)
        for _ in range(remaining_steps):
            mid = 0.5 * (low + high)
            pred_mid = self._retention_at(mid, weights, stabilities, decays)
            go_low = pred_mid > self._target
            low = self._torch.where(go_low, mid, low)
            high = self._torch.where(go_low, high, mid)
        if remaining_steps > 0:
            t = high
        return self._torch.clamp(t, min=self.min_interval, max=self.max_interval)

    def _retention_at(
        self,
        days: "torch.Tensor",
        weights: "torch.Tensor",
        stabilities: "torch.Tensor",
        decays: "torch.Tensor",
    ) -> "torch.Tensor":
        elapsed_scaled = self._torch.clamp(days, min=0.0) * self.interval_scale
        return self._lstm_retention(elapsed_scaled, weights, stabilities, decays)

    def _retention_and_derivative(
        self,
        days: "torch.Tensor",
        weights: "torch.Tensor",
        stabilities: "torch.Tensor",
        decays: "torch.Tensor",
    ) -> tuple["torch.Tensor", "torch.Tensor"]:
        elapsed_scaled = self._torch.clamp(days, min=0.0) * self.interval_scale
        denom = stabilities + EPS
        x = 1.0 + elapsed_scaled.unsqueeze(-1) / denom
        x_pow = x ** (-decays)
        total = self._torch.sum(weights * x_pow, dim=-1)
        deriv = (
            self._torch.sum(weights * (-decays / denom) * x_pow / x, dim=-1)
            * self.interval_scale
        )
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
