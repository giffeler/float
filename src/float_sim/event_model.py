from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SideName = Literal["outer", "inner"]


@dataclass(frozen=True)
class ModelParameters:
    body_length: float = 3.0
    body_width: float = 0.4
    domain_half_length: float = 6.0
    domain_half_width: float = 4.0
    attenuation_length: float = 2.5
    attenuation_power: float = 0.0
    mean_wave_amplitude: float = 1.0
    side_samples: int = 21
    mobility: float = 0.03
    source_padding: float = 1e-6


@dataclass(frozen=True)
class Body:
    name: str
    center_x: float
    center_y: float
    length: float
    width: float
    outward_sign: int

    @property
    def min_x(self) -> float:
        return self.center_x - 0.5 * self.length

    @property
    def max_x(self) -> float:
        return self.center_x + 0.5 * self.length

    @property
    def min_y(self) -> float:
        return self.center_y - 0.5 * self.width

    @property
    def max_y(self) -> float:
        return self.center_y + 0.5 * self.width

    def side_y(self, side: SideName) -> float:
        sign = self.outward_sign if side == "outer" else -self.outward_sign
        return self.center_y + sign * 0.5 * self.width

    def side_normal(self, side: SideName) -> np.ndarray:
        sign = self.outward_sign if side == "outer" else -self.outward_sign
        return np.array([0.0, float(sign)])


@dataclass(frozen=True)
class WaveEvents:
    positions: np.ndarray
    amplitudes: np.ndarray

    def __len__(self) -> int:
        return int(self.amplitudes.shape[0])


@dataclass(frozen=True)
class SideMetrics:
    hits: int
    cumulative_impulse: float


@dataclass(frozen=True)
class BodyMetrics:
    inner: SideMetrics
    outer: SideMetrics
    net_force_y: float
    gap_closing_force: float


@dataclass(frozen=True)
class BatchResult:
    gap: float
    blocking_enabled: bool
    upper: BodyMetrics
    lower: BodyMetrics
    bodies: tuple[Body, Body]
    events: WaveEvents

    @property
    def mean_gap_closing_force(self) -> float:
        return 0.5 * (self.upper.gap_closing_force + self.lower.gap_closing_force)


@dataclass(frozen=True)
class SweepPoint:
    gap: float
    blocking_enabled: bool
    mean_gap_closing_force: float
    std_gap_closing_force: float


@dataclass(frozen=True)
class TrajectoryPoint:
    step: int
    gap: float
    mean_gap_closing_force: float


def make_parallel_bodies(gap: float, params: ModelParameters) -> tuple[Body, Body]:
    center_offset = 0.5 * (gap + params.body_width)
    upper = Body(
        name="upper",
        center_x=0.0,
        center_y=center_offset,
        length=params.body_length,
        width=params.body_width,
        outward_sign=1,
    )
    lower = Body(
        name="lower",
        center_x=0.0,
        center_y=-center_offset,
        length=params.body_length,
        width=params.body_width,
        outward_sign=-1,
    )
    return upper, lower


def point_in_body(point: np.ndarray, body: Body, padding: float = 0.0) -> bool:
    return (
        (body.min_x - padding) <= point[0] <= (body.max_x + padding)
        and (body.min_y - padding) <= point[1] <= (body.max_y + padding)
    )


def attenuation(distance: np.ndarray, params: ModelParameters) -> np.ndarray:
    safe_distance = np.maximum(distance, 1e-9)
    envelope = np.exp(-safe_distance / params.attenuation_length)
    if params.attenuation_power <= 0.0:
        return envelope
    return envelope / np.power(safe_distance, params.attenuation_power)


def side_sample_points(body: Body, side: SideName, count: int) -> np.ndarray:
    x_values = np.linspace(body.min_x, body.max_x, count)
    y_values = np.full(count, body.side_y(side))
    return np.column_stack([x_values, y_values])


def segment_intersects_body(start: np.ndarray, end: np.ndarray, body: Body) -> bool:
    direction = end - start
    t_min = 0.0
    t_max = 1.0
    bounds = ((body.min_x, body.max_x), (body.min_y, body.max_y))

    for axis, (lower, upper) in enumerate(bounds):
        delta = direction[axis]
        if abs(delta) < 1e-12:
            if start[axis] < lower or start[axis] > upper:
                return False
            continue

        t1 = (lower - start[axis]) / delta
        t2 = (upper - start[axis]) / delta
        axis_min = min(t1, t2)
        axis_max = max(t1, t2)
        t_min = max(t_min, axis_min)
        t_max = min(t_max, axis_max)
        if t_min > t_max:
            return False

    return True


def sample_wave_events(
    rng: np.random.Generator,
    count: int,
    bodies: tuple[Body, Body],
    params: ModelParameters,
) -> WaveEvents:
    accepted: list[np.ndarray] = []
    while len(accepted) < count:
        draw_count = max(count - len(accepted), params.side_samples)
        x_values = rng.uniform(-params.domain_half_length, params.domain_half_length, size=draw_count)
        y_values = rng.uniform(-params.domain_half_width, params.domain_half_width, size=draw_count)
        candidates = np.column_stack([x_values, y_values])
        for point in candidates:
            if any(point_in_body(point, body, padding=params.source_padding) for body in bodies):
                continue
            accepted.append(point)
            if len(accepted) == count:
                break

    positions = np.asarray(accepted, dtype=float)
    amplitudes = rng.exponential(scale=params.mean_wave_amplitude, size=count)
    return WaveEvents(positions=positions, amplitudes=amplitudes)


def evaluate_side_metrics(
    body: Body,
    side: SideName,
    events: WaveEvents,
    params: ModelParameters,
    blocker: Body | None,
    blocking_enabled: bool,
) -> SideMetrics:
    points = side_sample_points(body, side, params.side_samples)
    normal = body.side_normal(side)
    hits = 0
    total_impulse = 0.0

    for source, amplitude in zip(events.positions, events.amplitudes, strict=True):
        vectors = points - source
        distances = np.linalg.norm(vectors, axis=1)
        incident = -np.einsum("ij,j->i", vectors / np.maximum(distances[:, None], 1e-12), normal)
        directional = np.maximum(incident, 0.0)
        local_impulse = amplitude * attenuation(distances, params) * directional

        if blocking_enabled and blocker is not None:
            blocked = np.array([segment_intersects_body(source, point, blocker) for point in points], dtype=bool)
            local_impulse = np.where(blocked, 0.0, local_impulse)

        event_impulse = float(local_impulse.mean())
        if event_impulse > 0.0:
            hits += 1
            total_impulse += event_impulse

    return SideMetrics(hits=hits, cumulative_impulse=total_impulse)


def compute_body_metrics(
    body: Body,
    blocker: Body | None,
    events: WaveEvents,
    params: ModelParameters,
    blocking_enabled: bool,
) -> BodyMetrics:
    outer = evaluate_side_metrics(
        body=body,
        side="outer",
        events=events,
        params=params,
        blocker=blocker,
        blocking_enabled=blocking_enabled,
    )
    inner = evaluate_side_metrics(
        body=body,
        side="inner",
        events=events,
        params=params,
        blocker=blocker,
        blocking_enabled=blocking_enabled,
    )

    outer_force_y = -body.side_normal("outer")[1] * outer.cumulative_impulse
    inner_force_y = -body.side_normal("inner")[1] * inner.cumulative_impulse
    net_force_y = outer_force_y + inner_force_y
    gap_closing_force = outer.cumulative_impulse - inner.cumulative_impulse

    return BodyMetrics(
        inner=inner,
        outer=outer,
        net_force_y=float(net_force_y),
        gap_closing_force=float(gap_closing_force),
    )


def simulate_batch(
    gap: float,
    params: ModelParameters,
    rng: np.random.Generator,
    n_events: int,
    blocking_enabled: bool = True,
    events: WaveEvents | None = None,
) -> BatchResult:
    bodies = make_parallel_bodies(gap=gap, params=params)
    sampled_events = events if events is not None else sample_wave_events(rng, n_events, bodies, params)
    upper = compute_body_metrics(
        body=bodies[0],
        blocker=bodies[1],
        events=sampled_events,
        params=params,
        blocking_enabled=blocking_enabled,
    )
    lower = compute_body_metrics(
        body=bodies[1],
        blocker=bodies[0],
        events=sampled_events,
        params=params,
        blocking_enabled=blocking_enabled,
    )
    return BatchResult(
        gap=gap,
        blocking_enabled=blocking_enabled,
        upper=upper,
        lower=lower,
        bodies=bodies,
        events=sampled_events,
    )


def run_distance_sweep(
    gaps: np.ndarray,
    params: ModelParameters,
    n_events: int,
    repeats: int,
    seed: int,
    blocking_enabled: bool = True,
) -> list[SweepPoint]:
    results: list[SweepPoint] = []
    for gap in gaps:
        closing_forces = []
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + repeat)
            batch = simulate_batch(
                gap=float(gap),
                params=params,
                rng=rng,
                n_events=n_events,
                blocking_enabled=blocking_enabled,
            )
            closing_forces.append(batch.mean_gap_closing_force)

        closing_array = np.asarray(closing_forces)
        results.append(
            SweepPoint(
                gap=float(gap),
                blocking_enabled=blocking_enabled,
                mean_gap_closing_force=float(closing_array.mean()),
                std_gap_closing_force=float(closing_array.std(ddof=0)),
            )
        )
    return results


def simulate_gap_trajectory(
    initial_gap: float,
    steps: int,
    n_events_per_step: int,
    params: ModelParameters,
    seed: int,
    blocking_enabled: bool = True,
) -> list[TrajectoryPoint]:
    rng = np.random.default_rng(seed)
    gap = float(initial_gap)
    trajectory: list[TrajectoryPoint] = [TrajectoryPoint(step=0, gap=gap, mean_gap_closing_force=0.0)]

    for step in range(1, steps + 1):
        batch = simulate_batch(
            gap=gap,
            params=params,
            rng=rng,
            n_events=n_events_per_step,
            blocking_enabled=blocking_enabled,
        )
        gap = max(0.0, gap - params.mobility * batch.mean_gap_closing_force)
        trajectory.append(
            TrajectoryPoint(
                step=step,
                gap=float(gap),
                mean_gap_closing_force=batch.mean_gap_closing_force,
            )
        )

    return trajectory


def plot_geometry(ax, batch: BatchResult, max_events: int = 400) -> None:
    import matplotlib.pyplot as plt  # noqa: F401

    events = batch.events.positions[:max_events]
    colors = np.where(events[:, 1] >= 0.0, "#457b9d", "#e76f51")
    ax.scatter(events[:, 0], events[:, 1], s=12, alpha=0.35, c=colors, label="wave events")

    for body, facecolor in zip(batch.bodies, ("#1d3557", "#264653"), strict=True):
        rectangle = plt.Rectangle(
            (body.min_x, body.min_y),
            body.length,
            body.width,
            facecolor=facecolor,
            edgecolor="black",
            alpha=0.65,
        )
        ax.add_patch(rectangle)

    ax.axhline(0.0, color="0.65", linestyle="--", linewidth=1.0)
    ax.set_title(f"Geometry and Sampled Wave Events (gap = {batch.gap:.2f})")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")


def plot_side_metrics(ax, batch: BatchResult, metric: Literal["hits", "impulse"]) -> None:
    label = "Hit count" if metric == "hits" else "Cumulative impulse"
    upper_values = [
        batch.upper.inner.hits if metric == "hits" else batch.upper.inner.cumulative_impulse,
        batch.upper.outer.hits if metric == "hits" else batch.upper.outer.cumulative_impulse,
    ]
    lower_values = [
        batch.lower.inner.hits if metric == "hits" else batch.lower.inner.cumulative_impulse,
        batch.lower.outer.hits if metric == "hits" else batch.lower.outer.cumulative_impulse,
    ]

    x = np.arange(2)
    width = 0.35
    ax.bar(x - width / 2, upper_values, width=width, color="#457b9d", label="upper body")
    ax.bar(x + width / 2, lower_values, width=width, color="#e76f51", label="lower body")
    ax.set_xticks(x, ["inner", "outer"])
    ax.set_ylabel(label)
    ax.set_title(f"{label} by side")
    ax.legend(loc="best")


def plot_distance_sweep(ax, sweep: list[SweepPoint], label: str, color: str) -> None:
    gaps = np.array([point.gap for point in sweep])
    means = np.array([point.mean_gap_closing_force for point in sweep])
    stds = np.array([point.std_gap_closing_force for point in sweep])
    ax.plot(gaps, means, marker="o", color=color, label=label)
    ax.fill_between(gaps, means - stds, means + stds, color=color, alpha=0.15)
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Edge-to-edge gap")
    ax.set_ylabel("Mean gap-closing force")
    ax.set_title("Gap-closing force vs gap")
    ax.legend(loc="best")


def plot_gap_trajectory(ax, trajectory: list[TrajectoryPoint], label: str, color: str) -> None:
    steps = np.array([point.step for point in trajectory])
    gaps = np.array([point.gap for point in trajectory])
    ax.plot(steps, gaps, marker="o", color=color, label=label)
    ax.set_xlabel("Update step")
    ax.set_ylabel("Edge-to-edge gap")
    ax.set_title("Gap evolution under repeated Monte Carlo forcing")
    ax.legend(loc="best")
