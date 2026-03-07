from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np

SideName = Literal["outer", "inner"]
SourceModel = Literal["uniform", "vertical_gradient", "body_surface"]
ShieldingMode = Literal["none", "binary", "graded"]


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
class SourceField:
    model: SourceModel = "uniform"
    vertical_bias: float = 0.0
    emission_offset: float = 0.1

    def __post_init__(self) -> None:
        if abs(self.vertical_bias) > 1.0:
            raise ValueError("vertical_bias must lie in [-1, 1]")
        if self.emission_offset <= 0.0:
            raise ValueError("emission_offset must be positive")
        if self.model not in {"uniform", "vertical_gradient", "body_surface"}:
            raise ValueError(f"unsupported source model: {self.model}")

    @property
    def label(self) -> str:
        if self.model == "body_surface":
            return f"body-surface emission (offset = {self.emission_offset:.2f})"
        if self.model == "uniform" or np.isclose(self.vertical_bias, 0.0):
            return "uniform"
        direction = "+y" if self.vertical_bias > 0.0 else "-y"
        return f"{self.model} ({direction}, |bias|={abs(self.vertical_bias):.2f})"


@dataclass(frozen=True)
class ShieldingModel:
    mode: ShieldingMode = "binary"
    minimum_transmission: float = 0.0
    occlusion_decay_length: float = 0.25

    def __post_init__(self) -> None:
        if self.mode not in {"none", "binary", "graded"}:
            raise ValueError(f"unsupported shielding mode: {self.mode}")
        if not 0.0 <= self.minimum_transmission <= 1.0:
            raise ValueError("minimum_transmission must lie in [0, 1]")
        if self.occlusion_decay_length <= 0.0:
            raise ValueError("occlusion_decay_length must be positive")

    @classmethod
    def none(cls) -> "ShieldingModel":
        return cls(mode="none")

    @classmethod
    def binary(cls) -> "ShieldingModel":
        return cls(mode="binary")

    @classmethod
    def graded(
        cls,
        *,
        minimum_transmission: float = 0.15,
        occlusion_decay_length: float = 0.25,
    ) -> "ShieldingModel":
        return cls(
            mode="graded",
            minimum_transmission=minimum_transmission,
            occlusion_decay_length=occlusion_decay_length,
        )

    @property
    def label(self) -> str:
        if self.mode == "none":
            return "no shielding"
        if self.mode == "binary":
            return "binary shielding"
        return (
            "graded shielding "
            f"(min transmission = {self.minimum_transmission:.2f}, "
            f"decay = {self.occlusion_decay_length:.2f})"
        )


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
    emitter_indices: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.amplitudes.shape[0])


@dataclass(frozen=True)
class BoundarySamples:
    positions: np.ndarray
    normals: np.ndarray
    weights: np.ndarray


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
    explicit_force_vector: np.ndarray
    explicit_net_force_y: float
    explicit_gap_closing_force: float


@dataclass(frozen=True)
class BatchResult:
    gap: float
    blocking_enabled: bool
    shielding_mode: ShieldingMode
    upper: BodyMetrics
    lower: BodyMetrics
    bodies: tuple[Body, Body]
    events: WaveEvents
    source_field: SourceField
    shielding_model: ShieldingModel

    @property
    def mean_gap_closing_force(self) -> float:
        return 0.5 * (self.upper.gap_closing_force + self.lower.gap_closing_force)

    @property
    def explicit_mean_gap_closing_force(self) -> float:
        return 0.5 * (self.upper.explicit_gap_closing_force + self.lower.explicit_gap_closing_force)


@dataclass(frozen=True)
class BatchDiagnostics:
    gap: float
    blocking_enabled: bool
    shielding_mode: ShieldingMode
    seed: int
    n_events: int
    source_model: SourceModel
    source_bias: float
    emission_offset: float
    minimum_transmission: float
    occlusion_decay_length: float
    mean_gap_closing_force: float
    explicit_mean_gap_closing_force: float
    system_net_force_y: float
    explicit_system_net_force_y: float
    mean_abs_net_force_y: float
    explicit_mean_abs_net_force_y: float
    upper_net_force_y: float
    lower_net_force_y: float
    upper_explicit_net_force_y: float
    lower_explicit_net_force_y: float
    mean_inner_hits: float
    mean_outer_hits: float
    mean_inner_impulse: float
    mean_outer_impulse: float

    @property
    def hit_imbalance(self) -> float:
        return self.mean_outer_hits - self.mean_inner_hits

    @property
    def impulse_imbalance(self) -> float:
        return self.mean_outer_impulse - self.mean_inner_impulse

    @property
    def force_law_gap_difference(self) -> float:
        return self.explicit_mean_gap_closing_force - self.mean_gap_closing_force

    @property
    def force_law_sign_agreement(self) -> bool:
        return same_force_sign(self.mean_gap_closing_force, self.explicit_mean_gap_closing_force)


@dataclass(frozen=True)
class EnsembleSummary:
    gap: float
    blocking_enabled: bool
    shielding_mode: ShieldingMode
    repeats: int
    n_events: int
    source_model: SourceModel
    source_bias: float
    emission_offset: float
    minimum_transmission: float
    occlusion_decay_length: float
    mean_gap_closing_force: float
    std_gap_closing_force: float
    sem_gap_closing_force: float
    explicit_mean_gap_closing_force: float
    explicit_std_gap_closing_force: float
    explicit_sem_gap_closing_force: float
    mean_system_net_force_y: float
    std_system_net_force_y: float
    explicit_mean_system_net_force_y: float
    explicit_std_system_net_force_y: float
    mean_abs_net_force_y: float
    std_abs_net_force_y: float
    explicit_mean_abs_net_force_y: float
    explicit_std_abs_net_force_y: float
    force_law_sign_agreement_rate: float
    mean_inner_hits: float
    std_inner_hits: float
    mean_outer_hits: float
    std_outer_hits: float
    mean_inner_impulse: float
    std_inner_impulse: float
    mean_outer_impulse: float
    std_outer_impulse: float

    @property
    def hit_imbalance(self) -> float:
        return self.mean_outer_hits - self.mean_inner_hits

    @property
    def impulse_imbalance(self) -> float:
        return self.mean_outer_impulse - self.mean_inner_impulse

    @property
    def force_law_gap_difference(self) -> float:
        return self.explicit_mean_gap_closing_force - self.mean_gap_closing_force


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


def source_domain_area(params: ModelParameters) -> float:
    return 4.0 * params.domain_half_length * params.domain_half_width


def event_count_from_source_density(source_density: float, params: ModelParameters) -> int:
    if source_density <= 0.0:
        raise ValueError("source_density must be positive")
    return max(1, int(round(source_density * source_domain_area(params))))


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


def sample_line_segment(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        raise ValueError("count must be positive")
    fractions = (np.arange(count, dtype=float) + 0.5) / count
    return start[None, :] + fractions[:, None] * (end - start)[None, :]


def sample_body_boundary(body: Body, side_samples: int) -> BoundarySamples:
    if side_samples <= 0:
        raise ValueError("side_samples must be positive")

    endcap_samples = max(1, int(round(side_samples * body.width / max(body.length, 1e-12))))

    top = sample_line_segment(np.array([body.min_x, body.max_y]), np.array([body.max_x, body.max_y]), side_samples)
    bottom = sample_line_segment(
        np.array([body.max_x, body.min_y]),
        np.array([body.min_x, body.min_y]),
        side_samples,
    )
    right = sample_line_segment(
        np.array([body.max_x, body.max_y]),
        np.array([body.max_x, body.min_y]),
        endcap_samples,
    )
    left = sample_line_segment(np.array([body.min_x, body.min_y]), np.array([body.min_x, body.max_y]), endcap_samples)

    positions = np.vstack([top, right, bottom, left])
    normals = np.vstack(
        [
            np.tile(np.array([[0.0, 1.0]]), (side_samples, 1)),
            np.tile(np.array([[1.0, 0.0]]), (endcap_samples, 1)),
            np.tile(np.array([[0.0, -1.0]]), (side_samples, 1)),
            np.tile(np.array([[-1.0, 0.0]]), (endcap_samples, 1)),
        ]
    )
    weights = np.concatenate(
        [
            np.full(side_samples, body.length / side_samples, dtype=float),
            np.full(endcap_samples, body.width / endcap_samples, dtype=float),
            np.full(side_samples, body.length / side_samples, dtype=float),
            np.full(endcap_samples, body.width / endcap_samples, dtype=float),
        ]
    )
    return BoundarySamples(positions=positions, normals=normals, weights=weights)


def sample_point_on_body_perimeter(
    rng: np.random.Generator,
    body: Body,
    emission_offset: float,
) -> np.ndarray:
    perimeter = 2.0 * (body.length + body.width)
    draw = rng.uniform(0.0, perimeter)

    if draw < body.length:
        x = body.min_x + draw
        point = np.array([x, body.max_y])
        normal = np.array([0.0, 1.0])
    elif draw < body.length + body.width:
        y = body.max_y - (draw - body.length)
        point = np.array([body.max_x, y])
        normal = np.array([1.0, 0.0])
    elif draw < 2.0 * body.length + body.width:
        x = body.max_x - (draw - body.length - body.width)
        point = np.array([x, body.min_y])
        normal = np.array([0.0, -1.0])
    else:
        y = body.min_y + (draw - 2.0 * body.length - body.width)
        point = np.array([body.min_x, y])
        normal = np.array([-1.0, 0.0])

    return point + emission_offset * normal


def segment_intersects_body(start: np.ndarray, end: np.ndarray, body: Body) -> bool:
    return segment_body_overlap_length(start, end, body) > 0.0


def segment_body_overlap_length(start: np.ndarray, end: np.ndarray, body: Body) -> float:
    direction = end - start
    t_min = 0.0
    t_max = 1.0
    bounds = ((body.min_x, body.max_x), (body.min_y, body.max_y))

    for axis, (lower, upper) in enumerate(bounds):
        delta = direction[axis]
        if abs(delta) < 1e-12:
            if start[axis] < lower or start[axis] > upper:
                return 0.0
            continue

        t1 = (lower - start[axis]) / delta
        t2 = (upper - start[axis]) / delta
        axis_min = min(t1, t2)
        axis_max = max(t1, t2)
        t_min = max(t_min, axis_min)
        t_max = min(t_max, axis_max)
        if t_min > t_max:
            return 0.0

    return float(np.linalg.norm(direction) * max(0.0, t_max - t_min))


def resolve_shielding_model(
    blocking_enabled: bool,
    shielding_model: ShieldingModel | None,
) -> ShieldingModel:
    if shielding_model is not None:
        return shielding_model
    return ShieldingModel.binary() if blocking_enabled else ShieldingModel.none()


def shielding_transmission(
    start: np.ndarray,
    end: np.ndarray,
    blocker: Body | None,
    shielding_model: ShieldingModel,
) -> float:
    if blocker is None or shielding_model.mode == "none":
        return 1.0

    overlap_length = segment_body_overlap_length(start, end, blocker)
    if overlap_length <= 0.0:
        return 1.0

    if shielding_model.mode == "binary":
        return 0.0

    decay = shielding_model.occlusion_decay_length
    return float(
        shielding_model.minimum_transmission
        + (1.0 - shielding_model.minimum_transmission) * np.exp(-overlap_length / decay)
    )


def same_force_sign(left: float, right: float, tolerance: float = 1e-9) -> bool:
    if abs(left) <= tolerance and abs(right) <= tolerance:
        return True
    if abs(left) <= tolerance or abs(right) <= tolerance:
        return False
    return bool(np.sign(left) == np.sign(right))


def sample_wave_events(
    rng: np.random.Generator,
    count: int,
    bodies: tuple[Body, Body],
    params: ModelParameters,
    source_field: SourceField | None = None,
) -> WaveEvents:
    field = source_field or SourceField()

    if field.model == "body_surface":
        perimeters = np.array([2.0 * (body.length + body.width) for body in bodies], dtype=float)
        probabilities = perimeters / perimeters.sum()
        emitter_indices = rng.choice(len(bodies), size=count, p=probabilities)
        positions = np.array(
            [
                sample_point_on_body_perimeter(rng, bodies[int(emitter_index)], field.emission_offset)
                for emitter_index in emitter_indices
            ],
            dtype=float,
        )
        amplitudes = rng.exponential(scale=params.mean_wave_amplitude, size=count)
        return WaveEvents(
            positions=positions,
            amplitudes=amplitudes,
            emitter_indices=np.asarray(emitter_indices, dtype=int),
        )

    accepted: list[np.ndarray] = []
    max_weight = 1.0 if field.model == "uniform" else 1.0 + abs(field.vertical_bias)

    while len(accepted) < count:
        draw_count = max(count - len(accepted), params.side_samples)
        x_values = rng.uniform(-params.domain_half_length, params.domain_half_length, size=draw_count)
        y_values = rng.uniform(-params.domain_half_width, params.domain_half_width, size=draw_count)
        candidates = np.column_stack([x_values, y_values])

        if field.model == "uniform" or np.isclose(field.vertical_bias, 0.0):
            accepted_mask = np.ones(draw_count, dtype=bool)
        else:
            normalized_y = candidates[:, 1] / max(params.domain_half_width, 1e-12)
            weights = 1.0 + field.vertical_bias * normalized_y
            weights = np.clip(weights, 0.0, None)
            accepted_mask = rng.random(draw_count) <= (weights / max_weight)

        for point, keep in zip(candidates, accepted_mask, strict=True):
            if not keep:
                continue
            if any(point_in_body(point, body, padding=params.source_padding) for body in bodies):
                continue
            accepted.append(point)
            if len(accepted) == count:
                break

    positions = np.asarray(accepted, dtype=float)
    amplitudes = rng.exponential(scale=params.mean_wave_amplitude, size=count)
    return WaveEvents(
        positions=positions,
        amplitudes=amplitudes,
        emitter_indices=np.full(count, -1, dtype=int),
    )


def evaluate_side_metrics(
    body: Body,
    side: SideName,
    events: WaveEvents,
    params: ModelParameters,
    blocker: Body | None,
    blocking_enabled: bool,
    shielding_model: ShieldingModel | None = None,
    body_emitter_index: int | None = None,
) -> SideMetrics:
    active_shielding = resolve_shielding_model(blocking_enabled, shielding_model)
    points = side_sample_points(body, side, params.side_samples)
    normal = body.side_normal(side)
    vectors = points[None, :, :] - events.positions[:, None, :]
    distances = np.linalg.norm(vectors, axis=2)
    unit_vectors = vectors / np.maximum(distances[:, :, None], 1e-12)
    incident = -np.einsum("epd,d->ep", unit_vectors, normal)
    local_impulse = events.amplitudes[:, None] * attenuation(distances, params) * np.maximum(incident, 0.0)

    if body_emitter_index is not None and events.emitter_indices is not None:
        self_emitted = events.emitter_indices == body_emitter_index
        local_impulse = np.where(self_emitted[:, None], 0.0, local_impulse)

    if active_shielding.mode != "none" and blocker is not None:
        transmission = np.array(
            [
                [shielding_transmission(source, point, blocker, active_shielding) for point in points]
                for source in events.positions
            ],
            dtype=float,
        )
        local_impulse = transmission * local_impulse

    event_impulse = local_impulse.mean(axis=1)
    positive = event_impulse > 0.0
    return SideMetrics(hits=int(np.count_nonzero(positive)), cumulative_impulse=float(event_impulse[positive].sum()))


def evaluate_explicit_force(
    body: Body,
    events: WaveEvents,
    params: ModelParameters,
    blocker: Body | None,
    blocking_enabled: bool,
    shielding_model: ShieldingModel | None = None,
    body_emitter_index: int | None = None,
) -> np.ndarray:
    active_shielding = resolve_shielding_model(blocking_enabled, shielding_model)
    boundary = sample_body_boundary(body, params.side_samples)
    vectors = boundary.positions[None, :, :] - events.positions[:, None, :]
    distances = np.linalg.norm(vectors, axis=2)
    unit_vectors = vectors / np.maximum(distances[:, :, None], 1e-12)
    incident = -np.einsum("epd,pd->ep", unit_vectors, boundary.normals)
    local_pressure = events.amplitudes[:, None] * attenuation(distances, params) * np.maximum(incident, 0.0)

    if body_emitter_index is not None and events.emitter_indices is not None:
        self_emitted = events.emitter_indices == body_emitter_index
        local_pressure = np.where(self_emitted[:, None], 0.0, local_pressure)

    if active_shielding.mode != "none" and blocker is not None:
        transmission = np.array(
            [
                [
                    shielding_transmission(source, point, blocker, active_shielding)
                    for point in boundary.positions
                ]
                for source in events.positions
            ],
            dtype=float,
        )
        local_pressure = transmission * local_pressure

    force_density = -local_pressure[:, :, None] * boundary.normals[None, :, :]
    weighted_force = force_density * boundary.weights[None, :, None]
    return weighted_force.sum(axis=(0, 1), dtype=float)


def compute_body_metrics(
    body: Body,
    blocker: Body | None,
    events: WaveEvents,
    params: ModelParameters,
    blocking_enabled: bool,
    shielding_model: ShieldingModel | None = None,
    body_emitter_index: int | None = None,
) -> BodyMetrics:
    outer = evaluate_side_metrics(
        body=body,
        side="outer",
        events=events,
        params=params,
        blocker=blocker,
        blocking_enabled=blocking_enabled,
        shielding_model=shielding_model,
        body_emitter_index=body_emitter_index,
    )
    inner = evaluate_side_metrics(
        body=body,
        side="inner",
        events=events,
        params=params,
        blocker=blocker,
        blocking_enabled=blocking_enabled,
        shielding_model=shielding_model,
        body_emitter_index=body_emitter_index,
    )

    outer_force_y = -body.side_normal("outer")[1] * outer.cumulative_impulse
    inner_force_y = -body.side_normal("inner")[1] * inner.cumulative_impulse
    net_force_y = outer_force_y + inner_force_y
    gap_closing_force = outer.cumulative_impulse - inner.cumulative_impulse
    explicit_force_vector = evaluate_explicit_force(
        body=body,
        events=events,
        params=params,
        blocker=blocker,
        blocking_enabled=blocking_enabled,
        shielding_model=shielding_model,
        body_emitter_index=body_emitter_index,
    )
    explicit_net_force_y = float(explicit_force_vector[1])
    explicit_gap_closing_force = float(-body.outward_sign * explicit_net_force_y)

    return BodyMetrics(
        inner=inner,
        outer=outer,
        net_force_y=float(net_force_y),
        gap_closing_force=float(gap_closing_force),
        explicit_force_vector=explicit_force_vector,
        explicit_net_force_y=explicit_net_force_y,
        explicit_gap_closing_force=explicit_gap_closing_force,
    )


def simulate_batch(
    gap: float,
    params: ModelParameters,
    rng: np.random.Generator,
    n_events: int,
    blocking_enabled: bool = True,
    events: WaveEvents | None = None,
    source_field: SourceField | None = None,
    shielding_model: ShieldingModel | None = None,
) -> BatchResult:
    field = source_field or SourceField()
    active_shielding = resolve_shielding_model(blocking_enabled, shielding_model)
    bodies = make_parallel_bodies(gap=gap, params=params)
    sampled_events = events if events is not None else sample_wave_events(rng, n_events, bodies, params, field)
    upper = compute_body_metrics(
        body=bodies[0],
        blocker=bodies[1],
        events=sampled_events,
        params=params,
        blocking_enabled=active_shielding.mode != "none",
        shielding_model=active_shielding,
        body_emitter_index=0,
    )
    lower = compute_body_metrics(
        body=bodies[1],
        blocker=bodies[0],
        events=sampled_events,
        params=params,
        blocking_enabled=active_shielding.mode != "none",
        shielding_model=active_shielding,
        body_emitter_index=1,
    )
    return BatchResult(
        gap=gap,
        blocking_enabled=active_shielding.mode != "none",
        shielding_mode=active_shielding.mode,
        upper=upper,
        lower=lower,
        bodies=bodies,
        events=sampled_events,
        source_field=field,
        shielding_model=active_shielding,
    )


def summarize_batch(batch: BatchResult, seed: int, n_events: int) -> BatchDiagnostics:
    mean_inner_hits = 0.5 * (batch.upper.inner.hits + batch.lower.inner.hits)
    mean_outer_hits = 0.5 * (batch.upper.outer.hits + batch.lower.outer.hits)
    mean_inner_impulse = 0.5 * (
        batch.upper.inner.cumulative_impulse + batch.lower.inner.cumulative_impulse
    )
    mean_outer_impulse = 0.5 * (
        batch.upper.outer.cumulative_impulse + batch.lower.outer.cumulative_impulse
    )
    system_net_force = batch.upper.net_force_y + batch.lower.net_force_y
    mean_abs_net_force = 0.5 * (abs(batch.upper.net_force_y) + abs(batch.lower.net_force_y))
    explicit_system_net_force = batch.upper.explicit_net_force_y + batch.lower.explicit_net_force_y
    explicit_mean_abs_net_force = 0.5 * (
        abs(batch.upper.explicit_net_force_y) + abs(batch.lower.explicit_net_force_y)
    )

    return BatchDiagnostics(
        gap=batch.gap,
        blocking_enabled=batch.blocking_enabled,
        shielding_mode=batch.shielding_model.mode,
        seed=seed,
        n_events=n_events,
        source_model=batch.source_field.model,
        source_bias=batch.source_field.vertical_bias,
        emission_offset=batch.source_field.emission_offset,
        minimum_transmission=batch.shielding_model.minimum_transmission,
        occlusion_decay_length=batch.shielding_model.occlusion_decay_length,
        mean_gap_closing_force=batch.mean_gap_closing_force,
        explicit_mean_gap_closing_force=batch.explicit_mean_gap_closing_force,
        system_net_force_y=float(system_net_force),
        explicit_system_net_force_y=float(explicit_system_net_force),
        mean_abs_net_force_y=float(mean_abs_net_force),
        explicit_mean_abs_net_force_y=float(explicit_mean_abs_net_force),
        upper_net_force_y=batch.upper.net_force_y,
        lower_net_force_y=batch.lower.net_force_y,
        upper_explicit_net_force_y=batch.upper.explicit_net_force_y,
        lower_explicit_net_force_y=batch.lower.explicit_net_force_y,
        mean_inner_hits=float(mean_inner_hits),
        mean_outer_hits=float(mean_outer_hits),
        mean_inner_impulse=float(mean_inner_impulse),
        mean_outer_impulse=float(mean_outer_impulse),
    )


def run_ensemble(
    gap: float,
    params: ModelParameters,
    n_events: int,
    repeats: int,
    seed: int,
    blocking_enabled: bool = True,
    source_field: SourceField | None = None,
    shielding_model: ShieldingModel | None = None,
) -> list[BatchDiagnostics]:
    field = source_field or SourceField()
    active_shielding = resolve_shielding_model(blocking_enabled, shielding_model)
    diagnostics: list[BatchDiagnostics] = []
    for repeat in range(repeats):
        run_seed = seed + repeat
        batch = simulate_batch(
            gap=gap,
            params=params,
            rng=np.random.default_rng(run_seed),
            n_events=n_events,
            blocking_enabled=active_shielding.mode != "none",
            source_field=field,
            shielding_model=active_shielding,
        )
        diagnostics.append(summarize_batch(batch=batch, seed=run_seed, n_events=n_events))
    return diagnostics


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=0))


def summarize_ensemble(records: Sequence[BatchDiagnostics]) -> EnsembleSummary:
    if not records:
        raise ValueError("records must be non-empty")

    first = records[0]
    repeats = len(records)
    gap_closing_mean, gap_closing_std = _mean_std([record.mean_gap_closing_force for record in records])
    explicit_gap_closing_mean, explicit_gap_closing_std = _mean_std(
        [record.explicit_mean_gap_closing_force for record in records]
    )
    system_mean, system_std = _mean_std([record.system_net_force_y for record in records])
    explicit_system_mean, explicit_system_std = _mean_std([record.explicit_system_net_force_y for record in records])
    abs_force_mean, abs_force_std = _mean_std([record.mean_abs_net_force_y for record in records])
    explicit_abs_force_mean, explicit_abs_force_std = _mean_std(
        [record.explicit_mean_abs_net_force_y for record in records]
    )
    inner_hits_mean, inner_hits_std = _mean_std([record.mean_inner_hits for record in records])
    outer_hits_mean, outer_hits_std = _mean_std([record.mean_outer_hits for record in records])
    inner_impulse_mean, inner_impulse_std = _mean_std([record.mean_inner_impulse for record in records])
    outer_impulse_mean, outer_impulse_std = _mean_std([record.mean_outer_impulse for record in records])
    sem = gap_closing_std / np.sqrt(repeats)
    explicit_sem = explicit_gap_closing_std / np.sqrt(repeats)
    sign_agreement_rate = float(np.mean([record.force_law_sign_agreement for record in records]))

    return EnsembleSummary(
        gap=first.gap,
        blocking_enabled=first.blocking_enabled,
        shielding_mode=first.shielding_mode,
        repeats=repeats,
        n_events=first.n_events,
        source_model=first.source_model,
        source_bias=first.source_bias,
        emission_offset=first.emission_offset,
        minimum_transmission=first.minimum_transmission,
        occlusion_decay_length=first.occlusion_decay_length,
        mean_gap_closing_force=gap_closing_mean,
        std_gap_closing_force=gap_closing_std,
        sem_gap_closing_force=float(sem),
        explicit_mean_gap_closing_force=explicit_gap_closing_mean,
        explicit_std_gap_closing_force=explicit_gap_closing_std,
        explicit_sem_gap_closing_force=float(explicit_sem),
        mean_system_net_force_y=system_mean,
        std_system_net_force_y=system_std,
        explicit_mean_system_net_force_y=explicit_system_mean,
        explicit_std_system_net_force_y=explicit_system_std,
        mean_abs_net_force_y=abs_force_mean,
        std_abs_net_force_y=abs_force_std,
        explicit_mean_abs_net_force_y=explicit_abs_force_mean,
        explicit_std_abs_net_force_y=explicit_abs_force_std,
        force_law_sign_agreement_rate=sign_agreement_rate,
        mean_inner_hits=inner_hits_mean,
        std_inner_hits=inner_hits_std,
        mean_outer_hits=outer_hits_mean,
        std_outer_hits=outer_hits_std,
        mean_inner_impulse=inner_impulse_mean,
        std_inner_impulse=inner_impulse_std,
        mean_outer_impulse=outer_impulse_mean,
        std_outer_impulse=outer_impulse_std,
    )


def run_gap_ensemble_sweep(
    gaps: np.ndarray,
    params: ModelParameters,
    n_events: int,
    repeats: int,
    seed: int,
    blocking_enabled: bool = True,
    source_field: SourceField | None = None,
    shielding_model: ShieldingModel | None = None,
) -> list[EnsembleSummary]:
    summaries: list[EnsembleSummary] = []
    for gap in gaps:
        records = run_ensemble(
            gap=float(gap),
            params=params,
            n_events=n_events,
            repeats=repeats,
            seed=seed,
            blocking_enabled=blocking_enabled,
            source_field=source_field,
            shielding_model=shielding_model,
        )
        summaries.append(summarize_ensemble(records))
    return summaries


def run_distance_sweep(
    gaps: np.ndarray,
    params: ModelParameters,
    n_events: int,
    repeats: int,
    seed: int,
    blocking_enabled: bool = True,
    source_field: SourceField | None = None,
    shielding_model: ShieldingModel | None = None,
) -> list[SweepPoint]:
    summaries = run_gap_ensemble_sweep(
        gaps=gaps,
        params=params,
        n_events=n_events,
        repeats=repeats,
        seed=seed,
        blocking_enabled=blocking_enabled,
        source_field=source_field,
        shielding_model=shielding_model,
    )
    return [
        SweepPoint(
            gap=summary.gap,
            blocking_enabled=summary.blocking_enabled,
            mean_gap_closing_force=summary.mean_gap_closing_force,
            std_gap_closing_force=summary.std_gap_closing_force,
        )
        for summary in summaries
    ]


def simulate_gap_trajectory(
    initial_gap: float,
    steps: int,
    n_events_per_step: int,
    params: ModelParameters,
    seed: int,
    blocking_enabled: bool = True,
    source_field: SourceField | None = None,
    shielding_model: ShieldingModel | None = None,
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
            source_field=source_field,
            shielding_model=shielding_model,
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
    ax.set_title(
        f"Geometry and Sampled Wave Events ({batch.source_field.label}, {batch.shielding_model.label}, gap = {batch.gap:.2f})"
    )
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


def plot_distance_sweep(
    ax,
    sweep: Sequence[SweepPoint] | Sequence[EnsembleSummary],
    label: str,
    color: str,
    uncertainty: Literal["std", "sem"] = "std",
) -> None:
    gaps = np.array([point.gap for point in sweep], dtype=float)
    means = np.array([point.mean_gap_closing_force for point in sweep], dtype=float)
    if uncertainty == "sem" and all(hasattr(point, "sem_gap_closing_force") for point in sweep):
        spreads = np.array([getattr(point, "sem_gap_closing_force") for point in sweep], dtype=float)
    else:
        spreads = np.array([point.std_gap_closing_force for point in sweep], dtype=float)

    ax.plot(gaps, means, marker="o", color=color, label=label)
    ax.fill_between(gaps, means - spreads, means + spreads, color=color, alpha=0.15)
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Edge-to-edge gap")
    ax.set_ylabel("Mean gap-closing force")
    ax.set_title("Gap-closing force vs gap")
    ax.legend(loc="best")


def plot_ensemble_metric(
    ax,
    summaries: Sequence[EnsembleSummary],
    metric: Literal[
        "gap_closing_force",
        "explicit_gap_closing_force",
        "force_law_gap_difference",
        "system_net_force",
        "explicit_system_net_force",
        "mean_abs_net_force",
        "explicit_mean_abs_net_force",
    ],
    label: str,
    color: str,
    uncertainty: Literal["std", "sem"] = "sem",
    title: str | None = None,
) -> None:
    gaps = np.array([summary.gap for summary in summaries], dtype=float)

    if metric == "gap_closing_force":
        means = np.array([summary.mean_gap_closing_force for summary in summaries], dtype=float)
        stds = np.array([summary.std_gap_closing_force for summary in summaries], dtype=float)
        sems = np.array([summary.sem_gap_closing_force for summary in summaries], dtype=float)
        ylabel = "Mean gap-closing force"
    elif metric == "explicit_gap_closing_force":
        means = np.array([summary.explicit_mean_gap_closing_force for summary in summaries], dtype=float)
        stds = np.array([summary.explicit_std_gap_closing_force for summary in summaries], dtype=float)
        sems = np.array([summary.explicit_sem_gap_closing_force for summary in summaries], dtype=float)
        ylabel = "Explicit mean gap-closing force"
    elif metric == "force_law_gap_difference":
        means = np.array([summary.force_law_gap_difference for summary in summaries], dtype=float)
        stds = np.zeros_like(means)
        sems = np.zeros_like(means)
        ylabel = "Explicit minus bookkeeping force"
    elif metric == "system_net_force":
        means = np.array([summary.mean_system_net_force_y for summary in summaries], dtype=float)
        stds = np.array([summary.std_system_net_force_y for summary in summaries], dtype=float)
        sems = stds / np.sqrt(np.maximum([summary.repeats for summary in summaries], 1))
        ylabel = "System net lateral force"
    elif metric == "explicit_system_net_force":
        means = np.array([summary.explicit_mean_system_net_force_y for summary in summaries], dtype=float)
        stds = np.array([summary.explicit_std_system_net_force_y for summary in summaries], dtype=float)
        sems = stds / np.sqrt(np.maximum([summary.repeats for summary in summaries], 1))
        ylabel = "Explicit system net lateral force"
    elif metric == "mean_abs_net_force":
        means = np.array([summary.mean_abs_net_force_y for summary in summaries], dtype=float)
        stds = np.array([summary.std_abs_net_force_y for summary in summaries], dtype=float)
        sems = stds / np.sqrt(np.maximum([summary.repeats for summary in summaries], 1))
        ylabel = "Mean |body net lateral force|"
    else:
        means = np.array([summary.explicit_mean_abs_net_force_y for summary in summaries], dtype=float)
        stds = np.array([summary.explicit_std_abs_net_force_y for summary in summaries], dtype=float)
        sems = stds / np.sqrt(np.maximum([summary.repeats for summary in summaries], 1))
        ylabel = "Explicit mean |body net lateral force|"

    spreads = sems if uncertainty == "sem" else stds
    ax.plot(gaps, means, marker="o", color=color, label=label)
    ax.fill_between(gaps, means - spreads, means + spreads, color=color, alpha=0.15)
    ax.axhline(0.0, color="0.4", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Edge-to-edge gap")
    ax.set_ylabel(ylabel)
    ax.set_title(title or ylabel)
    ax.legend(loc="best")


def plot_inner_outer_summary(
    ax,
    summaries: Sequence[EnsembleSummary],
    metric: Literal["hits", "impulse"],
    uncertainty: Literal["std", "sem"] = "sem",
) -> None:
    gaps = np.array([summary.gap for summary in summaries], dtype=float)
    if metric == "hits":
        inner_means = np.array([summary.mean_inner_hits for summary in summaries], dtype=float)
        outer_means = np.array([summary.mean_outer_hits for summary in summaries], dtype=float)
        inner_stds = np.array([summary.std_inner_hits for summary in summaries], dtype=float)
        outer_stds = np.array([summary.std_outer_hits for summary in summaries], dtype=float)
        ylabel = "Mean hit count"
        title = "Inner vs outer hit counts"
    else:
        inner_means = np.array([summary.mean_inner_impulse for summary in summaries], dtype=float)
        outer_means = np.array([summary.mean_outer_impulse for summary in summaries], dtype=float)
        inner_stds = np.array([summary.std_inner_impulse for summary in summaries], dtype=float)
        outer_stds = np.array([summary.std_outer_impulse for summary in summaries], dtype=float)
        ylabel = "Mean cumulative impulse"
        title = "Inner vs outer cumulative impulse"

    repeats = np.maximum(np.array([summary.repeats for summary in summaries], dtype=float), 1.0)
    inner_spreads = inner_stds / np.sqrt(repeats) if uncertainty == "sem" else inner_stds
    outer_spreads = outer_stds / np.sqrt(repeats) if uncertainty == "sem" else outer_stds

    ax.plot(gaps, inner_means, marker="o", color="#e76f51", label="inner side")
    ax.fill_between(gaps, inner_means - inner_spreads, inner_means + inner_spreads, color="#e76f51", alpha=0.15)
    ax.plot(gaps, outer_means, marker="o", color="#1d3557", label="outer side")
    ax.fill_between(gaps, outer_means - outer_spreads, outer_means + outer_spreads, color="#1d3557", alpha=0.15)
    ax.set_xlabel("Edge-to-edge gap")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")


def plot_gap_trajectory(ax, trajectory: Sequence[TrajectoryPoint], label: str, color: str) -> None:
    steps = np.array([point.step for point in trajectory], dtype=float)
    gaps = np.array([point.gap for point in trajectory], dtype=float)
    ax.plot(steps, gaps, marker="o", color=color, label=label)
    ax.set_xlabel("Update step")
    ax.set_ylabel("Edge-to-edge gap")
    ax.set_title("Gap evolution under repeated Monte Carlo forcing")
    ax.legend(loc="best")
