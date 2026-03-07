from __future__ import annotations

import numpy as np

from float_sim.event_model import (
    Body,
    ModelParameters,
    WaveEvents,
    evaluate_side_metrics,
    make_parallel_bodies,
    segment_intersects_body,
    simulate_batch,
)


def test_segment_intersects_body_for_crossing_segment() -> None:
    body = Body(
        name="test",
        center_x=0.0,
        center_y=0.0,
        length=2.0,
        width=1.0,
        outward_sign=1,
    )

    assert segment_intersects_body(np.array([-2.0, 0.0]), np.array([2.0, 0.0]), body)
    assert not segment_intersects_body(np.array([-2.0, 2.0]), np.array([2.0, 2.0]), body)


def test_mirrored_events_give_equal_inner_and_outer_impulse_without_blocker() -> None:
    params = ModelParameters(side_samples=9, attenuation_length=3.0)
    body = Body(
        name="upper",
        center_x=0.0,
        center_y=0.75,
        length=3.0,
        width=0.4,
        outward_sign=1,
    )
    events = WaveEvents(
        positions=np.array(
            [
                [0.0, 2.0],
                [0.0, -0.5],
                [1.0, 2.3],
                [1.0, -0.8],
            ],
            dtype=float,
        ),
        amplitudes=np.ones(4, dtype=float),
    )

    outer = evaluate_side_metrics(body, "outer", events, params, blocker=None, blocking_enabled=False)
    inner = evaluate_side_metrics(body, "inner", events, params, blocker=None, blocking_enabled=False)

    assert outer.hits == inner.hits
    assert np.isclose(outer.cumulative_impulse, inner.cumulative_impulse)


def test_blocking_only_removes_contributions() -> None:
    params = ModelParameters(side_samples=11, attenuation_length=2.0)
    bodies = make_parallel_bodies(gap=0.3, params=params)
    events = WaveEvents(
        positions=np.array(
            [
                [0.0, 0.0],
                [0.0, 2.5],
                [0.0, -2.5],
                [1.0, 0.1],
                [-1.0, -0.1],
            ],
            dtype=float,
        ),
        amplitudes=np.ones(5, dtype=float),
    )

    upper_free = evaluate_side_metrics(
        bodies[0],
        "inner",
        events,
        params,
        blocker=bodies[1],
        blocking_enabled=False,
    )
    upper_blocked = evaluate_side_metrics(
        bodies[0],
        "inner",
        events,
        params,
        blocker=bodies[1],
        blocking_enabled=True,
    )

    assert upper_blocked.hits <= upper_free.hits
    assert upper_blocked.cumulative_impulse <= upper_free.cumulative_impulse


def test_simulation_is_reproducible_for_fixed_seed() -> None:
    params = ModelParameters()
    batch_a = simulate_batch(
        gap=0.8,
        params=params,
        rng=np.random.default_rng(7),
        n_events=250,
        blocking_enabled=True,
    )
    batch_b = simulate_batch(
        gap=0.8,
        params=params,
        rng=np.random.default_rng(7),
        n_events=250,
        blocking_enabled=True,
    )

    assert np.isclose(batch_a.mean_gap_closing_force, batch_b.mean_gap_closing_force)
    assert np.allclose(batch_a.events.positions, batch_b.events.positions)
    assert np.allclose(batch_a.events.amplitudes, batch_b.events.amplitudes)
