from __future__ import annotations

import numpy as np

from float_sim.event_model import (
    Body,
    ModelParameters,
    ShieldingModel,
    SourceField,
    WaveEvents,
    event_count_from_source_density,
    evaluate_explicit_force,
    evaluate_side_metrics,
    make_parallel_bodies,
    point_in_body,
    resolve_events_for_source_support,
    run_ensemble,
    sample_body_boundary,
    sample_wave_events,
    segment_body_overlap_length,
    segment_intersects_body,
    simulate_batch,
    summarize_batch,
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
    assert np.isclose(segment_body_overlap_length(np.array([-2.0, 0.0]), np.array([2.0, 0.0]), body), 2.0)
    assert np.isclose(segment_body_overlap_length(np.array([-2.0, 2.0]), np.array([2.0, 2.0]), body), 0.0)


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
    upper_graded = evaluate_side_metrics(
        bodies[0],
        "inner",
        events,
        params,
        blocker=bodies[1],
        blocking_enabled=True,
        shielding_model=ShieldingModel.graded(minimum_transmission=0.2, occlusion_decay_length=0.2),
    )

    assert upper_blocked.hits <= upper_free.hits
    assert upper_blocked.cumulative_impulse <= upper_free.cumulative_impulse
    assert upper_blocked.cumulative_impulse <= upper_graded.cumulative_impulse <= upper_free.cumulative_impulse


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
    assert np.isclose(batch_a.explicit_mean_gap_closing_force, batch_b.explicit_mean_gap_closing_force)
    assert np.allclose(batch_a.events.positions, batch_b.events.positions)
    assert np.allclose(batch_a.events.amplitudes, batch_b.events.amplitudes)


def test_binary_shielding_model_matches_legacy_boolean_blocking() -> None:
    params = ModelParameters()
    batch_legacy = simulate_batch(
        gap=0.8,
        params=params,
        rng=np.random.default_rng(19),
        n_events=250,
        blocking_enabled=True,
    )
    batch_model = simulate_batch(
        gap=0.8,
        params=params,
        rng=np.random.default_rng(19),
        n_events=250,
        blocking_enabled=True,
        shielding_model=ShieldingModel.binary(),
    )

    assert np.isclose(batch_legacy.mean_gap_closing_force, batch_model.mean_gap_closing_force)
    assert np.isclose(batch_legacy.upper.inner.cumulative_impulse, batch_model.upper.inner.cumulative_impulse)
    assert np.isclose(batch_legacy.lower.outer.cumulative_impulse, batch_model.lower.outer.cumulative_impulse)


def test_body_surface_sources_are_tagged_and_outside_bodies() -> None:
    params = ModelParameters(domain_half_length=6.0, domain_half_width=4.0)
    bodies = make_parallel_bodies(gap=0.8, params=params)
    events = sample_wave_events(
        rng=np.random.default_rng(31),
        count=400,
        bodies=bodies,
        params=params,
        source_field=SourceField(model="body_surface", emission_offset=0.15),
    )

    assert events.emitter_indices is not None
    assert set(np.unique(events.emitter_indices)).issubset({0, 1})
    assert not any(point_in_body(point, body) for point in events.positions for body in bodies)


def test_self_emitted_body_surface_event_does_not_push_emitter() -> None:
    params = ModelParameters(side_samples=9, attenuation_length=3.0)
    bodies = make_parallel_bodies(gap=0.8, params=params)
    upper = bodies[0]
    events = WaveEvents(
        positions=np.array([[0.0, upper.max_y + 0.15]], dtype=float),
        amplitudes=np.ones(1, dtype=float),
        emitter_indices=np.array([0], dtype=int),
    )

    batch = simulate_batch(
        gap=0.8,
        params=params,
        rng=np.random.default_rng(5),
        n_events=len(events),
        blocking_enabled=False,
        events=events,
    )

    assert np.isclose(batch.upper.inner.cumulative_impulse, 0.0)
    assert np.isclose(batch.upper.outer.cumulative_impulse, 0.0)
    assert np.allclose(batch.upper.explicit_force_vector, np.zeros(2))
    assert np.isclose(batch.upper.explicit_gap_closing_force, 0.0)


def test_body_surface_source_model_is_reproducible() -> None:
    params = ModelParameters()
    source_field = SourceField(model="body_surface", emission_offset=0.12)
    batch_a = simulate_batch(
        gap=0.8,
        params=params,
        rng=np.random.default_rng(41),
        n_events=250,
        blocking_enabled=False,
        source_field=source_field,
    )
    batch_b = simulate_batch(
        gap=0.8,
        params=params,
        rng=np.random.default_rng(41),
        n_events=250,
        blocking_enabled=False,
        source_field=source_field,
    )

    assert np.allclose(batch_a.events.positions, batch_b.events.positions)
    assert np.array_equal(batch_a.events.emitter_indices, batch_b.events.emitter_indices)
    assert np.allclose(batch_a.events.amplitudes, batch_b.events.amplitudes)
    assert np.isclose(batch_a.explicit_mean_gap_closing_force, batch_b.explicit_mean_gap_closing_force)


def test_periodic_support_repeats_all_image_tiles() -> None:
    params = ModelParameters(domain_half_length=2.0, domain_half_width=1.5)
    source_field = SourceField(model="uniform", support="periodic_rectangle", periodic_image_layers=1)
    events = WaveEvents(
        positions=np.array([[0.2, 0.3], [-0.4, -0.6]], dtype=float),
        amplitudes=np.array([1.0, 2.0], dtype=float),
        emitter_indices=np.array([-1, -1], dtype=int),
    )

    resolved = resolve_events_for_source_support(events, params, source_field)

    assert resolved.positions.shape == (18, 2)
    assert resolved.amplitudes.shape == (18,)
    assert np.array_equal(resolved.emitter_indices, np.tile(events.emitter_indices, 9))


def test_periodic_support_is_invariant_under_whole_tile_translation() -> None:
    params = ModelParameters(domain_half_length=2.0, domain_half_width=1.5, side_samples=9, attenuation_length=2.0)
    source_field = SourceField(model="uniform", support="periodic_rectangle", periodic_image_layers=1)
    base_event = WaveEvents(
        positions=np.array([[0.1, 1.2]], dtype=float),
        amplitudes=np.ones(1, dtype=float),
        emitter_indices=np.array([-1], dtype=int),
    )
    translated_event = WaveEvents(
        positions=np.array([[0.1 + 4.0, 1.2 - 3.0]], dtype=float),
        amplitudes=np.ones(1, dtype=float),
        emitter_indices=np.array([-1], dtype=int),
    )

    batch_a = simulate_batch(
        gap=0.8,
        params=params,
        rng=np.random.default_rng(9),
        n_events=1,
        blocking_enabled=False,
        events=base_event,
        source_field=source_field,
    )
    batch_b = simulate_batch(
        gap=0.8,
        params=params,
        rng=np.random.default_rng(9),
        n_events=1,
        blocking_enabled=False,
        events=translated_event,
        source_field=source_field,
    )

    assert np.isclose(batch_a.mean_gap_closing_force, batch_b.mean_gap_closing_force)
    assert np.isclose(batch_a.explicit_mean_gap_closing_force, batch_b.explicit_mean_gap_closing_force)


def test_periodic_support_rejects_vertical_gradient() -> None:
    try:
        SourceField(model="vertical_gradient", support="periodic_rectangle", vertical_bias=0.5)
    except ValueError as exc:
        assert "vertical_gradient" in str(exc)
    else:
        raise AssertionError("expected vertical_gradient periodic support to be rejected")


def test_periodic_support_reduces_domain_size_sensitivity_for_uniform_field() -> None:
    small = ModelParameters(domain_half_length=4.0, domain_half_width=3.0, attenuation_length=2.5, side_samples=17)
    large = ModelParameters(domain_half_length=8.0, domain_half_width=6.0, attenuation_length=2.5, side_samples=17)
    shielding = ShieldingModel.graded(minimum_transmission=0.15, occlusion_decay_length=0.25)
    finite_field = SourceField()
    periodic_field = SourceField(support="periodic_rectangle", periodic_image_layers=1)

    finite_small = simulate_batch(
        gap=0.6,
        params=small,
        rng=np.random.default_rng(240),
        n_events=event_count_from_source_density(1.0, small),
        source_field=finite_field,
        shielding_model=shielding,
    )
    finite_large = simulate_batch(
        gap=0.6,
        params=large,
        rng=np.random.default_rng(240),
        n_events=event_count_from_source_density(1.0, large),
        source_field=finite_field,
        shielding_model=shielding,
    )
    periodic_small = simulate_batch(
        gap=0.6,
        params=small,
        rng=np.random.default_rng(240),
        n_events=event_count_from_source_density(1.0, small),
        source_field=periodic_field,
        shielding_model=shielding,
    )
    periodic_large = simulate_batch(
        gap=0.6,
        params=large,
        rng=np.random.default_rng(240),
        n_events=event_count_from_source_density(1.0, large),
        source_field=periodic_field,
        shielding_model=shielding,
    )

    finite_delta = abs(finite_small.mean_gap_closing_force - finite_large.mean_gap_closing_force)
    periodic_delta = abs(periodic_small.mean_gap_closing_force - periodic_large.mean_gap_closing_force)
    finite_explicit_delta = abs(
        finite_small.explicit_mean_gap_closing_force - finite_large.explicit_mean_gap_closing_force
    )
    periodic_explicit_delta = abs(
        periodic_small.explicit_mean_gap_closing_force - periodic_large.explicit_mean_gap_closing_force
    )

    assert periodic_delta < finite_delta
    assert periodic_explicit_delta < finite_explicit_delta


def test_boundary_sampling_weights_sum_to_body_perimeter() -> None:
    body = Body(
        name="test",
        center_x=0.0,
        center_y=0.0,
        length=3.0,
        width=0.4,
        outward_sign=1,
    )

    boundary = sample_body_boundary(body, side_samples=17)

    assert np.isclose(boundary.weights.sum(), 2.0 * (body.length + body.width))
    assert boundary.positions.shape == boundary.normals.shape
    assert boundary.positions.shape[1] == 2


def test_explicit_force_tracks_endcap_loading_not_present_in_side_bookkeeping() -> None:
    params = ModelParameters(side_samples=9, attenuation_length=2.5)
    body = Body(
        name="upper",
        center_x=0.0,
        center_y=0.75,
        length=3.0,
        width=0.4,
        outward_sign=1,
    )
    events = WaveEvents(
        positions=np.array([[body.max_x + 0.15, body.center_y]], dtype=float),
        amplitudes=np.ones(1, dtype=float),
        emitter_indices=np.array([-1], dtype=int),
    )

    explicit_force = evaluate_explicit_force(
        body=body,
        events=events,
        params=params,
        blocker=None,
        blocking_enabled=False,
    )

    assert explicit_force[0] < 0.0
    assert np.isfinite(explicit_force[1])


def test_source_density_to_event_count_scales_with_domain_area() -> None:
    density = 3.5
    params_small = ModelParameters(domain_half_length=4.0, domain_half_width=3.0)
    params_large = ModelParameters(domain_half_length=8.0, domain_half_width=3.0)

    assert event_count_from_source_density(density, params_large) == 2 * event_count_from_source_density(
        density, params_small
    )


def test_vertical_gradient_bias_shifts_sources_toward_positive_y() -> None:
    params = ModelParameters(domain_half_length=6.0, domain_half_width=4.0)
    bodies = make_parallel_bodies(gap=0.8, params=params)

    unbiased = sample_wave_events(
        rng=np.random.default_rng(12),
        count=2500,
        bodies=bodies,
        params=params,
        source_field=SourceField(),
    )
    biased = sample_wave_events(
        rng=np.random.default_rng(12),
        count=2500,
        bodies=bodies,
        params=params,
        source_field=SourceField(model="vertical_gradient", vertical_bias=0.8),
    )

    assert biased.positions[:, 1].mean() > unbiased.positions[:, 1].mean() + 0.5
    assert not any(
        point_in_body(point, body, padding=params.source_padding)
        for point in biased.positions
        for body in bodies
    )


def test_run_ensemble_is_reproducible_for_fixed_seed_and_source_field() -> None:
    params = ModelParameters()
    source_field = SourceField(model="vertical_gradient", vertical_bias=0.6)
    records_a = run_ensemble(
        gap=0.6,
        params=params,
        n_events=200,
        repeats=4,
        seed=21,
        blocking_enabled=True,
        source_field=source_field,
    )
    records_b = run_ensemble(
        gap=0.6,
        params=params,
        n_events=200,
        repeats=4,
        seed=21,
        blocking_enabled=True,
        source_field=source_field,
    )

    assert np.allclose(
        [record.mean_gap_closing_force for record in records_a],
        [record.mean_gap_closing_force for record in records_b],
    )
    assert np.allclose(
        [record.explicit_mean_gap_closing_force for record in records_a],
        [record.explicit_mean_gap_closing_force for record in records_b],
    )
    assert np.allclose(
        [record.mean_inner_impulse for record in records_a],
        [record.mean_inner_impulse for record in records_b],
    )


def test_mirrored_configuration_has_zero_system_net_force_without_blocking() -> None:
    params = ModelParameters(side_samples=9, attenuation_length=3.0)
    events = WaveEvents(
        positions=np.array(
            [
                [0.0, 2.0],
                [0.0, -2.0],
                [1.0, 2.3],
                [1.0, -2.3],
                [-1.4, 1.8],
                [-1.4, -1.8],
            ],
            dtype=float,
        ),
        amplitudes=np.ones(6, dtype=float),
    )

    batch = simulate_batch(
        gap=0.8,
        params=params,
        rng=np.random.default_rng(3),
        n_events=len(events),
        blocking_enabled=False,
        events=events,
    )
    diagnostics = summarize_batch(batch=batch, seed=3, n_events=len(events))

    assert np.isclose(diagnostics.system_net_force_y, 0.0)
    assert np.isclose(diagnostics.explicit_system_net_force_y, 0.0)
