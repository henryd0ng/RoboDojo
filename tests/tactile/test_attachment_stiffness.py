"""CPU evidence for mesh-independent material-relative backing penalties."""

from copy import deepcopy
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from env.tactile.attachment import material_backing_strengths
from env.tactile.config import TactileConfigError, load_tactile_config
from env.tactile.gelsight import _TacExGelSight, _backplate_indices

POINTS = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
TETS = np.array([[0, 1, 2, 3]])
PARAMETERS = dict(youngs_modulus_pa=60000., poisson_ratio=.25, mass_density=1000., dt=.01, multiplier=10.)


def strengths(points=POINTS, tetrahedra=TETS, backing_indices=(0, 1, 2), **overrides):
    return material_backing_strengths(points, tetrahedra, backing_indices, **(PARAMETERS | overrides))


def test_single_tetrahedron_matches_analytic_nodal_spring_and_preserves_order():
    # V=1/6 m³, m_i=1000/24 kg, μ=λ=24000 Pa. Rest K_ii has
    # largest eigenvalue V(λ+2μ)|grad N_i|²: 36000 at node 0,
    # 12000 at the three axis nodes. Independent closed-form reference.
    result = strengths(backing_indices=[2, 0, 3, 1])
    np.testing.assert_allclose(result, [.288, .864, .288, .288], rtol=1e-13)
    assert result.dtype == np.float64 and result.shape == (4,)


def test_rigid_rotation_and_translation_preserve_strengths_and_inputs():
    axis = np.array([1., 2., 3.]) / np.sqrt(14.)
    skew = np.array([[0., -axis[2], axis[1]], [axis[2], 0., -axis[0]], [-axis[1], axis[0], 0.]])
    angle = .71
    rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
    rotated = POINTS @ rotation.T + [3., -4., 7.]
    before_points, before_tets = rotated.copy(), TETS.copy()
    np.testing.assert_allclose(strengths(rotated), strengths(), rtol=1e-13)
    np.testing.assert_array_equal(rotated, before_points)
    np.testing.assert_array_equal(TETS, before_tets)


@pytest.mark.parametrize("scale", [.001, .4, 2.5])
def test_geometric_rescaling_changes_strength_by_inverse_length_squared(scale):
    np.testing.assert_allclose(strengths(POINTS * scale), strengths() / scale**2, rtol=1e-13)


@pytest.mark.parametrize("field, value, factor", [
    ("youngs_modulus_pa", 120000., 2.),
    ("mass_density", 2000., .5),
    ("dt", .02, 4.),
    ("multiplier", 30., 3.),
])
def test_material_density_timestep_and_multiplier_have_consistent_units(field, value, factor):
    np.testing.assert_allclose(strengths(**{field: value}), strengths() * factor, rtol=1e-13)


def test_small_mass_node_gets_same_spring_to_material_ratio_as_large_node():
    # A small boundary tetrahedron must not receive a weaker backing merely
    # because its lumped mass is tiny. The equivalent physical spring is
    # s_i*m_i/dt², and its ratio to local K_ii should equal the multiplier.
    scale = 1e-4
    points = np.vstack([POINTS, POINTS * scale + [2., 0., 0.]])
    tets = np.vstack([TETS, TETS + 4])
    result = strengths(points, tets, [0, 4])
    masses = np.array([1000. / 24, 1000. / 24 * scale**3])
    local_stiffness = np.array([36000., 36000. * scale])
    np.testing.assert_allclose(result * masses / PARAMETERS["dt"]**2 / local_stiffness, [10., 10.], rtol=1e-10)
    assert result[1] > 1e7 * result[0]


@pytest.mark.parametrize("points, tets, indices", [
    ([], TETS, [0]),
    ([[0., 0.]], TETS, [0]),
    (POINTS * np.nan, TETS, [0]),
    (POINTS, [], [0]),
    (POINTS, TETS.astype(float), [0]),
    (POINTS, [[0, 1, 2, 4]], [0]),
    (POINTS, [[0, 1, 2, -1]], [0]),
    (POINTS, [[0, 1, 1, 3]], [0]),
    (POINTS, [[0, 2, 1, 3]], [0]),
    (POINTS, TETS, []),
    (POINTS, TETS, [0., 1.]),
    (POINTS, TETS, [0, 0]),
    (POINTS, TETS, [-1]),
    (POINTS, TETS, [4]),
    (np.vstack([POINTS, [3., 3., 3.]]), TETS, [4]),
])
def test_invalid_mesh_or_backing_is_rejected(points, tets, indices):
    with pytest.raises(ValueError):
        strengths(points, tets, indices)


@pytest.mark.parametrize("field", ["youngs_modulus_pa", "mass_density", "dt", "multiplier"])
@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
def test_nonphysical_scalar_parameters_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        strengths(**{field: value})


@pytest.mark.parametrize("value", [-1., .5, np.nan, np.inf])
def test_poisson_ratio_has_finite_elastic_range(value):
    with pytest.raises(ValueError, match="poisson_ratio"):
        strengths(poisson_ratio=value)


def raw_config():
    return dict(enabled=True, **{
        side: dict(body_name=f"{side}_finger", gelpad_prim_path=f"/{side}/gel",
                   sensor_prim_path=f"/{side}/sensor", mounting="surface",
                   optical_field_mm=[20, 20])
        for side in ("left", "right")
    })


def test_optional_multiplier_keeps_legacy_ratio_and_is_per_sensor():
    raw = raw_config()
    raw["left"]["attachment_strength_ratio"] = 7e5
    config = load_tactile_config(raw)
    assert config.left.attachment_strength_ratio == 7e5
    assert config.left.attachment_stiffness_multiplier is None
    assert config.right.attachment_strength_ratio == 1e5
    assert config.right.attachment_stiffness_multiplier is None
    raw["left"]["attachment_stiffness_multiplier"] = 10
    config = load_tactile_config(raw)
    assert config.left.attachment_stiffness_multiplier == 10
    assert config.left.attachment_strength_ratio == 7e5
    assert config.right.attachment_stiffness_multiplier is None
    raw["left"]["attachment_stiffness_multiplier"] = None
    assert load_tactile_config(raw).left.attachment_stiffness_multiplier is None


@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, "invalid", []])
def test_optional_multiplier_rejects_invalid_config_values(value):
    raw = deepcopy(raw_config())
    raw["left"]["attachment_stiffness_multiplier"] = value
    with pytest.raises(TactileConfigError, match="left.attachment_stiffness_multiplier"):
        load_tactile_config(raw)


def adapter_fixture(geometry):
    sensor = _TacExGelSight.__new__(_TacExGelSight)
    sensor.sensor_cfg = SimpleNamespace(attachment_stiffness_multiplier=10.)
    sensor.attachment = SimpleNamespace(attachment_points_idx=[2, 0])
    sensor.gelpad = SimpleNamespace(
        geo_slot_list=[SimpleNamespace(geometry=lambda: geometry)],
        uipc_sim=SimpleNamespace(cfg=SimpleNamespace(dt=.01)),
    )
    gelpad_cfg = SimpleNamespace(
        constitution_cfg=SimpleNamespace(youngs_modulus=.06, poisson_rate=.25), mass_density=1000.,
    )
    return sensor, gelpad_cfg


def test_adapter_uses_mutable_api_actual_timestep_mpa_units_and_only_changes_backing_nodes(monkeypatch):
    values = np.array([1e5, 2e5, 3e5, 4e5])
    readonly = values.view()
    readonly.flags.writeable = False
    attribute = SimpleNamespace(view=lambda: readonly)
    mutable_calls = []

    def mutable_view(requested):
        assert requested is attribute
        mutable_calls.append(requested)
        return values

    monkeypatch.setitem(sys.modules, "uipc", SimpleNamespace(view=mutable_view))
    geometry = SimpleNamespace(
        positions=lambda: SimpleNamespace(view=lambda: POINTS.reshape(4, 3, 1)),
        tetrahedra=lambda: SimpleNamespace(topo=lambda: SimpleNamespace(view=lambda: TETS.reshape(1, 4, 1))),
        vertices=lambda: SimpleNamespace(find=lambda name: attribute),
    )
    sensor, gelpad_cfg = adapter_fixture(geometry)
    sensor._set_attachment_stiffness(gelpad_cfg)
    assert mutable_calls == [attribute] and not readonly.flags.writeable
    np.testing.assert_allclose(values, [.864, 2e5, .288, 4e5], rtol=1e-13)


def test_adapter_writes_real_uipc_attribute_with_const_and_mutable_views_without_engine():
    uipc = pytest.importorskip("uipc")
    from uipc.constitution import SoftPositionConstraint
    from uipc.geometry import tetmesh

    geometry = tetmesh(POINTS.copy(), TETS.astype(np.int32))
    SoftPositionConstraint().apply_to(geometry, 1e5)
    attribute = geometry.vertices().find("strength_ratio")
    assert attribute.view().dtype == np.float64 and attribute.view().shape == (4,)
    assert not attribute.view().flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        attribute.view()[0] = 7.
    mutable = uipc.view(attribute)
    assert mutable.flags.writeable
    mutable[:] = [1e5, 2e5, 3e5, 4e5]
    sensor, gelpad_cfg = adapter_fixture(geometry)
    sensor._set_attachment_stiffness(gelpad_cfg)
    # Read the actual const attribute after the adapter write, rather than
    # accepting a changed temporary NumPy array as proof of scene mutation.
    np.testing.assert_allclose(attribute.view(), [.864, 2e5, .288, 4e5], rtol=1e-13)


def test_refined_backplate_selects_new_nodes_in_local_coordinates():
    points = np.array([[0, 0, 0], [.01, 0, 0], [0, .01, 0], [0, 0, .004]])
    assert _backplate_indices(points) == [0, 1, 2]
    # Shared-edge midpoint nodes on the backing plane must also attach.
    refined = np.vstack([points, points[[0, 1]].mean(axis=0), points[[0, 3]].mean(axis=0)])
    assert _backplate_indices(refined) == [0, 1, 2, 4]
    shifted = refined + [1, 2, .3]
    assert _backplate_indices(shifted) == [0, 1, 2, 4]


@pytest.mark.parametrize("points", [[], [[0, 0]], [[0, 0, float('nan')]], [[0, 0, 0], [1, 1, 0]]])
def test_invalid_or_entirely_fixed_gel_is_rejected(points):
    with pytest.raises(ValueError):
        _backplate_indices(points)
