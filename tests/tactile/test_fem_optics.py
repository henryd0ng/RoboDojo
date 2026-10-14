"""CPU checks for optical rendering of a solver's signed gel surface."""

import ast
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("torch_scatter")


@pytest.fixture(scope="module")
def taxim():
    spec = importlib.util.find_spec("tacex")
    if spec is None or spec.origin is None:
        pytest.skip("Optional TacEx source is not installed")
    package_dir = Path(spec.origin).parent
    # Import the independent optical model without launching Isaac Sim.
    name = "_robodojo_fem_optics_test"
    package = ModuleType(name)
    package.__path__ = [str(package_dir / "simulation_approaches/gpu_taxim/sim")]
    sys.modules[name] = package
    model = importlib.import_module(f"{name}.taxim_torch").TaximTorch
    calibration = package_dir.parents[1] / "tacex_assets/tacex_assets/data/Sensors/GelSight_Mini/calibs/640x480"
    return model(calib_folder=calibration, device="cpu")


def coordinates(taxim, height, width):
    yy, xx = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    x = (xx - width / 2) * taxim.sensor_params.pixmm * taxim.width / width
    y = (yy - height / 2) * taxim.sensor_params.pixmm * taxim.height / height
    return x, y


@pytest.mark.parametrize("shape", [(128, 128), (240, 320), (480, 640)])
def test_plane_normals_use_the_calibrated_physical_scale(taxim, shape):
    x, y = coordinates(taxim, *shape)
    surface = (0.2 * x + 0.1 * y)[None]
    angles, _ = taxim._TaximTorch__generate_normals(surface / taxim.sensor_params.pixmm)
    expected = np.arctan(np.hypot(0.2, 0.1))
    np.testing.assert_allclose(angles[:, 2:-2, 2:-2], expected, atol=2e-5)


def test_custom_surface_field_preserves_analytic_physical_slopes(taxim):
    from env.tactile.gelsight import _TacExGelSight

    wrapper = _TacExGelSight.__new__(_TacExGelSight)
    wrapper.sensor_cfg = SimpleNamespace(optical_field_mm=(20.4, 15.3))
    wrapper.sensor = SimpleNamespace(optical_simulator=SimpleNamespace(_taxim=taxim))
    original_scale = taxim.sensor_params.pixmm
    try:
        wrapper._configure_surface_optics()
        # Construct a known physical plane independently of Taxim's scale.
        yy, xx = torch.meshgrid(torch.arange(240), torch.arange(320), indexing="ij")
        x, y = xx * (20.4 / 320), yy * (15.3 / 240)
        surface = 0.2 * x + 0.1 * y
        angles, _ = taxim._TaximTorch__generate_normals(surface[None] / taxim.sensor_params.pixmm)
        np.testing.assert_allclose(angles[:, 2:-2, 2:-2], np.arctan(np.hypot(0.2, 0.1)), atol=2e-5)
    finally:
        taxim.sensor_params.pixmm = original_scale


def test_custom_surface_field_rejects_calibration_aspect_mismatch(taxim):
    from env.tactile.gelsight import _TacExGelSight

    wrapper = _TacExGelSight.__new__(_TacExGelSight)
    wrapper.sensor_cfg = SimpleNamespace(optical_field_mm=(20.4, 10.2))
    wrapper.sensor = SimpleNamespace(optical_simulator=SimpleNamespace(_taxim=taxim))
    original_scale = taxim.sensor_params.pixmm
    with pytest.raises(ValueError, match="calibration aspect ratio"):
        wrapper._configure_surface_optics()
    assert taxim.sensor_params.pixmm == original_scale


def test_spherical_surface_normals_match_analytic_geometry(taxim):
    x, y = coordinates(taxim, 240, 320)
    radius = 20.0
    rho2 = x * x + y * y
    surface = radius - torch.sqrt(radius * radius - rho2)
    angles, _ = taxim._TaximTorch__generate_normals(surface[None] / taxim.sensor_params.pixmm)
    expected = torch.asin(torch.sqrt(rho2) / radius)
    interior = rho2 < 16.0
    # Includes finite-difference truncation and float32 cancellation in z.
    np.testing.assert_allclose(angles[0][interior], expected[interior], atol=3e-5)


def test_surface_render_preserves_a_narrow_indent_and_positive_bulge(taxim, monkeypatch):
    x, y = coordinates(taxim, 240, 320)
    surface = -0.8 * torch.clamp(1 - x.abs() / 0.12, min=0) * torch.clamp(1 - y.abs() / 4, min=0)
    surface += 0.1 * torch.exp(-((x - 3) ** 2 + y ** 2) / 0.5)
    original = surface.clone()
    captured = []
    optical_stage = taxim._TaximTorch__render_optics

    def capture_shape(gel, *args, **kwargs):
        captured.append(gel.clone())
        return optical_stage(gel, *args, **kwargs)

    def forbid_deformation(*args, **kwargs):
        raise AssertionError("The physical surface must not be deformed again")

    monkeypatch.setattr(taxim, "_TaximTorch__render_optics", capture_shape)
    monkeypatch.setattr(taxim, "_TaximTorch__compute_gel_pad_deformation", forbid_deformation)
    image = taxim.render_surface(surface)
    assert image.shape == (3, 240, 320)
    assert torch.isfinite(image).all()
    assert torch.equal(surface, original)
    assert torch.equal(captured[0][0], original)
    assert captured[0].max() > 0
    assert captured[0].min() == pytest.approx(-0.8)


def test_angular_interpolation_is_continuous_across_a_magnitude_bin(taxim):
    lookup = taxim._TaximTorch__interpolate_poly_parameters
    direction = torch.full((1, 1, 1), 45.3)
    below = lookup(torch.full_like(direction, 20 - 1e-5), direction)
    above = lookup(torch.full_like(direction, 20 + 1e-5), direction)
    step = lookup(torch.full_like(direction, 21.0), direction) - lookup(torch.full_like(direction, 19.0), direction)
    assert torch.linalg.vector_norm(above - below) < torch.linalg.vector_norm(step) * 1e-4


def test_angular_lookup_is_periodic_and_finite_at_the_direction_seam(taxim):
    lookup = taxim._TaximTorch__interpolate_poly_parameters
    magnitude = torch.full((1, 1, 1), 20.4)
    intervals = taxim.sensor_params.num_bins - 1
    np.testing.assert_array_equal(lookup(magnitude, torch.zeros_like(magnitude)),
                                  lookup(magnitude, torch.full_like(magnitude, intervals)))
    below = lookup(magnitude, torch.full_like(magnitude, -1e-5))
    above = lookup(magnitude, torch.full_like(magnitude, 1e-5))
    np.testing.assert_allclose(below, above, atol=2e-5, rtol=2e-5)
    assert torch.isfinite(lookup(torch.full_like(magnitude, 1e6), torch.zeros_like(magnitude))).all()


def test_surface_render_rejects_shadows_and_nonfinite_input(taxim):
    with pytest.raises(ValueError, match="physical contact mask"):
        taxim.render_surface(torch.zeros(8, 8), with_shadow=True)
    with pytest.raises(ValueError, match="finite floating-point tensor"):
        taxim.render_surface(torch.full((8, 8), float("nan")))


@pytest.mark.parametrize("shape", [(2, 8), (8, 2), (0, 8, 8)])
def test_surface_render_rejects_undersized_or_empty_fields(taxim, shape):
    with pytest.raises(ValueError, match="nonempty batch and spatial dimensions"):
        taxim.render_surface(torch.zeros(shape))


def test_surface_render_rejects_integer_height_fields(taxim):
    with pytest.raises(ValueError, match="floating-point tensor"):
        taxim.render_surface(torch.zeros(8, 8, dtype=torch.int32))


def test_surface_render_accepts_double_precision_input(taxim):
    field = torch.linspace(-0.1, 0.1, 64).reshape(8, 8)
    np.testing.assert_array_equal(taxim.render_surface(field.double()), taxim.render_surface(field))


@pytest.mark.parametrize("direction", [0, 0.3, np.pi / 2, np.pi, -np.pi / 3])
def test_rgb_converges_to_flat_as_tilt_approaches_zero(taxim, direction):
    x, y = coordinates(taxim, 48, 64)
    flat = taxim.render_surface(torch.zeros_like(x))
    errors = []
    for tilt in (1e-3, 1e-4, 1e-5, 1e-6):
        plane = tilt * (np.cos(direction) * x + np.sin(direction) * y)
        image = taxim.render_surface(plane)
        errors.append(float((image - flat).abs().mean()))
    assert errors[-1] < 2e-6
    assert all(later < earlier * 0.15 for earlier, later in zip(errors, errors[1:]))


def test_zero_tilt_is_independent_of_direction(taxim):
    lookup = taxim._TaximTorch__interpolate_poly_parameters
    directions = torch.linspace(-124, 248, 37).reshape(1, 1, -1)
    parameters = lookup(torch.zeros_like(directions), directions)
    np.testing.assert_allclose(parameters, parameters[..., :1, :].expand_as(parameters), atol=0, rtol=0)


def test_masked_hole_cannot_create_a_false_edge_and_valid_interior_is_unchanged(taxim):
    x, y = coordinates(taxim, 48, 64)
    plane = 0.03 * x + 0.02 * y - 0.2
    ridge = -0.8 * torch.clamp(1 - x.abs() / 0.6, min=0) * torch.clamp(1 - y.abs() / 4, min=0)
    fields = torch.stack([plane, ridge]).reshape(1, 2, 48, 64)
    valid = torch.ones_like(fields, dtype=torch.bool)
    valid[..., 21:26, 29:34] = False
    damaged = fields.clone()
    damaged[~valid] = 0
    clean_rgb = taxim.render_surface(fields)
    masked_rgb = taxim.render_surface(damaged, valid_mask=valid)
    flat_rgb = taxim.render_surface(torch.zeros_like(fields))
    # Independent finite-difference domain oracle: a hole and its four
    # adjacent pixel strips cannot provide a complete derivative stencil.
    affected = ~valid.clone()
    affected[..., 20:27, 29:34] = True
    affected[..., 21:26, 28:35] = True
    affected[..., 0, :] = True
    affected[..., -1, :] = True
    affected[..., :, 0] = True
    affected[..., :, -1] = True
    channels = affected.unsqueeze(-3).expand_as(masked_rgb)
    np.testing.assert_array_equal(masked_rgb[channels], flat_rgb[channels])
    np.testing.assert_allclose(masked_rgb[~channels], clean_rgb[~channels], atol=1e-7, rtol=0)


def test_validity_mask_requires_matching_shape_and_boolean_dtype(taxim):
    with pytest.raises(ValueError, match="boolean and match"):
        taxim.render_surface(torch.zeros(8, 8), valid_mask=torch.ones(8, 8))
    with pytest.raises(ValueError, match="boolean and match"):
        taxim.render_surface(torch.zeros(8, 8), valid_mask=torch.ones(7, 8, dtype=torch.bool))


def test_absolute_response_matches_existing_surface_render_for_a_flat_reference(taxim):
    x, y = coordinates(taxim, 48, 64)
    current = -0.1 * torch.exp(-(x * x + y * y) / 4)
    valid = torch.ones_like(current, dtype=torch.bool)
    expected = taxim.render_surface(current, valid_mask=valid)
    actual = taxim.render_surface_difference(current, torch.zeros_like(current), valid, valid)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=0)


def test_unchanged_curved_reference_returns_flat_baseline_exactly(taxim):
    x, y = coordinates(taxim, 48, 64)
    curved = (0.08 * x * x + 0.03 * y * y)[None, None]
    actual = taxim.render_surface_difference(curved, curved)
    np.testing.assert_array_equal(actual, taxim.render_surface(torch.zeros_like(curved)))


def test_curved_reference_uses_absolute_normal_response_not_difference_normals(taxim):
    x, y = coordinates(taxim, 48, 64)
    reference = 0.35 * x + 0.1 * y + 0.01 * (x * x + y * y)
    current = reference + 0.1 * x
    raw_optics = taxim._TaximTorch__render_optics
    responses = [raw_optics(field[None], field[None] < 0, interpolate_angles=True, clip_output=False)[0]
                 for field in (torch.zeros_like(reference), current, reference)]
    expected = (responses[0] + (responses[1] - responses[2])).clamp(0, 1)
    actual = taxim.render_surface_difference(current, reference)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    old_difference_normals = taxim.render_surface(current - reference)
    assert float((actual - old_difference_normals).abs().mean()) > 1e-3


def test_response_difference_is_formed_before_saturation(taxim, monkeypatch):
    seen = []

    def optical_response(field, contact_mask, **kwargs):
        assert kwargs["interpolate_angles"] is True
        seen.append(kwargs["clip_output"])
        value = {0.0: 0.25, 1.0: 1.1, 2.0: 1.2}[float(field.mean())]
        image = torch.full((1, 3, 8, 8), value)
        return image.clamp(0, 1) if kwargs["clip_output"] else image

    monkeypatch.setattr(taxim, "_TaximTorch__render_optics", optical_response)
    image = taxim.render_surface_difference(torch.full((8, 8), 2.), torch.ones(8, 8))
    torch.testing.assert_close(image, torch.full((3, 8, 8), 0.35), atol=1e-7, rtol=0)
    assert seen == [False, False, False]
    assert not torch.allclose(image, torch.full_like(image, 0.25))


def test_different_current_and_reference_masks_share_one_derivative_domain(taxim):
    x, y = coordinates(taxim, 48, 64)
    reference = 0.03 * x * x + 0.04 * y * y
    current = reference - 0.2 * torch.exp(-(x * x + y * y) / 2)
    current_valid = torch.ones_like(current, dtype=torch.bool)
    reference_valid = current_valid.clone()
    current_valid[20:25, 20:25] = False
    reference_valid[26:31, 35:40] = False
    damaged_current, damaged_reference = current.clone(), reference.clone()
    damaged_current[~current_valid] = -50
    damaged_reference[~reference_valid] = 50
    actual = taxim.render_surface_difference(damaged_current, damaged_reference, current_valid, reference_valid)
    clean = taxim.render_surface_difference(current, reference)
    flat = taxim.render_surface(torch.zeros_like(current))
    # Independently enumerate the union of missing pixels and each derivative
    # stencil touching them. No missing value may create an optical contour.
    affected = ~(current_valid & reference_valid)
    original = affected.clone()
    affected[1:] |= original[:-1]
    affected[:-1] |= original[1:]
    affected[:, 1:] |= original[:, :-1]
    affected[:, :-1] |= original[:, 1:]
    affected[0] = affected[-1] = True
    affected[:, 0] = affected[:, -1] = True
    channels = affected[None].expand_as(actual)
    np.testing.assert_array_equal(actual[channels], flat[channels])
    torch.testing.assert_close(actual[~channels], clean[~channels], atol=0, rtol=0)


@pytest.mark.parametrize("mask_target", ["current", "reference"])
def test_absolute_response_validates_each_mask(taxim, mask_target):
    options = {"valid_mask" if mask_target == "current" else "reference_valid_mask": torch.ones(7, 8, dtype=torch.bool)}
    with pytest.raises(ValueError, match="validity mask must be boolean and match"):
        taxim.render_surface_difference(torch.zeros(8, 8), torch.zeros(8, 8), **options)


@pytest.mark.parametrize("bad_reference", [torch.zeros(7, 8), torch.full((8, 8), float("inf")),
                                         torch.zeros(8, 8, dtype=torch.int32)])
def test_absolute_response_rejects_invalid_reference_fields(taxim, bad_reference):
    with pytest.raises(ValueError):
        taxim.render_surface_difference(torch.zeros(8, 8), bad_reference)


@pytest.mark.parametrize("input_mode", ["object", "deformed_surface"])
def test_actual_depth_extraction_preserves_signed_shape_and_current_validity(taxim, input_mode):
    package_dir = Path(importlib.util.find_spec("tacex").origin).parent
    source = ast.parse((package_dir / "gelsight_sensor.py").read_text())
    method = next(node for node in ast.walk(source)
                  if isinstance(node, ast.FunctionDef) and node.name == "_get_height_map")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "gelsight_sensor._get_height_map", "exec"), namespace)
    depth = torch.tensor([[[0.027, 0.029, 0.05], [float("inf"), 0.020, 0.023],
                           [0.028, 0.0399995, 0.026]]])
    nominal_mask = torch.ones_like(depth, dtype=torch.bool)
    nominal_mask[0, 2, 2] = False
    sensor = SimpleNamespace(
        _robodojo_nominal_depth=torch.full_like(depth, 0.028), _robodojo_nominal_mask=nominal_mask,
        camera=SimpleNamespace(data=SimpleNamespace(output={"depth": depth[..., None]})),
        _data=SimpleNamespace(output={"height_map": torch.zeros_like(depth)}),
        cfg=SimpleNamespace(optical_sim_cfg=SimpleNamespace(input_mode=input_mode),
            sensor_camera_cfg=SimpleNamespace(clipping_range=(0.020, 0.040)),
            gelpad_dimensions=SimpleNamespace(height=0.004)),
    )
    result = namespace["_get_height_map"](sensor)
    if input_mode == "deformed_surface":
        expected = [[[-1, 1, 0], [0, 0, -5], [0, 0, 0]]]
        np.testing.assert_array_equal(sensor._robodojo_depth_valid,
            [[[True, True, False], [False, False, True], [True, False, False]]])
        np.testing.assert_allclose(sensor._robodojo_surface_height_mm,
            [[[7, 9, 0], [0, 0, 3], [8, 0, 6]]], atol=2e-6)
        np.testing.assert_allclose(sensor._robodojo_reference_surface_height_mm,
            [[[8, 8, 8], [8, 8, 8], [8, 8, 0]]], atol=2e-6)
        assert sensor._robodojo_surface_valid[0, 2, 2]
        assert not sensor._robodojo_reference_surface_valid[0, 2, 2]
    else:
        expected = [[[-1, 0, 0], [0, -4, -4], [0, 0, 0]]]
        assert not hasattr(sensor, "_robodojo_depth_valid")
    np.testing.assert_allclose(result, expected, atol=2e-6)


def test_actual_optical_wrapper_forwards_signed_height_and_current_mask(taxim):
    package_dir = Path(importlib.util.find_spec("tacex").origin).parent
    source = ast.parse((package_dir / "simulation_approaches/gpu_taxim/taxim_sim.py").read_text())
    method = next(node for node in ast.walk(source)
                  if isinstance(node, ast.FunctionDef) and node.name == "optical_simulation")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "taxim_sim.optical_simulation", "exec"), namespace)
    field = torch.linspace(-0.1, 0.1, 64).reshape(1, 8, 8)
    valid = torch.ones_like(field, dtype=torch.bool)
    valid[:, 3, 3] = False
    seen = []

    def render_surface(height, valid_mask=None):
        seen.append((height.clone(), valid_mask.clone()))
        return torch.zeros(1, 3, 8, 8)

    simulator = SimpleNamespace(
        sensor=SimpleNamespace(_data=SimpleNamespace(output={"height_map": field}), _robodojo_depth_valid=valid),
        cfg=SimpleNamespace(input_mode="deformed_surface", tactile_img_res=(8, 8)),
        _device="cpu", _taxim=SimpleNamespace(render_surface=render_surface),
        tactile_rgb_img=torch.zeros(1, 8, 8, 3),
    )
    image = namespace["optical_simulation"](simulator)
    assert image.shape == (1, 8, 8, 3)
    np.testing.assert_array_equal(seen[0][0], field)
    np.testing.assert_array_equal(seen[0][1], valid)
    simulator.cfg.tactile_img_res = (16, 16)
    with pytest.raises(ValueError, match="matching camera and output resolutions"):
        namespace["optical_simulation"](simulator)


def test_actual_optical_wrapper_forwards_absolute_fields_and_separate_masks(taxim):
    package_dir = Path(importlib.util.find_spec("tacex").origin).parent
    source = ast.parse((package_dir / "simulation_approaches/gpu_taxim/taxim_sim.py").read_text())
    method = next(node for node in ast.walk(source)
                  if isinstance(node, ast.FunctionDef) and node.name == "optical_simulation")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "taxim_sim.optical_simulation", "exec"), namespace)
    reference = torch.linspace(0.2, 0.9, 64).reshape(1, 8, 8)
    current = reference - 0.1
    current_valid = torch.ones_like(current, dtype=torch.bool)
    reference_valid = current_valid.clone()
    current_valid[:, 3, 3] = False
    reference_valid[:, 4, 4] = False
    delta = current - reference
    seen = []

    def render_surface_difference(height, reference_height, valid_mask, reference_valid_mask):
        seen.append((height.clone(), reference_height.clone(), valid_mask.clone(), reference_valid_mask.clone()))
        return torch.full((1, 3, 8, 8), 0.4)

    simulator = SimpleNamespace(
        sensor=SimpleNamespace(_data=SimpleNamespace(output={"height_map": delta}),
            _robodojo_nominal_depth=reference * 0.001,
            _robodojo_depth_valid=current_valid & reference_valid,
            _robodojo_surface_height_mm=current, _robodojo_reference_surface_height_mm=reference,
            _robodojo_surface_valid=current_valid, _robodojo_reference_surface_valid=reference_valid),
        cfg=SimpleNamespace(input_mode="deformed_surface", tactile_img_res=(8, 8)),
        _device="cpu", _taxim=SimpleNamespace(render_surface_difference=render_surface_difference),
        tactile_rgb_img=torch.zeros(1, 8, 8, 3),
    )
    image = namespace["optical_simulation"](simulator)
    np.testing.assert_array_equal(image, torch.full_like(image, 0.4))
    for actual, expected in zip(seen[0], (current, reference, current_valid, reference_valid)):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(simulator.sensor._data.output["height_map"], delta)


def test_wrapper_does_not_use_stale_absolute_fields_after_nominal_is_removed(taxim):
    package_dir = Path(importlib.util.find_spec("tacex").origin).parent
    source = ast.parse((package_dir / "simulation_approaches/gpu_taxim/taxim_sim.py").read_text())
    method = next(node for node in ast.walk(source)
                  if isinstance(node, ast.FunctionDef) and node.name == "optical_simulation")
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "taxim_sim.optical_simulation", "exec"), namespace)
    zero = torch.zeros(1, 8, 8)
    seen = []

    def render_surface(height, valid_mask=None):
        seen.append(height.clone())
        return torch.zeros(1, 3, 8, 8)

    def stale_surface(*args, **kwargs):
        raise AssertionError("A removed nominal reference invalidates the old absolute capture")

    simulator = SimpleNamespace(
        sensor=SimpleNamespace(_data=SimpleNamespace(output={"height_map": zero}),
            _robodojo_surface_height_mm=torch.ones_like(zero),
            _robodojo_reference_surface_height_mm=torch.ones_like(zero)),
        cfg=SimpleNamespace(input_mode="deformed_surface", tactile_img_res=(8, 8)),
        _device="cpu", _taxim=SimpleNamespace(render_surface=render_surface,
            render_surface_difference=stale_surface),
        tactile_rgb_img=torch.zeros(1, 8, 8, 3),
    )
    namespace["optical_simulation"](simulator)
    np.testing.assert_array_equal(seen[0], zero)


def test_actual_reset_restores_reference_normals_only_for_selected_environments(taxim):
    package_dir = Path(importlib.util.find_spec("tacex").origin).parent
    sensor_source = ast.parse((package_dir / "gelsight_sensor.py").read_text())
    reset = next(node for node in ast.walk(sensor_source)
                 if isinstance(node, ast.FunctionDef) and node.name == "reset")
    reset.returns = None
    for argument in reset.args.args:
        argument.annotation = None

    class ResetBase:
        def reset(self, env_ids):
            pass

    sensor_class = ast.ClassDef(name="ResetSensor", bases=[ast.Name(id="ResetBase", ctx=ast.Load())],
                               keywords=[], body=[reset], decorator_list=[])
    namespace = {"ResetBase": ResetBase, "torch": torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[sensor_class], type_ignores=[])),
                 "gelsight_sensor.reset", "exec"), namespace)
    wrapper_source = ast.parse((package_dir / "simulation_approaches/gpu_taxim/taxim_sim.py").read_text())
    optical = next(node for node in ast.walk(wrapper_source)
                   if isinstance(node, ast.FunctionDef) and node.name == "optical_simulation")
    exec(compile(ast.Module(body=[optical], type_ignores=[]), "taxim_sim.optical_simulation", "exec"), namespace)
    x, y = coordinates(taxim, 48, 64)
    reference = torch.stack([0.03 * x * x, 0.04 * y * y])
    current = reference - 0.1 * torch.exp(-(x * x + y * y) / 2)
    valid = torch.ones_like(current, dtype=torch.bool)
    current_valid = valid.clone()
    current_valid[0, 20:25, 20:25] = False
    current_valid[1, 30:33, 30:33] = False
    unselected_valid = current_valid[1].clone()
    sensor = namespace["ResetSensor"]()
    sensor._ALL_INDICES = torch.arange(2)
    sensor._frame = torch.ones(2, dtype=torch.int64)
    sensor.camera = SimpleNamespace(reset=lambda: None)
    sensor._indentation_depth = torch.ones(2)
    sensor._data = SimpleNamespace(output={"height_map": current-reference,
        "tactile_rgb": torch.zeros(2, 48, 64, 3, dtype=torch.uint8)})
    sensor._robodojo_nominal_depth = reference * 0.001
    sensor._robodojo_surface_height_mm = current.clone()
    sensor._robodojo_reference_surface_height_mm = reference.clone()
    sensor._robodojo_surface_valid = current_valid
    sensor._robodojo_reference_surface_valid = valid
    sensor._robodojo_depth_valid = valid
    sensor.marker_motion_simulator = None
    sensor._update_marker_rgb = lambda: None
    sensor._as_uint8_rgb = lambda image: (image.clamp(0, 1) * 255).to(torch.uint8)
    simulator = SimpleNamespace(sensor=sensor, cfg=SimpleNamespace(input_mode="deformed_surface",
        tactile_img_res=(64, 48)), _device="cpu", _taxim=taxim, tactile_rgb_img=torch.zeros(2, 48, 64, 3))
    sensor.optical_simulator = SimpleNamespace(
        optical_simulation=lambda: namespace["optical_simulation"](simulator), reset=lambda: None)
    before = namespace["optical_simulation"](simulator).clone()
    assert float((before[0] - taxim.render_surface(torch.zeros_like(x)).movedim(0, 2)).abs().mean()) > 1e-4
    sensor.reset(torch.tensor([0]))
    baseline = sensor._as_uint8_rgb(taxim.render_surface(torch.zeros_like(x)).movedim(0, 2))
    np.testing.assert_array_equal(sensor._data.output["tactile_rgb"][0], baseline)
    np.testing.assert_array_equal(sensor._data.output["tactile_rgb"][1], sensor._as_uint8_rgb(before[1]))
    np.testing.assert_array_equal(sensor._robodojo_surface_height_mm[0], reference[0])
    np.testing.assert_array_equal(sensor._robodojo_surface_height_mm[1], current[1])
    np.testing.assert_array_equal(sensor._robodojo_surface_valid[0], valid[0])
    np.testing.assert_array_equal(sensor._robodojo_surface_valid[1], unselected_valid)
