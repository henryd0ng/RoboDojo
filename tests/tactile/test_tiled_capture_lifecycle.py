"""Exercise real capture class methods with CPU fakes; no Kit/Warp launch."""

import ast
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest

ROOT = Path(__file__).resolve().parents[2]
CAPTURE = ROOT / 'env/camera_manager/capture'


def load_class(file, name, namespace):
    tree = ast.parse((CAPTURE / file).read_text())
    cls = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), cls], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(CAPTURE / file), 'exec'), namespace)
    return namespace[name]


@pytest.fixture
def capture(monkeypatch):
    events = []
    launches = []
    uint8 = object()
    spec = {'rgb': {'name': 'rgba', 'channels': 4, 'dtype': uint8}}

    class Buffer:
        def __init__(self, shape=(2, 2, 4), dtype=uint8, device='cuda:0', ptr=1234):
            self.shape, self.dtype, self.device, self.ptr = shape, dtype, device, ptr
        def flatten(self):
            return self

    class Product:
        def __init__(self, path):
            self.path = path
            self.destroyed = False
            self.graph_valid = True
        def destroy(self):
            assert not self.destroyed, 'Product destroyed twice'
            # Pinned Replicator HydraTexture.destroy uses startswith, not a
            # path-component boundary, when removing SyntheticData graphs.
            for product in products:
                if product.path.startswith(self.path):
                    product.graph_valid = False
            self.destroyed = True
            events.append(('destroy', self.path))

    class Annotator:
        def __init__(self):
            self.path = None
            self.buffer = Buffer()
        def attach(self, path):
            self.path = path
            events.append(('attach', path))
        def detach(self, paths):
            assert paths == [self.path]
            assert next(product for product in reversed(products) if product.path == self.path).graph_valid, 'Invalid NodeObj'
            events.append(('detach', self.path))
            self.path = None
        def get_data(self, **kwargs):
            assert self.path is not None, 'Reading a detached old annotator'
            return self.buffer

    products = []
    annotators = []
    def create_product(**kwargs):
        base = '/Render/' + kwargs['name']
        path, suffix = base, 0
        while any(product.path == path and not product.destroyed for product in products):
            suffix += 1
            path = f'{base}_{suffix:02d}'
        product = Product(path)
        products.append(product)
        events.append(('create', product.path))
        return product
    def create_annotator(*args, **kwargs):
        annotator = Annotator()
        annotators.append(annotator)
        return annotator
    rep = SimpleNamespace(create=SimpleNamespace(render_product_tiled=create_product),
                          AnnotatorRegistry=SimpleNamespace(get_annotator=create_annotator))
    class XForm:
        def __init__(self, prim_paths_expr, name, **kwargs):
            self.name = name
            self.prim_paths = list(prim_paths_expr)
            self.prims = [object() for _ in prim_paths_expr]
        def __del__(self):
            pass
    wp = SimpleNamespace(zeros=lambda shape, dtype, device: Buffer(shape, dtype, device),
                         launch=lambda **kwargs: launches.append(kwargs))
    monkeypatch.setitem(__import__('sys').modules, 'warp', wp)
    namespace = {'XFormPrim': XForm, 'np': np, 'wp': wp, 'rep': rep,
                 'ANNOTATOR_SPEC': spec, 'reshape_tiled_image': object()}
    view_cls = load_class('camera_view.py', 'CameraView', namespace)
    monkeypatch.setitem(__import__('sys').modules, 'env.camera_manager.capture.camera_view',
                        SimpleNamespace(ANNOTATOR_SPEC=spec, CameraView=view_cls))
    manager_cls = load_class('tiled_capture_manager.py', 'TiledCaptureManager',
                            {'CameraView': view_cls, 'OmegaConf': OmegaConf, 'deepcopy': deepcopy,
                             'get_current_stage': lambda: stage})
    stage = SimpleNamespace(GetPrimAtPath=lambda path: SimpleNamespace(
        IsValid=lambda: any(product.path == path and not product.destroyed for product in products)))
    camera_manager = SimpleNamespace(cameras=[[SimpleNamespace(prim_path='/World/cam0', _resolution=(2, 2))]],
                                     camera_names=[['cam_head']])
    cfg = OmegaConf.create({'annotator': {'common': {'enabled': True, 'color': {'type': 'rgb'}}}})
    manager = manager_cls(1, cfg, camera_manager, 'cpu')
    manager.initialize(object())
    manager.reset()
    yield SimpleNamespace(manager=manager, cameras=camera_manager, products=products, annotators=annotators,
                          launches=launches, events=events, Buffer=Buffer, CameraView=view_cls)
    manager.destroy()


def test_reset_detaches_before_destroy_and_drops_every_old_handle(capture):
    manager = capture.manager
    old_view = manager.tiled_cameras[0]
    old_product = capture.products[0]
    old_annotator = capture.annotators[0]
    old_buffer = manager._output_buffers[0]['rgb']
    capture.cameras.cameras = [[SimpleNamespace(prim_path='/World/new_cam0', _resolution=(2, 2))]]
    capture.cameras.camera_names = [['cam_head']]
    manager.reset()
    assert capture.events == [('create', old_product.path), ('attach', old_product.path),
                              ('detach', old_product.path), ('destroy', old_product.path),
                              ('create', capture.products[1].path), ('attach', capture.products[1].path)]
    assert old_view._render_product is None and old_view._render_product_path is None
    assert old_view._annotators == {} and old_annotator.path is None
    assert manager.tiled_cameras[0].prim_paths == ['/World/new_cam0']
    assert manager._output_buffers[0]['rgb'] is not old_buffer
    assert manager.tiled_render_products == [capture.products[1]]
    old_view._clean_up_tiled_sensor()  # Destructor/repeated cleanup cannot destroy the replacement.
    manager.destroy()
    manager.destroy()
    assert not manager.tiled_cameras and not manager.tiled_render_products and not manager._output_buffers
    assert len(capture.cameras.cameras) == 1  # No mutation through a shared list alias.
    assert len([event for event in capture.events if event[0] == 'destroy']) == 2


def test_invalid_or_empty_annotator_output_never_launches_warp(capture):
    view = capture.manager.tiled_cameras[0]
    annotator = capture.annotators[0]
    for data in (None, capture.Buffer(shape=(0,)), capture.Buffer(ptr=0),
                 capture.Buffer(shape=(2, 1, 4)), capture.Buffer(dtype=object())):
        annotator.buffer = data
        with pytest.raises(RuntimeError, match='has no valid rgb output'):
            view.get_data('rgb', out=capture.manager._output_buffers[0]['rgb'])
        assert capture.launches == []
    annotator.buffer = capture.Buffer()
    out, info = view.get_data('rgb', out=capture.manager._output_buffers[0]['rgb'])
    assert len(capture.launches) == 1
    assert out is capture.manager._output_buffers[0]['rgb'] and info == {}
    assert capture.launches[0]['inputs'][0] is annotator.buffer


def test_detach_failure_does_not_create_replacement_or_silently_continue(capture):
    old = capture.annotators[0].detach
    capture.annotators[0].detach = lambda paths: (_ for _ in ()).throw(RuntimeError('detach failed'))
    with pytest.raises(RuntimeError, match='detach failed'):
        capture.manager.destroy()
    assert len(capture.products) == 1 and not capture.products[0].destroyed
    capture.annotators[0].detach = old


def test_soft_reset_reuses_products_and_buffers_but_reads_fresh_annotator_data(capture):
    manager = capture.manager
    view, product = manager.tiled_cameras[0], capture.products[0]
    buffer = manager._output_buffers[0]['rgb']
    for _ in range(3):
        # Replacing a Python Camera handle with the same stage/path/resolution
        # must not allocate a new render product on every soft reset.
        capture.cameras.cameras = [[SimpleNamespace(prim_path='/World/cam0', _resolution=(2, 2))]]
        manager.reset()
        fresh = capture.Buffer(ptr=2000 + len(capture.launches))
        capture.annotators[0].buffer = fresh
        view.get_data('rgb', out=buffer)
        assert capture.launches[-1]['inputs'][0] is fresh
    assert manager.tiled_cameras == [view] and capture.products == [product]
    assert manager._output_buffers[0]['rgb'] is buffer
    assert capture.events == [('create', product.path), ('attach', product.path)]


def test_release_detaches_all_views_before_prefix_colliding_products_are_destroyed(capture):
    manager = capture.manager
    # Reproduce the old default name and its Replicator-generated suffix.
    first = manager.tiled_cameras[0]
    second = capture.CameraView(['/World/cam1'], name=first.name,
                                camera_resolution=(2, 2), output_annotators=['rgb'])
    manager.tiled_cameras.append(second)
    manager.tiled_render_products.append(second._render_product)
    assert second._render_product.path.startswith(first._render_product.path)
    capture.events.clear()
    manager.destroy()
    assert [event[0] for event in capture.events] == ['detach', 'detach', 'destroy', 'destroy']


def test_distinct_cameras_have_non_prefix_product_names(capture):
    manager = capture.manager
    capture.cameras.cameras[0].append(SimpleNamespace(prim_path='/World/cam1', _resolution=(2, 2)))
    capture.cameras.camera_names[0].append('cam_wrist')
    manager.reset()
    paths = [product.path for product in manager.tiled_render_products]
    assert len(paths) == 2
    assert not paths[0].startswith(paths[1]) and not paths[1].startswith(paths[0])
