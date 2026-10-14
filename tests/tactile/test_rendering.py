"""The optical proxy must use solver topology, even when USD visuals differ."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("pxr")
pytest.importorskip("uipc")
from pxr import Usd, UsdGeom
from uipc import view
from uipc.geometry import label_surface, tetmesh

from env.tactile.gelsight import _TacExGelSight


def test_proxy_uses_live_solver_vertices_instead_of_stale_visual_topology(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    source = UsdGeom.Mesh.Define(stage, "/Gel/mesh")
    source.CreatePointsAttr([(0, 0, 0)] * 984)
    source.CreateFaceVertexCountsAttr([3])
    source.CreateFaceVertexIndicesAttr([981, 982, 983])
    # One tetrahedron has three exposed points and a fixed rear point.
    points = np.array([[0, 0, 0], [-.01, -.01, .004], [.01, -.01, .004], [0, .01, .004]])
    geometry = tetmesh(points, np.array([[0, 1, 2, 3]], dtype=np.int32))
    label_surface(geometry)
    usd = ModuleType("omni.usd")
    usd.get_context = lambda: SimpleNamespace(get_stage=lambda: stage)
    monkeypatch.setitem(sys.modules, "omni.usd", usd)
    transform = SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.eye(4))))
    sensor = _TacExGelSight.__new__(_TacExGelSight)
    sensor._render_proxy_mesh = None
    sensor.gelpad = SimpleNamespace(
        geo_slot_list=[SimpleNamespace(geometry=lambda: geometry)],
        init_world_transform=transform,
        _usd_mesh_prim=source.GetPrim(),
    )
    sensor.setup_render_proxy("/Proxy")
    proxy = sensor._render_proxy_mesh
    assert len(proxy.GetPointsAttr().Get()) == 4
    assert max(proxy.GetFaceVertexIndicesAttr().Get()) == 3
    assert source.GetVisibilityAttr().Get() == "invisible"
    view(geometry.positions())[3, 2, 0] += .0003
    sensor.update_render_proxy()
    np.testing.assert_allclose(np.asarray(proxy.GetPointsAttr().Get())[3], [0, .01, .0043], atol=1e-8)
