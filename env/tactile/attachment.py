"""Material-relative soft backing penalties without a simulation dependency."""

from __future__ import annotations

import math

import numpy as np


def _positive_scalar(value, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be positive and finite") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def material_backing_strengths(
    points,
    tetrahedra,
    backing_indices,
    *,
    youngs_modulus_pa: float,
    poisson_ratio: float,
    mass_density: float,
    dt: float,
    multiplier: float,
) -> np.ndarray:
    """Return UIPC strengths in the supplied backing-index order.

    Coordinates are metres, Young's modulus is Pa, density is kg/m³, and
    ``dt`` is seconds. UIPC's soft-position energy is ``s_i m_i |dx|² / 2``;
    its elastic energy includes ``dt²``. Set the dimensionless strength to
    ``s_i = multiplier * dt² * lambda_max(K_ii) / m_i``, where ``K_ii`` is
    the rest-state elastic stiffness diagonal block in N/m and ``m_i`` is
    lumped nodal mass. This is a numerical penalty, not a hard clamp or a
    material calibration. The Stable Neo-Hookean rest tangent has the
    configured isotropic Lamé moduli.

    Inputs are not modified. Every tetrahedron must have positive volume;
    selected nodes must belong to at least one tetrahedron.
    """
    youngs = _positive_scalar(youngs_modulus_pa, "youngs_modulus_pa")
    density = _positive_scalar(mass_density, "mass_density")
    timestep = _positive_scalar(dt, "dt")
    factor = _positive_scalar(multiplier, "multiplier")
    try:
        poisson = float(poisson_ratio)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("poisson_ratio must be finite and satisfy -1 < value < 0.5") from error
    if not math.isfinite(poisson) or not -1 < poisson < 0.5:
        raise ValueError("poisson_ratio must be finite and satisfy -1 < value < 0.5")

    vertices = np.asarray(points, dtype=np.float64)
    elements = np.asarray(tetrahedra)
    indices = np.asarray(backing_indices)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("points must be a finite nonempty N x 3 array")
    if elements.ndim != 2 or elements.shape[1] != 4 or not len(elements) or elements.dtype.kind not in "iu":
        raise ValueError("tetrahedra must be a nonempty integer M x 4 array")
    if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu":
        raise ValueError("backing_indices must be a nonempty integer vector")
    if np.any(elements < 0) or np.any(elements >= len(vertices)):
        raise ValueError("tetrahedra contain out-of-range vertex indices")
    if np.any(indices < 0) or np.any(indices >= len(vertices)) or len(np.unique(indices)) != len(indices):
        raise ValueError("backing_indices must be unique, in-range vertex indices")
    elements = elements.astype(np.int64, copy=False)
    indices = indices.astype(np.int64, copy=False)

    tet_points = vertices[elements]
    rest_edges = np.stack([tet_points[:, j] - tet_points[:, 0] for j in (1, 2, 3)], axis=2)
    volumes = np.linalg.det(rest_edges) / 6.0
    if not np.isfinite(volumes).all() or np.any(volumes <= 0):
        raise ValueError("tetrahedra must have finite positive volume")
    inverse = np.linalg.inv(rest_edges)
    gradients = np.concatenate((-inverse.sum(axis=1)[:, None, :], inverse), axis=1)

    mu = youngs / (2 * (1 + poisson))
    lame_lambda = youngs * poisson / ((1 + poisson) * (1 - 2 * poisson))
    nodal_mass = np.zeros(len(vertices), dtype=np.float64)
    stiffness = np.zeros((len(vertices), 3, 3), dtype=np.float64)
    for local_vertex in range(4):
        node_ids = elements[:, local_vertex]
        gradient = gradients[:, local_vertex]
        squared_norm = np.einsum("ij,ij->i", gradient, gradient)
        block = volumes[:, None, None] * (
            mu * squared_norm[:, None, None] * np.eye(3)
            + (lame_lambda + mu) * gradient[:, :, None] * gradient[:, None, :]
        )
        np.add.at(nodal_mass, node_ids, density * volumes / 4.0)
        np.add.at(stiffness, node_ids, block)
    if (np.any(nodal_mass[indices] <= 0) or not np.isfinite(nodal_mass[indices]).all()
            or not np.isfinite(stiffness[indices]).all()):
        raise ValueError("selected backing nodes must have positive finite mass and stiffness")
    eigenvalues = np.linalg.eigvalsh(stiffness[indices])[:, -1]
    strengths = factor * (timestep * timestep) * eigenvalues / nodal_mass[indices]
    if not np.isfinite(strengths).all() or np.any(strengths <= 0):
        raise ValueError("computed backing strengths must be positive and finite")
    return strengths
