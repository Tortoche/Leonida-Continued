#!/usr/bin/env python3
"""Navmesh surface triangulation, cross-tile portal stitching, and A* navigation runtime hooks.

RAGE navmesh tiles store walkable polygonal boundaries with up to 15 vertices per polygon.
Existing inventory tools extract 2D/3D boundary polylines but lack interior surface triangulation
and inter-tile portal links.

This tool provides:
1. Robust clean-room ear-clipping triangulation with Newell plane normal estimation and dominant
   axis projection (handling convex and concave simple 3D polygons).
2. Barycentric surface height interpolation: computes exact height at any (x, y) coordinate
   across sloped polygons, ramps, stairs, and terrain surfaces.
3. Inter-tile portal link extraction: detects open border edges (target 0xFFFFFFFF, 0xFFFF)
   along shared tile boundaries and stitches them into traversable navigation portals.
4. Runtime A* pathfinder: computes shortest traversable paths across triangulated polygons,
   inter-tile portals, and special links.
5. Wavefront OBJ surface export (with triangle faces 'f v0 v1 v2') and structured JSON export.

Pure Python standard library (clean-room implementation, zero game data).
"""

import argparse
from collections import defaultdict, deque
import heapq
import json
import math
import os
from pathlib import Path
import struct
import sys
import tempfile

# Ensure sibling imports work under python -I
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import navmesh_decode as nd
    from navmesh_nodes import navmesh_resources, sector_of
    from report_paths import same_report_path
except ImportError:
    nd = None
    navmesh_resources = None
    sector_of = None
    same_report_path = None


# -----------------------------------------------------------------------------
# 1. Computational Geometry: Normals, Projections & Ear Clipping
# -----------------------------------------------------------------------------

def polygon_normal_newell(points):
    """Compute Newell's plane normal for a 3D polygon.

    points: list of (x, y, z) tuples.
    Returns (nx, ny, nz) normalized, or (0.0, 0.0, 1.0) if degenerate.
    """
    n = len(points)
    if n < 3:
        return 0.0, 0.0, 1.0
    nx = ny = nz = 0.0
    for i in range(n):
        curr = points[i]
        nxt = points[(i + 1) % n]
        nx += (curr[1] - nxt[1]) * (curr[2] + nxt[2])
        ny += (curr[2] - nxt[2]) * (curr[0] + nxt[0])
        nz += (curr[0] - nxt[0]) * (curr[1] + nxt[1])
    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if length < 1e-12:
        return 0.0, 0.0, 1.0
    return nx / length, ny / length, nz / length


def project_to_dominant_plane(points, normal=None):
    """Project 3D vertices to 2D by dropping the axis along the normal's largest component.

    Returns (pts_2d, dropped_axis, is_inverted) where:
    - pts_2d: list of (u, v) tuples
    - dropped_axis: 0 for X, 1 for Y, 2 for Z
    - is_inverted: whether orientation was flipped to ensure counter-clockwise (CCW) order.
    """
    if normal is None:
        normal = polygon_normal_newell(points)
    abs_n = [abs(normal[0]), abs(normal[1]), abs(normal[2])]
    if abs_n[2] >= abs_n[0] and abs_n[2] >= abs_n[1]:
        dropped_axis = 2
        pts_2d = [(p[0], p[1]) for p in points]
        sign_factor = 1.0 if normal[2] >= 0.0 else -1.0
    elif abs_n[0] >= abs_n[1] and abs_n[0] >= abs_n[2]:
        dropped_axis = 0
        pts_2d = [(p[1], p[2]) for p in points]
        sign_factor = 1.0 if normal[0] >= 0.0 else -1.0
    else:
        dropped_axis = 1
        pts_2d = [(p[2], p[0]) for p in points]
        sign_factor = 1.0 if normal[1] >= 0.0 else -1.0

    area = signed_area_2d(pts_2d) * sign_factor
    is_inverted = area < 0.0
    return pts_2d, dropped_axis, is_inverted


def signed_area_2d(pts_2d):
    """Compute 2D signed polygon area using the shoelace formula. Positive = CCW."""
    n = len(pts_2d)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        x1, y1 = pts_2d[i]
        x2, y2 = pts_2d[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area


def cross_product_2d(a, b, c):
    """2D cross product of vector (b - a) and (c - b): (bx - ax)*(cy - by) - (by - ay)*(cx - bx)."""
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])


def is_point_inside_triangle_2d(pt, a, b, c, eps=1e-9):
    """Check if point pt is inside 2D triangle (a, b, c). Works for both CCW and CW triangles."""
    cp1 = cross_product_2d(a, b, pt)
    cp2 = cross_product_2d(b, c, pt)
    cp3 = cross_product_2d(c, a, pt)
    has_neg = (cp1 < -eps) or (cp2 < -eps) or (cp3 < -eps)
    has_pos = (cp1 > eps) or (cp2 > eps) or (cp3 > eps)
    return not (has_neg and has_pos)


def triangulate_polygon_2d(pts_2d):
    """Triangulate a 2D simple polygon using robust ear clipping.

    pts_2d: list of (x, y) coordinates.
    Returns: list of triangle tuples (i0, i1, i2) indexing pts_2d.
    """
    n = len(pts_2d)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    area = signed_area_2d(pts_2d)
    if abs(area) < 1e-12:
        # Collinear or degenerate: simple fan
        return [(0, i, i + 1) for i in range(1, n - 1)]

    # Ensure counter-clockwise winding
    indices = list(range(n))
    if area < 0.0:
        indices.reverse()

    triangles = []
    max_iterations = n * n
    iteration = 0

    while len(indices) > 3 and iteration < max_iterations:
        iteration += 1
        ear_found = False
        m = len(indices)
        for i in range(m):
            prev_idx = indices[(i - 1 + m) % m]
            curr_idx = indices[i]
            next_idx = indices[(i + 1) % m]

            a = pts_2d[prev_idx]
            b = pts_2d[curr_idx]
            c = pts_2d[next_idx]

            # In CCW, an ear vertex must be strictly convex
            if cross_product_2d(a, b, c) <= 1e-9:
                continue

            # Check if any other remaining vertex is inside the ear triangle
            any_inside = False
            for j in range(m):
                test_idx = indices[j]
                if test_idx in (prev_idx, curr_idx, next_idx):
                    continue
                if is_point_inside_triangle_2d(pts_2d[test_idx], a, b, c):
                    any_inside = True
                    break

            if not any_inside:
                triangles.append((prev_idx, curr_idx, next_idx))
                del indices[i]
                ear_found = True
                break

        if not ear_found:
            # Fallback: clip vertex with greatest positive angle or fan
            break

    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    elif len(indices) > 3:
        # Graceful fallback for non-simple or perturbed polygons: triangle fan
        for i in range(1, len(indices) - 1):
            triangles.append((indices[0], indices[i], indices[i + 1]))

    return triangles


def triangulate_polygon_3d(points):
    """Triangulate a 3D polygon with 3 to 15 vertices into non-overlapping triangle faces.

    points: list of (x, y, z) 3D tuples.
    Returns: list of (i0, i1, i2) indices referencing points.
    """
    n = len(points)
    if n < 3:
        return []
    if n == 3:
        return [(0, 1, 2)]

    normal = polygon_normal_newell(points)
    pts_2d, dropped_axis, is_inverted = project_to_dominant_plane(points, normal)
    triangles = triangulate_polygon_2d(pts_2d)

    aligned = []
    for i0, i1, i2 in triangles:
        p0, p1, p2 = points[i0], points[i1], points[i2]
        ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
        vx, vy, vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
        tx = uy * vz - uz * vy
        ty = uz * vx - ux * vz
        tz = ux * vy - uy * vx
        dot = tx * normal[0] + ty * normal[1] + tz * normal[2]
        if dot < 0.0:
            aligned.append((i0, i2, i1))
        else:
            aligned.append((i0, i1, i2))
    return aligned


# -----------------------------------------------------------------------------
# 2. Exact Barycentric Surface Height Query
# -----------------------------------------------------------------------------

def barycentric_coords_2d(px, py, a, b, c):
    """Calculate 2D barycentric coordinates (w0, w1, w2) of point (px, py) relative to (a, b, c)."""
    denom = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(denom) < 1e-12:
        return None, None, None
    w0 = ((b[1] - c[1]) * (px - c[0]) + (c[0] - b[0]) * (py - c[1])) / denom
    w1 = ((c[1] - a[1]) * (px - c[0]) + (a[0] - c[0]) * (py - c[1])) / denom
    w2 = 1.0 - w0 - w1
    return w0, w1, w2


def triangle_height_at(x, y, v0, v1, v2, tolerance=1e-4):
    """Interpolate exact height z at plan coordinates (x, y) over 3D triangle (v0, v1, v2).

    Returns z if (x, y) is inside the triangle plan projection, else None.
    """
    w0, w1, w2 = barycentric_coords_2d(x, y, v0, v1, v2)
    if w0 is None or w1 is None or w2 is None:
        return None
    if w0 >= -tolerance and w1 >= -tolerance and w2 >= -tolerance:
        return w0 * v0[2] + w1 * v1[2] + w2 * v2[2]
    return None


def surface_height_at(polygons_with_triangles, x, y, max_dz=None, target_z=None):
    """Query exact interpolated surface height at (x, y) across triangulated polygons.

    polygons_with_triangles: list of (points_3d, triangle_indices)
    Returns: (best_z, poly_index, height_difference) or None.
    """
    best = None
    for poly_idx, (points, tri_indices) in enumerate(polygons_with_triangles):
        for i0, i1, i2 in tri_indices:
            v0, v1, v2 = points[i0], points[i1], points[i2]
            z = triangle_height_at(x, y, v0, v1, v2)
            if z is not None:
                dz = abs(z - target_z) if target_z is not None else 0.0
                if max_dz is not None and dz > max_dz:
                    continue
                if best is None or dz < best[2]:
                    best = (z, poly_idx, dz)
    return best


# -----------------------------------------------------------------------------
# 3. NavMesh Tile Triangulation
# -----------------------------------------------------------------------------

class TriangulatedPolygon:
    """A triangulated navmesh polygon with 3D vertices, triangles, and metadata."""

    __slots__ = ("poly_id", "points", "triangles", "flags18", "flags28", "area_2d", "centroid")

    def __init__(self, poly_id, points, flags18=0, flags28=0):
        self.poly_id = poly_id
        self.points = list(points)
        self.flags18 = flags18
        self.flags28 = flags28
        self.triangles = triangulate_polygon_3d(self.points)
        pts_2d = [(p[0], p[1]) for p in self.points]
        self.area_2d = abs(signed_area_2d(pts_2d))
        n = len(self.points)
        if n > 0:
            self.centroid = (
                sum(p[0] for p in self.points) / n,
                sum(p[1] for p in self.points) / n,
                sum(p[2] for p in self.points) / n,
            )
        else:
            self.centroid = (0.0, 0.0, 0.0)

    def to_dict(self):
        return {
            "poly_id": self.poly_id,
            "vertex_count": len(self.points),
            "triangle_count": len(self.triangles),
            "triangles": self.triangles,
            "flags18": self.flags18,
            "flags28": self.flags28,
            "area_2d": round(self.area_2d, 3),
            "centroid": [round(c, 3) for c in self.centroid],
        }


class TriangulatedTile:
    """A full navmesh tile with all polygons triangulated."""

    __slots__ = ("sector", "origin", "extents", "polygons", "total_vertices", "total_triangles", "total_area")

    def __init__(self, sector, polygons_data, origin=None, extents=None):
        self.sector = sector
        self.origin = origin or (0.0, 0.0, 0.0)
        self.extents = extents or (128.0, 128.0, 16.0)
        self.polygons = []
        self.total_vertices = 0
        self.total_triangles = 0
        self.total_area = 0.0

        for i, item in enumerate(polygons_data):
            if isinstance(item, (tuple, list)) and len(item) in (2, 3) and isinstance(item[0], (tuple, list)) and (not item[0] or isinstance(item[0][0], (tuple, list, int, float)) and not isinstance(item[1], (tuple, list))):
                pts = item[0]
                f18 = item[1]
                f28 = item[2] if len(item) == 3 else 0
            else:
                pts = item
                f18 = f28 = 0
            tp = TriangulatedPolygon(i, pts, f18, f28)
            self.polygons.append(tp)
            self.total_vertices += len(tp.points)
            self.total_triangles += len(tp.triangles)
            self.total_area += tp.area_2d

    def to_dict(self):
        return {
            "sector": self.sector,
            "origin": [round(c, 3) for c in self.origin],
            "extents": [round(c, 3) for c in self.extents],
            "polygon_count": len(self.polygons),
            "total_vertices": self.total_vertices,
            "total_triangles": self.total_triangles,
            "total_area_m2": round(self.total_area, 2),
            "polygons": [p.to_dict() for p in self.polygons],
        }


def triangulate_tile_geometry(geometry):
    """Triangulate geometry returned by navmesh_nodes.tile_geometry()."""
    if geometry is None:
        return None
    sector = geometry.get("sector", (0, 0))
    sx, sy = sector
    x0 = (sx - 256) * 128.0
    y0 = (sy - 256) * 128.0
    return TriangulatedTile(sector, geometry.get("polygons", []), origin=(x0, y0, 0.0))


# -----------------------------------------------------------------------------
# 4. Inter-Tile Portal Linking
# -----------------------------------------------------------------------------

class NavPortal:
    """A traversable portal linking two adjacent navmesh polygons across tile boundaries."""

    __slots__ = ("tile_a", "poly_a", "tile_b", "poly_b", "p1", "p2", "midpoint", "width")

    def __init__(self, tile_a, poly_a, tile_b, poly_b, p1, p2):
        self.tile_a = tile_a
        self.poly_a = poly_a
        self.tile_b = tile_b
        self.poly_b = poly_b
        self.p1 = p1
        self.p2 = p2
        self.midpoint = (
            0.5 * (p1[0] + p2[0]),
            0.5 * (p1[1] + p2[1]),
            0.5 * (p1[2] + p2[2]),
        )
        self.width = math.sqrt(
            (p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2
        )

    def to_dict(self):
        return {
            "tile_a": self.tile_a,
            "poly_a": self.poly_a,
            "tile_b": self.tile_b,
            "poly_b": self.poly_b,
            "p1": [round(c, 3) for c in self.p1],
            "p2": [round(c, 3) for c in self.p2],
            "midpoint": [round(c, 3) for c in self.midpoint],
            "width": round(self.width, 3),
        }


def _interp_edge_z(p1, p2, tangent_axis, t):
    """Linearly interpolate z coordinate along edge (p1, p2) at tangent coordinate t."""
    t1, t2 = p1[tangent_axis], p2[tangent_axis]
    dt = t2 - t1
    if abs(dt) < 1e-6:
        return 0.5 * (p1[2] + p2[2])
    alpha = (t - t1) / dt
    return p1[2] + alpha * (p2[2] - p1[2])


def find_tile_border_portals(tile_a, tile_b, tolerance=0.1, max_dz=1.5):
    """Find traversable portal edges connecting adjacent navmesh tiles A and B.

    tile_a, tile_b: TriangulatedTile objects or geometry dicts.
    Returns: list of NavPortal instances.
    """
    sec_a = tile_a.sector if hasattr(tile_a, "sector") else tile_a["sector"]
    sec_b = tile_b.sector if hasattr(tile_b, "sector") else tile_b["sector"]

    dsx = sec_b[0] - sec_a[0]
    dsy = sec_b[1] - sec_a[1]

    # Only orthogonally adjacent tiles share 128m borders
    if abs(dsx) + abs(dsy) != 1:
        return []

    polys_a = tile_a.polygons if hasattr(tile_a, "polygons") else tile_a.get("polygons", [])
    polys_b = tile_b.polygons if hasattr(tile_b, "polygons") else tile_b.get("polygons", [])

    portals = []

    # Determine boundary equation
    if dsx == 1:  # B is East of A: shared border is x = max_x_a = min_x_b
        border_x = (sec_a[0] - 256 + 1) * 128.0

        def is_on_border(pt):
            return abs(pt[0] - border_x) <= tolerance

        tangent_axis = 1  # Y axis
    elif dsx == -1:  # B is West of A
        border_x = (sec_a[0] - 256) * 128.0

        def is_on_border(pt):
            return abs(pt[0] - border_x) <= tolerance

        tangent_axis = 1
    elif dsy == 1:  # B is North of A: shared border is y = max_y_a = min_y_b
        border_y = (sec_a[1] - 256 + 1) * 128.0

        def is_on_border(pt):
            return abs(pt[1] - border_y) <= tolerance

        tangent_axis = 0  # X axis
    else:  # B is South of A
        border_y = (sec_a[1] - 256) * 128.0

        def is_on_border(pt):
            return abs(pt[1] - border_y) <= tolerance

        tangent_axis = 0

    # Extract border segments from A
    border_edges_a = []
    for pa_idx, pa in enumerate(polys_a):
        pts = pa.points if hasattr(pa, "points") else pa[0]
        n = len(pts)
        for i in range(n):
            p1, p2 = pts[i], pts[(i + 1) % n]
            if is_on_border(p1) and is_on_border(p2):
                t1, t2 = p1[tangent_axis], p2[tangent_axis]
                t_min, t_max = min(t1, t2), max(t1, t2)
                if t_max - t_min > 1e-4:
                    border_edges_a.append((pa_idx, p1, p2, t_min, t_max))

    # Extract border segments from B
    border_edges_b = []
    for pb_idx, pb in enumerate(polys_b):
        pts = pb.points if hasattr(pb, "points") else pb[0]
        n = len(pts)
        for i in range(n):
            p1, p2 = pts[i], pts[(i + 1) % n]
            if is_on_border(p1) and is_on_border(p2):
                t1, t2 = p1[tangent_axis], p2[tangent_axis]
                t_min, t_max = min(t1, t2), max(t1, t2)
                if t_max - t_min > 1e-4:
                    border_edges_b.append((pb_idx, p1, p2, t_min, t_max))

    # Match overlapping segments
    for pa_idx, a_p1, a_p2, a_min, a_max in border_edges_a:
        for pb_idx, b_p1, b_p2, b_min, b_max in border_edges_b:
            overlap_min = max(a_min, b_min)
            overlap_max = min(a_max, b_max)
            if overlap_max - overlap_min > 1e-4:
                # Interpolate 3D coordinates at overlap interval
                za_min = _interp_edge_z(a_p1, a_p2, tangent_axis, overlap_min)
                za_max = _interp_edge_z(a_p1, a_p2, tangent_axis, overlap_max)
                zb_min = _interp_edge_z(b_p1, b_p2, tangent_axis, overlap_min)
                zb_max = _interp_edge_z(b_p1, b_p2, tangent_axis, overlap_max)
                if abs(za_min - zb_min) <= max_dz and abs(za_max - zb_max) <= max_dz:
                    z1 = 0.5 * (za_min + zb_min)
                    z2 = 0.5 * (za_max + zb_max)
                    if tangent_axis == 1:
                        pt1 = (border_x, overlap_min, z1)
                        pt2 = (border_x, overlap_max, z2)
                    else:
                        pt1 = (overlap_min, border_y, z1)
                        pt2 = (overlap_max, border_y, z2)
                    portals.append(NavPortal(sec_a, pa_idx, sec_b, pb_idx, pt1, pt2))

    return portals


# -----------------------------------------------------------------------------
# 5. NavGraph & A* Pathfinding Runtime Hook
# -----------------------------------------------------------------------------

class NavGraph:
    """Multi-tile navigation graph linking polygons via adjacent edges and portals."""

    def __init__(self):
        self.nodes = {}  # (sector, poly_id) -> TriangulatedPolygon or centroid
        self.edges = defaultdict(list)  # (sector, poly_id) -> list of (neighbor_node, cost, portal_info)

    def add_tile(self, triangulated_tile, intra_tile_adjacencies=None):
        """Add all polygons of a tile to the graph."""
        sec = triangulated_tile.sector
        for p in triangulated_tile.polygons:
            self.nodes[(sec, p.poly_id)] = p

        # If adjacent edges list provided (e.g. from navmesh_decode +0x88 records)
        if intra_tile_adjacencies:
            for src_poly, dst_poly in intra_tile_adjacencies:
                node_a = (sec, src_poly)
                node_b = (sec, dst_poly)
                if node_a in self.nodes and node_b in self.nodes:
                    ca = self.nodes[node_a].centroid
                    cb = self.nodes[node_b].centroid
                    dist = math.sqrt((cb[0] - ca[0]) ** 2 + (cb[1] - ca[1]) ** 2 + (cb[2] - ca[2]) ** 2)
                    self.edges[node_a].append((node_b, dist, None))
                    self.edges[node_b].append((node_a, dist, None))

    def add_portals(self, portals):
        """Add bidirectional inter-tile portal connections to the graph."""
        for p in portals:
            node_a = (p.tile_a, p.poly_a)
            node_b = (p.tile_b, p.poly_b)
            if node_a in self.nodes and node_b in self.nodes:
                ca = self.nodes[node_a].centroid
                cb = self.nodes[node_b].centroid
                dist = math.sqrt((cb[0] - ca[0]) ** 2 + (cb[1] - ca[1]) ** 2 + (cb[2] - ca[2]) ** 2)
                self.edges[node_a].append((node_b, dist, p))
                self.edges[node_b].append((node_a, dist, p))

    def astar(self, start_node, goal_node):
        """A* search between two nodes on the navmesh graph.

        Returns (path_nodes, total_cost, portals_traversed) or (None, inf, None).
        """
        if start_node not in self.nodes or goal_node not in self.nodes:
            return None, float("inf"), None

        goal_pos = self.nodes[goal_node].centroid

        def heuristic(node):
            pos = self.nodes[node].centroid
            return math.sqrt((goal_pos[0] - pos[0]) ** 2 + (goal_pos[1] - pos[1]) ** 2 + (goal_pos[2] - pos[2]) ** 2)

        open_set = []
        heapq.heappush(open_set, (heuristic(start_node), 0.0, start_node))
        came_from = {}
        edge_used = {}
        g_score = {start_node: 0.0}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            if current == goal_node:
                # Reconstruct path
                path = []
                portals_traversed = []
                curr = current
                while curr in came_from:
                    path.append(curr)
                    if curr in edge_used and edge_used[curr] is not None:
                        portals_traversed.append(edge_used[curr])
                    curr = came_from[curr]
                path.append(start_node)
                path.reverse()
                portals_traversed.reverse()
                return path, current_g, portals_traversed

            for neighbor, cost, portal in self.edges[current]:
                tentative_g = current_g + cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    edge_used[neighbor] = portal
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return None, float("inf"), None


# -----------------------------------------------------------------------------
# 6. OBJ Surface & JSON Exporters
# -----------------------------------------------------------------------------

def write_triangulated_obj(out, triangulated_tile, tile_ordinal=0):
    """Write Wavefront OBJ with triangulated surface polygon faces ('f v0 v1 v2')."""
    out.write(f"# Leonida NavMesh Surface Triangulation — Tile {tile_ordinal}\n")
    out.write(f"o tile_{tile_ordinal}\n\n")

    vert_offset = 1
    for p in triangulated_tile.polygons:
        out.write(f"g polygon_{p.poly_id}\n")
        out.write(f"# area: {p.area_2d:.3f} m2, flags18: 0x{p.flags18:X}, flags28: 0x{p.flags28:X}\n")
        for x, y, z in p.points:
            out.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for i0, i1, i2 in p.triangles:
            out.write(f"f {vert_offset + i0} {vert_offset + i1} {vert_offset + i2}\n")
        vert_offset += len(p.points)
        out.write("\n")


# -----------------------------------------------------------------------------
# 7. CLI Main Entry Point
# -----------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("archive", help="navmeshes.rpf archive or carved bin directory")
    parser.add_argument("--tile", type=int, default=0, help="zero-based tile ordinal (default: 0)")
    parser.add_argument("--out", type=Path, help="destination file (.obj or .json)")
    parser.add_argument("--format", choices=("obj", "json"), default="obj", help="export format (default: obj)")
    parser.add_argument("--portals-with", type=int, default=None, help="secondary tile ordinal to compute portals with")
    args = parser.parse_args(argv)

    if args.tile < 0:
        parser.error("--tile must be non-negative")
    if args.portals_with is not None and args.portals_with < 0:
        parser.error("--portals-with must be non-negative")
    if not os.path.exists(args.archive):
        print(f"ERROR: archive not found: {args.archive}", file=sys.stderr)
        return 2

    if args.out:
        if same_report_path and same_report_path(args.archive, args.out):
            print("ERROR: --out aliases the archive", file=sys.stderr)
            return 2
        if os.path.isdir(args.out):
            print("ERROR: --out path is a directory", file=sys.stderr)
            return 2

    try:
        from contextlib import closing
        with closing(navmesh_resources(args.archive)) as resources:
            tile_a_geom = None
            tile_b_geom = None
            for ordinal, tile_bytes in enumerate(resources):
                if ordinal == args.tile:
                    from navmesh_nodes import tile_geometry
                    tile_a_geom = tile_geometry(tile_bytes)
                if args.portals_with is not None and ordinal == args.portals_with:
                    from navmesh_nodes import tile_geometry
                    tile_b_geom = tile_geometry(tile_bytes)
                if tile_a_geom is not None and (args.portals_with is None or tile_b_geom is not None):
                    break

        if tile_a_geom is None:
            print("ERROR: selected tile not found or unreadable", file=sys.stderr)
            return 2

        tile_a = triangulate_tile_geometry(tile_a_geom)
        portals = []
        if tile_b_geom is not None:
            tile_b = triangulate_tile_geometry(tile_b_geom)
            portals = find_tile_border_portals(tile_a, tile_b)

        if args.format == "json":
            result = {
                "tile": tile_a.to_dict(),
                "portals": [p.to_dict() for p in portals],
            }
            content = json.dumps(result, indent=2) + "\n"
        else:
            # Wavefront OBJ
            import io
            buf = io.StringIO()
            write_triangulated_obj(buf, tile_a, args.tile)
            content = buf.getvalue()

        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=args.out.parent, delete=False
            ) as tmp:
                tmp_path = Path(tmp.name)
                tmp.write(content)
            os.replace(tmp_path, args.out)
        else:
            sys.stdout.write(content)

    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
