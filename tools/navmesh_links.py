#!/usr/bin/env python3
"""Decode and verify navmesh +0x88 adjacent-polygon edges and +0xA8/+0xB0 special links.

Reads a retail RPF9 navmeshes.rpf archive (or a directory of carved .bin streams)
in-place and validates the 12-byte adjacent-polygon edge records (+0x88, count u32@0x90),
the 32-byte special-link records (+0xA8, count u32@0xD4), the u16 special-link index
table (+0xB0, count u32@0xD8), and the polygon +0x20 special-link prefix-sum chain
across all 0x00010015 navmesh tiles. Emits aggregate counts and invariant pass/fail
totals only (no game data).
Usage: `python -I tools/navmesh_links.py --help` lists the options.
"""

import argparse
from collections import Counter, defaultdict
import heapq
import json
import math
import mmap
import os
import struct
import sys

try:
    from tools import navmesh_decode as nd
    from tools.report_paths import same_report_path
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import navmesh_decode as nd
    from report_paths import same_report_path


CHECK_NAMES = (
    "edges_valid",
    "poly_link_chain",
    "special_links_valid",
    "link_indices_endpoints",
)


def _counter_top(counter, limit=32):
    return {str(k): v for k, v in counter.most_common(limit)}


def _count_reciprocity(b, polygons, edges, sector, polygon_count, totals):
    """Match directed polygon counts within one tile whose pool sizes agree."""
    records = (record for offset, count in edges
               for record in struct.iter_unpack("<IH6x", memoryview(b)[offset:offset + count * 12]))
    directed = Counter()
    source = position = same = invalid = unlinked = 0
    for offset, count in polygons:
        for i in range(count):
            packed = nd._u32(b, offset + 44 * i + 4)
            if packed >> 4 != position:
                return  # Overlapping or missing spans make edge ownership ambiguous.
            for _ in range(packed & 15):
                target_sector, target = next(records)
                if target_sector == 0xFFFFFFFF and target == 0xFFFF:
                    unlinked += 1
                elif target_sector == sector and target < polygon_count:
                    directed[source, target] += 1
                    same += 1
                else:
                    invalid += 1
            position += packed & 15
            source += 1

    excess = sum(max(0, n - directed[target, source])
                 for (source, target), n in directed.items())
    totals["tiles_counted"] += 1
    totals["same_tile"] += same
    totals["excess_directed"] += excess
    totals["invalid_targets"] += invalid
    totals["unlinked"] += unlinked


ENDPOINT_CLASSES = (
    "reversed", "same_direction", "degenerate", "repeated", "no_exact_return",
)


def _turn(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _xy_winding(points):
    """Sign of twice the area of a simple projected ring, or zero if ambiguous."""
    n = len(points)
    if n < 3:
        return 0
    if n == 3:
        area = _turn(*points)
        return (area > 0) - (area < 0)
    if len(set(points)) != n:
        return 0
    area = 0
    for i, a in enumerate(points):
        b = points[(i + 1) % n]
        area += a[0] * b[1] - b[0] * a[1]
        prev = points[i - 1]
        if (_turn(prev, a, b) == 0 and
                (prev[0] - a[0]) * (b[0] - a[0]) + (prev[1] - a[1]) * (b[1] - a[1]) > 0):
            return 0  # Adjacent segments double back over one another.
        for j in range(i + 2, n if i else n - 1):
            c, d = points[j], points[(j + 1) % n]
            if (max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0]) or
                    max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])):
                continue
            if _turn(a, b, c) * _turn(a, b, d) <= 0 and _turn(c, d, a) * _turn(c, d, b) <= 0:
                return 0
    return (area > 0) - (area < 0)


def _plane_ring(points):
    """Return a canonical plane normal, winding sign and undirected corner cycle."""
    if len(points) < 3 or len(set(points)) != len(points):
        return None
    a, b = points[:2]
    dx, dy, dz = (v - u for u, v in zip(a, b))
    for c in points[2:]:
        ex, ey, ez = (v - u for u, v in zip(a, c))
        normal = (dy * ez - dz * ey, dz * ex - dx * ez, dx * ey - dy * ex)
        if any(normal):
            break
    else:
        return None
    if any(sum(n * (v - u) for n, v, u in zip(normal, point, a)) for point in points):
        return None
    axis = next(i for i, n in enumerate(normal) if n)
    divisor = math.gcd(*normal) * (1 if normal[axis] > 0 else -1)
    plane = tuple(n // divisor for n in normal)
    projected = [(p[(axis + 1) % 3], p[(axis + 2) % 3]) for p in points]
    sign = _xy_winding(projected)
    if not sign:
        return None
    corners = [p for i, p in enumerate(projected)
               if _turn(projected[i - 1], p, projected[(i + 1) % len(points)])]
    boundary = frozenset(tuple(sorted((p, corners[i - 1]))) for i, p in enumerate(corners))
    return plane, sign, boundary


def _plane_relation(left, right):
    if left is None or right is None:
        return "undefined_ring"
    # The exact shared edge supplies a common point, so parallel planes coincide.
    if left[0] != right[0]:
        return "different_planes"
    if left[2] == right[2]:
        return "coincident"
    return "same" if left[1] == right[1] else "opposite"


def _endpoint_counts(b, tables, sector, polygon_count, totals):
    """Compare each slot under both cyclic boundary conventions within this tile."""
    vertices, indices, edges, polygons = tables
    coords = [xyz for offset, count in vertices
              for (xyz,) in struct.iter_unpack("<6s", b[offset:offset + count * 6])]
    ring_indices = [index for offset, count in indices
                    for (index,) in struct.iter_unpack("<H", b[offset:offset + count * 2])]
    targets = (target for offset, count in edges for target in
               struct.iter_unpack("<IH6x", b[offset:offset + count * 12]))
    hypotheses = {"outgoing": Counter(), "incoming": Counter()}
    winding = totals.get("relative_winding")
    if winding is not None:
        xy = [struct.unpack_from("<HH", vertex) for vertex in coords]
        valid_xy = all(math.isfinite(v) and v > 0 for v in nd._f32s(b, 0x60, 2))
        signs = []
    coplanar = totals.get("coplanar_winding")
    if coplanar is not None:
        rings = []
        planes = {}
        extents = nd._f32s(b, 0x60, 3)
        valid_xyz = all(math.isfinite(v) and v >= 0 for v in extents)

        def plane(index):
            if index not in planes:
                points = [tuple(v if extent else 0 for v, extent in zip(struct.unpack("<3H", raw), extents))
                          for raw in rings[index]]
                planes[index] = _plane_ring(points) if valid_xyz else None
            return planes[index]

    position = source = self_refs = unlinked = invalid = 0
    for offset, count in polygons:
        for i in range(count):
            span = nd._u32(b, offset + 44 * i + 4)
            if span >> 4 != position:
                return
            end = position + (span & 15)
            ring = [coords[index] for index in ring_indices[position:end]]
            if coplanar is not None:
                rings.append(ring)
            if winding is not None:
                signs.append(_xy_winding([xy[index] for index in ring_indices[position:end]])
                             if valid_xy else 0)
            for j, vertex in enumerate(ring):
                target_sector, target = next(targets)
                if (target_sector, target) == (0xffffffff, 0xffff):
                    unlinked += 1
                elif target_sector != sector or target >= polygon_count:
                    invalid += 1
                elif target == source:
                    self_refs += 1
                else:
                    hypotheses["outgoing"][source, target, vertex, ring[(j + 1) % len(ring)]] += 1
                    hypotheses["incoming"][source, target, ring[j - 1], vertex] += 1
            position = end
            source += 1

    totals["tiles_counted"] += 1
    totals["self_references"] += self_refs
    totals["unlinked"] += unlinked
    totals["invalid_targets"] += invalid
    for convention, counts in hypotheses.items():
        for (source, target, start, end), n in counts.items():
            reverse = counts[target, source, end, start]
            same = counts[target, source, start, end] if start != end else 0
            if not (same or reverse):
                category = "no_exact_return"
            elif (n > 1 or reverse + same > 1 or
                  (start != end and counts[source, target, end, start])):
                category = "repeated"
            elif start == end:
                category = "degenerate"
            elif reverse:
                category = "reversed"
            else:
                category = "same_direction"
            totals[convention][category] += n
            if winding is not None and convention == "outgoing" and category in winding:
                product = signs[source] * signs[target]
                relation = "ambiguous" if product == 0 else "same" if product > 0 else "opposite"
                winding[category][relation] += n
                if coplanar is not None and category == "same_direction" and relation == "ambiguous":
                    coplanar[_plane_relation(plane(source), plane(target))] += n


class NavPortal:
    """A traversable portal linking two adjacent navmesh polygons across tile boundaries."""

    __slots__ = ("tile_a", "poly_a", "tile_b", "poly_b", "p1", "p2", "midpoint", "width")

    def __init__(self, tile_a, poly_a, tile_b, poly_b, p1, p2, midpoint=None, width=None):
        self.tile_a = tile_a
        self.poly_a = poly_a
        self.tile_b = tile_b
        self.poly_b = poly_b
        self.p1 = tuple(p1)
        self.p2 = tuple(p2)
        if midpoint is not None:
            self.midpoint = tuple(midpoint)
        else:
            self.midpoint = (
                0.5 * (self.p1[0] + self.p2[0]),
                0.5 * (self.p1[1] + self.p2[1]),
                0.5 * (self.p1[2] + self.p2[2]),
            )
        if width is not None:
            self.width = round(float(width), 3)
        else:
            self.width = round(math.sqrt(
                (self.p2[0] - self.p1[0]) ** 2 +
                (self.p2[1] - self.p1[1]) ** 2 +
                (self.p2[2] - self.p1[2]) ** 2
            ), 3)

    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        return {
            "tile_a": self.tile_a,
            "poly_a": self.poly_a,
            "tile_b": self.tile_b,
            "poly_b": self.poly_b,
            "p1": [round(c, 3) for c in self.p1],
            "p2": [round(c, 3) for c in self.p2],
            "midpoint": [round(c, 3) for c in self.midpoint],
            "width": self.width,
        }


def extract_open_border_edges(b, tolerance=0.1):
    """Extract unlinked border edges (adj_sector == 0xFFFFFFFF, adj_poly == 0xFFFF)
    from a navmesh tile byte buffer.

    Returns a list of dicts with 3D coordinates, polygon ID, edge index, and border location.
    """
    if len(b) < 0xF0:
        return []
    c0 = nd._u32(b, 0xC0)
    c4 = nd._u32(b, 0xC4)
    c8 = nd._u32(b, 0xC8)
    pool = nd._u32(b, 0x90)

    b70 = nd.split_array(b, 0x70, 6, c0)
    b80 = nd.split_array(b, 0x80, 2, pool)
    b88 = nd.split_array(b, 0x88, 12, pool)
    b98 = nd.split_array(b, 0x98, 44, c4)

    root = nd.offset_of(nd._u64(b, 0xA0))
    if not (b70 and b80 and b88 and b98 and root and root + 0x20 <= len(b)):
        return []

    lo = nd._f32s(b, root, 3)
    ext = nd._f32s(b, 0x60, 3)
    if not all(math.isfinite(v) for v in (*lo, *ext)):
        return []

    verts = []
    for at, count in b70:
        for vx, vy, vz in struct.iter_unpack("<3H", b[at:at + 6 * count]):
            verts.append((
                lo[0] + vx / 65535.0 * ext[0],
                lo[1] + vy / 65535.0 * ext[1],
                lo[2] + vz / 65535.0 * ext[2],
            ))

    indices = []
    for at, count in b80:
        indices.extend(struct.unpack_from("<%dH" % count, b, at))

    edge_targets = []
    for bo, cnt in b88:
        for asec, apoly in struct.iter_unpack("<IH6x", b[bo:bo + 12 * cnt]):
            edge_targets.append((asec, apoly))

    sx = c8 & 0x1FF
    sy = c8 >> 9
    min_x = (sx - 256) * 128.0
    max_x = min_x + 128.0
    min_y = (sy - 256) * 128.0
    max_y = min_y + 128.0

    border_edges = []
    pos = 0
    global_poly_id = 0
    for po, pcnt in b98:
        for pi in range(pcnt):
            packed = nd._u32(b, po + 44 * pi + 4)
            n_verts = packed & 0xF
            vpos = packed >> 4
            poly_id = global_poly_id
            global_poly_id += 1
            for edge_idx in range(n_verts):
                global_edge = vpos + edge_idx
                if global_edge >= len(edge_targets):
                    break
                asec, apoly = edge_targets[global_edge]
                if asec == 0xFFFFFFFF and apoly == 0xFFFF:
                    v1_idx = indices[vpos + edge_idx]
                    v2_idx = indices[vpos + ((edge_idx + 1) % n_verts)]
                    if v1_idx < len(verts) and v2_idx < len(verts):
                        p1 = verts[v1_idx]
                        p2 = verts[v2_idx]
                        # Classify border
                        border = "interior"
                        if abs(p1[0] - min_x) <= tolerance and abs(p2[0] - min_x) <= tolerance:
                            border = "min_x"
                        elif abs(p1[0] - max_x) <= tolerance and abs(p2[0] - max_x) <= tolerance:
                            border = "max_x"
                        elif abs(p1[1] - min_y) <= tolerance and abs(p2[1] - min_y) <= tolerance:
                            border = "min_y"
                        elif abs(p1[1] - max_y) <= tolerance and abs(p2[1] - max_y) <= tolerance:
                            border = "max_y"
                        border_edges.append({
                            "poly_idx": poly_id,
                            "edge_idx": edge_idx,
                            "p1": p1,
                            "p2": p2,
                            "border": border,
                            "sector": (sx, sy),
                        })
            pos += n_verts

    return border_edges


def stitch_border_portals(tile_a_bytes, tile_b_bytes, tolerance=0.1, max_dz=1.5):
    """Match unlinked border edges across two adjacent navmesh tiles to create traversable portals.

    Returns a list of NavPortal instances.
    """
    edges_a = extract_open_border_edges(tile_a_bytes, tolerance=tolerance)
    edges_b = extract_open_border_edges(tile_b_bytes, tolerance=tolerance)
    if not edges_a or not edges_b:
        return []

    sec_a = edges_a[0]["sector"]
    sec_b = edges_b[0]["sector"]
    dsx = sec_b[0] - sec_a[0]
    dsy = sec_b[1] - sec_a[1]

    if abs(dsx) + abs(dsy) != 1:
        return []

    if dsx == 1:
        border_a, border_b = "max_x", "min_x"
        tangent_idx = 1
        border_coord = (sec_a[0] - 256 + 1) * 128.0
    elif dsx == -1:
        border_a, border_b = "min_x", "max_x"
        tangent_idx = 1
        border_coord = (sec_a[0] - 256) * 128.0
    elif dsy == 1:
        border_a, border_b = "max_y", "min_y"
        tangent_idx = 0
        border_coord = (sec_a[1] - 256 + 1) * 128.0
    else:
        border_a, border_b = "min_y", "max_y"
        tangent_idx = 0
        border_coord = (sec_a[1] - 256) * 128.0

    cands_a = [e for e in edges_a if e["border"] == border_a]
    cands_b = [e for e in edges_b if e["border"] == border_b]

    portals = []
    for ea in cands_a:
        t_a1, t_a2 = ea["p1"][tangent_idx], ea["p2"][tangent_idx]
        min_ta, max_ta = min(t_a1, t_a2), max(t_a1, t_a2)
        z_a1, z_a2 = ea["p1"][2], ea["p2"][2]
        dt_a = t_a2 - t_a1

        for eb in cands_b:
            t_b1, t_b2 = eb["p1"][tangent_idx], eb["p2"][tangent_idx]
            min_tb, max_tb = min(t_b1, t_b2), max(t_b1, t_b2)
            z_b1, z_b2 = eb["p1"][2], eb["p2"][2]
            dt_b = t_b2 - t_b1

            overlap_min = max(min_ta, min_tb)
            overlap_max = min(max_ta, max_tb)
            if overlap_max - overlap_min > 1e-4:
                if abs(dt_a) > 1e-6:
                    za_min = z_a1 + (overlap_min - t_a1) / dt_a * (z_a2 - z_a1)
                    za_max = z_a1 + (overlap_max - t_a1) / dt_a * (z_a2 - z_a1)
                else:
                    za_min = za_max = 0.5 * (z_a1 + z_a2)

                if abs(dt_b) > 1e-6:
                    zb_min = z_b1 + (overlap_min - t_b1) / dt_b * (z_b2 - z_b1)
                    zb_max = z_b1 + (overlap_max - t_b1) / dt_b * (z_b2 - z_b1)
                else:
                    zb_min = zb_max = 0.5 * (z_b1 + z_b2)

                if abs(za_min - zb_min) <= max_dz and abs(za_max - zb_max) <= max_dz:
                    z1 = 0.5 * (za_min + zb_min)
                    z2 = 0.5 * (za_max + zb_max)
                    if tangent_idx == 1:
                        p1 = (border_coord, overlap_min, z1)
                        p2 = (border_coord, overlap_max, z2)
                    else:
                        p1 = (overlap_min, border_coord, z1)
                        p2 = (overlap_max, border_coord, z2)
                    midpoint = (0.5 * (p1[0] + p2[0]), 0.5 * (p1[1] + p2[1]), 0.5 * (z1 + z2))
                    width = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
                    portals.append(NavPortal(
                        tile_a=sec_a,
                        poly_a=ea["poly_idx"],
                        tile_b=sec_b,
                        poly_b=eb["poly_idx"],
                        p1=p1,
                        p2=p2,
                        midpoint=midpoint,
                        width=round(width, 3),
                    ))

    return portals


def find_tile_border_portals(tile_a, tile_b, max_dz=1.5):
    """Match border edges between adjacent triangulated tiles with linear Z interpolation.

    Accepts TriangulatedTile instances or objects with .sector and .polygons attributes.
    """
    sec_a = tile_a.sector if hasattr(tile_a, "sector") else tile_a[0]
    sec_b = tile_b.sector if hasattr(tile_b, "sector") else tile_b[0]
    dsx = sec_b[0] - sec_a[0]
    dsy = sec_b[1] - sec_a[1]
    if abs(dsx) + abs(dsy) != 1:
        return []

    if dsx == 1:
        border_coord = (sec_a[0] - 256 + 1) * 128.0
        tangent_axis = 1
        is_on_border = lambda p: abs(p[0] - border_coord) < 1e-3
    elif dsx == -1:
        border_coord = (sec_a[0] - 256) * 128.0
        tangent_axis = 1
        is_on_border = lambda p: abs(p[0] - border_coord) < 1e-3
    elif dsy == 1:
        border_coord = (sec_a[1] - 256 + 1) * 128.0
        tangent_axis = 0
        is_on_border = lambda p: abs(p[1] - border_coord) < 1e-3
    else:
        border_coord = (sec_a[1] - 256) * 128.0
        tangent_axis = 0
        is_on_border = lambda p: abs(p[1] - border_coord) < 1e-3

    portals = []
    polys_a = tile_a.polygons if hasattr(tile_a, "polygons") else tile_a[1]
    polys_b = tile_b.polygons if hasattr(tile_b, "polygons") else tile_b[1]

    border_edges_a = []
    for pa_idx, pa in enumerate(polys_a):
        pts = pa.points if hasattr(pa, "points") else (pa.vertices if hasattr(pa, "vertices") else pa[0])
        n = len(pts)
        for i in range(n):
            p1, p2 = pts[i], pts[(i + 1) % n]
            if is_on_border(p1) and is_on_border(p2):
                t1, t2 = p1[tangent_axis], p2[tangent_axis]
                t_min, t_max = min(t1, t2), max(t1, t2)
                if t_max - t_min > 1e-4:
                    border_edges_a.append((pa_idx, p1, p2, t_min, t_max))

    border_edges_b = []
    for pb_idx, pb in enumerate(polys_b):
        pts = pb.points if hasattr(pb, "points") else (pb.vertices if hasattr(pb, "vertices") else pb[0])
        n = len(pts)
        for i in range(n):
            p1, p2 = pts[i], pts[(i + 1) % n]
            if is_on_border(p1) and is_on_border(p2):
                t1, t2 = p1[tangent_axis], p2[tangent_axis]
                t_min, t_max = min(t1, t2), max(t1, t2)
                if t_max - t_min > 1e-4:
                    border_edges_b.append((pb_idx, p1, p2, t_min, t_max))

    for pa_idx, a_p1, a_p2, a_min, a_max in border_edges_a:
        t_a1, t_a2 = a_p1[tangent_axis], a_p2[tangent_axis]
        z_a1, z_a2 = a_p1[2], a_p2[2]
        dt_a = t_a2 - t_a1

        for pb_idx, b_p1, b_p2, b_min, b_max in border_edges_b:
            overlap_min = max(a_min, b_min)
            overlap_max = min(a_max, b_max)
            if overlap_max - overlap_min > 1e-4:
                t_b1, t_b2 = b_p1[tangent_axis], b_p2[tangent_axis]
                z_b1, z_b2 = b_p1[2], b_p2[2]
                dt_b = t_b2 - t_b1

                if abs(dt_a) > 1e-6:
                    za_min = z_a1 + (overlap_min - t_a1) / dt_a * (z_a2 - z_a1)
                    za_max = z_a1 + (overlap_max - t_a1) / dt_a * (z_a2 - z_a1)
                else:
                    za_min = za_max = 0.5 * (z_a1 + z_a2)

                if abs(dt_b) > 1e-6:
                    zb_min = z_b1 + (overlap_min - t_b1) / dt_b * (z_b2 - z_b1)
                    zb_max = z_b1 + (overlap_max - t_b1) / dt_b * (z_b2 - z_b1)
                else:
                    zb_min = zb_max = 0.5 * (z_b1 + z_b2)

                if abs(za_min - zb_min) <= max_dz and abs(za_max - zb_max) <= max_dz:
                    z1 = 0.5 * (za_min + zb_min)
                    z2 = 0.5 * (za_max + zb_max)
                    if tangent_axis == 1:
                        pt1 = (border_coord, overlap_min, z1)
                        pt2 = (border_coord, overlap_max, z2)
                    else:
                        pt1 = (overlap_min, border_coord, z1)
                        pt2 = (overlap_max, border_coord, z2)
                    midpoint = (0.5 * (pt1[0] + pt2[0]), 0.5 * (pt1[1] + pt2[1]), 0.5 * (z1 + z2))
                    width = math.sqrt((pt2[0] - pt1[0]) ** 2 + (pt2[1] - pt1[1]) ** 2 + (pt2[2] - pt1[2]) ** 2)
                    portals.append(NavPortal(
                        tile_a=sec_a,
                        poly_a=pa_idx,
                        tile_b=sec_b,
                        poly_b=pb_idx,
                        p1=pt1,
                        p2=pt2,
                        midpoint=midpoint,
                        width=round(width, 3),
                    ))

    return portals


class NavNode:
    """A node in the navigation graph representing a walkable navmesh polygon."""

    __slots__ = ("sector", "poly_id", "centroid", "vertices", "area", "flags18", "flags28", "w20")

    def __init__(self, sector, poly_id, centroid, vertices=None, area=None, flags18=0, flags28=0, w20=0):
        self.sector = sector
        self.poly_id = poly_id
        self.centroid = tuple(centroid)
        self.vertices = [tuple(v) for v in vertices] if vertices else []
        if area is not None:
            self.area = float(area)
        elif self.vertices and len(self.vertices) >= 3:
            pts_2d = [(v[0], v[1]) for v in self.vertices]
            n = len(pts_2d)
            a = 0.0
            for i in range(n):
                x1, y1 = pts_2d[i]
                x2, y2 = pts_2d[(i + 1) % n]
                a += x1 * y2 - x2 * y1
            self.area = abs(0.5 * a)
        else:
            self.area = 0.0
        self.flags18 = flags18
        self.flags28 = flags28
        self.w20 = w20

    @property
    def key(self):
        return (self.sector, self.poly_id)

    @property
    def sector_xy(self):
        if isinstance(self.sector, tuple):
            return self.sector
        return (self.sector & 0x1FF, self.sector >> 9)

    def to_dict(self):
        return {
            "sector": self.sector,
            "poly_id": self.poly_id,
            "centroid": [round(c, 3) for c in self.centroid],
            "vertex_count": len(self.vertices),
            "area": round(self.area, 3),
            "flags18": self.flags18,
            "flags28": self.flags28,
        }


class NavEdge:
    """A directed or bidirectional traversal edge between two NavNodes."""

    __slots__ = ("source", "target", "cost", "edge_type", "portal", "link_data")

    def __init__(self, source, target, cost, edge_type="intra_tile", portal=None, link_data=None):
        self.source = source  # (sector, poly_id)
        self.target = target  # (sector, poly_id)
        self.cost = float(cost)
        self.edge_type = edge_type  # "intra_tile", "portal", "special_link"
        self.portal = portal
        self.link_data = link_data

    def to_dict(self):
        d = {
            "source": self.source,
            "target": self.target,
            "cost": round(self.cost, 3),
            "edge_type": self.edge_type,
        }
        if self.portal:
            d["portal"] = self.portal.to_dict() if hasattr(self.portal, "to_dict") else self.portal
        if self.link_data:
            d["link_data"] = self.link_data
        return d


class PathResult:
    """Result of an A* navigation path query."""

    __slots__ = ("path", "cost", "portals", "waypoints", "edges")

    def __init__(self, path=None, cost=float("inf"), portals=None, waypoints=None, edges=None):
        self.path = path or []
        self.cost = float(cost)
        self.portals = portals or []
        self.waypoints = waypoints or []
        self.edges = edges or []

    def __bool__(self):
        return bool(self.path) and self.cost < float("inf")

    def __len__(self):
        return len(self.path)

    def __iter__(self):
        # Allow unpacking as `path, cost, portals = result`
        yield self.path
        yield self.cost
        yield self.portals

    def to_dict(self):
        return {
            "found": bool(self),
            "node_count": len(self.path),
            "total_cost": round(self.cost, 3) if math.isfinite(self.cost) else None,
            "path": self.path,
            "portals": [p.to_dict() if hasattr(p, "to_dict") else p for p in self.portals],
            "waypoints": [[round(c, 3) for c in wp] for wp in self.waypoints],
        }


def point_in_polygon_2d(pt, vertices):
    """Check if 2D point pt=(x, y) is inside simple polygon vertices [(x0, y0), ...]."""
    x, y = pt[0], pt[1]
    inside = False
    n = len(vertices)
    if n < 3:
        return False
    p1x, p1y = vertices[0][0], vertices[0][1]
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n][0], vertices[i % n][1]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


class NavGraph:
    """Multi-tile navigation graph linking polygons via adjacent edges, portals, and special links."""

    def __init__(self):
        self.nodes = {}  # (sector, poly_id) -> NavNode
        self.adjacency = defaultdict(list)  # (sector, poly_id) -> list of NavEdge
        self.portals = []  # list of NavPortal
        self.special_links = []  # list of special link dicts

    def _resolve_node_key(self, node):
        """Normalize node identifiers to (sector, poly_id) key."""
        if isinstance(node, NavNode):
            return node.key
        if isinstance(node, (tuple, list)):
            if len(node) == 2:
                sec, pid = node
                if isinstance(sec, (tuple, list)) and len(sec) == 2:
                    c8 = (sec[1] << 9) | sec[0]
                    if (c8, pid) in self.nodes:
                        return (c8, pid)
                    if (tuple(sec), pid) in self.nodes:
                        return (tuple(sec), pid)
                    return (c8, pid)
                return (sec, pid)
            if len(node) == 3:
                sx, sy, pid = node
                c8 = (sy << 9) | sx
                if (c8, pid) in self.nodes:
                    return (c8, pid)
                if ((sx, sy), pid) in self.nodes:
                    return ((sx, sy), pid)
                return (c8, pid)
        return node

    def add_node(self, sector, poly_id, centroid, vertices=None, area=None, flags18=0, flags28=0, w20=0):
        key = (sector, poly_id)
        node = NavNode(sector, poly_id, centroid, vertices, area, flags18, flags28, w20)
        self.nodes[key] = node
        return node

    def add_edge(self, source, target, cost=None, edge_type="intra_tile", portal=None, link_data=None, bidirectional=True):
        src_key = self._resolve_node_key(source)
        tgt_key = self._resolve_node_key(target)
        if src_key not in self.nodes or tgt_key not in self.nodes:
            return None
        if cost is None:
            c1 = self.nodes[src_key].centroid
            c2 = self.nodes[tgt_key].centroid
            cost = math.sqrt((c2[0] - c1[0]) ** 2 + (c2[1] - c1[1]) ** 2 + (c2[2] - c1[2]) ** 2)
            if cost <= 0:
                cost = 1e-4

        edge = NavEdge(src_key, tgt_key, cost, edge_type=edge_type, portal=portal, link_data=link_data)
        self.adjacency[src_key].append(edge)
        if bidirectional:
            rev_edge = NavEdge(tgt_key, src_key, cost, edge_type=edge_type, portal=portal, link_data=link_data)
            self.adjacency[tgt_key].append(rev_edge)
        return edge

    def add_portal(self, portal, bidirectional=True):
        if not isinstance(portal, NavPortal):
            portal = NavPortal(
                portal["tile_a"], portal["poly_a"],
                portal["tile_b"], portal["poly_b"],
                portal["p1"], portal["p2"],
                midpoint=portal.get("midpoint"),
                width=portal.get("width"),
            )
        self.portals.append(portal)
        node_a = self._resolve_node_key((portal.tile_a, portal.poly_a))
        node_b = self._resolve_node_key((portal.tile_b, portal.poly_b))
        if node_a in self.nodes and node_b in self.nodes:
            ca = self.nodes[node_a].centroid
            cb = self.nodes[node_b].centroid
            mp = portal.midpoint
            cost = (math.sqrt((mp[0] - ca[0]) ** 2 + (mp[1] - ca[1]) ** 2 + (mp[2] - ca[2]) ** 2) +
                    math.sqrt((cb[0] - mp[0]) ** 2 + (cb[1] - mp[1]) ** 2 + (cb[2] - mp[2]) ** 2))
            if cost <= 0:
                cost = 1e-4
            self.add_edge(node_a, node_b, cost=cost, edge_type="portal", portal=portal, bidirectional=bidirectional)

    def add_tile_bytes(self, b):
        """Parse a navmesh tile byte buffer and add its nodes, intra-tile edges, and special links."""
        if len(b) < 0xF0:
            return None
        c0 = nd._u32(b, 0xC0)
        c4 = nd._u32(b, 0xC4)
        c8 = nd._u32(b, 0xC8)
        pool = nd._u32(b, 0x90)
        flags = nd._u32(b, 0x10)
        d4 = nd._u32(b, 0xD4)

        b70 = nd.split_array(b, 0x70, 6, c0)
        b80 = nd.split_array(b, 0x80, 2, pool)
        b88 = nd.split_array(b, 0x88, 12, pool)
        b98 = nd.split_array(b, 0x98, 44, c4)

        root = nd.offset_of(nd._u64(b, 0xA0))
        if not (b70 and b80 and b88 and b98 and root and root + 0x20 <= len(b)):
            return None

        lo = nd._f32s(b, root, 3)
        ext = nd._f32s(b, 0x60, 3)
        if not all(math.isfinite(v) for v in (*lo, *ext)):
            return None

        verts = []
        for at, count in b70:
            for vx, vy, vz in struct.iter_unpack("<3H", b[at:at + 6 * count]):
                verts.append((
                    lo[0] + vx / 65535.0 * ext[0],
                    lo[1] + vy / 65535.0 * ext[1],
                    lo[2] + vz / 65535.0 * ext[2],
                ))

        indices = []
        for at, count in b80:
            indices.extend(struct.unpack_from("<%dH" % count, b, at))

        edge_targets = []
        for bo, cnt in b88:
            for asec, apoly in struct.iter_unpack("<IH6x", b[bo:bo + 12 * cnt]):
                edge_targets.append((asec, apoly))

        # Decode polygons
        global_poly_id = 0
        intra_edges = []
        for po, pcnt in b98:
            for pi in range(pcnt):
                packed = nd._u32(b, po + 44 * pi + 4)
                n_verts = packed & 0xF
                vpos = packed >> 4
                f18 = nd._u32(b, po + 44 * pi + 0x18)
                w20 = nd._u32(b, po + 44 * pi + 0x20)
                f28 = nd._u32(b, po + 44 * pi + 0x28)
                poly_verts = []
                for vi in range(n_verts):
                    if vpos + vi < len(indices):
                        v_idx = indices[vpos + vi]
                        if v_idx < len(verts):
                            poly_verts.append(verts[v_idx])
                if poly_verts:
                    cx = sum(v[0] for v in poly_verts) / len(poly_verts)
                    cy = sum(v[1] for v in poly_verts) / len(poly_verts)
                    cz = sum(v[2] for v in poly_verts) / len(poly_verts)
                    centroid = (cx, cy, cz)
                else:
                    centroid = (lo[0], lo[1], lo[2])

                self.add_node(c8, global_poly_id, centroid, vertices=poly_verts, flags18=f18, flags28=f28, w20=w20)

                for edge_idx in range(n_verts):
                    global_edge = vpos + edge_idx
                    if global_edge < len(edge_targets):
                        asec, apoly = edge_targets[global_edge]
                        if asec == c8 and apoly < c4 and apoly != global_poly_id:
                            intra_edges.append((global_poly_id, apoly))

                global_poly_id += 1

        # Add intra-tile edges (avoid duplicate symmetric edges)
        seen_edges = set()
        for src_poly, dst_poly in intra_edges:
            pair = (min(src_poly, dst_poly), max(src_poly, dst_poly))
            if pair not in seen_edges:
                seen_edges.add(pair)
                self.add_edge((c8, src_poly), (c8, dst_poly), edge_type="intra_tile", bidirectional=True)

        # Special links (+0xA8)
        if (flags & 2) and d4 > 0:
            a8 = nd._u64(b, 0xA8)
            la = nd.offset_of(a8) if a8 else None
            if la and la + 32 * d4 <= len(b):
                for i in range(d4):
                    p0, p1, s0, s1, f0, f1, heading, pad = struct.unpack_from("<HHIIHHHH", b, la + 32 * i + 12)
                    pos0_raw = struct.unpack_from("<3H", b, la + 32 * i)
                    pos1_raw = struct.unpack_from("<3H", b, la + 32 * i + 6)
                    wp0 = (lo[0] + pos0_raw[0] / 65535.0 * ext[0],
                           lo[1] + pos0_raw[1] / 65535.0 * ext[1],
                           lo[2] + pos0_raw[2] / 65535.0 * ext[2])
                    wp1 = (lo[0] + pos1_raw[0] / 65535.0 * ext[0],
                           lo[1] + pos1_raw[1] / 65535.0 * ext[1],
                           lo[2] + pos1_raw[2] / 65535.0 * ext[2])
                    cost = math.sqrt((wp1[0] - wp0[0]) ** 2 + (wp1[1] - wp0[1]) ** 2 + (wp1[2] - wp0[2]) ** 2)
                    if cost <= 0:
                        cost = 1e-4
                    link_info = {
                        "p0": p0, "p1": p1, "s0": s0, "s1": s1,
                        "f0": f0, "f1": f1, "heading": heading,
                        "pos0": wp0, "pos1": wp1,
                    }
                    self.special_links.append(link_info)
                    node0 = (s0, p0)
                    node1 = (s1, p1)
                    if node0 in self.nodes and node1 in self.nodes:
                        self.add_edge(node0, node1, cost=cost, edge_type="special_link", link_data=link_info, bidirectional=True)

        return c8

    def astar(self, start_node, goal_node, *, allowed_types=None, weight_heuristic=1.0):
        """A* shortest path query between start_node and goal_node.

        start_node, goal_node: (sector, poly_id) tuples or NavNode instances.
        Returns a PathResult object (can be unpacked as path, cost, portals).
        """
        start_key = self._resolve_node_key(start_node)
        goal_key = self._resolve_node_key(goal_node)

        if start_key not in self.nodes or goal_key not in self.nodes:
            return PathResult(path=None, cost=float("inf"))

        if start_key == goal_key:
            ca = self.nodes[start_key].centroid
            return PathResult(path=[start_key], cost=0.0, waypoints=[ca])

        goal_pos = self.nodes[goal_key].centroid

        def heuristic(key):
            pos = self.nodes[key].centroid
            return math.sqrt(
                (goal_pos[0] - pos[0]) ** 2 +
                (goal_pos[1] - pos[1]) ** 2 +
                (goal_pos[2] - pos[2]) ** 2
            ) * weight_heuristic

        open_set = []
        counter = 0
        heapq.heappush(open_set, (heuristic(start_key), 0.0, counter, start_key))
        came_from = {}
        edge_taken = {}
        g_score = {start_key: 0.0}

        while open_set:
            f, current_g, _, current = heapq.heappop(open_set)

            if current == goal_key:
                # Reconstruct path
                path = []
                edges = []
                portals = []
                waypoints = []
                curr = current
                while curr in came_from:
                    path.append(curr)
                    edge = edge_taken[curr]
                    edges.append(edge)
                    if edge.portal is not None:
                        portals.append(edge.portal)
                    curr = came_from[curr]
                path.append(start_key)
                path.reverse()
                edges.reverse()
                portals.reverse()

                # Build 3D waypoints: start centroid -> edge midpoints -> goal centroid
                waypoints.append(self.nodes[start_key].centroid)
                for edge in edges:
                    if edge.portal is not None:
                        mp = edge.portal.midpoint if hasattr(edge.portal, "midpoint") else edge.portal["midpoint"]
                        waypoints.append(mp)
                    elif edge.link_data is not None:
                        pos1 = edge.link_data["pos1"]
                        waypoints.append(pos1)
                    else:
                        waypoints.append(self.nodes[edge.target].centroid)

                return PathResult(
                    path=path,
                    cost=current_g,
                    portals=portals,
                    waypoints=waypoints,
                    edges=edges,
                )

            if current_g > g_score.get(current, float("inf")):
                continue

            for edge in self.adjacency[current]:
                if allowed_types and edge.edge_type not in allowed_types:
                    continue
                neighbor = edge.target
                tentative_g = current_g + edge.cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    edge_taken[neighbor] = edge
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor)
                    counter += 1
                    heapq.heappush(open_set, (f_score, tentative_g, counter, neighbor))

        return PathResult(path=None, cost=float("inf"))

    def find_path(self, start_node, goal_node, *, allowed_types=None, weight_heuristic=1.0):
        """Alias for astar."""
        return self.astar(start_node, goal_node, allowed_types=allowed_types, weight_heuristic=weight_heuristic)

    def find_nearest_polygon(self, x, y, z=0.0, sector=None):
        """Find the navmesh polygon closest to 3D point (x, y, z)."""
        candidates = []
        sec_key = None
        if sector is not None:
            if isinstance(sector, (tuple, list)):
                sec_key = (sector[1] << 9) | sector[0]
            else:
                sec_key = sector

        for key, node in self.nodes.items():
            if sec_key is not None and node.sector != sec_key and node.sector_xy != sector:
                continue
            candidates.append(node)

        if not candidates:
            return None

        # Pass 1: test 2D containment
        contained = []
        for node in candidates:
            if node.vertices and point_in_polygon_2d((x, y), node.vertices):
                contained.append((abs(z - node.centroid[2]), node.key))

        if contained:
            contained.sort(key=lambda item: item[0])
            return contained[0][1]

        # Pass 2: nearest centroid
        best_dist = float("inf")
        best_key = None
        for node in candidates:
            c = node.centroid
            d = (c[0] - x) ** 2 + (c[1] - y) ** 2 + (c[2] - z) ** 2
            if d < best_dist:
                best_dist = d
                best_key = node.key

        return best_key

    def find_path_between_points(self, start_point, goal_point, start_sector=None, goal_sector=None, allowed_types=None):
        """Compute shortest path between two 3D world coordinates."""
        start_node = self.find_nearest_polygon(*start_point, sector=start_sector)
        goal_node = self.find_nearest_polygon(*goal_point, sector=goal_sector)
        if start_node is None or goal_node is None:
            return PathResult(path=None, cost=float("inf"))

        result = self.astar(start_node, goal_node, allowed_types=allowed_types)
        if not result:
            return result

        c_start = self.nodes[start_node].centroid
        c_goal = self.nodes[goal_node].centroid
        d_start = math.sqrt((start_point[0] - c_start[0]) ** 2 + (start_point[1] - c_start[1]) ** 2 + (start_point[2] - c_start[2]) ** 2)
        d_goal = math.sqrt((goal_point[0] - c_goal[0]) ** 2 + (goal_point[1] - c_goal[1]) ** 2 + (goal_point[2] - c_goal[2]) ** 2)

        waypoints = [tuple(start_point)] + result.waypoints + [tuple(goal_point)]
        total_cost = result.cost + d_start + d_goal

        return PathResult(
            path=result.path,
            cost=total_cost,
            portals=result.portals,
            waypoints=waypoints,
            edges=result.edges,
        )

    def connected_components(self):
        """Find connected components of the graph. Returns list of sets of node keys."""
        visited = set()
        components = []

        for node_key in self.nodes:
            if node_key not in visited:
                comp = set()
                queue = [node_key]
                visited.add(node_key)
                while queue:
                    curr = queue.pop()
                    comp.add(curr)
                    for edge in self.adjacency[curr]:
                        if edge.target not in visited:
                            visited.add(edge.target)
                            queue.append(edge.target)
                components.append(comp)

        components.sort(key=len, reverse=True)
        return components

    def to_dict(self):
        """Export structured dictionary summary of the graph."""
        sectors = sorted(list(set(node.sector for node in self.nodes.values())))
        return {
            "node_count": len(self.nodes),
            "edge_count": sum(len(edges) for edges in self.adjacency.values()) // 2,
            "portal_count": len(self.portals),
            "special_link_count": len(self.special_links),
            "connected_components": len(self.connected_components()),
            "sectors": sectors,
            "portals": [p.to_dict() for p in self.portals],
        }


def build_navgraph(tiles, stitch_portals=True, tolerance=0.1, max_dz=1.5):
    """Build a complete navigation graph from an iterable of navmesh tile byte buffers.

    Automatically stitches cross-tile border portals between adjacent sectors.
    """
    graph = NavGraph()
    tile_dict = {}

    for b in tiles:
        if len(b) >= 0xF0:
            c8 = nd._u32(b, 0xC8)
            sx = c8 & 0x1FF
            sy = c8 >> 9
            tile_dict[(sx, sy)] = b
            graph.add_tile_bytes(b)

    if stitch_portals and len(tile_dict) > 1:
        for (sx, sy), tile_bytes in tile_dict.items():
            east = (sx + 1, sy)
            if east in tile_dict:
                portals = stitch_border_portals(tile_bytes, tile_dict[east], tolerance=tolerance, max_dz=max_dz)
                for p in portals:
                    graph.add_portal(p)
            north = (sx, sy + 1)
            if north in tile_dict:
                portals = stitch_border_portals(tile_bytes, tile_dict[north], tolerance=tolerance, max_dz=max_dz)
                for p in portals:
                    graph.add_portal(p)

    return graph


def verify_tile(b, stats):
    """Verify +0x88 adjacent edges, polygon +0x20 link chain, and +0xA8/+0xB0 special links."""
    chk = {}
    if len(b) < 0xF0:
        for name in CHECK_NAMES:
            chk[name] = False
        return chk

    flags = nd._u32(b, 0x10)
    c0, c4, c8, _, d4, d8 = (nd._u32(b, o) for o in (0xC0, 0xC4, 0xC8, 0xCC, 0xD4, 0xD8))
    pool = nd._u32(b, 0x90)
    a8, b0 = nd._u64(b, 0xA8), nd._u64(b, 0xB0)

    # Cross-check u32@0xC0 (c0) and u32@0x90 (pool) against +0x70 (6-byte vertices),
    # +0x80 (2-byte vertex indices < c0), +0x88 (12-byte edges), and the sum of polygon vertex counts in +0x98.
    b70 = nd.split_array(b, 0x70, 6, c0)
    b80 = nd.split_array(b, 0x80, 2, pool)
    b88 = nd.split_array(b, 0x88, 12, pool)
    b98 = nd.split_array(b, 0x98, 44, c4)
    poly_vert_sum = 0
    if b98 is not None:
        for po, pcnt in b98:
            for pi in range(pcnt):
                poly_vert_sum += struct.unpack_from("<I", b, po + 44 * pi + 4)[0] & 0xF
    vidx_in_bounds = b80 is not None and all(nd._u16_max(b, o, n) < c0 for o, n in b80)
    pool_confirmed = b70 is not None and vidx_in_bounds and b98 is not None and poly_vert_sum == pool

    # 1. 12-byte adjacent-polygon edges at +0x88 (count = pool = u32@0x90).
    edges_ok = b88 is not None and pool_confirmed
    if edges_ok and "reciprocity" in stats:
        _count_reciprocity(b, b98, b88, c8, c4, stats["reciprocity"])
    if edges_ok and "endpoint_agreement" in stats:
        _endpoint_counts(b, (b70, b80, b88, b98), c8, c4, stats["endpoint_agreement"])
    same_tile = 0
    unlinked = 0
    edge_flags_lo = Counter()
    edge_meta_hi = Counter()
    if edges_ok:
        for bo, cnt in b88:
            for asec, apoly, meta16, flags32 in struct.iter_unpack("<IHHI", b[bo:bo + 12 * cnt]):
                if asec == c8 and apoly < c4:
                    same_tile += 1
                elif asec == 0xFFFFFFFF and apoly == 0xFFFF:
                    unlinked += 1
                else:
                    edges_ok = False
                    break
                edge_flags_lo[flags32 & 0xFF] += 1
                edge_meta_hi[meta16 >> 8] += 1
            if not edges_ok:
                break
    chk["edges_valid"] = edges_ok
    if edges_ok:
        est = stats["edges"]
        est["total"] += pool
        est["same_tile"] += same_tile
        est["unlinked"] += unlinked
        est["flags_low_byte"].update(edge_flags_lo)
        est["meta_high_byte"].update(edge_meta_hi)

    # 2. 32-byte special links at +0xA8 (count = d4 = u32@0xD4) and u16 index table at +0xB0 (count = d8 = u32@0xD8).
    has_links = bool(flags & 2)
    la = nd.offset_of(a8) if a8 else None
    lb = nd.offset_of(b0) if b0 else None
    if not has_links:
        links_bounds_ok = (a8 == 0 and b0 == 0 and d4 == 0 and d8 == 0)
    else:
        links_bounds_ok = (
            d4 > 0
            and d8 > 0
            and la is not None
            and lb is not None
            and la + 32 * d4 <= len(b)
            and lb + 2 * d8 <= len(b)
        )

    links_ok = links_bounds_ok
    intra = cross_s0 = cross_s1 = 0
    f0_hist = Counter()
    f1_hist = Counter()
    if links_ok and d4 > 0:
        for i in range(d4):
            p0, p1, s0, s1, f0, f1, heading, pad = struct.unpack_from("<HHIIHHHH", b, la + 32 * i + 12)
            if heading >= 256 or pad != 0:
                links_ok = False
                break
            if s0 == c8 and s1 == c8:
                if p0 >= c4 or p1 >= c4:
                    links_ok = False
                    break
                intra += 1
            elif s0 == c8:
                if p0 >= c4:
                    links_ok = False
                    break
                cross_s0 += 1
            elif s1 == c8:
                if p1 >= c4:
                    links_ok = False
                    break
                cross_s1 += 1
            else:
                links_ok = False
                break
            f0_hist[f0] += 1
            f1_hist[f1] += 1
    chk["special_links_valid"] = links_ok
    if links_ok and d4 > 0:
        lst = stats["special_links"]
        lst["link_tiles"] += 1
        lst["total"] += d4
        lst["indices_total"] += d8
        lst["intra_tile"] += intra
        lst["cross_tile_s0"] += cross_s0
        lst["cross_tile_s1"] += cross_s1
        lst["flags0"].update(f0_hist)
        lst["flags1"].update(f1_hist)

    # 3. 44-byte polygons at +0x98: u32@0x20 packs (flags_hi9 << 23) | (link_start << 6) | link_count,
    #    chaining across all c4 polygons to d8 and indexing +0xB0 -> +0xA8 endpoints touching that polygon.
    chain_ok = b98 is not None and links_bounds_ok
    endpoints_ok = chain_ok and links_ok
    lidx = struct.unpack_from(f"<{d8}H", b, lb) if (links_bounds_ok and d8 > 0) else ()
    nxt = 0
    global_poly_id = 0
    max_poly_links = 0
    w20_hi9 = Counter()
    by_li = {}
    if b98 is not None:
        for po, pcnt in b98:
            for pi in range(pcnt):
                w20 = struct.unpack_from("<I", b, po + 44 * pi + 0x20)[0]
                cnt = w20 & 0x3F
                start = (w20 >> 6) & 0x1FFFF
                w20_hi9[w20 >> 23] += 1
                if cnt > max_poly_links:
                    max_poly_links = cnt
                if start != nxt or start + cnt > d8:
                    chain_ok = False
                    endpoints_ok = False
                    break
                if cnt > 0 and endpoints_ok:
                    for k in range(start, start + cnt):
                        li = lidx[k]
                        if li >= d4:
                            endpoints_ok = False
                            break
                        p0, p1, s0, s1 = struct.unpack_from("<HHII", b, la + 32 * li + 12)
                        if not ((p0 == global_poly_id or p1 == global_poly_id) and (s0 == c8 or s1 == c8)):
                            endpoints_ok = False
                            break
                        by_li.setdefault(li, set()).add(global_poly_id)
                nxt += cnt
                global_poly_id += 1
            if not chain_ok:
                break
        if nxt != d8:
            chain_ok = False
            endpoints_ok = False

    # Every link referenced in +0xB0 must be listed by all of its local-tile endpoints (p0 when s0 == c8,
    # and p1 when s1 == c8).
    if endpoints_ok and by_li:
        for li, gpis in by_li.items():
            p0, p1, s0, s1 = struct.unpack_from("<HHII", b, la + 32 * li + 12)
            if (s0 == c8 and p0 not in gpis) or (s1 == c8 and p1 not in gpis):
                endpoints_ok = False
                break

    chk["poly_link_chain"] = chain_ok
    chk["link_indices_endpoints"] = endpoints_ok
    if chain_ok:
        pst = stats["polygons"]
        pst["total"] += c4
        if max_poly_links > pst["max_links_per_poly"]:
            pst["max_links_per_poly"] = max_poly_links
        pst["w20_hi9_flags"].update(w20_hi9)
    return chk


def new_stats():
    return {
        "tiles": 0,
        "checks": {name: {"pass": 0, "fail": 0} for name in CHECK_NAMES},
        "edges": {
            "total": 0,
            "same_tile": 0,
            "unlinked": 0,
            "flags_low_byte": Counter(),
            "meta_high_byte": Counter(),
        },
        "polygons": {
            "total": 0,
            "max_links_per_poly": 0,
            "w20_hi9_flags": Counter(),
        },
        "special_links": {
            "link_tiles": 0,
            "total": 0,
            "indices_total": 0,
            "intra_tile": 0,
            "cross_tile_s0": 0,
            "cross_tile_s1": 0,
            "flags0": Counter(),
            "flags1": Counter(),
        },
    }


def _iter_resources(path, start=0, limit=None):
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path) if f.endswith(".bin"))
        idx = 0
        while idx < len(files):
            with open(os.path.join(path, files[idx]), "rb") as fh:
                d0 = fh.read()
            idx += 1
            if len(d0) < 64:
                continue
            cls = nd.classify(d0)
            buf = bytearray(d0)
            n = 1
            while nd.wants_more(cls, buf, n) and idx < len(files):
                with open(os.path.join(path, files[idx]), "rb") as fh:
                    dn = fh.read()
                if len(dn) >= 64 and nd.classify(dn) != "other":
                    break
                buf.extend(dn)
                idx += 1
                n += 1
            if cls == "navmesh":
                yield bytes(buf)
        return

    size = os.path.getsize(path)
    if size < 8:
        return
    with open(path, "rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as m:
        first = struct.unpack_from("<I", m, 4)[0]
        scan_start = max(first, start)
        scan_end = size if limit is None else min(size, scan_start + limit)
        if scan_start >= scan_end:
            return
        gen = nd.walk(m, scan_start, size)
        pending = None
        while True:
            item = pending if pending is not None else next(gen, None)
            pending = None
            if item is None:
                break
            off0, _, d0 = item
            if off0 >= scan_end:
                break
            if len(d0) < 64:
                continue
            cls = nd.classify(d0)
            buf = bytearray(d0)
            n = 1
            while nd.wants_more(cls, buf, n):
                nxt = next(gen, None)
                if nxt is None:
                    break
                if len(nxt[2]) >= 64 and nd.classify(nxt[2]) != "other":
                    pending = nxt
                    break
                buf.extend(nxt[2])
                n += 1
            if cls == "navmesh":
                yield bytes(buf)


def scan(path, start=0, limit=None, *, reciprocity=False, endpoint_agreement=False, relative_winding=False,
         coplanar_winding=False, stitch_portals=False, build_graph=False):
    relative_winding = relative_winding or coplanar_winding
    stats = new_stats()
    if endpoint_agreement or relative_winding:
        stats["endpoint_agreement"] = {
            "tiles_counted": 0, "self_references": 0, "unlinked": 0, "invalid_targets": 0,
            **{name: dict.fromkeys(ENDPOINT_CLASSES, 0) for name in ("incoming", "outgoing")},
        }
    if relative_winding:
        stats["endpoint_agreement"]["relative_winding"] = {
            category: dict.fromkeys(("same", "opposite", "ambiguous"), 0)
            for category in ("reversed", "same_direction")
        }
    if coplanar_winding:
        stats["endpoint_agreement"]["coplanar_winding"] = dict.fromkeys(
            ("same", "opposite", "different_planes", "coincident", "undefined_ring"), 0)
    if reciprocity:
        stats["reciprocity"] = dict.fromkeys(
            ("tiles_counted", "same_tile", "excess_directed", "invalid_targets", "unlinked"), 0)
    tile_buffers = []
    for buf in _iter_resources(path, start=start, limit=limit):
        stats["tiles"] += 1
        chk = verify_tile(buf, stats)
        for name, ok in chk.items():
            stats["checks"][name]["pass" if ok else "fail"] += 1
        if stitch_portals or build_graph:
            tile_buffers.append(buf)

    extra = {}
    if stitch_portals or build_graph:
        graph = build_navgraph(tile_buffers, stitch_portals=stitch_portals)
        if stitch_portals:
            extra["portals"] = [p.to_dict() for p in graph.portals]
        if build_graph:
            extra["graph"] = graph.to_dict()

    return {
        **({"endpoint_agreement": stats["endpoint_agreement"]} if endpoint_agreement or relative_winding else {}),
        **({"reciprocity": stats["reciprocity"]} if reciprocity else {}),
        "tiles": stats["tiles"],
        "checks": stats["checks"],
        "edges": {
            "total": stats["edges"]["total"],
            "same_tile": stats["edges"]["same_tile"],
            "unlinked": stats["edges"]["unlinked"],
            "flags_low_byte": _counter_top(stats["edges"]["flags_low_byte"]),
            "meta_high_byte": _counter_top(stats["edges"]["meta_high_byte"]),
        },
        "polygons": {
            "total": stats["polygons"]["total"],
            "max_links_per_poly": stats["polygons"]["max_links_per_poly"],
            "w20_hi9_flags": _counter_top(stats["polygons"]["w20_hi9_flags"]),
        },
        "special_links": {
            "link_tiles": stats["special_links"]["link_tiles"],
            "total": stats["special_links"]["total"],
            "indices_total": stats["special_links"]["indices_total"],
            "intra_tile": stats["special_links"]["intra_tile"],
            "cross_tile_s0": stats["special_links"]["cross_tile_s0"],
            "cross_tile_s1": stats["special_links"]["cross_tile_s1"],
            "flags0": _counter_top(stats["special_links"]["flags0"]),
            "flags1": _counter_top(stats["special_links"]["flags1"]),
        },
        **extra,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help="Path to navmeshes.rpf or a directory of carved .bin streams")
    ap.add_argument("--start", type=int, default=0, help="Start byte offset inside an RPF9 archive")
    ap.add_argument("--limit", type=int, default=None, help="Maximum byte span to scan")
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    ap.add_argument("--quiet", action="store_true", help="Suppress stdout summary")
    ap.add_argument("--reciprocity", action="store_true",
                    help="Count same-tile directed edges in excess of their reverse multiplicity")
    ap.add_argument("--endpoint-agreement", action="store_true",
                    help="Compare incoming and outgoing boundary endpoints for same-tile edges")
    ap.add_argument("--coplanar-winding", action="store_true",
                    help="Also classify XY-ambiguous same-direction slots using coplanar 3D rings")
    ap.add_argument("--relative-winding", action="store_true",
                    help="Include endpoint agreement and compare polygon XY winding for unique outgoing matches")
    ap.add_argument("--portals", action="store_true",
                    help="Detect and stitch cross-tile boundary portals across all adjacent tiles")
    ap.add_argument("--graph", action="store_true",
                    help="Construct navigation graph and report connectivity statistics")
    ap.add_argument("--find-path", nargs=2, metavar=("START", "GOAL"),
                    help="Query A* shortest path between two nodes (e.g. 'sector,poly')")
    ap.add_argument("--export-graph", metavar="PATH",
                    help="Export navigation graph structure to JSON")
    args = ap.parse_args(argv)

    if args.start < 0:
        sys.stderr.write("error: --start must be >= 0\n")
        return 2
    if args.limit is not None and args.limit < 1:
        sys.stderr.write("error: --limit must be >= 1\n")
        return 2
    if not os.path.exists(args.input):
        sys.stderr.write(f"error: input path not found: {args.input}\n")
        return 2
    if os.path.isdir(args.input) and (args.start != 0 or args.limit is not None):
        sys.stderr.write("error: --start and --limit only apply to RPF9 archive files\n")
        return 2
    if args.out is not None:
        if os.path.isdir(args.out):
            sys.stderr.write(f"error: --out path is a directory: {args.out}\n")
            return 2
        if same_report_path(args.out, args.input):
            sys.stderr.write("error: --out must not overwrite the input path\n")
            return 2
        if os.path.isdir(args.input):
            for fname in os.listdir(args.input):
                if fname.endswith(".bin") and same_report_path(args.out, os.path.join(args.input, fname)):
                    sys.stderr.write("error: --out must not overwrite a carved input stream\n")
                    return 2
    if args.export_graph is not None:
        if os.path.isdir(args.export_graph):
            sys.stderr.write(f"error: --export-graph path is a directory: {args.export_graph}\n")
            return 2
        if same_report_path(args.export_graph, args.input):
            sys.stderr.write("error: --export-graph must not overwrite the input path\n")
            return 2

    stitch_portals = args.portals or bool(args.find_path)
    build_graph = args.graph or bool(args.find_path) or bool(args.export_graph)

    doc = scan(args.input, start=args.start, limit=args.limit, reciprocity=args.reciprocity,
               endpoint_agreement=args.endpoint_agreement, relative_winding=args.relative_winding,
               coplanar_winding=args.coplanar_winding, stitch_portals=stitch_portals,
               build_graph=build_graph)

    if args.export_graph is not None and "graph" in doc:
        parent = os.path.dirname(os.path.abspath(args.export_graph))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.export_graph, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc["graph"], fh, indent=2)
            fh.write("\n")

    if args.find_path is not None:
        def parse_node(s):
            parts = [int(p.strip()) for p in s.split(",")]
            if len(parts) == 2:
                return (parts[0], parts[1])
            if len(parts) == 3:
                return ((parts[0], parts[1]), parts[2])
            raise ValueError(f"invalid node string: {s}")

        try:
            start_node = parse_node(args.find_path[0])
            goal_node = parse_node(args.find_path[1])
            # Reconstruct graph from buffers if needed or load from doc
            buffers = list(_iter_resources(args.input, start=args.start, limit=args.limit))
            nav_graph = build_navgraph(buffers, stitch_portals=True)
            path_res = nav_graph.astar(start_node, goal_node)
            doc["path_query"] = path_res.to_dict()
        except (ValueError, KeyError) as e:
            sys.stderr.write(f"error parsing path query: {e}\n")
            return 2

    if args.out is not None:
        parent = os.path.dirname(os.path.abspath(args.out))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(doc, fh, indent=2)
            fh.write("\n")
    if not args.quiet:
        print(json.dumps(doc, indent=2))
    return 0


if __name__ == "__main__":
    try:  # a redirected stdout is strict ANSI code page under -I; write '?' for what it cannot encode
        sys.stdout.reconfigure(errors="replace")
    except (AttributeError, ValueError, OSError):
        pass
    sys.exit(main())
