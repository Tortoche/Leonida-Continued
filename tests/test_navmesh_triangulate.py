#!/usr/bin/env python3
"""Unit tests for tools/navmesh_triangulate.py and cross-tile portal stitching."""

import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import navmesh_triangulate as nt
import navmesh_links as nl
from test_navmesh_decode import build_navmesh, make_rpf, nd


TOOL = ROOT / "tools" / "navmesh_triangulate.py"


class GeometryTriangulationTest(unittest.TestCase):
    def test_polygon_normal_newell(self):
        # Flat horizontal square CCW
        pts = [(0.0, 0.0, 5.0), (10.0, 0.0, 5.0), (10.0, 10.0, 5.0), (0.0, 10.0, 5.0)]
        nx, ny, nz = nt.polygon_normal_newell(pts)
        self.assertAlmostEqual(nx, 0.0, places=5)
        self.assertAlmostEqual(ny, 0.0, places=5)
        self.assertAlmostEqual(nz, 1.0, places=5)

        # Flat horizontal square CW (should point downward)
        nx, ny, nz = nt.polygon_normal_newell(list(reversed(pts)))
        self.assertAlmostEqual(nz, -1.0, places=5)

        # Degenerate points (< 3 or collinear)
        self.assertEqual(nt.polygon_normal_newell([]), (0.0, 0.0, 1.0))
        self.assertEqual(nt.polygon_normal_newell([(0, 0, 0), (1, 1, 1)]), (0.0, 0.0, 1.0))
        self.assertEqual(nt.polygon_normal_newell([(0, 0, 0), (1, 0, 0), (2, 0, 0)]), (0.0, 0.0, 1.0))

    def test_project_to_dominant_plane(self):
        # Horizontal polygon drops Z (axis 2)
        pts = [(0.0, 0.0, 1.0), (5.0, 0.0, 1.0), (5.0, 5.0, 1.0)]
        pts_2d, dropped, inv = nt.project_to_dominant_plane(pts)
        self.assertEqual(dropped, 2)
        self.assertFalse(inv)
        self.assertEqual(len(pts_2d), 3)

        # Vertical polygon along X=0 drops X (axis 0)
        pts_v = [(0.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, 5.0, 5.0)]
        pts_2d, dropped, _ = nt.project_to_dominant_plane(pts_v)
        self.assertEqual(dropped, 0)

    def test_signed_area_2d(self):
        # 10x10 square: area = 100
        square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        self.assertAlmostEqual(nt.signed_area_2d(square), 100.0)
        self.assertAlmostEqual(nt.signed_area_2d(list(reversed(square))), -100.0)
        self.assertEqual(nt.signed_area_2d([]), 0.0)
        self.assertEqual(nt.signed_area_2d([(0, 0), (1, 1)]), 0.0)

    def test_point_in_triangle_2d(self):
        a, b, c = (0.0, 0.0), (10.0, 0.0), (0.0, 10.0)
        self.assertTrue(nt.is_point_inside_triangle_2d((2.0, 2.0), a, b, c))
        self.assertTrue(nt.is_point_inside_triangle_2d((0.0, 0.0), a, b, c))  # on corner
        self.assertTrue(nt.is_point_inside_triangle_2d((5.0, 0.0), a, b, c))  # on edge
        self.assertFalse(nt.is_point_inside_triangle_2d((8.0, 8.0), a, b, c))
        self.assertFalse(nt.is_point_inside_triangle_2d((-1.0, 2.0), a, b, c))

    def test_triangulate_convex_polygon(self):
        # 4-gon (square)
        square = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 10.0, 0.0)]
        tris = nt.triangulate_polygon_3d(square)
        self.assertEqual(len(tris), 2)
        total_area = sum(nt.signed_area_2d([(square[i][0], square[i][1]) for i in t]) for t in tris)
        self.assertAlmostEqual(total_area, 100.0)

        # 6-gon (hexagon)
        r = 10.0
        hex_pts = [(r * math.cos(i * math.pi / 3), r * math.sin(i * math.pi / 3), 0.0) for i in range(6)]
        hex_tris = nt.triangulate_polygon_3d(hex_pts)
        self.assertEqual(len(hex_tris), 4)  # n - 2 = 4
        hex_area = abs(nt.signed_area_2d([(p[0], p[1]) for p in hex_pts]))
        tri_area_sum = sum(abs(nt.signed_area_2d([(hex_pts[i][0], hex_pts[i][1]) for i in t])) for t in hex_tris)
        self.assertAlmostEqual(tri_area_sum, hex_area, places=4)

    def test_triangulate_concave_polygon(self):
        # L-shaped polygon (6 vertices)
        #  (0, 10) --- (5, 10)
        #    |            |
        #    |         (5, 5) --- (10, 5)
        #    |                       |
        #  (0, 0) --------------- (10, 0)
        l_pts = [
            (0.0, 0.0, 2.0),
            (10.0, 0.0, 2.0),
            (10.0, 5.0, 2.0),
            (5.0, 5.0, 2.0),
            (5.0, 10.0, 2.0),
            (0.0, 10.0, 2.0),
        ]
        tris = nt.triangulate_polygon_3d(l_pts)
        self.assertEqual(len(tris), 4)  # 6 - 2 = 4 triangles
        l_area = abs(nt.signed_area_2d([(p[0], p[1]) for p in l_pts]))
        self.assertAlmostEqual(l_area, 75.0)
        tri_sum = sum(abs(nt.signed_area_2d([(l_pts[i][0], l_pts[i][1]) for i in t])) for t in tris)
        self.assertAlmostEqual(tri_sum, l_area, places=4)

    def test_triangulate_degenerate_and_tiny_polygons(self):
        self.assertEqual(nt.triangulate_polygon_3d([]), [])
        self.assertEqual(nt.triangulate_polygon_3d([(0, 0, 0), (1, 1, 1)]), [])
        self.assertEqual(nt.triangulate_polygon_3d([(0, 0, 0), (1, 0, 0), (0, 1, 0)]), [(0, 1, 2)])

    def test_triangulate_star_polygon(self):
        # 10-vertex non-convex star polygon (5 points)
        star = []
        for i in range(10):
            r = 10.0 if i % 2 == 0 else 4.0
            angle = i * math.pi / 5.0
            star.append((r * math.cos(angle), r * math.sin(angle), 0.0))
        tris = nt.triangulate_polygon_3d(star)
        self.assertEqual(len(tris), 8)  # 10 - 2 = 8
        star_area = abs(nt.signed_area_2d([(p[0], p[1]) for p in star]))
        tri_area_sum = sum(abs(nt.signed_area_2d([(star[i][0], star[i][1]) for i in t])) for t in tris)
        self.assertAlmostEqual(tri_area_sum, star_area, places=4)

    def test_triangulate_collinear_vertices(self):
        # 5-gon with collinear edge vertices (0,0), (5,0), (10,0)
        coll = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 10.0, 0.0)]
        tris = nt.triangulate_polygon_3d(coll)
        self.assertEqual(len(tris), 3)  # 5 - 2 = 3
        coll_area = abs(nt.signed_area_2d([(p[0], p[1]) for p in coll]))
        self.assertAlmostEqual(coll_area, 100.0)
        tri_area_sum = sum(abs(nt.signed_area_2d([(coll[i][0], coll[i][1]) for i in t])) for t in tris)
        self.assertAlmostEqual(tri_area_sum, coll_area, places=4)

    def test_triangulate_3d_sloped_and_vertical(self):
        # Sloped 3D ramp polygon
        ramp = [(0.0, 0.0, 0.0), (10.0, 0.0, 5.0), (10.0, 10.0, 5.0), (0.0, 10.0, 0.0)]
        ramp_tris = nt.triangulate_polygon_3d(ramp)
        self.assertEqual(len(ramp_tris), 2)

        # Vertical polygon along YZ (facing X)
        vert_x = [(0.0, 0.0, 0.0), (0.0, 10.0, 0.0), (0.0, 10.0, 10.0), (0.0, 0.0, 10.0)]
        tris_x = nt.triangulate_polygon_3d(vert_x)
        self.assertEqual(len(tris_x), 2)

        # Vertical polygon along XZ (facing Y)
        vert_y = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 0.0, 10.0), (0.0, 0.0, 10.0)]
        tris_y = nt.triangulate_polygon_3d(vert_y)
        self.assertEqual(len(tris_y), 2)


class BarycentricHeightTest(unittest.TestCase):
    def test_barycentric_height_on_sloped_triangle(self):
        # Plane equation: z = 2*x + 3*y + 10
        v0 = (0.0, 0.0, 10.0)
        v1 = (10.0, 0.0, 30.0)
        v2 = (0.0, 10.0, 40.0)

        # Corner queries
        self.assertAlmostEqual(nt.triangle_height_at(0.0, 0.0, v0, v1, v2), 10.0)
        self.assertAlmostEqual(nt.triangle_height_at(10.0, 0.0, v0, v1, v2), 30.0)
        self.assertAlmostEqual(nt.triangle_height_at(0.0, 10.0, v0, v1, v2), 40.0)

        # Interior query: x=3, y=4 -> z = 2*3 + 3*4 + 10 = 28
        self.assertAlmostEqual(nt.triangle_height_at(3.0, 4.0, v0, v1, v2), 28.0)

        # Exterior query -> None
        self.assertIsNone(nt.triangle_height_at(15.0, 15.0, v0, v1, v2))

    def test_surface_height_at(self):
        # Two polygons: ground (z=0) and elevated roof (z=15)
        ground = ([(0, 0, 0), (20, 0, 0), (20, 20, 0), (0, 20, 0)], [(0, 1, 2), (0, 2, 3)])
        roof = ([(5, 5, 15), (15, 5, 15), (15, 15, 15), (5, 15, 15)], [(0, 1, 2), (0, 2, 3)])
        mesh = [ground, roof]

        # Query over roof area picking target_z=14
        hit = nt.surface_height_at(mesh, 10.0, 10.0, target_z=14.0)
        self.assertIsNotNone(hit)
        z, poly_idx, dz = hit
        self.assertAlmostEqual(z, 15.0)
        self.assertEqual(poly_idx, 1)

        # Query over roof area picking target_z=1 (closer to ground)
        hit_ground = nt.surface_height_at(mesh, 10.0, 10.0, target_z=1.0)
        self.assertIsNotNone(hit_ground)
        self.assertAlmostEqual(hit_ground[0], 0.0)
        self.assertEqual(hit_ground[1], 0)


class TileTriangulationAndPortalsTest(unittest.TestCase):
    def test_triangulated_tile_and_obj_export(self):
        poly1 = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0), (0.0, 10.0, 0.0)]
        poly2 = [(10.0, 0.0, 0.0), (20.0, 0.0, 0.0), (20.0, 10.0, 0.0)]
        geom = {
            "sector": (256, 256),
            "polygons": [(poly1, 0x2100, 0), (poly2, 0x1000, 0)],
        }
        tile = nt.triangulate_tile_geometry(geom)
        self.assertEqual(tile.sector, (256, 256))
        self.assertEqual(len(tile.polygons), 2)
        self.assertEqual(tile.total_vertices, 7)
        self.assertEqual(tile.total_triangles, 3)  # 2 for quad + 1 for tri
        self.assertAlmostEqual(tile.total_area, 150.0)

        # Test OBJ export format
        import io
        buf = io.StringIO()
        nt.write_triangulated_obj(buf, tile, tile_ordinal=5)
        obj_text = buf.getvalue()
        self.assertIn("o tile_5", obj_text)
        self.assertIn("g polygon_0", obj_text)
        self.assertIn("g polygon_1", obj_text)
        self.assertIn("f 5 6 7", obj_text)
        self.assertIn("v 0.0000 0.0000 0.0000", obj_text)

    def test_inter_tile_portal_stitching(self):
        # Tile A: sector (256, 256) -> x in [0, 128], y in [0, 128]
        # Polygon A touches East border at x = 128.0 along y in [30, 70]
        poly_a = [(100.0, 30.0, 5.0), (128.0, 30.0, 5.0), (128.0, 70.0, 5.0), (100.0, 70.0, 5.0)]
        tile_a = nt.TriangulatedTile((256, 256), [poly_a])

        # Tile B: sector (257, 256) -> East neighbor, x in [128, 256]
        # Polygon B touches West border at x = 128.0 along y in [40, 80]
        poly_b = [(128.0, 40.0, 5.0), (150.0, 40.0, 5.0), (150.0, 80.0, 5.0), (128.0, 80.0, 5.0)]
        tile_b = nt.TriangulatedTile((257, 256), [poly_b])

        portals = nt.find_tile_border_portals(tile_a, tile_b)
        self.assertEqual(len(portals), 1)
        p = portals[0]
        self.assertEqual(p.tile_a, (256, 256))
        self.assertEqual(p.tile_b, (257, 256))
        self.assertEqual(p.poly_a, 0)
        self.assertEqual(p.poly_b, 0)
        # Overlap is y in [40, 70] -> width = 30
        self.assertAlmostEqual(p.width, 30.0, delta=0.01)
        self.assertAlmostEqual(p.midpoint[0], 128.0, delta=0.01)
        self.assertAlmostEqual(p.midpoint[1], 55.0, delta=0.01)

    def test_astar_navmesh_pathfinding(self):
        # Two connected tiles
        # Tile A has 2 polygons (0 connected to 1 internally)
        poly_a0 = [(0.0, 0.0, 0.0), (50.0, 0.0, 0.0), (50.0, 50.0, 0.0), (0.0, 50.0, 0.0)]
        poly_a1 = [(50.0, 0.0, 0.0), (128.0, 0.0, 0.0), (128.0, 50.0, 0.0), (50.0, 50.0, 0.0)]
        tile_a = nt.TriangulatedTile((256, 256), [poly_a0, poly_a1])

        # Tile B has 1 polygon touching Tile A at x = 128.0
        poly_b0 = [(128.0, 0.0, 0.0), (200.0, 0.0, 0.0), (200.0, 50.0, 0.0), (128.0, 50.0, 0.0)]
        tile_b = nt.TriangulatedTile((257, 256), [poly_b0])

        portals = nt.find_tile_border_portals(tile_a, tile_b)
        self.assertEqual(len(portals), 1)

        graph = nt.NavGraph()
        graph.add_tile(tile_a, intra_tile_adjacencies=[(0, 1)])
        graph.add_tile(tile_b)
        graph.add_portals(portals)

        # Path from tile A poly 0 to tile B poly 0
        start = ((256, 256), 0)
        goal = ((257, 256), 0)
        path, cost, portals_used = graph.astar(start, goal)
        self.assertIsNotNone(path)
        self.assertEqual(path, [start, ((256, 256), 1), goal])
        self.assertEqual(len(portals_used), 1)
        self.assertGreater(cost, 0.0)

    def test_astar_navmesh_edge_cases(self):
        poly = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 0.0)]
        tile = nt.TriangulatedTile((256, 256), [poly])
        graph = nt.NavGraph()
        graph.add_tile(tile)

        node = ((256, 256), 0)
        # Start equals goal: zero cost, path is [node]
        path, cost, portals = graph.astar(node, node)
        self.assertEqual(path, [node])
        self.assertEqual(cost, 0.0)
        self.assertEqual(portals, [])

        # Non-existent start or goal
        non_existent = ((999, 999), 0)
        path, cost, _ = graph.astar(node, non_existent)
        self.assertIsNone(path)
        self.assertEqual(cost, float("inf"))

        path, cost, _ = graph.astar(non_existent, node)
        self.assertIsNone(path)
        self.assertEqual(cost, float("inf"))


class NavmeshLinksPortalIntegrationTest(unittest.TestCase):
    def test_navmesh_links_open_border_edges_extraction(self):
        # Build synthetic tile using test_navmesh_decode builder
        # Poly 0 is a 4-gon, Poly 1 is a 4-gon
        # Create unlinked edges (0xFFFFFFFF, 0xFFFF)
        tile_bytes, = build_navmesh(
            sector=(256 << 9) | 256,
            poly_sizes=(4,),
            n_verts=4,
            flags=1,
        )
        tile = bytearray(tile_bytes)
        (v_off, _), = nd.split_array(tile, 0x70, 6, 4)
        # Put vertices on East border (x=128.0: u16 65535)
        # (0, 0), (128, 0), (128, 50), (0, 50)
        struct.pack_into("<3H", tile, v_off + 0, 0, 0, 0)
        struct.pack_into("<3H", tile, v_off + 6, 65535, 0, 0)
        struct.pack_into("<3H", tile, v_off + 12, 65535, int(50.0 / 128.0 * 65535), 0)
        struct.pack_into("<3H", tile, v_off + 18, 0, int(50.0 / 128.0 * 65535), 0)

        # Put edge 1 (from v1 to v2 along x=128) as unlinked (0xFFFFFFFF, 0xFFFF)
        (e_off, _), = nd.split_array(tile, 0x88, 12, 4)
        struct.pack_into("<IH", tile, e_off + 12, 0xFFFFFFFF, 0xFFFF)

        edges = nl.extract_open_border_edges(bytes(tile))
        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["border"], "max_x")
        self.assertEqual(edges[0]["poly_idx"], 0)
        self.assertAlmostEqual(edges[0]["p1"][0], 128.0, delta=0.01)
        self.assertAlmostEqual(edges[0]["p2"][0], 128.0, delta=0.01)


class CliTest(unittest.TestCase):
    def test_cli_subprocesses_and_validation(self):
        tile_bytes, = build_navmesh(sector=(256 << 9) | 256, poly_sizes=(3,), n_verts=3, flags=1)
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive = Path(tmp_dir) / "navmeshes.rpf"
            make_rpf(archive, [tile_bytes])
            out_obj = Path(tmp_dir) / "tile.obj"
            out_json = Path(tmp_dir) / "tile.json"

            # 1. Valid OBJ export
            res = subprocess.run(
                [sys.executable, "-I", str(TOOL), str(archive), "--tile", "0", "--out", str(out_obj), "--format", "obj"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, res.stderr)
            self.assertTrue(out_obj.exists())
            self.assertIn("v ", out_obj.read_text(encoding="utf-8"))
            self.assertIn("f ", out_obj.read_text(encoding="utf-8"))

            # 2. Valid JSON export
            res_json = subprocess.run(
                [sys.executable, "-I", str(TOOL), str(archive), "--tile", "0", "--out", str(out_json), "--format", "json"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res_json.returncode, 0, res_json.stderr)
            self.assertTrue(out_json.exists())
            data = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertIn("tile", data)
            self.assertEqual(data["tile"]["polygon_count"], 1)

            # 3. Argument validation errors (must return code 2)
            for bad_args in (
                [str(archive), "--tile", "-1"],
                [str(archive), "--portals-with", "-2"],
                [str(Path(tmp_dir) / "missing.rpf")],
                [str(archive), "--out", str(archive)],
                [str(archive), "--out", tmp_dir],
            ):
                err_res = subprocess.run([sys.executable, "-I", str(TOOL), *bad_args], capture_output=True)
                self.assertEqual(err_res.returncode, 2)


if __name__ == "__main__":
    unittest.main()
