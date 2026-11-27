"""Synthetic unit tests for skeletal animation and skinning pipeline.

Tests joint hierarchy transformations, forward kinematics, inverse bind matrices,
skin weight normalization, skinning matrix computation, linear blend skinning,
and glTF 2.0 binary skin chunk export.

Run with:
    python -I tests/test_anim_skinning.py
"""
from __future__ import annotations

import json
import math
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import anim_glb  # noqa: E402
import skinned_glb  # noqa: E402


def _identity_mat4() -> list[float]:
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ]


def _translation_mat4(tx: float, ty: float, tz: float) -> list[float]:
    return [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        tx, ty, tz, 1.0,
    ]


def _mat_approx_equal(a: list[float], b: list[float], tol: float = 1e-5) -> bool:
    if len(a) != 16 or len(b) != 16:
        return False
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _vec_approx_equal(a: tuple[float, ...], b: tuple[float, ...], tol: float = 1e-5) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(x - y) <= tol for x, y in zip(a, b))


class TestJointHierarchyTransformations(unittest.TestCase):
    """Test joint hierarchy forward kinematics and multi-root traversal."""

    def test_single_root_forward_kinematics(self):
        """Root at origin with local translation moves child joint accordingly."""
        ped_skeleton = [
            {"name": "SKEL_ROOT", "parent": None, "bind_matrix": _identity_mat4()},
            {"name": "SKEL_Spine", "parent": 0, "bind_matrix": _translation_mat4(0.0, 0.0, 1.0)},
        ]
        joints = skinned_glb.JointSet(ped_skeleton)
        self.assertEqual(joints.count, 2)
        self.assertIsNone(joints.parents[0])
        self.assertEqual(joints.parents[1], 0)

        local_mats = [_identity_mat4(), _translation_mat4(0.0, 0.0, 1.0)]
        world_mats = skinned_glb.forward_kinematics(joints, local_mats)
        self.assertEqual(len(world_mats), 2)
        # Root world matrix should match local
        self.assertTrue(_mat_approx_equal(world_mats[0], _identity_mat4()))
        # Spine world matrix should have translation (0, 0, 1)
        self.assertTrue(_mat_approx_equal(world_mats[1], _translation_mat4(0.0, 0.0, 1.0)))

    def test_deep_hierarchy_composition(self):
        """Chain of 4 joints accumulates parent translations."""
        ped_skeleton = [
            {"name": "SKEL_ROOT", "parent": None, "bind_matrix": _identity_mat4()},
            {"name": "SKEL_Pelvis", "parent": 0, "bind_matrix": _translation_mat4(0.0, 0.0, 0.5)},
            {"name": "SKEL_Thigh", "parent": 1, "bind_matrix": _translation_mat4(0.2, 0.0, 0.5)},
            {"name": "SKEL_Calf", "parent": 2, "bind_matrix": _translation_mat4(0.2, 0.0, 0.0)},
        ]
        joints = skinned_glb.JointSet(ped_skeleton)
        local_mats = [
            _identity_mat4(),
            _translation_mat4(0.0, 0.0, 0.5),
            _translation_mat4(0.2, 0.0, 0.0),
            _translation_mat4(0.0, 0.0, -0.5),
        ]
        world_mats = skinned_glb.forward_kinematics(joints, local_mats)
        self.assertEqual(len(world_mats), 4)
        # Joint 3 world translation: (0 + 0 + 0.2 + 0, 0, 0 + 0.5 + 0 - 0.5) = (0.2, 0.0, 0.0)
        expected_calf = _translation_mat4(0.2, 0.0, 0.0)
        self.assertTrue(_mat_approx_equal(world_mats[3], expected_calf))

    def test_multi_root_skeleton_not_orphaned(self):
        """Multi-root skeleton (e.g. helper/attachment roots) preserves detached roots."""
        ped_skeleton = [
            {"name": "SKEL_ROOT", "parent": None, "bind_matrix": _identity_mat4()},
            {"name": "SKEL_Spine", "parent": 0, "bind_matrix": _translation_mat4(0.0, 0.0, 1.0)},
            {"name": "ATTACH_PROP", "parent": None, "bind_matrix": _translation_mat4(0.5, 0.5, 0.5)},
        ]
        joints = skinned_glb.JointSet(ped_skeleton)
        nodes, scene_nodes, skin = skinned_glb.assemble_skeleton_hierarchy(joints, mesh_node_index=3)

        # Both roots (0 and 2) plus the mesh node (3) must be in scene_nodes
        self.assertIn(0, scene_nodes)
        self.assertIn(2, scene_nodes)
        self.assertIn(3, scene_nodes)
        self.assertEqual(skin["joints"], [0, 1, 2])
        self.assertEqual(nodes[0]["name"], "SKEL_ROOT")
        self.assertEqual(nodes[1]["name"], "SKEL_Spine")
        self.assertEqual(nodes[2]["name"], "ATTACH_PROP")
        self.assertEqual(nodes[0]["children"], [1])
        self.assertNotIn("children", nodes[2])

    def test_forward_kinematics_invalid_inputs(self):
        ped_skeleton = [{"name": "ROOT", "parent": None, "bind_matrix": _identity_mat4()}]
        joints = skinned_glb.JointSet(ped_skeleton)
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.forward_kinematics(joints, [_identity_mat4(), _identity_mat4()])
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.forward_kinematics(joints, [[float("nan")] * 16])

    def test_forward_kinematics_cycle_detection(self):
        """Cyclic parent references must be caught by cycle guard and raise SkinError."""
        ped_skeleton = [
            {"name": "ROOT", "parent": None, "bind_matrix": _identity_mat4()},
            {"name": "BONE_A", "parent": 2, "bind_matrix": _identity_mat4()},
            {"name": "BONE_B", "parent": 1, "bind_matrix": _identity_mat4()},
        ]
        joints = skinned_glb.JointSet(ped_skeleton)
        self.assertEqual(joints.parents[1], 2)
        self.assertEqual(joints.parents[2], 1)
        local_mats = [_identity_mat4(), _identity_mat4(), _identity_mat4()]
        with self.assertRaises(skinned_glb.SkinError) as ctx:
            skinned_glb.forward_kinematics(joints, local_mats)
        self.assertIn("cycle detected", str(ctx.exception).lower())

    def test_forward_kinematics_self_cycle(self):
        """Self-referencing joint (parent = self) must raise SkinError."""
        ped_skeleton = [
            {"name": "ROOT", "parent": None, "bind_matrix": _identity_mat4()},
            {"name": "BONE_SELF", "parent": 1, "bind_matrix": _identity_mat4()},
        ]
        joints = skinned_glb.JointSet(ped_skeleton)
        local_mats = [_identity_mat4(), _identity_mat4()]
        with self.assertRaises(skinned_glb.SkinError) as ctx:
            skinned_glb.forward_kinematics(joints, local_mats)
        self.assertIn("cycle detected", str(ctx.exception).lower())

    def test_forward_kinematics_out_of_range_parent(self):
        """Parent index out of bounds must raise SkinError."""
        ped_skeleton = [{"name": "ROOT", "parent": None, "bind_matrix": _identity_mat4()}]
        joints = skinned_glb.JointSet(ped_skeleton)
        joints.parents = [99]
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.forward_kinematics(joints, [_identity_mat4()])


class TestInverseBindMatrices(unittest.TestCase):
    """Test inverse bind matrix computation, validation, and glTF conversion."""

    def test_invert_affine_matrix(self):
        m = _translation_mat4(1.0, -2.5, 3.0)
        inv = skinned_glb.invert_affine(m)
        expected = _translation_mat4(-1.0, 2.5, -3.0)
        self.assertTrue(_mat_approx_equal(inv, expected))
        prod = skinned_glb.mat_mul(m, inv)
        self.assertTrue(_mat_approx_equal(prod, _identity_mat4()))

    def test_invert_affine_mat4_anim_glb(self):
        m = _translation_mat4(2.0, 3.0, 4.0)
        inv = anim_glb.invert_affine_mat4(m)
        prod = anim_glb.mat4_mul(m, inv)
        self.assertTrue(_mat_approx_equal(prod, _identity_mat4()))

    def test_singular_matrix_raises_error(self):
        singular = [
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.invert_affine(singular)
        with self.assertRaises(ValueError):
            anim_glb.invert_affine_mat4(singular)

    def test_bind_to_gltf_yup_conversion(self):
        """RAGE Z-up to glTF Y-up conversion maps (x, y, z) -> (x, z, -y)."""
        m = _translation_mat4(1.0, 2.0, 3.0)
        gltf_m = skinned_glb.bind_to_gltf(m)
        # Translation in column 3 (indices 12, 13, 14):
        # x -> 1.0, y -> 3.0 (from z), z -> -2.0 (from -y)
        self.assertAlmostEqual(gltf_m[12], 1.0, places=5)
        self.assertAlmostEqual(gltf_m[13], 3.0, places=5)
        self.assertAlmostEqual(gltf_m[14], -2.0, places=5)

    def test_quat_to_mat4_and_inverse(self):
        # 90-degree rotation around X axis: q = (sin(45), 0, 0, cos(45))
        s = math.sin(math.pi / 4)
        c = math.cos(math.pi / 4)
        rot_x = (s, 0.0, 0.0, c)
        trans = (0.0, 1.0, 2.0)
        mat = anim_glb.quat_to_mat4(rot_x, trans)
        inv = anim_glb.invert_affine_mat4(mat)
        prod = anim_glb.mat4_mul(mat, inv)
        self.assertTrue(_mat_approx_equal(prod, _identity_mat4(), tol=1e-5))


class TestSkinWeightNormalization(unittest.TestCase):
    """Test 8-influence skin weight normalization and palette remapping."""

    def test_normalise_weights_standard(self):
        count = 2
        local_bones = 3
        local_to_joint = [10, 20, 30]

        # Vertex 0: bone 0 (wt 100), bone 1 (wt 100), bone 2 (wt 55) -> sum 255
        # Vertex 1: bone 1 (wt 128), bone 2 (wt 127) -> sum 255
        j0 = bytes([0, 1, 2, 0, 1, 2, 0, 0])
        j1 = bytes([0, 0, 0, 0, 0, 0, 0, 0])
        w0 = bytes([100, 100, 55, 0, 128, 127, 0, 0])
        w1 = bytes([0, 0, 0, 0, 0, 0, 0, 0])

        res = skinned_glb.normalise_weights(
            j0, j1, w0, w1, count, local_bones, local_to_joint, joint_count=50
        )
        self.assertEqual(len(res["j0"]), count * 8)
        self.assertEqual(len(res["w0"]), count * 16)

        # Check weights sum to 1.0
        w0_v0 = struct.unpack_from("<4f", res["w0"], 0)
        self.assertAlmostEqual(sum(w0_v0), 1.0, places=5)
        self.assertAlmostEqual(w0_v0[0], 100 / 255.0, places=5)
        self.assertAlmostEqual(w0_v0[1], 100 / 255.0, places=5)
        self.assertAlmostEqual(w0_v0[2], 55 / 255.0, places=5)

        # Check remapped joints
        j0_v0 = struct.unpack_from("<4H", res["j0"], 0)
        self.assertEqual(j0_v0[0], 10)
        self.assertEqual(j0_v0[1], 20)
        self.assertEqual(j0_v0[2], 30)
        self.assertEqual(j0_v0[3], 0)  # unweighted slot defaults to 0

    def test_normalise_weights_rejects_zero_weight(self):
        count = 1
        local_bones = 2
        local_to_joint = [0, 1]
        j0 = bytes([0, 0, 0, 0])
        j1 = bytes([0, 0, 0, 0])
        w0 = bytes([0, 0, 0, 0])
        w1 = bytes([0, 0, 0, 0])
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.normalise_weights(j0, j1, w0, w1, count, local_bones, local_to_joint)

    def test_normalise_weights_rejects_out_of_bounds_palette(self):
        count = 1
        local_bones = 2
        local_to_joint = [0, 1]
        # Bone index 5 is out of local_bones=2
        j0 = bytes([5, 0, 0, 0])
        j1 = bytes([0, 0, 0, 0])
        w0 = bytes([255, 0, 0, 0])
        w1 = bytes([0, 0, 0, 0])
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.normalise_weights(j0, j1, w0, w1, count, local_bones, local_to_joint)

    def test_normalise_weights_non_255_sum(self):
        """Arbitrary raw weights summing to non-255 (e.g. 100) must normalize to 1.0."""
        count = 1
        local_bones = 2
        local_to_joint = [5, 6]
        j0 = bytes([0, 1, 0, 0])
        j1 = bytes([0, 0, 0, 0])
        w0 = bytes([60, 40, 0, 0])  # sum = 100
        w1 = bytes([0, 0, 0, 0])
        res = skinned_glb.normalise_weights(j0, j1, w0, w1, count, local_bones, local_to_joint)
        w0_out = struct.unpack_from("<4f", res["w0"], 0)
        self.assertAlmostEqual(sum(w0_out), 1.0, places=5)
        self.assertAlmostEqual(w0_out[0], 0.6, places=5)
        self.assertAlmostEqual(w0_out[1], 0.4, places=5)

    def test_normalise_weights_eight_influences(self):
        """All 8 influence slots normalized together so total sum across both vectors is 1.0."""
        count = 1
        local_bones = 8
        local_to_joint = list(range(8))
        j0 = bytes([0, 1, 2, 3])
        j1 = bytes([4, 5, 6, 7])
        w0 = bytes([32, 32, 32, 32])
        w1 = bytes([32, 32, 32, 31])  # total = 255
        res = skinned_glb.normalise_weights(j0, j1, w0, w1, count, local_bones, local_to_joint)
        w0_out = struct.unpack_from("<4f", res["w0"], 0)
        w1_out = struct.unpack_from("<4f", res["w1"], 0)
        self.assertAlmostEqual(sum(w0_out) + sum(w1_out), 1.0, places=5)
        self.assertAlmostEqual(w0_out[0], 32.0 / 255.0, places=5)
        self.assertAlmostEqual(w1_out[3], 31.0 / 255.0, places=5)

    def test_normalise_weights_rejects_joint_index_ge_65536(self):
        """Palette mapping to joint index >= 65536 violates glTF uint16 limit."""
        count = 1
        local_bones = 1
        local_to_joint = [65536]
        j0 = bytes([0, 0, 0, 0])
        j1 = bytes([0, 0, 0, 0])
        w0 = bytes([255, 0, 0, 0])
        w1 = bytes([0, 0, 0, 0])
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.normalise_weights(j0, j1, w0, w1, count, local_bones, local_to_joint)


class TestSkinningMatrixComputation(unittest.TestCase):
    """Test skinning matrix (World * InverseBind) calculation."""

    def test_rest_pose_skinning_matrices_are_identity(self):
        ped_skeleton = [
            {"name": "SKEL_ROOT", "parent": None, "bind_matrix": _identity_mat4()},
            {"name": "SKEL_Spine", "parent": 0, "bind_matrix": _translation_mat4(0.0, 0.0, 1.0)},
            {"name": "SKEL_Head", "parent": 1, "bind_matrix": _translation_mat4(0.0, 0.0, 1.5)},
        ]
        joints = skinned_glb.JointSet(ped_skeleton)
        # In rest pose, skinning matrices must be identity 4x4
        matrices = skinned_glb.compute_skinning_matrices(joints)
        self.assertEqual(len(matrices), 3)
        for m in matrices:
            self.assertTrue(_mat_approx_equal(m, _identity_mat4()))

    def test_posed_skinning_matrices(self):
        ped_skeleton = [
            {"name": "SKEL_ROOT", "parent": None, "bind_matrix": _identity_mat4()},
        ]
        joints = skinned_glb.JointSet(ped_skeleton)
        # Displace root by (1, 2, 3)
        posed = [_translation_mat4(1.0, 2.0, 3.0)]
        matrices = skinned_glb.compute_skinning_matrices(joints, posed)
        self.assertEqual(len(matrices), 1)
        self.assertTrue(_mat_approx_equal(matrices[0], _translation_mat4(1.0, 2.0, 3.0)))

    def test_anim_glb_skinning_matrices(self):
        joints = [
            {
                "hash": 0x1234,
                "name": "bone0",
                "parent": None,
                "rotation": (0.0, 0.0, 0.0, 1.0),
                "translation": (0.0, 0.0, 0.0),
                "scale": (1.0, 1.0, 1.0),
                "inverse_bind": _identity_mat4(),
            }
        ]
        # In rest pose:
        mats = anim_glb.compute_skinning_matrices(joints)
        self.assertEqual(len(mats), 1)
        self.assertTrue(_mat_approx_equal(mats[0], _identity_mat4()))

    def test_skinning_matrix_w_times_b_inv_analytical(self):
        """Verify skinning matrix explicitly satisfies M = W * B^-1."""
        b = _translation_mat4(10.0, 20.0, 30.0)
        w = _translation_mat4(15.0, 25.0, 35.0)
        b_inv = skinned_glb.invert_affine(b)
        expected_m = skinned_glb.mat_mul(w, b_inv)
        self.assertTrue(_mat_approx_equal(expected_m, _translation_mat4(5.0, 5.0, 5.0)))

        ped_skeleton = [{"name": "SKEL_ROOT", "parent": None, "bind_matrix": b}]
        joints = skinned_glb.JointSet(ped_skeleton)
        gltf_w = skinned_glb.bind_to_gltf(w)
        mats = skinned_glb.compute_skinning_matrices(joints, [gltf_w])
        expected_gltf_m = skinned_glb.mat_mul(gltf_w, joints.inverse_bind(0))
        self.assertTrue(_mat_approx_equal(mats[0], expected_gltf_m))


class TestLinearBlendSkinning(unittest.TestCase):
    """Test linear blend skinning on vertex positions and normals."""

    def test_rest_pose_preserves_positions(self):
        positions = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (-1.0, 0.5, 2.0)]
        vertex_count = len(positions)
        joints_0 = struct.pack(f"<{4 * vertex_count}H", *([0, 0, 0, 0] * vertex_count))
        joints_1 = struct.pack(f"<{4 * vertex_count}H", *([0, 0, 0, 0] * vertex_count))
        weights_0 = struct.pack(f"<{4 * vertex_count}f", *([1.0, 0.0, 0.0, 0.0] * vertex_count))
        weights_1 = struct.pack(f"<{4 * vertex_count}f", *([0.0, 0.0, 0.0, 0.0] * vertex_count))
        skinning_matrices = [_identity_mat4()]

        out_pos, out_norm = skinned_glb.apply_skinning(
            positions, joints_0, joints_1, weights_0, weights_1, skinning_matrices
        )
        self.assertIsNone(out_norm)
        self.assertEqual(len(out_pos), vertex_count)
        for orig, skinned in zip(positions, out_pos):
            self.assertTrue(_vec_approx_equal(orig, skinned))

    def test_blended_skinning_positions(self):
        # 2 joints: joint 0 translated (0, 0, 0), joint 1 translated (2, 0, 0)
        skinning_matrices = [
            _identity_mat4(),
            _translation_mat4(2.0, 0.0, 0.0),
        ]
        positions = [(0.0, 0.0, 0.0)]
        # 50% joint 0, 50% joint 1
        joints_0 = struct.pack("<4H", 0, 1, 0, 0)
        joints_1 = struct.pack("<4H", 0, 0, 0, 0)
        weights_0 = struct.pack("<4f", 0.5, 0.5, 0.0, 0.0)
        weights_1 = struct.pack("<4f", 0.0, 0.0, 0.0, 0.0)

        out_pos, _ = skinned_glb.apply_skinning(
            positions, joints_0, joints_1, weights_0, weights_1, skinning_matrices
        )
        # Position should be 0.5 * (0,0,0) + 0.5 * (2,0,0) = (1, 0, 0)
        self.assertTrue(_vec_approx_equal(out_pos[0], (1.0, 0.0, 0.0)))

    def test_skinning_with_normals(self):
        # Joint translated and rotated (normal should rotate but not translate)
        # 90 deg around Z: (x, y) -> (-y, x)
        cos45 = math.cos(math.pi / 4)
        sin45 = math.sin(math.pi / 4)
        rot_z = anim_glb.quat_to_mat4((0.0, 0.0, sin45, cos45), (10.0, 20.0, 30.0))
        skinning_matrices = [rot_z]

        positions = [(1.0, 0.0, 0.0)]
        normals = [(1.0, 0.0, 0.0)]
        joints_0 = struct.pack("<4H", 0, 0, 0, 0)
        joints_1 = struct.pack("<4H", 0, 0, 0, 0)
        weights_0 = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
        weights_1 = struct.pack("<4f", 0.0, 0.0, 0.0, 0.0)

        out_pos, out_norm = skinned_glb.apply_skinning(
            positions, joints_0, joints_1, weights_0, weights_1, skinning_matrices, normals=normals
        )
        self.assertIsNotNone(out_norm)
        # Normal (1, 0, 0) rotated by 90 deg around Z becomes (0, 1, 0)
        self.assertTrue(_vec_approx_equal(out_norm[0], (0.0, 1.0, 0.0), tol=1e-5))

    def test_anim_glb_skin_mesh_positions(self):
        positions = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
        joint_indices = [[0], [0, 1]]
        weights = [[1.0], [0.5, 0.5]]
        skinning_matrices = [_identity_mat4(), _translation_mat4(2.0, 0.0, 0.0)]

        skinned = anim_glb.skin_mesh_positions(
            positions, joint_indices, weights, skinning_matrices
        )
        self.assertEqual(len(skinned), 2)
        self.assertTrue(_vec_approx_equal(skinned[0], (0.0, 0.0, 0.0)))
        # Vertex 1: 0.5 * (1, 1, 1) + 0.5 * (3, 1, 1) = (2, 1, 1)
        self.assertTrue(_vec_approx_equal(skinned[1], (2.0, 1.0, 1.0)))

    def test_linear_blend_skinning_8_influences(self):
        """Vertex influenced equally by 8 joints across both influence sets."""
        skinning_matrices = [_translation_mat4(float(k), 0.0, 0.0) for k in range(8)]
        positions = [(0.0, 0.0, 0.0)]
        normals = [(0.0, 1.0, 0.0)]
        joints_0 = struct.pack("<4H", 0, 1, 2, 3)
        joints_1 = struct.pack("<4H", 4, 5, 6, 7)
        weights_0 = struct.pack("<4f", 0.125, 0.125, 0.125, 0.125)
        weights_1 = struct.pack("<4f", 0.125, 0.125, 0.125, 0.125)

        out_pos, out_norm = skinned_glb.apply_skinning(
            positions, joints_0, joints_1, weights_0, weights_1, skinning_matrices, normals=normals
        )
        # Expected X: 0.125 * (0 + 1 + 2 + 3 + 4 + 5 + 6 + 7) = 0.125 * 28 = 3.5
        self.assertTrue(_vec_approx_equal(out_pos[0], (3.5, 0.0, 0.0)))
        # Normal (0, 1, 0) should remain (0, 1, 0) because all transforms are pure translations
        self.assertTrue(_vec_approx_equal(out_norm[0], (0.0, 1.0, 0.0)))
        length = math.sqrt(sum(c * c for c in out_norm[0]))
        self.assertAlmostEqual(length, 1.0, places=5)

    def test_skinned_normals_unit_length_under_rotation(self):
        """Skinned normal vectors maintain unit length under 3D rotation."""
        # 45-degree rotation around Y axis
        cos45 = math.cos(math.pi / 4)
        sin45 = math.sin(math.pi / 4)
        rot_y = anim_glb.quat_to_mat4((0.0, sin45, 0.0, cos45), (1.0, 2.0, 3.0))
        skinning_matrices = [rot_y]
        positions = [(1.0, 0.0, 0.0)]
        normals = [(1.0, 0.0, 0.0)]
        joints_0 = struct.pack("<4H", 0, 0, 0, 0)
        joints_1 = struct.pack("<4H", 0, 0, 0, 0)
        weights_0 = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
        weights_1 = struct.pack("<4f", 0.0, 0.0, 0.0, 0.0)

        _, out_norm = skinned_glb.apply_skinning(
            positions, joints_0, joints_1, weights_0, weights_1, skinning_matrices, normals=normals
        )
        self.assertIsNotNone(out_norm)
        norm = out_norm[0]
        length = math.sqrt(sum(c * c for c in norm))
        self.assertAlmostEqual(length, 1.0, places=5)


class TestGLTFBinarySkinExport(unittest.TestCase):
    """Test glTF 2.0 binary skin chunk and container formatting."""

    def test_assemble_skeleton_nodes_anim_glb(self):
        joints = [
            {
                "hash": 0x100,
                "name": "root",
                "parent": None,
                "rotation": (0.0, 0.0, 0.0, 1.0),
                "translation": (0.0, 0.0, 0.0),
                "scale": (1.0, 1.0, 1.0),
                "inverse_bind": _identity_mat4(),
            },
            {
                "hash": 0x101,
                "name": "child",
                "parent": 0,
                "rotation": (0.0, 0.0, 0.0, 1.0),
                "translation": (0.0, 0.0, 1.0),
                "scale": (1.0, 1.0, 1.0),
                "inverse_bind": _translation_mat4(0.0, 0.0, -1.0),
            },
        ]
        nodes, skin, first_joint, mover = anim_glb.assemble_skeleton_nodes(joints, root_motion=True)
        # Root node 0: rage_z_up
        self.assertEqual(nodes[0]["name"], "rage_z_up")
        self.assertIsNotNone(mover)
        self.assertEqual(nodes[mover]["name"], "mover")
        # Joints start after rage_z_up and mover
        self.assertEqual(first_joint, 2)
        self.assertEqual(nodes[first_joint]["name"], "root")
        self.assertEqual(nodes[first_joint + 1]["name"], "child")
        self.assertEqual(skin["joints"], [2, 3])
        self.assertEqual(skin["skeleton"], 2)

    def test_pack_glb_chunk_framing(self):
        """Test glTF 2.0 binary container chunk alignment and headers."""
        gltf = {
            "asset": {"version": "2.0"},
            "buffers": [{"byteLength": 0}],
        }
        buf = b"\x01\x02\x03\x04"
        glb_bytes = anim_glb.pack_glb(gltf, buf)

        # GLB Header (12 bytes)
        magic, version, length = struct.unpack_from("<4sII", glb_bytes, 0)
        self.assertEqual(magic, b"glTF")
        self.assertEqual(version, 2)
        self.assertEqual(length, len(glb_bytes))

        # JSON Chunk (8 bytes header)
        json_len, json_type = struct.unpack_from("<II", glb_bytes, 12)
        self.assertEqual(json_type, 0x4E4F534A)
        json_data = glb_bytes[20:20 + json_len]
        parsed = json.loads(json_data.decode("utf-8"))
        self.assertEqual(parsed["asset"]["version"], "2.0")

        # BIN Chunk (8 bytes header)
        bin_offset = 20 + json_len
        bin_len, bin_type = struct.unpack_from("<II", glb_bytes, bin_offset)
        self.assertEqual(bin_type, 0x004E4942)
        self.assertGreaterEqual(bin_len, 4)
        bin_data = glb_bytes[bin_offset + 8:bin_offset + 8 + bin_len]
        self.assertEqual(bin_data[:4], b"\x01\x02\x03\x04")

    def test_assemble_skeleton_hierarchy_structure(self):
        """Test assemble_skeleton_hierarchy builds glTF nodes, scenes, and skin object."""
        ped_skeleton = [
            {"name": "SKEL_ROOT", "parent": None, "bind_matrix": _identity_mat4()},
            {"name": "SKEL_Spine", "parent": 0, "bind_matrix": _translation_mat4(0.0, 0.0, 1.0)},
        ]
        joints = skinned_glb.JointSet(ped_skeleton)
        nodes, scene_nodes, skin = skinned_glb.assemble_skeleton_hierarchy(
            joints, mesh_node_index=2, skin_name="test_skin", ibm_accessor=5
        )
        self.assertEqual(skin["name"], "test_skin")
        self.assertEqual(skin["joints"], [0, 1])
        self.assertEqual(skin["skeleton"], 0)
        self.assertEqual(skin["inverseBindMatrices"], 5)
        self.assertIn(0, scene_nodes)
        self.assertIn(2, scene_nodes)
        self.assertEqual(nodes[0]["children"], [1])
        self.assertEqual(nodes[2]["skin"], 0)

    def test_glb_builder_alignment_and_accessors(self):
        """Test _GlbBuilder ensures 4-byte buffer alignment and accessor padding."""
        builder = skinned_glb._GlbBuilder()
        # View 1: 3 bytes -> placed at offset 0, padded to 4 for next
        v0 = builder.view(b"\x01\x02\x03")
        self.assertEqual(builder.views[v0]["byteOffset"], 0)
        self.assertEqual(builder.views[v0]["byteLength"], 3)

        # View 2: 5 bytes -> offset should be padded to 4
        v1 = builder.view(b"\x04\x05\x06\x07\x08")
        self.assertEqual(builder.views[v1]["byteOffset"], 4)
        self.assertEqual(builder.views[v1]["byteLength"], 5)

        # Accessor with ARRAY_BUFFER and element size 2 (COMPONENT_UNSIGNED_SHORT SCALAR)
        # 1 element = 2 bytes -> requires 2 bytes padding to 4-byte stride
        acc_idx = builder.accessor(
            b"\x01\x00",
            skinned_glb.COMPONENT_UNSIGNED_SHORT,
            "SCALAR",
            1,
            skinned_glb.ARRAY_BUFFER,
        )
        acc = builder.accessors[acc_idx]
        acc_view = builder.views[acc["bufferView"]]
        self.assertEqual(acc_view["byteStride"], 4)
        self.assertEqual(acc_view["byteOffset"] % 4, 0)


if __name__ == "__main__":
    unittest.main()
