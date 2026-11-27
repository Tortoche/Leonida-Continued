"""Adversarial stress tests for Skeletal Animation and Skinning Pipeline (Track 10).

Stress-tests:
1. Degenerate quaternions (zeros, unnormalized, NaNs, Infs).
2. Non-invertible and singular bind matrices (rank deficient, near-singular, non-finite, non-affine).
3. Zero-weight vertices and weight distribution anomalies.
4. High bone counts (>256 bones, 560, 596, 1024, >65535 bones).
5. Extreme scale values (zero scale, tiny scale, float32 overflow scale, negative scale).
6. Hierarchy topology edge cases (cycles, self-parenting, invalid parents).

Run with:
    python -I tests/test_anim_skinning_adversarial.py
"""
from __future__ import annotations

import math
import multiprocessing
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


class TestDegenerateQuaternions(unittest.TestCase):
    """Stress-test quaternion handling across animation and skinning math routines."""

    def test_zero_quaternion_quat_to_mat4(self):
        """Zero quaternion (0,0,0,0) has norm 0; quat_to_mat4 silently outputs identity linear block."""
        q_zero = (0.0, 0.0, 0.0, 0.0)
        t = (1.0, 2.0, 3.0)
        m = anim_glb.quat_to_mat4(q_zero, t)
        # 3x3 block evaluates to identity matrix despite degenerate 0-norm quaternion
        self.assertEqual(m[0], 1.0)
        self.assertEqual(m[5], 1.0)
        self.assertEqual(m[10], 1.0)
        self.assertEqual(m[12:15], [1.0, 2.0, 3.0])

    def test_unnormalized_quaternion_causes_distortion(self):
        """Quaternions with norm != 1.0 produce non-orthogonal, scaling/shearing matrices."""
        # Quaternion (2.0, 0, 0, 0) has norm 2.0
        q_unnorm = (2.0, 0.0, 0.0, 0.0)
        m = anim_glb.quat_to_mat4(q_unnorm, (0.0, 0.0, 0.0))
        # 1.0 - (yy + zz) = 1.0, but 1.0 - (xx + zz) = 1.0 - 8.0 = -7.0!
        self.assertEqual(m[0], 1.0)
        self.assertEqual(m[5], -7.0)
        self.assertEqual(m[10], -7.0)
        # Determinant is 49.0 != 1.0 (shearing / reflection rather than rotation)
        det = (m[0] * (m[5] * m[10] - m[6] * m[9])
               - m[1] * (m[4] * m[10] - m[6] * m[8])
               + m[2] * (m[4] * m[9] - m[5] * m[8]))
        self.assertAlmostEqual(det, 49.0, places=5)

    def test_nan_and_inf_quaternion_propagation(self):
        """Quaternions with NaN or Inf propagate through forward kinematics and skinning matrices."""
        q_nan = (float("nan"), 0.0, 0.0, 1.0)
        joints = [
            {
                "hash": 0x1,
                "name": "root",
                "parent": None,
                "rotation": q_nan,
                "translation": (0.0, 0.0, 0.0),
                "scale": (1.0, 1.0, 1.0),
                "inverse_bind": _identity_mat4(),
            }
        ]
        skinning_mats = anim_glb.compute_skinning_matrices(joints)
        self.assertTrue(any(math.isnan(v) for v in skinning_mats[0]))

        # Skinning with NaN matrix produces NaN position
        skinned = anim_glb.skin_mesh_positions(
            positions=[(1.0, 2.0, 3.0)],
            joint_indices=[[0]],
            weights=[[1.0]],
            skinning_matrices=skinning_mats,
        )
        self.assertTrue(math.isnan(skinned[0][0]))

    def test_quaternion_multiplication_deep_hierarchy_accumulation(self):
        """Unnormalized quaternions in a deep hierarchy compound exponentially."""
        depth = 12
        # Slight overshoot: norm = 1.05
        q_overshoot = (0.0, 0.0, 0.0, 1.05)
        joints = []
        for i in range(depth):
            joints.append({
                "hash": i,
                "name": f"joint_{i}",
                "parent": None if i == 0 else i - 1,
                "rotation": q_overshoot,
                "translation": (0.0, 0.0, 1.0),
                "scale": (1.0, 1.0, 1.0),
                "inverse_bind": _identity_mat4(),
            })
        world = anim_glb.compute_forward_kinematics(joints)
        last_q = world[-1][0]
        # Cumulative norm is (1.05)^12 ~= 1.795856
        self.assertGreater(last_q[3], 1.79)


class TestNonInvertibleBindMatrices(unittest.TestCase):
    """Stress-test matrix inversion and singular/malformed matrix rejection."""

    def test_singular_zero_matrix_rejected(self):
        """All-zero 4x4 matrix must be rejected by both invert_affine implementations."""
        zero_mat = [0.0] * 16
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.invert_affine(zero_mat)
        with self.assertRaises(ValueError):
            anim_glb.invert_affine_mat4(zero_mat)

    def test_rank_deficient_matrices_rejected(self):
        """Rank 1 and rank 2 matrices with det=0 must be rejected."""
        # Rank 1: all columns identical
        rank1 = [1.0, 2.0, 3.0, 0.0] * 3 + [0.0, 0.0, 0.0, 1.0]
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.invert_affine(rank1)
        with self.assertRaises(ValueError):
            anim_glb.invert_affine_mat4(rank1)

        # Rank 2: third column is zero
        rank2 = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.invert_affine(rank2)
        with self.assertRaises(ValueError):
            anim_glb.invert_affine_mat4(rank2)

    def test_near_singular_matrices_boundary(self):
        """Matrices right at the 1e-12 determinant threshold."""
        # Below threshold: det = 1e-13 -> rejected
        m_below = [
            1e-5, 0.0, 0.0, 0.0,
            0.0, 1e-4, 0.0, 0.0,
            0.0, 0.0, 1e-4, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.invert_affine(m_below)

        # Above threshold: det = 1e-11 -> accepted and numerically stable
        m_above = [
            1e-4, 0.0, 0.0, 0.0,
            0.0, 1e-4, 0.0, 0.0,
            0.0, 0.0, 1e-3, 0.0,
            1.0, 2.0, 3.0, 1.0,
        ]
        inv = skinned_glb.invert_affine(m_above)
        prod = skinned_glb.mat_mul(m_above, inv)
        error = max(abs(prod[i] - (1.0 if i in (0, 5, 10, 15) else 0.0)) for i in range(16))
        self.assertLess(error, 1e-12)

    def test_non_finite_translation_vulnerability_in_invert_affine(self):
        """EMPIRICAL FINDING: invert_affine only checks determinant of upper-left 3x3 block;
        if translation (tx, ty, tz) has NaN or Inf, invert_affine does NOT raise SkinError
        and silently outputs NaN/Inf."""
        m_nan_t = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            float("nan"), 0.0, 0.0, 1.0,
        ]
        # Invert affine succeeds without raising SkinError
        inv = skinned_glb.invert_affine(m_nan_t)
        self.assertTrue(math.isnan(inv[12]))

        m_inf_t = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            float("inf"), 0.0, 0.0, 1.0,
        ]
        inv_inf = skinned_glb.invert_affine(m_inf_t)
        self.assertTrue(math.isinf(inv_inf[12]))

    def test_non_affine_bottom_row_handling(self):
        """invert_affine ignores bottom row elements m[3], m[7], m[11] and sets out[15]=1.0.
        When given a projective matrix (m[3]=1), invert_affine returns an incorrect matrix
        whose product with input is NOT identity."""
        m_projective = [
            1.0, 0.0, 0.0, 1.0,  # projective component
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        inv = skinned_glb.invert_affine(m_projective)
        prod = skinned_glb.mat_mul(m_projective, inv)
        # prod is not identity because bottom row was ignored
        self.assertNotAlmostEqual(prod[3], 0.0)


class TestZeroWeightVerticesAndWeightAnomalies(unittest.TestCase):
    """Stress-test zero-weight vertices, unnormalized weights, and weight distribution edge cases."""

    def test_normalise_weights_rejects_zero_weight_vertex(self):
        """normalise_weights raises SkinError if any vertex has sum(weights) == 0."""
        count = 1
        local_bones = 2
        local_to_joint = [0, 1]
        j0 = bytes([0, 0, 0, 0])
        j1 = bytes([0, 0, 0, 0])
        w0 = bytes([0, 0, 0, 0])
        w1 = bytes([0, 0, 0, 0])
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.normalise_weights(j0, j1, w0, w1, count, local_bones, local_to_joint)

    def test_apply_skinning_zero_weight_vertex_collapses_to_origin(self):
        """EMPIRICAL FINDING: apply_skinning does NOT validate that vertex weights sum to 1.0;
        a zero-weight vertex collapses silently to position (0,0,0) and normal (0,0,0)."""
        positions = [(15.0, 25.0, 35.0)]
        normals = [(0.0, 1.0, 0.0)]
        joints_0 = struct.pack("<4H", 0, 0, 0, 0)
        joints_1 = struct.pack("<4H", 0, 0, 0, 0)
        weights_zero = struct.pack("<4f", 0.0, 0.0, 0.0, 0.0)
        matrices = [_identity_mat4()]

        pos, norm = skinned_glb.apply_skinning(
            positions, joints_0, joints_1, weights_zero, weights_zero, matrices, normals=normals
        )
        self.assertEqual(pos[0], (0.0, 0.0, 0.0))
        self.assertEqual(norm[0], (0.0, 0.0, 0.0))

    def test_apply_skinning_unnormalized_weights_scale_geometry(self):
        """apply_skinning with weights summing to 0.5 shrinks vertex position by 0.5."""
        positions = [(10.0, 20.0, 30.0)]
        joints_0 = struct.pack("<4H", 0, 0, 0, 0)
        joints_1 = struct.pack("<4H", 0, 0, 0, 0)
        weights_half = struct.pack("<4f", 0.5, 0.0, 0.0, 0.0)
        weights_zero = struct.pack("<4f", 0.0, 0.0, 0.0, 0.0)
        matrices = [_identity_mat4()]

        pos, _ = skinned_glb.apply_skinning(
            positions, joints_0, joints_1, weights_half, weights_zero, matrices
        )
        self.assertEqual(pos[0], (5.0, 10.0, 15.0))

    def test_unweighted_oob_joint_divergence_numpy_vs_pure_python(self):
        """EMPIRICAL BUG / DIVERGENCE: In apply_skinning, if an UNWEIGHTED slot (weight=0)
        has a joint index >= joint_count, NumPy path raises SkinError while pure Python
        ignores it and succeeds."""
        positions = [(1.0, 2.0, 3.0)]
        # Slot 0 has joint 0 and weight 1.0; Slot 1 has joint 999 and weight 0.0
        joints_0 = struct.pack("<4H", 0, 999, 0, 0)
        joints_1 = struct.pack("<4H", 0, 0, 0, 0)
        weights_0 = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
        weights_1 = struct.pack("<4f", 0.0, 0.0, 0.0, 0.0)
        matrices = [_identity_mat4()]  # count = 1

        # 1. NumPy path raises SkinError
        if skinned_glb._np is not None:
            with self.assertRaises(skinned_glb.SkinError):
                skinned_glb.apply_skinning(positions, joints_0, joints_1, weights_0, weights_1, matrices)

        # 2. Pure Python path ignores unweighted joint and succeeds
        old_np = skinned_glb._np
        skinned_glb._np = None
        try:
            pos_py, _ = skinned_glb.apply_skinning(
                positions, joints_0, joints_1, weights_0, weights_1, matrices
            )
            self.assertEqual(pos_py[0], (1.0, 2.0, 3.0))
        finally:
            skinned_glb._np = old_np


class TestHighBoneCounts(unittest.TestCase):
    """Stress-test high bone count skeletons (>256 bones, 560 male, 596 female, >65535 limit)."""

    def test_humanoid_596_bone_skeleton_pipeline(self):
        """596-bone skeleton successfully executes JointSet, forward_kinematics, and assemble_skeleton."""
        bone_count = 596
        skeleton = []
        for i in range(bone_count):
            parent = None if i == 0 else i - 1
            skeleton.append({
                "name": f"SKEL_BONE_{i:04d}",
                "parent": parent,
                "bind_matrix": _translation_mat4(0.0, 0.0, float(i)),
            })
        joints = skinned_glb.JointSet(skeleton)
        self.assertEqual(joints.count, bone_count)

        # Forward kinematics along 596-bone spine
        local_mats = [_translation_mat4(0.0, 0.0, 0.1) for _ in range(bone_count)]
        world_mats = skinned_glb.forward_kinematics(joints, local_mats)
        self.assertEqual(len(world_mats), bone_count)

        # Skinning matrices computation
        skinning_mats = skinned_glb.compute_skinning_matrices(joints, world_mats)
        self.assertEqual(len(skinning_mats), bone_count)

        # glTF assembly: skin joints contains all 596 joints
        nodes, scene_nodes, skin = skinned_glb.assemble_skeleton_hierarchy(joints, mesh_node_index=bone_count)
        self.assertEqual(len(skin["joints"]), bone_count)
        self.assertEqual(skin["joints"][-1], 595)

    def test_palette_mapping_to_uint16_bones_over_256(self):
        """normalise_weights correctly remaps palette indices to bone IDs in range 256..65535."""
        count = 1
        local_bones = 2
        # Local bone 0 -> joint 500, local bone 1 -> joint 595
        local_to_joint = [500, 595]
        j0 = bytes([0, 1, 0, 0])
        j1 = bytes([0, 0, 0, 0])
        w0 = bytes([128, 127, 0, 0])
        w1 = bytes([0, 0, 0, 0])

        res = skinned_glb.normalise_weights(
            j0, j1, w0, w1, count, local_bones, local_to_joint, joint_count=600
        )
        remapped_j0 = struct.unpack_from("<4H", res["j0"], 0)
        self.assertEqual(remapped_j0[0], 500)
        self.assertEqual(remapped_j0[1], 595)

    def test_joint_exceeding_uint16_limit_rejected(self):
        """normalise_weights raises SkinError if a weighted bone index exceeds 65535 (glTF UNSIGNED_SHORT)."""
        count = 1
        local_bones = 1
        # Joint index 70000 exceeds 0xFFFF
        local_to_joint = [70000]
        j0 = bytes([0, 0, 0, 0])
        j1 = bytes([0, 0, 0, 0])
        w0 = bytes([255, 0, 0, 0])
        w1 = bytes([0, 0, 0, 0])

        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.normalise_weights(j0, j1, w0, w1, count, local_bones, local_to_joint)


class TestExtremeScaleValues(unittest.TestCase):
    """Stress-test scale extremes: zero scale, near-zero scale, float32 overflow, and negative scale."""

    def test_zero_scale_rejected_as_singular(self):
        """Scale (0,0,0) produces zero determinant and is rejected by invert_affine_mat4."""
        m_zero = anim_glb.quat_to_mat4((0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), scale=(0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            anim_glb.invert_affine_mat4(m_zero)

    def test_float32_overflow_scale_produces_nans(self):
        """Scale of 1e39 overflows IEEE 754 float32, creating Infs/NaNs during matrix operations."""
        mat_huge = [
            1e39, 0.0, 0.0, 0.0,
            0.0, 1e39, 0.0, 0.0,
            0.0, 0.0, 1e39, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ]
        positions = [(1.0, 1.0, 1.0)]
        joints_0 = struct.pack("<4H", 0, 0, 0, 0)
        joints_1 = struct.pack("<4H", 0, 0, 0, 0)
        weights_0 = struct.pack("<4f", 1.0, 0.0, 0.0, 0.0)
        weights_1 = struct.pack("<4f", 0.0, 0.0, 0.0, 0.0)

        # In numpy (float32), 1e39 overflows to inf, and position becomes nan or inf
        if skinned_glb._np is not None:
            pos_np, _ = skinned_glb.apply_skinning(
                positions, joints_0, joints_1, weights_0, weights_1, [mat_huge]
            )
            self.assertTrue(math.isnan(pos_np[0][0]) or math.isinf(pos_np[0][0]))

    def test_negative_scale_reflection_inversion(self):
        """Negative scale creates reflection (negative determinant); invert_affine inverts it properly."""
        m_reflect = [
            -1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            2.0, 0.0, 0.0, 1.0,
        ]
        inv = skinned_glb.invert_affine(m_reflect)
        prod = skinned_glb.mat_mul(m_reflect, inv)
        for i in range(16):
            expected = 1.0 if i in (0, 5, 10, 15) else 0.0
            self.assertAlmostEqual(prod[i], expected, places=5)


def _run_fk_helper(joints, local_mats):
    skinned_glb.forward_kinematics(joints, local_mats)


class TestHierarchyTopologyEdgeCases(unittest.TestCase):
    """Stress-test skeleton topology edge cases: cyclic parents, self-parenting, detached roots."""

    def test_cyclic_parent_hierarchy_raises_skin_error(self):
        """Cyclic parent links are detected and cleanly raise SkinError without hanging."""
        skeleton_cycle = [
            {"name": "root", "parent": None, "bind_matrix": _identity_mat4()},
            {"name": "bone1", "parent": 2, "bind_matrix": _identity_mat4()},
            {"name": "bone2", "parent": 1, "bind_matrix": _identity_mat4()},
        ]
        joints = skinned_glb.JointSet(skeleton_cycle)
        local_mats = [_identity_mat4() for _ in range(3)]
        with self.assertRaises(skinned_glb.SkinError):
            skinned_glb.forward_kinematics(joints, local_mats)

    def test_invalid_parent_remapped_to_root(self):
        """JointSet re-maps invalid parent indices (e.g. negative or >= count) to root."""
        skeleton = [
            {"name": "root", "parent": None, "bind_matrix": _identity_mat4()},
            {"name": "bone1", "parent": 999, "bind_matrix": _identity_mat4()},
        ]
        joints = skinned_glb.JointSet(skeleton)
        self.assertEqual(joints.parents[1], 0)  # Remapped to root


if __name__ == "__main__":
    unittest.main()
