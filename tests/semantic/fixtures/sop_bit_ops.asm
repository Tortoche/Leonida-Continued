; Scalar bit compare, bitfield extract (signed), and bitfield mask.
; Inputs: s4 (val), s5 (bit_idx / bfe_ctrl), s6 (bfm_width), s7 (bfm_offset).
; Outputs in buffer s[0:3]: [bitcmp0_scc, bitcmp1_scc, bfe_i32, bfm_b32]
s_bitcmp0_b32 s4, s5
s_cselect_b32 s8, 1, 0
s_bitcmp1_b32 s4, s5
s_cselect_b32 s9, 1, 0
s_bfe_i32 s10, s4, s5
s_bfm_b32 s11, s6, s7
v_mov_b32_e32 v1, s8
v_mov_b32_e32 v2, s9
v_mov_b32_e32 v3, s10
v_mov_b32_e32 v4, s11
v_mov_b32_e32 v0, 0
buffer_store_dword v1, v0, s[0:3], 0 offen
v_mov_b32_e32 v0, 4
buffer_store_dword v2, v0, s[0:3], 0 offen
v_mov_b32_e32 v0, 8
buffer_store_dword v3, v0, s[0:3], 0 offen
v_mov_b32_e32 v0, 12
buffer_store_dword v4, v0, s[0:3], 0 offen
s_endpgm
