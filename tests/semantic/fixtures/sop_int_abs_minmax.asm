; Scalar integer abs, signed min/max, and unsigned min/max.
; Inputs: s4 (a), s5 (b) via push constants.
; Outputs in buffer s[0:3]: [abs_i32(a), min_i32(a,b), max_i32(a,b), min_u32(a,b), max_u32(a,b)]
s_abs_i32 s6, s4
s_min_i32 s7, s4, s5
s_max_i32 s8, s4, s5
s_min_u32 s9, s4, s5
s_max_u32 s10, s4, s5
v_mov_b32_e32 v1, s6
v_mov_b32_e32 v2, s7
v_mov_b32_e32 v3, s8
v_mov_b32_e32 v4, s9
v_mov_b32_e32 v5, s10
v_mov_b32_e32 v0, 0
buffer_store_dword v1, v0, s[0:3], 0 offen
v_mov_b32_e32 v0, 4
buffer_store_dword v2, v0, s[0:3], 0 offen
v_mov_b32_e32 v0, 8
buffer_store_dword v3, v0, s[0:3], 0 offen
v_mov_b32_e32 v0, 12
buffer_store_dword v4, v0, s[0:3], 0 offen
v_mov_b32_e32 v0, 16
buffer_store_dword v5, v0, s[0:3], 0 offen
s_endpgm
