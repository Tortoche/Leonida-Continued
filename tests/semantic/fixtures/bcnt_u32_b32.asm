; Bit-count plus accumulator with v_bcnt_u32_b32.
; Expected MRT0 words: 00000000 00000020 0000001a 00000015.
	v_mov_b32 v0, 0
	v_mov_b32 v1, 0xffffffff
	v_mov_b32 v2, 0xa5a55a5a
	v_mov_b32 v3, 0x0f0f0f0f
	v_bcnt_u32_b32 v4, v0, 0
	v_bcnt_u32_b32 v5, v1, 0
	v_bcnt_u32_b32 v6, v2, 10
	v_bcnt_u32_b32 v7, v3, 5
	exp mrt0 v4, v5, v6, v7 done vm
	s_endpgm
