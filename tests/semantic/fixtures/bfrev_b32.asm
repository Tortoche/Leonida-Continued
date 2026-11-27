; Bit-reverse 32-bit values with v_bfrev_b32.
; Expected MRT0 words: 00000000 80000000 00000001 1e6a2c48.
	v_mov_b32 v0, 0
	v_mov_b32 v1, 1
	v_mov_b32 v2, 0x80000000
	v_mov_b32 v3, 0x12345678
	v_bfrev_b32 v4, v0
	v_bfrev_b32 v5, v1
	v_bfrev_b32 v6, v2
	v_bfrev_b32 v7, v3
	exp mrt0 v4, v5, v6, v7 done vm
	s_endpgm
