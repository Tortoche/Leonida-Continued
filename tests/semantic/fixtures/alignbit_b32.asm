; Bit-field align across two 32-bit sources with v_alignbit_b32.
; Expected MRT0 words: 9abcdef0 789abcde 56789abc 3456789a.
	v_mov_b32 v0, 0x12345678
	v_mov_b32 v1, 0x9abcdef0
	v_alignbit_b32 v2, v0, v1, 0
	v_alignbit_b32 v3, v0, v1, 8
	v_alignbit_b32 v4, v0, v1, 16
	v_alignbit_b32 v5, v0, v1, 24
	exp mrt0 v2, v3, v4, v5 done vm
	s_endpgm
