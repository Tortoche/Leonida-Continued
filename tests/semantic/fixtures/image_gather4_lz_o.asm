; Regression fixture for image_gather4_lz_o (2D typed-alias path and 2D array path).
	v_mov_b32 v4, 0x033e
	v_mov_b32 v5, 0.25
	v_mov_b32 v6, 0.75
	v_mov_b32 v7, 1.0
	image_gather4_lz_o v[0:3], v[4:6], s[8:15], s[16:19] dmask:0x2 dim:SQ_RSRC_IMG_2D
	image_gather4_lz_o v[10:13], v[4:7], s[8:15], s[16:19] dmask:0x4 dim:SQ_RSRC_IMG_2D_ARRAY
	v_add_f32 v0, v0, v10
	exp mrt0 v0, v1, v2, v3 done vm
	s_endpgm
