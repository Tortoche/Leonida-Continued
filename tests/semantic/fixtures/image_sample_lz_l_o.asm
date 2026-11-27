; Regression fixture for image_sample_lz_o, image_sample_l_o, and image_sample_c_lz_o.
	v_mov_b32 v4, 0x023f
	v_mov_b32 v5, 0.25
	v_mov_b32 v6, 0.5
	v_mov_b32 v7, 2.0
	v_mov_b32 v8, 0.75
	image_sample_lz_o v[0:3], v[4:6], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
	image_sample_l_o v[10:13], v[4:7], s[8:15], s[16:19] dmask:0xf dim:SQ_RSRC_IMG_2D
	image_sample_c_lz_o v9, v[4:7], s[8:15], s[16:19] dmask:0x1 dim:SQ_RSRC_IMG_2D
	v_add_f32 v0, v0, v10
	exp mrt0 v0, v1, v2, v9 done vm
	s_endpgm
