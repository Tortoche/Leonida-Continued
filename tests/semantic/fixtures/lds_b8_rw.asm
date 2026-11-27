; LDS 8-bit write and unsigned/signed read (ds_write_b8, ds_read_u8, ds_read_i8).
; Input: s4 (byte value) via push constant.
; Outputs in buffer s[0:3]: [ds_read_u8(val), ds_read_i8(val)]
v_lshlrev_b32_e32 v1, 2, v0
v_mov_b32_e32 v2, s4
ds_write_b8 v1, v2
s_waitcnt lgkmcnt(0)
ds_read_u8 v3, v1
ds_read_i8 v4, v1
s_waitcnt lgkmcnt(0)
v_mov_b32_e32 v5, 0
buffer_store_dword v3, v5, s[0:3], 0 offen
v_mov_b32_e32 v5, 4
buffer_store_dword v4, v5, s[0:3], 0 offen
s_endpgm
