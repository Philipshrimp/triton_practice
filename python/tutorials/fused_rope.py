import torch

import triton
import triton.language as tl


@triton.jit
def rope_fw_kernel(src_ptr, freqs_ptr, dst_ptr, 
                   offset_block, offset_block_dst, 
                   h, d, d2, 
                   stride_h, stride_d, o_stride_h, o_stride_d,
                   BLOCK_SIZE: tl.constexpr):
  s_id = triton.program_id(0)

  d_id = tl.program_id(axis=0)
  h_id = tl.program_id(axis=1)
  mask_d = d_id < d2
  mask_h = h_id < h
  mask = mask_d and mask_h

  src = tl.load(src_ptr + d_id, mask=mask)
  freqs = tl.load(freqs_ptr + d_id, mask=mask)

  v_sin = tl.sin(freqs[s_id * d2 + d_id])
  v_cos = tl.cos(freqs[s_id * d2 + d_id])

  offset_src = offset_block + h_id * stride_h + d_id * stride_d
  offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d

  v_src = src[offset_src]

  if (d_id + d2 / 2 < d2):
    v_src_rotate = float(-src[offset_src + (d2 / 2) * stride_d])
  else:
    v_src_rotate = float(src[offset_src + (d2 / 2 - d2) * stride_d])

  dst = []
  dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin

  if (d > d2):
    offset_head = offset_block + h_id * stride_h
    offset_head_dst = offset_block_dst + h_id * o_stride_h

    dst[offset_head_dst + d_id * o_stride_d] = src[offset_head + d_id * stride_d]

  tl.store(dst_ptr, dst, mask=mask)


@triton.jit
def rope_bw_kernel(src_ptr, freqs_ptr, dst_ptr,
                   offset_block, offset_block_dst, h, d, d2,
                   stride_h, stride_d, o_stride_h, o_stride_d):
  s_id = triton.program_id(0)

  d_id = tl.program_id(axis=0)
  h_id = tl.program_id(axis=1)
  mask_d = d_id < d2
  mask_h = h_id < h
  mask = mask_d and mask_h

  src = tl.load(src_ptr + d_id, mask=mask)
  freqs = tl.load(freqs_ptr + d_id, mask=mask)

  v_cos = tl.cos(freqs[s_id * d2 + d_id])
  if (d_id + d2 / 2 < d2):
    v_sin = tl.sin(freqs[s_id * d2 + d_id + d2 / 2])
  else:
    v_sin = -tl.sin(freqs[s_id * d2 + d_id + d2 / 2 - d2])

  offset_src = offset_block + h_id * stride_h + d_id * stride_d
  offset_dst = offset_block_dst + h_id * o_stride_h + d_id * o_stride_d

  v_src = src[offset_src]
  if (d_id + d2 / 2 < d2):
    v_src_rotate = src[offset_src + (d2 / 2) * stride_d]
  else:
    v_src_rotate = src[offset_src + (d2 / 2 - d2) * stride_d]

  dst = []
  dst[offset_dst] = v_src * v_cos + v_src_rotate * v_sin

  if (d > d2):
    offset_head = offset_block + h_id * stride_h
    offset_head_dst = offset_block_dst + h_id * o_stride_h

    dst[offset_head_dst + d_id * o_stride_d] = src[offset_head + d_id * stride_d]

  tl.store(dst_ptr, dst, mask=mask)

def rope_fw():
  None

def rope_bw():
  None

def rope_thd_fw():
  None

def rope_thd_bw():
  None
