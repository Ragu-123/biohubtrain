import triton
import triton.language as tl
import torch

@triton.jit
def trilinear_interpolate_kernel(
    input_ptr,  # (C, Z, Y, X)
    coords_ptr, # (N, 3) - z, y, x in float
    output_ptr, # (N, C)
    C, Z, Y, X,
    stride_ic, stride_iz, stride_iy, stride_ix,
    stride_on, stride_oc,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load coordinates [N, 3]
    z_f = tl.load(coords_ptr + offsets * 3 + 0)
    y_f = tl.load(coords_ptr + offsets * 3 + 1)
    x_f = tl.load(coords_ptr + offsets * 3 + 2)
    
    z0 = tl.where(z_f < 0, 0, tl.cast(tl.floor(z_f), tl.int32))
    y0 = tl.where(y_f < 0, 0, tl.cast(tl.floor(y_f), tl.int32))
    x0 = tl.where(x_f < 0, 0, tl.cast(tl.floor(x_f), tl.int32))
    
    z1 = tl.minimum(z0 + 1, Z - 1)
    y1 = tl.minimum(y0 + 1, Y - 1)
    x1 = tl.minimum(x0 + 1, X - 1)
    
    dz = z_f - tl.cast(z0, tl.float32)
    dy = y_f - tl.cast(y0, tl.float32)
    dx = x_f - tl.cast(x0, tl.float32)
    
    # Fused Read-Interpolate (Optimized for Register Reuse)
    for c in range(C):
        # Fetch 8 neighbors
        base_c = c * stride_ic
        v000 = tl.load(input_ptr + base_c + z0 * stride_iz + y0 * stride_iy + x0 * stride_ix)
        v001 = tl.load(input_ptr + base_c + z0 * stride_iz + y0 * stride_iy + x1 * stride_ix)
        v010 = tl.load(input_ptr + base_c + z0 * stride_iz + y1 * stride_iy + x0 * stride_ix)
        v011 = tl.load(input_ptr + base_c + z0 * stride_iz + y1 * stride_iy + x1 * stride_ix)
        v100 = tl.load(input_ptr + base_c + z1 * stride_iz + y0 * stride_iy + x0 * stride_ix)
        v101 = tl.load(input_ptr + base_c + z1 * stride_iz + y0 * stride_iy + x1 * stride_ix)
        v110 = tl.load(input_ptr + base_c + z1 * stride_iz + y1 * stride_iy + x0 * stride_ix)
        v111 = tl.load(input_ptr + base_c + z1 * stride_iz + y1 * stride_iy + x1 * stride_ix)
        
        # Bilinear on top Z-plane
        v_z0 = (1-dy)*((1-dx)*v000 + dx*v001) + dy*((1-dx)*v010 + dx*v011)
        # Bilinear on bottom Z-plane
        v_z1 = (1-dy)*((1-dx)*v100 + dx*v101) + dy*((1-dx)*v110 + dx*v111)
        # Final Linear on Z
        res = (1-dz)*v_z0 + dz*v_z1
        
        tl.store(output_ptr + offsets * stride_on + c * stride_oc, res)

def trilinear_interpolate_triton(input, coords):
    # input: (C, Z, Y, X), coords: (N, 3)
    C, Z, Y, X = input.shape
    N = coords.shape[0]
    output = torch.empty((N, C), device=input.device, dtype=input.dtype)
    BLOCK_SIZE = 128
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    trilinear_interpolate_kernel[grid](
        input, coords, output,
        C, Z, Y, X,
        *input.stride(),
        coords.stride(0), coords.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output

# Benchmark or Verify
if __name__ == "__main__":
    device = 'cuda'
    input_tensor = torch.randn(32, 64, 64, 64, device=device)
    coords_tensor = torch.rand(1024, 3, device=device) * 60.0
    out = trilinear_interpolate_triton(input_tensor, coords_tensor)
    print("Output shape:", out.shape)
