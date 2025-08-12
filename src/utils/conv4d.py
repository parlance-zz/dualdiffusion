from typing import Optional, Union

import torch
import torch.nn.functional as F


def conv4d(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None,
           stride: Union[int, tuple[int, int, int, int]] = 1,
           padding: Union[int, tuple[int, int, int, int]] = 0,
           dilation: Union[int, tuple[int, int, int, int]] = 1) -> torch.Tensor:
    
    # input:  [B, C_in, D1, D2, D3, D4]
    # weight: [C_out, C_in, k1, k2, k3, k4]
    
    B, C_in, D1, D2, D3, D4 = input.shape
    C_out, _, k1, k2, k3, k4 = weight.shape

    # promote args to tuples if needed
    if isinstance(stride, int): stride = (stride,) * 4
    if isinstance(padding, int): padding = (padding,) * 4
    if isinstance(dilation, int): dilation = (dilation,) * 4

    # pad input
    input_padded = F.pad(input, (
        padding[3], padding[3],  # dim4
        padding[2], padding[2],  # dim3
        padding[1], padding[1],  # dim2
        padding[0], padding[0],  # dim1
    ))

    # output spatial sizes
    D1_out = (D1 + 2*padding[0] - dilation[0]*(k1-1) - 1) // stride[0] + 1
    D2_out = (D2 + 2*padding[1] - dilation[1]*(k2-1) - 1) // stride[1] + 1
    D3_out = (D3 + 2*padding[2] - dilation[2]*(k3-1) - 1) // stride[2] + 1
    D4_out = (D4 + 2*padding[3] - dilation[3]*(k4-1) - 1) // stride[3] + 1

    # get strides of padded input
    s = input_padded.stride()

    # extract sliding local blocks (as a view, no copy)
    patches = input_padded.as_strided(
        size=(B, C_in, D1_out, D2_out, D3_out, D4_out, k1, k2, k3, k4),
        stride=(
            s[0], s[1],
            s[2] * stride[0], s[3] * stride[1], s[4] * stride[2], s[5] * stride[3],
            s[2] * dilation[0], s[3] * dilation[1], s[4] * dilation[2], s[5] * dilation[3],
        )
    )

    # collapse everything except batch+spatial into one axis for matmul
    patches_matrix = patches.reshape(B * D1_out * D2_out * D3_out * D4_out, -1)  # [N, C_in*k1*k2*k3*k4]
    weight_matrix = weight.reshape(C_out, -1)  # [C_out, C_in*k1*k2*k3*k4]

    # matmul: [N, K] @ [K, C_out]T -> [N, C_out]
    out_matrix = patches_matrix @ weight_matrix.T  # [N, C_out]

    # add bias if needed
    if bias is not None:
        out_matrix = out_matrix +bias.view(1, -1)

    # reshape back to [B, C_out, D1_out, D2_out, D3_out, D4_out]
    output = out_matrix.view(B, D1_out, D2_out, D3_out, D4_out, C_out).permute(0, 5, 1, 2, 3, 4).contiguous()

    return output

if __name__ == "__main__":

    import timeit
    
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    x = torch.randn(4, 32, 8, 8, 33, 129, device="cuda")
    w = torch.randn(32, 32, 3, 3, 3, 3, device="cuda")
    n_runs = 200

    conv4d = torch.compile(conv4d, fullgraph=True, dynamic=False)

    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16): 

        out = conv4d(x, w, stride=1, padding=1) # compile
        print("output shape:", out.shape) # should be [16, 4, 8, 8, 33, 65]

        time = timeit.timeit(lambda: conv4d(x, w, stride=1, padding=1), number=n_runs)
        print(f"time: {time:.6f} seconds for {n_runs} runs")