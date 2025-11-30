import pycuda.autoinit
from pycuda.compiler import SourceModule

# ==============================================================================
# DEFINIZIONE DEL KERNEL CUDA
# ==============================================================================

_KERNEL_CODE_RGB = """
__global__ void morphological_gradient_rgb_kernel(unsigned char *input, unsigned char *output, 
                                                  int W, int H, int kH, int kW)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int pad_h = kH / 2;
    int pad_w = kW / 2;

    if (y < H && x < W) {
        unsigned char min_r = 255, max_r = 0;
        unsigned char min_g = 255, max_g = 0;
        unsigned char min_b = 255, max_b = 0;

        for (int ki = 0; ki < kH; ki++) {
            for (int kj = 0; kj < kW; kj++) {
                int img_i = y - pad_h + ki;
                int img_j = x - pad_w + kj;
                // Padding 'edge' virtuale (clamp)
                int safe_i = max(0, min(H - 1, img_i));
                int safe_j = max(0, min(W - 1, img_j));
                
                int idx = (safe_i * W + safe_j) * 3;

                unsigned char r = input[idx];
                unsigned char g = input[idx + 1];
                unsigned char b = input[idx + 2];

                if (r < min_r) min_r = r; if (r > max_r) max_r = r;
                if (g < min_g) min_g = g; if (g > max_g) max_g = g;
                if (b < min_b) min_b = b; if (b > max_b) max_b = b;
            }
        }
        int out_idx = (y * W + x) * 3;
        output[out_idx]     = max_r - min_r;
        output[out_idx + 1] = max_g - min_g;
        output[out_idx + 2] = max_b - min_b;
    }
}
"""

# Compilazione del modulo (eseguita una volta sola all'importazione di questo file)
_mod = SourceModule(_KERNEL_CODE_RGB)

def get_gradient_function():
    """Restituisce la funzione CUDA compilata pronta per l'uso."""
    return _mod.get_function("morphological_gradient_rgb_kernel")