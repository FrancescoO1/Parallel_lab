import time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from kernels import get_gradient_function

_edge_func = get_gradient_function()

class EdgeDetectorGPU:
    """Versione senza timer"""
    def detect(self, input_img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        H, W, C = input_img.shape
        input_img = np.ascontiguousarray(input_img.astype(np.uint8))
        
        d_input = cuda.mem_alloc(input_img.nbytes)
        d_output = cuda.mem_alloc(input_img.nbytes)
        
        cuda.memcpy_htod(d_input, input_img)
        
        block_dim = (16, 16, 1)
        grid_dim = (int(np.ceil(W / 16)), int(np.ceil(H / 16)))
        kH, kW = kernel.shape
        
        _edge_func(d_input, d_output, np.int32(W), np.int32(H), np.int32(kH), np.int32(kW),
                   block=block_dim, grid=grid_dim)
        
        output_img = np.empty_like(input_img)
        cuda.memcpy_dtoh(output_img, d_output)
        
        return output_img

class EdgeDetectorGPU_Instrumented:
    """Versione per i grafici """
    def detect_with_metrics(self, input_img: np.ndarray, kernel: np.ndarray):
        """ Restituisce: (immagine, tempo_trasferimento, tempo_calcolo) """
        H, W, C = input_img.shape
        input_img = np.ascontiguousarray(input_img.astype(np.uint8))
        
        d_input = cuda.mem_alloc(input_img.nbytes)
        d_output = cuda.mem_alloc(input_img.nbytes)
        
        # --- TIMER 1: H -> D ---
        cuda.Context.synchronize()
        t0 = time.perf_counter()
        cuda.memcpy_htod(d_input, input_img)
        cuda.Context.synchronize()
        t_htod = time.perf_counter() - t0
        
        # --- TIMER 2: Calcolo ---
        block_dim = (16, 16, 1)
        grid_dim = (int(np.ceil(W / 16)), int(np.ceil(H / 16)))
        kH, kW = kernel.shape
        
        cuda.Context.synchronize()
        t0 = time.perf_counter()
        _edge_func(d_input, d_output, np.int32(W), np.int32(H), np.int32(kH), np.int32(kW),
                   block=block_dim, grid=grid_dim)
        cuda.Context.synchronize()
        t_compute = time.perf_counter() - t0
        
        # --- TIMER 3: D -> H ---
        output_img = np.empty_like(input_img)
        cuda.Context.synchronize()
        t0 = time.perf_counter()
        cuda.memcpy_dtoh(output_img, d_output)
        cuda.Context.synchronize()
        t_dtoh = time.perf_counter() - t0
        
        return output_img, (t_htod + t_dtoh), t_compute