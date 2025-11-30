import numpy as np

class EdgeDetectorCPU:
    def detect(self, input_img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        H, W, C = input_img.shape
        kH, kW = kernel.shape
        pad_h, pad_w = kH // 2, kW // 2
        
        output_img = np.zeros_like(input_img, dtype=np.uint8)
        # Padding edge sui bordi spaziali
        padded = np.pad(input_img, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='edge')
        
        for i in range(H):
            for j in range(W):
                patch = padded[i : i + kH, j : j + kW, :]
                max_val = np.max(patch, axis=(0, 1))
                min_val = np.min(patch, axis=(0, 1))
                output_img[i, j, :] = max_val - min_val
        return output_img