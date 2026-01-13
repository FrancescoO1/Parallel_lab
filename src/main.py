import os
import time
import statistics
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from cpu import EdgeDetectorCPU
from gpu import EdgeDetectorGPU, EdgeDetectorGPU_Instrumented

IMAGE_PATH = "/media/francesco/DATA/dev/vsCode/Lab_Parallel/IMMAGINE/aquila.jpg"
OUTPUT_DIR = "final_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('fast') 
COLORS = ['#4c72b0', '#55a868', '#c44e52', '#8172b2', '#ccb974']

# BENCHMARK (CPU vs GPU)
def run_text_benchmark(pil_img):
    print("\n" + "="*50)
    print("  FASE 1: BENCHMARK CLASSICO (Speedup Check)")
    print("="*50)
    
    img_bench = pil_img 
    
    input_arr = np.array(img_bench, dtype=np.uint8)
    H, W, C = input_arr.shape
    kernel = np.ones((3, 3), dtype=np.uint8)
    
    cpu = EdgeDetectorCPU()
    gpu = EdgeDetectorGPU()
    
    print(f"Input: {W}x{H}x{C}")

    # --- Run CPU ---
    print("Avvio Sequenziale (CPU)... [1 run]")
    t0 = time.perf_counter()
    res_cpu = cpu.detect(input_arr, kernel)
    t_cpu = time.perf_counter() - t0
    print(f"  -> Tempo CPU: {t_cpu:.4f} s")

    # --- Run GPU ---
    print("Avvio Parallelo (GPU)... [20 run]")
    times_gpu = []
    gpu.detect(input_arr, kernel) 
    for _ in range(20):
        t0 = time.perf_counter()
        res_gpu = gpu.detect(input_arr, kernel)
        times_gpu.append(time.perf_counter() - t0)
    
    mean_gpu_time = statistics.mean(times_gpu)
    
    # --- Risultati ---
    speedup = t_cpu / mean_gpu_time
    throughput = (W*H) / mean_gpu_time / 1e6
    
    print(f"\nRISULTATI:")
    print(f"  CPU Time:       {t_cpu*1000:.2f} ms")
    print(f"  GPU Time (Average of 20 runs):{mean_gpu_time*1000:.2f} ms")
    print(f"  SPEEDUP:        {speedup:.2f} X")
    print(f"  EFFICIENZA:     {throughput:.2f} MPixel/s")
    
    out_path = os.path.join(OUTPUT_DIR, "output_benchmark_result.jpg")
    Image.fromarray(res_gpu).save(out_path)
    print(f"  Immagine salvata in: {out_path}")


# GRAFICI
def run_scientific_experiments(pil_original):
    print("\n" + "="*50)
    print("  FASE 2: GENERAZIONE GRAFICI SCIENTIFICI")
    print("="*50)
    
    detector = EdgeDetectorGPU_Instrumented()
    kernel_3x3 = np.ones((3, 3), dtype=np.uint8)
    
    # --- GRAFICO 1: Pie chart ---
    print("Generazione grafico 1 (Bottleneck)...")
    img_full = np.array(pil_original, dtype=np.uint8)
    detector.detect_with_metrics(img_full, kernel_3x3) 
    _, t_trans, t_comp = detector.detect_with_metrics(img_full, kernel_3x3)
    
    plt.figure(figsize=(6, 6))
    plt.pie([t_trans, t_comp], labels=['Transfer (PCIe)', 'Compute (GPU)'], 
            colors=[COLORS[2], COLORS[0]], autopct='%1.1f%%', startangle=90)
    plt.title('Collo di Bottiglia (Compute vs Transfer)')
    plt.savefig(os.path.join(OUTPUT_DIR, "Exp1_Pie_Bottleneck.png"))
    plt.close()
    
    # --- GRAFICO 2: Scalabilità Risoluzione ---
    print("Generazione grafico 2 (Scalabilità)...")
    resolutions = [(640, 360), (1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]
    res_labels = ["360p", "720p", "1080p", "2K", "4K"]
    times_ms = []
    fps_val = []
    
    for W, H in resolutions:
        resized = np.array(pil_original.resize((W, H)), dtype=np.uint8)
        _, t_t, t_c = detector.detect_with_metrics(resized, kernel_3x3)
        total = t_t + t_c
        times_ms.append(total * 1000)
        fps_val.append(1.0/total)
        
    plt.figure(figsize=(10, 5))
    plt.plot(res_labels, times_ms, marker='o', linewidth=3, color=COLORS[0])
    plt.ylabel('Tempo (ms)')
    plt.title('Tempo esecuzione vs Risoluzione')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "Exp2_Line_Scalability.png"))
    plt.close()
    
    # --- GRAFICO 3: FPS Real-time ---
    print("Generazione grafico 3 (FPS)...")
    plt.figure(figsize=(10, 5))
    bars = plt.bar(res_labels, fps_val, color=COLORS[1], alpha=0.8)
    plt.axhline(y=60, color='r', linestyle='--', label='60 FPS Target')
    plt.ylabel('FPS')
    plt.title('Frame Per Secondo (Real-Time)')
    plt.legend()
    
    for bar in bars:
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                 f"{int(bar.get_height())}", ha='center', va='bottom')
    plt.savefig(os.path.join(OUTPUT_DIR, "Exp3_Bar_FPS.png"))
    plt.close()

    # --- GRAFICO 4: Complessità Kernel ---
    print("Generazione grafico 4 (Kernel Size)...")
    kernel_sizes = [3, 5, 9, 15, 21]
    times_k = []
    img_1080 = np.array(pil_original.resize((1920, 1080)), dtype=np.uint8)
    
    for k in kernel_sizes:
        ks = np.ones((k, k), dtype=np.uint8)
        res_img, _, t_c = detector.detect_with_metrics(img_1080, ks)
        times_k.append(t_c * 1000)
        Image.fromarray(res_img).save(os.path.join(OUTPUT_DIR, f"Visual_Kernel_{k}x{k}.jpg"))
        
    plt.figure(figsize=(10, 5))
    plt.plot(kernel_sizes, times_k, marker='D', color=COLORS[3])
    plt.xlabel('Dimensione Kernel (NxN)')
    plt.ylabel('Tempo GPU (ms)')
    plt.title('Impatto Dimensione Filtro (su 1080p)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "Exp4_Line_KernelSize.png"))
    plt.close()

    print(f"\n Tutto completato. Controlla la cartella: {OUTPUT_DIR}/")


if __name__ == "__main__":
    if not os.path.exists(IMAGE_PATH):
        print("ERRORE: Immagine non trovata.")
        pil_img = Image.fromarray(np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8))
    else:
        pil_img = Image.open(IMAGE_PATH).convert('RGB')
    
    run_text_benchmark(pil_img)
    run_scientific_experiments(pil_img)