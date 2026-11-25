import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os


def create_sample_image(size=128):
    """Creates a clean synthetic grayscale test image with geometric shapes."""
    img = np.full((size, size), 40, dtype=np.float64)
    # Bright square
    img[20:60, 20:60] = 210
    # Medium circle
    y, x = np.ogrid[:size, :size]
    mask = (x - 85) ** 2 + (y - 80) ** 2 <= 25 ** 2
    img[mask] = 160
    # Diagonal bright bar
    for i in range(15, 55):
        img[i, i + 55:i + 62] = 240
    return img


def add_salt_and_pepper_noise(img, noise_ratio=0.12):
    """Corrupts random pixels with 0 (pepper) or 255 (salt)."""
    noisy = np.copy(img)
    rng = np.random.default_rng(42)
    rand_matrix = rng.random(img.shape)
    noisy[rand_matrix < (noise_ratio / 2)] = 0
    noisy[rand_matrix > 1 - (noise_ratio / 2)] = 255
    return noisy


def denoise_row_rule(args):
    """Step 4a: Parallel cellular transition rule for denoising (3x3 Moore Median)."""
    i, G = args
    rows, cols = G.shape
    new_row = np.copy(G[i])

    # Boundary cells remain unchanged
    if i == 0 or i == rows - 1:
        return i, new_row

    for j in range(1, cols - 1):
        # Gather 3x3 Moore neighborhood
        neighborhood = G[i - 1:i + 2, j - 1:j + 2]
        # New[i][j] = Rule(G[i][j], neighbors)
        new_row[j] = np.median(neighborhood)

    return i, new_row


def edge_row_rule(args):
    """Step 4a: Parallel cellular transition rule for Edge Detection (Sobel Operator)."""
    i, G = args
    rows, cols = G.shape
    new_row = np.zeros(cols, dtype=np.float64)

    if i == 0 or i == rows - 1:
        return i, new_row

    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float64)
    ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float64)

    for j in range(1, cols - 1):
        neighborhood = G[i - 1:i + 2, j - 1:j + 2]
        gx = np.sum(neighborhood * kx)
        gy = np.sum(neighborhood * ky)
        new_row[j] = np.hypot(gx, gy)

    return i, new_row


def run_parallel_cellular_image_processing():
    # 1. Start & 2. Initialize the cellular grid G
    clean_img = create_sample_image(128)
    G = add_salt_and_pepper_noise(clean_img, noise_ratio=0.15)
    initial_noisy = np.copy(G)

    rows, cols = G.shape
    T = 3  # Number of parallel denoising iterations
    num_workers = min(rows, os.cpu_count() or 4)

    # 3. Create temporary grid New
    New = np.copy(G)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # --- STAGE 1: PARALLEL CELLULAR DENOISING ---
        # 4. For t = 1 to T do
        for t in range(1, T + 1):
            tasks = [(i, G) for i in range(rows)]
            # 4a. Parallel for each cell [i][j] & 4b. Sync all processes
            for i, updated_row in executor.map(denoise_row_rule, tasks):
                New[i] = updated_row

            # 4c. Update Grid: G = New
            G = np.copy(New)
            print(f"Denoising iteration t = {t} completed and synchronized.")

        denoised_img = np.copy(G)

        # --- STAGE 2: PARALLEL CELLULAR EDGE DETECTION ---
        edge_tasks = [(i, G) for i in range(rows)]
        for i, edge_row in executor.map(edge_row_rule, edge_tasks):
            New[i] = edge_row

        # Normalize edge magnitudes to 0-255 for display
        edges_img = np.clip((New / np.max(New)) * 255, 0, 255)

    # 5. Display final grids visually & 6. Stop
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = [
        "1. Original Ground Truth",
        "2. Noisy Input Grid (G₀)",
        f"3. Parallel Denoised (t = {T})",
        "4. Parallel Edge Detection"
    ]
    images = [clean_img, initial_noisy, denoised_img, edges_img]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("parallel_ca_output.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run_parallel_cellular_image_processing()