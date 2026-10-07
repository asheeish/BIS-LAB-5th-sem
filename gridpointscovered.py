import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# -------------------------------------------------------------------------
# Simulation Setup & Parameters
# -------------------------------------------------------------------------
GRID_SIZE = 100          # 100x100 grid (coordinates range 0-99)
SENSING_RADIUS = 15     # Radius around each insect covered by sensing
NUM_INSECTS = 30         # Size of swarm
MAX_ITERATIONS = 50     # Number of optimization steps

# Velocity update hyper-parameters
w = 0.5                  # Inertia weight
c1 = 1.5                 # Personal attraction weight
c2 = 1.5                 # Social attraction weight

# Synthetic Fitness Function: Multi-peak food quality field across the grid
def evaluate_food_source(position):
    x, y = position
    peak1 = 100 * np.exp(-((x - 25)**2 + (y - 75)**2) / 400)
    peak2 = 150 * np.exp(-((x - 80)**2 + (y - 30)**2) / 600)
    return peak1 + peak2

def clamp_to_grid(pos, limit=GRID_SIZE - 1):
    return np.clip(pos, 0, limit)

# -------------------------------------------------------------------------
# Swarm Initialization
# -------------------------------------------------------------------------
np.random.seed(42)  # For reproducible visual output
positions = np.random.uniform(0, GRID_SIZE - 1, size=(NUM_INSECTS, 2))
velocities = np.random.uniform(-2, 2, size=(NUM_INSECTS, 2))

pbest_positions = np.copy(positions)
pbest_fitnesses = np.array([evaluate_food_source(p) for p in pbest_positions])

gbest_idx = np.argmax(pbest_fitnesses)
gbest_position = np.copy(pbest_positions[gbest_idx])
gbest_fitness = pbest_fitnesses[gbest_idx]

# -------------------------------------------------------------------------
# Optimization Loop
# -------------------------------------------------------------------------
for iteration in range(MAX_ITERATIONS):
    for i in range(NUM_INSECTS):
        current_fitness = evaluate_food_source(positions[i])
        
        if current_fitness > pbest_fitnesses[i]:
            pbest_fitnesses[i] = current_fitness
            pbest_positions[i] = np.copy(positions[i])
            
            if current_fitness > gbest_fitness:
                gbest_fitness = current_fitness
                gbest_position = np.copy(positions[i])
                
    for i in range(NUM_INSECTS):
        rand1, rand2 = np.random.rand(), np.random.rand()
        
        inertia = w * velocities[i]
        personal_attraction = c1 * rand1 * (pbest_positions[i] - positions[i])
        social_attraction = c2 * rand2 * (gbest_position - positions[i])
        
        velocities[i] = inertia + personal_attraction + social_attraction
        positions[i] = clamp_to_grid(positions[i] + velocities[i])

# -------------------------------------------------------------------------
# Grid Coverage Calculation
# -------------------------------------------------------------------------
grid_x, grid_y = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE))

distances = np.sqrt(
    (grid_x[:, :, np.newaxis] - positions[:, 0])**2 + 
    (grid_y[:, :, np.newaxis] - positions[:, 1])**2
)

coverage_mask = np.any(distances <= SENSING_RADIUS, axis=-1)
covered_points = np.sum(coverage_mask)
total_points = GRID_SIZE * GRID_SIZE
coverage_percentage = (covered_points / total_points) * 100

print(f"Optimal Food Site (gbest): X={gbest_position[0]:.2f}, Y={gbest_position[1]:.2f}")
print(f"Best Food Quality Found:   {gbest_fitness:.2f}")
print("--- Grid Coverage Analysis ---")
print(f"Total Grid Points:         {total_points}")
print(f"Points Covered by Swarm:   {covered_points}")
print(f"Percentage Covered:        {coverage_percentage:.2f}%\n")

# -------------------------------------------------------------------------
# Visualization with Matplotlib
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# --- Plot 1: Food Quality Field & Insect Sensing Radii ---
field_vals = evaluate_food_source([grid_x, grid_y])
contour = axes[0].contourf(grid_x, grid_y, field_vals, levels=20, cmap="viridis", alpha=0.7)
plt.colorbar(contour, ax=axes[0], label="Food Quality")

for pos in positions:
    circle = patches.Circle(
        (pos[0], pos[1]), 
        SENSING_RADIUS, 
        color='cyan', 
        alpha=0.15, 
        ec='blue', 
        linewidth=1
    )
    axes[0].add_patch(circle)

axes[0].scatter(positions[:, 0], positions[:, 1], color='red', s=25, label='Insects')
axes[0].scatter(gbest_position[0], gbest_position[1], color='gold', marker='*', s=150, edgecolors='black', label='gbest')

axes[0].set_xlim(0, GRID_SIZE)
axes[0].set_ylim(0, GRID_SIZE)
axes[0].set_title("Swarm Positions & Sensing Radii")
axes[0].set_xlabel("X Grid Coordinate")
axes[0].set_ylabel("Y Grid Coordinate")
axes[0].legend(loc="upper left")

# --- Plot 2: Binary Coverage Mask ---
axes[1].imshow(coverage_mask, cmap="Blues", origin="lower", extent=[0, GRID_SIZE, 0, GRID_SIZE])
axes[1].scatter(positions[:, 0], positions[:, 1], color='red', s=15, alpha=0.8)

axes[1].set_title(f"Grid Coverage Mask ({coverage_percentage:.2f}% Covered)")
axes[1].set_xlabel("X Grid Coordinate")
axes[1].set_ylabel("Y Grid Coordinate")

plt.tight_layout()
plt.show()