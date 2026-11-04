import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. GRID DATA & PARAMETER INITIALIZATION
# ==========================================
np.random.seed(42)

# Grid Entities
num_loads = 20         # Number of load demand centers/substations
num_generators = 5     # Number of microgrid generators/sources

# Random power demands (kW) and generator capacities (kW)
loads = np.random.uniform(15, 60, size=num_loads)
capacities = np.array([220, 200, 250, 180, 210])

# Coordinates for load centers and generators (in km)
load_coords = np.random.uniform(0, 50, size=(num_loads, 2))
gen_coords = np.random.uniform(0, 50, size=(num_generators, 2))

# Transmission distance matrix (Euclidean distance)
distance_matrix = np.linalg.norm(
    load_coords[:, np.newaxis, :] - gen_coords[np.newaxis, :, :], axis=2
)

# ACO Parameters
num_ants = 30
max_iterations = 100
alpha = 1.0           # Pheromone importance
beta = 2.5            # Heuristic importance (line distance)
rho = 0.1             # Pheromone evaporation rate
Q = 100.0             # Pheromone deposit factor
initial_pheromone = 1.0

# Initialize Matrices
pheromone_matrix = np.full((num_loads, num_generators), initial_pheromone)
heuristic_matrix = 1.0 / (distance_matrix + 1e-5)  # Favor closer generators

best_assignment = None
best_cost = float('inf')
cost_history = []

# ==========================================
# 2. ANT COLONY OPTIMIZATION LOOP
# ==========================================
for iteration in range(max_iterations):
    ant_assignments = []
    ant_costs = []

    for k in range(num_ants):
        assignment = np.zeros(num_loads, dtype=int)
        gen_current_loads = np.zeros(num_generators)

        # Shuffle load processing order for search diversity
        load_order = np.random.permutation(num_loads)

        for j in load_order:
            # Pheromone and Heuristic influence
            tau = pheromone_matrix[j] ** alpha
            eta = heuristic_matrix[j] ** beta
            probabilities = tau * eta

            # Dynamic capacity-awareness: reduce probability for overloaded generators
            for i in range(num_generators):
                if gen_current_loads[i] + loads[j] > capacities[i]:
                    probabilities[i] *= 0.01  # Soft penalty during construction

            # Normalize probabilities
            prob_sum = np.sum(probabilities)
            if prob_sum > 0:
                probabilities /= prob_sum
            else:
                probabilities = np.ones(num_generators) / num_generators

            # Select generator probabilistically
            selected_gen = np.random.choice(num_generators, p=probabilities)
            assignment[j] = selected_gen
            gen_current_loads[selected_gen] += loads[j]

        # Calculate Total Cost Function:
        # Cost = Transmission Loss + Load Imbalance Penalty + Overload Penalty
        transmission_loss = np.sum(distance_matrix[np.arange(num_loads), assignment] * loads)
        load_ratios = gen_current_loads / capacities
        imbalance_cost = np.var(load_ratios) * 800.0  # Penalize uneven loading
        overload_penalty = np.sum(np.maximum(0, gen_current_loads - capacities)) * 2000.0

        total_cost = transmission_loss + imbalance_cost + overload_penalty

        ant_assignments.append(assignment)
        ant_costs.append(total_cost)

        # Track global best solution
        if total_cost < best_cost:
            best_cost = total_cost
            best_assignment = assignment.copy()

    cost_history.append(best_cost)

    # Pheromone Evaporation
    pheromone_matrix = (1.0 - rho) * pheromone_matrix

    # Pheromone Deposit (Iterative updates based on solution quality)
    for assignment, cost in zip(ant_assignments, ant_costs):
        for j in range(num_loads):
            gen_idx = assignment[j]
            pheromone_matrix[j, gen_idx] += Q / cost

# ==========================================
# 3. RESULTS & VISUALIZATION
# ==========================================
print(f"Optimal Cost Achieved: {best_cost:.2f}\n")

# Compute final generator loads
final_gen_loads = np.zeros(num_generators)
for j in range(num_loads):
    final_gen_loads[best_assignment[j]] += loads[j]

for i in range(num_generators):
    utilization = (final_gen_loads[i] / capacities[i]) * 100
    print(f"Generator {i+1}: Demand Assigned = {final_gen_loads[i]:.2f} kW / Capacity = {capacities[i]} kW (Utilization: {utilization:.1f}%)")

# Plot Convergence and Load Allocation
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Convergence
axes[0].plot(cost_history, color='navy', linewidth=2)
axes[0].set_title('ACO Convergence Curve')
axes[0].set_xlabel('Iteration')
axes[0].set_ylabel('Best Cost (Loss + Imbalance)')
axes[0].grid(True)

# Plot 2: Load Distribution per Generator
x = np.arange(num_generators)
width = 0.35
axes[1].bar(x - width/2, capacities, width, label='Capacity (kW)', color='lightgray', edgecolor='black')
axes[1].bar(x + width/2, final_gen_loads, width, label='Assigned Load (kW)', color='teal')
axes[1].set_title('Generator Load Balancing')
axes[1].set_xlabel('Microgrid Generator')
axes[1].set_ylabel('Power (kW)')
axes[1].set_xticks(x)
axes[1].set_xticklabels([f'Gen {i+1}' for i in range(num_generators)])
axes[1].legend()
axes[1].grid(True, axis='y')

plt.tight_layout()
plt.show()
