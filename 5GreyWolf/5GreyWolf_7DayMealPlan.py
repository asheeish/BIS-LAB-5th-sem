
import numpy as np
import random

# 1. Food database: name, calories, protein, cost
breakfast = [
    ("Oats", 300, 12, 40),
    ("Eggs", 250, 18, 35),
    ("Idli", 280, 8, 30),
    ("Poha", 320, 7, 25),
    ("Dosa", 350, 10, 40)
]

lunch = [
    ("Rice Dal", 550, 20, 70),
    ("Paneer Rice", 600, 28, 90),
    ("Chicken Rice", 650, 40, 110),
    ("Roti Sabzi", 500, 15, 60),
    ("Rajma Rice", 580, 22, 75)
]

dinner = [
    ("Chapati Dal", 450, 18, 55),
    ("Paneer Roti", 500, 25, 80),
    ("Chicken Roti", 550, 38, 100),
    ("Vegetable Rice", 480, 12, 60),
    ("Egg Rice", 500, 25, 70)
]

foods = [breakfast, lunch, dinner]

# 2. Initialize GWO parameters
WOLVES = 20
ITERATIONS = 100
DIM = 21

CALORIE_TARGET = 2000
PROTEIN_TARGET = 80
BUDGET = 250

# 3. Initialize population randomly
wolves = np.random.uniform(0, 4, (WOLVES, DIM))

# 4. Fitness function
def fitness(wolf):
    total_error = 0
    weekly_cost = 0
    selected = []

    for day in range(7):
        calories = 0
        protein = 0

        for meal in range(3):
            index = int(round(wolf[day * 3 + meal]))
            index = max(0, min(4, index))

            food = foods[meal][index]

            calories += food[1]
            protein += food[2]
            weekly_cost += food[3]

            selected.append(food[0])

        total_error += abs(calories - CALORIE_TARGET) / CALORIE_TARGET
        total_error += abs(protein - PROTEIN_TARGET) / PROTEIN_TARGET

    # Budget penalty
    total_error += max(0, weekly_cost - BUDGET * 7) / (BUDGET * 7)

    # Meal repetition penalty
    total_error += (21 - len(set(selected))) * 0.05

    return total_error

# 5. Identify alpha, beta and delta
alpha = np.zeros(DIM)
beta = np.zeros(DIM)
delta = np.zeros(DIM)

alpha_score = float('inf')
beta_score = float('inf')
delta_score = float('inf')

# 6. Repeat until maximum iterations
for iteration in range(ITERATIONS):

    # Evaluate all wolves
    for wolf in wolves:
        score = fitness(wolf)

        if score < alpha_score:
            delta_score, delta = beta_score, beta.copy()
            beta_score, beta = alpha_score, alpha.copy()
            alpha_score, alpha = score, wolf.copy()

        elif score < beta_score:
            delta_score, delta = beta_score, beta.copy()
            beta_score, beta = score, wolf.copy()

        elif score < delta_score:
            delta_score, delta = score, wolf.copy()

    # Update parameter a from 2 to 0
    a = 2 - iteration * (2 / ITERATIONS)

    # Update position of each wolf
    for i in range(WOLVES):
        A1 = 2 * a * np.random.rand(DIM) - a
        C1 = 2 * np.random.rand(DIM)
        D1 = abs(C1 * alpha - wolves[i])
        X1 = alpha - A1 * D1

        A2 = 2 * a * np.random.rand(DIM) - a
        C2 = 2 * np.random.rand(DIM)
        D2 = abs(C2 * beta - wolves[i])
        X2 = beta - A2 * D2

        A3 = 2 * a * np.random.rand(DIM) - a
        C3 = 2 * np.random.rand(DIM)
        D3 = abs(C3 * delta - wolves[i])
        X3 = delta - A3 * D3

        # Average the three positions
        wolves[i] = (X1 + X2 + X3) / 3

        # Check and adjust boundaries
        wolves[i] = np.clip(wolves[i], 0, 4)

# 7. Display best weekly meal plan
print("\nSMART WEEKLY MEAL PLAN")
print("=" * 50)

total_week_cost = 0

for day in range(7):
    print("\nDay", day + 1)

    daily_calories = 0
    daily_protein = 0

    for meal in range(3):
        index = int(round(alpha[day * 3 + meal]))
        index = max(0, min(4, index))

        food = foods[meal][index]

        print(meal + 1, food[0])
        daily_calories += food[1]
        daily_protein += food[2]
        total_week_cost += food[3]

    print("Calories:", daily_calories)
    print("Protein:", daily_protein, "g")

print("\nWeekly Cost: Rs.", total_week_cost)
print("Best Fitness:", round(alpha_score, 4))
print("Optimization Completed!")
