"""
Equitable A* Pathfinding with AHP, α-Hurwicz, and Gini Penalty
--------------------------------------------------------------
Implements:
- AHP (Analytic Hierarchy Process) to weight decision criteria.
- α-Hurwicz criterion to combine best- and worst-case scenario costs per edge.
- Gini coefficient to measure inequality of edge costs along a path.
- Modified A* that minimises total expected cost + λ * Gini(path).
- Admissible heuristic for A* based on minimum cost per distance unit.
"""

import numpy as np
import heapq
import math
from typing import List, Tuple, Dict, Any

# ========================== 1. AHP: Criteria Weights ==========================

def ahp_weights(pairwise_matrix: np.ndarray) -> np.ndarray:
    """
    Compute weights from a pairwise comparison matrix using the eigenvector method.
    Also checks consistency ratio (CR); prints warning if CR > 0.1.
    """
    n = pairwise_matrix.shape[0]
    # Compute principal eigenvector (normalised)
    eigvals, eigvecs = np.linalg.eig(pairwise_matrix)
    principal_eigvec = eigvecs[:, np.argmax(eigvals.real)]
    weights = principal_eigvec.real / np.sum(principal_eigvec.real)

    # Consistency check
    lambda_max = np.max(eigvals.real)
    ci = (lambda_max - n) / (n - 1)
    ri = {1:0, 2:0, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}
    cr = ci / ri.get(n, 1.49)
    if cr > 0.1:
        print(f"Warning: Consistency ratio = {cr:.3f} > 0.1. Pairwise matrix may be inconsistent.")
    return weights

# Example: pairwise comparison matrix for 4 criteria
# Criteria order: Travel Time (T), Financial Cost (C), Security Risk (S), Environmental Impact (E)
# Using Saaty scale: 1 = equal, 3 = moderate, 5 = strong, 7 = very strong, 9 = extreme.
# Rows: T, C, S, E
pairwise = np.array([
    [1,   2,   1/3, 3],   # T vs: T(1), C(2), S(1/3), E(3)
    [1/2, 1,   1/4, 2],   # C vs: T(1/2), C(1), S(1/4), E(2)
    [3,   4,   1,   5],   # S vs: T(3), C(4), S(1), E(5)
    [1/3, 1/2, 1/5, 1]    # E vs: T(1/3), C(1/2), S(1/5), E(1)
])

weights = ahp_weights(pairwise)
print("AHP Weights (TravelTime, Cost, Security, EnvImpact):", np.round(weights, 3))

# ========================== 2. Graph & Edge Data ==========================

class City:
    def __init__(self, id: int, name: str, x: float, y: float):
        self.id = id
        self.name = name
        self.x = x
        self.y = y

def euclidean_distance(c1: City, c2: City) -> float:
    return math.hypot(c1.x - c2.x, c1.y - c2.y)

# Synthetic graph: 6 cities with coordinates
cities = [
    City(0, "A", 0, 0),
    City(1, "B", 4, 0),
    City(2, "C", 2, 3),
    City(3, "D", 6, 2),
    City(4, "E", 8, 1),
    City(5, "F", 5, 5)   # goal
]
start = cities[0]    # A
goal = cities[5]     # F

# For each directed edge (i,j) we store:
# - base values per criterion (time, cost, risk, env) under nominal conditions
# - best and worst multipliers (scenario factors) for each criterion
#   e.g., time can vary by 0.7 (best) to 1.5 (worst) due to traffic/weather
criterion_names = ["Time", "Cost", "Risk", "Env"]
# Scenario multipliers: each criterion has (min_factor, max_factor)
scenario_factors = {
    "Time": (0.70, 1.50),   # optimistic (fast) vs pessimistic (slow)
    "Cost": (0.80, 1.40),   # fuel price fluctuation
    "Risk": (0.60, 2.00),   # accidents can be much lower or much higher
    "Env":  (0.85, 1.30)    # emission variability
}

# We will create a complete graph for simplicity (each pair connected)
edges = {}   # key: (i,j) -> dict with base_criteria, min_cost, max_cost, expected_cost
dist_matrix = {}

for i, c1 in enumerate(cities):
    for j, c2 in enumerate(cities):
        if i == j:
            continue
        dist = euclidean_distance(c1, c2)
        dist_matrix[(i,j)] = dist

        # Base criteria values (proportional to distance, with some variation)
        # Travel time (hours) : assume speed 50 km/h, but distance in arbitrary units
        time_base = dist / 50.0
        # Financial cost ($) : fuel + wear, 2 $ per distance unit
        cost_base = dist * 2.0
        # Security risk (probability of incident per trip) : 0.001 * dist
        risk_base = 0.001 * dist
        # Environmental impact (kg CO2) : 0.5 * dist
        env_base = 0.5 * dist

        base_criteria = np.array([time_base, cost_base, risk_base, env_base])

        # Best and worst scenario values per criterion
        min_vals = []
        max_vals = []
        for idx, crit in enumerate(criterion_names):
            min_f, max_f = scenario_factors[crit]
            min_vals.append(base_criteria[idx] * min_f)
            max_vals.append(base_criteria[idx] * max_f)

        # Weighted cost under best and worst scenarios (using AHP weights)
        min_weighted_cost = np.dot(weights, min_vals)
        max_weighted_cost = np.dot(weights, max_vals)

        # α-Hurwicz (α = optimism coefficient, here 0.6)
        alpha = 0.6
        expected_cost = alpha * min_weighted_cost + (1 - alpha) * max_weighted_cost

        edges[(i,j)] = {
            "base": base_criteria,
            "min_cost": min_weighted_cost,
            "max_cost": max_weighted_cost,
            "expected_cost": expected_cost,
            "distance": dist
        }

print("\nEdge expected costs (sample):")
for (i,j), data in list(edges.items())[:5]:
    print(f"{cities[i].name}->{cities[j].name}: {data['expected_cost']:.3f}")

# ========================== 3. Gini Coefficient ==========================

def gini_coefficient(values: List[float]) -> float:
    """Compute Gini index for a list of positive numbers. Returns 0 if less than 2 values."""
    n = len(values)
    if n < 2:
        return 0.0
    sorted_vals = sorted(values)
    # Gini = (Σ_i Σ_j |x_i - x_j|) / (2 n² μ)
    # Efficient computation using sorted list
    sum_abs_diff = 0.0
    for i, val_i in enumerate(sorted_vals):
        sum_abs_diff += val_i * (2*i - n + 1)
    gini = (2 * sum_abs_diff) / (n * sum(sorted_vals))
    return gini

# ========================== 4. Equitable A* Algorithm ==========================

def equitable_astar(start: City, goal: City, edges: Dict, lambda_penalty: float = 0.5):
    """
    A* search that minimises f(n) = g_equitable(n) + h(n)
    where g_equitable = sum(expected_costs) + λ * Gini(list of edge costs so far)
    h(n) = min_cost_per_distance * Euclidean distance to goal (admissible).
    """
    # Precompute minimum cost per unit distance across all edges (for heuristic)
    min_cost_per_dist = min(data["expected_cost"] / data["distance"] for data in edges.values())
    print(f"Minimum cost per distance unit: {min_cost_per_dist:.4f}")

    # Heuristic function
    def heuristic(city: City) -> float:
        return min_cost_per_dist * euclidean_distance(city, goal)

    # Data structures
    open_set = []  # priority queue (f, node_id, path_cost_list, g_sum, path)
    # state: (node_id, tuple of visited nodes? Actually standard A* does not need visited set if graph is directed/acyclic,
    # but we allow revisiting? For pathfinding without cycles we keep best g for each node.
    # However g_equitable depends on entire path history (Gini), so we must store per-state.
    # We'll use a dictionary to record best (g_total + λ*Gini) for each (node, frozenset?) - too large.
    # For simplicity, we use a visited dict that stores the best "equitable cost" for each node,
    # but because Gini is not additive, we cannot prune only by node. We'll implement a best-known
    # map from node to the best sum_cost (without penalty) as an optimistic bound.
    # Alternatively, we just run without heavy pruning – graph small. Here we do simple BFS-like A*
    # with path tracking and avoid cycles (do not revisit nodes on same path).
    best_known = {}  # node_id -> best total expected cost (sum EC) found so far (for pruning)

    start_id = start.id
    goal_id = goal.id
    # Initial state: path = [start_id], edge_costs = [], sum_cost = 0
    heapq.heappush(open_set, (heuristic(start), start_id, [], 0.0, [start_id]))

    best_path = None
    best_f = float('inf')
    iterations = 0

    while open_set:
        iterations += 1
        f, node_id, edge_costs_sofar, sum_cost_sofar, path = heapq.heappop(open_set)

        if node_id == goal_id:
            if f < best_f:
                best_f = f
                best_path = path
            # Continue to possibly find better path (since Gini may produce different f)
            continue

        # Prune: if we already reached this node with a lower sum_cost (without penalty),
        # then any future path will have sum_cost >= that, and because Gini is non-negative,
        # the equitable cost will be higher than the previous sum_cost. But we might miss
        # a path with slightly higher sum_cost but much lower Gini? Actually Gini is not monotonic.
        # For correctness, we keep a simple threshold: only prune if sum_cost_sofar > best_known.get(node_id, float('inf')) + 1e-6.
        # But this is not strictly admissible. For small graphs we skip pruning.
        # Instead we implement a visited set for (node, tuple_of_edges?) - not practical.
        # We'll rely on small graph size and early termination when we have found a path
        # and the current f is already above best_f.

        if best_path is not None and f >= best_f:
            continue

        # Expand neighbours
        for neighbor in cities:
            nid = neighbor.id
            if nid == node_id or nid in path:  # avoid immediate cycles
                continue
            edge_key = (node_id, nid)
            if edge_key not in edges:
                continue
            edge_data = edges[edge_key]
            ec = edge_data["expected_cost"]

            new_edge_costs = edge_costs_sofar + [ec]
            new_sum_cost = sum_cost_sofar + ec
            # Compute Gini penalty
            gini = gini_coefficient(new_edge_costs)
            g_equitable = new_sum_cost + lambda_penalty * gini
            h_val = heuristic(neighbor)
            new_f = g_equitable + h_val

            # Optional pruning with best_known (optimistic)
            if nid in best_known and new_sum_cost > best_known[nid] + 1e-6:
                # If we already found a path to this node with strictly lower sum_cost,
                # we skip because any extension will have sum_cost >= that, but Gini could be lower.
                # For safety we skip only if new_sum_cost > best_known[nid] + margin.
                # To be correct, we don't prune aggressively.
                pass
            best_known[nid] = min(best_known.get(nid, float('inf')), new_sum_cost)

            heapq.heappush(open_set, (new_f, nid, new_edge_costs, new_sum_cost, path + [nid]))

    if best_path is None:
        return None, None, None

    # Compute final metrics for best path
    path_cities = [cities[i] for i in best_path]
    edge_costs = []
    for k in range(len(best_path)-1):
        ec = edges[(best_path[k], best_path[k+1])]["expected_cost"]
        edge_costs.append(ec)
    total_cost = sum(edge_costs)
    gini = gini_coefficient(edge_costs)
    equitable_obj = total_cost + lambda_penalty * gini
    return path_cities, total_cost, gini, equitable_obj

# ========================== 5. Standard A* (without Gini) for Comparison ==========================

def standard_astar(start: City, goal: City, edges: Dict):
    """Standard A* minimising only total expected cost."""
    min_cost_per_dist = min(data["expected_cost"] / data["distance"] for data in edges.values())
    def heuristic(city):
        return min_cost_per_dist * euclidean_distance(city, goal)

    open_set = [(heuristic(start), start.id, 0.0, [start.id])]
    best_known_cost = {start.id: 0.0}
    best_path = None
    best_total = float('inf')

    while open_set:
        f, node_id, g, path = heapq.heappop(open_set)
        if node_id == goal.id:
            if g < best_total:
                best_total = g
                best_path = path
            continue
        if g > best_known_cost.get(node_id, float('inf')):
            continue
        for neighbor in cities:
            nid = neighbor.id
            if nid in path:
                continue
            edge_key = (node_id, nid)
            if edge_key not in edges:
                continue
            new_g = g + edges[edge_key]["expected_cost"]
            if nid not in best_known_cost or new_g < best_known_cost[nid]:
                best_known_cost[nid] = new_g
                h_val = heuristic(neighbor)
                heapq.heappush(open_set, (new_g + h_val, nid, new_g, path + [nid]))
    if best_path is None:
        return None, None
    path_cities = [cities[i] for i in best_path]
    return path_cities, best_total

# ========================== 6. Run Experiments ==========================

print("\n" + "="*60)
print("STANDARD A* (minimises total expected cost)")
std_path, std_cost = standard_astar(start, goal, edges)
if std_path:
    std_edge_costs = [edges[(std_path[i].id, std_path[i+1].id)]["expected_cost"] for i in range(len(std_path)-1)]
    std_gini = gini_coefficient(std_edge_costs)
    print(f"Path: {' -> '.join([c.name for c in std_path])}")
    print(f"Total expected cost: {std_cost:.4f}")
    print(f"Gini coefficient: {std_gini:.4f}")

print("\n" + "="*60)
print("EQUITABLE A* (minimises total cost + λ * Gini) with λ = 0.5")
eq_path, eq_total, eq_gini, eq_obj = equitable_astar(start, goal, edges, lambda_penalty=0.5)
if eq_path:
    print(f"Path: {' -> '.join([c.name for c in eq_path])}")
    print(f"Total expected cost: {eq_total:.4f}")
    print(f"Gini coefficient: {eq_gini:.4f}")
    print(f"Objective (cost + λ*Gini): {eq_obj:.4f}")

# Try a higher penalty to see the effect
print("\n" + "="*60)
print("EQUITABLE A* with λ = 2.0 (strong fairness preference)")
eq_path2, eq_total2, eq_gini2, eq_obj2 = equitable_astar(start, goal, edges, lambda_penalty=2.0)
if eq_path2:
    print(f"Path: {' -> '.join([c.name for c in eq_path2])}")
    print(f"Total expected cost: {eq_total2:.4f}")
    print(f"Gini coefficient: {eq_gini2:.4f}")
    print(f"Objective: {eq_obj2:.4f}")