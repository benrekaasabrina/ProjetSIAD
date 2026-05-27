#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Equitable A* Algorithm incorporating AHP, alpha-Hurwicz, and Gini Equity Coefficients.
"""

import heapq
import math
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict

# =============================================================================
# STAGE 1: Qualitative Weighting (AHP)
# =============================================================================
def ahp_weights(pairwise_matrix: np.ndarray) -> np.ndarray:
    """Calculates criteria weights from an AHP pairwise comparison matrix."""
    eigvals, eigvecs = np.linalg.eig(pairwise_matrix)
    principal = eigvecs[:, np.argmax(eigvals.real)]
    weights = principal.real / np.sum(principal.real)
    return weights

# Pairwise comparison matrix for 4 criteria:
# [Travel Time, Financial Cost, Security/Accident Risk, Environmental Impact]
pairwise = np.array([
    [1, 2, 1/3, 3],
    [1/2, 1, 1/4, 2],
    [3, 4, 1, 5],
    [1/3, 1/2, 1/5, 1]
])
weights = ahp_weights(pairwise)
print("="*60)
print("STAGE 1: AHP Criteria Weights")
print(f"Time: {weights[0]:.4f}, Cost: {weights[1]:.4f}, Risk: {weights[2]:.4f}, Eco: {weights[3]:.4f}")
print("="*60)

# =============================================================================
# STAGE 2: Scenario Arbitrage (alpha-Hurwicz) & Building Edge Weights
# =============================================================================
class City:
    def __init__(self, id: int, name: str, x: float, y: float):
        self.id = id
        self.name = name
        self.x = x
        self.y = y

cities = [
    City(0, "A", 0, 0),
    City(1, "B", 2, 1),
    City(2, "C", 4, 0),
    City(3, "D", 5, 3),
    City(4, "E", 2, 4),
    City(5, "F", 5, 5)
]
start = cities[0]
goal = cities[5]

def euclidean_distance(c1: City, c2: City) -> float:
    return math.hypot(c1.x - c2.x, c1.y - c2.y)

# Raw multi-criteria scenario data for each valid edge.
# Format: (node_i, node_j): {"best": [time, cost, risk, eco], "worst": [time, cost, risk, eco]}
# Values are engineered to preserve your original problem's core trade-off after weighting.
edge_scenarios = {
    (0, 1): {"best": [0.4, 0.3, 0.5, 0.4], "worst": [0.6, 0.5, 0.6, 0.6]}, # Target GC ~ 0.5
    (1, 4): {"best": [1.8, 1.5, 2.1, 1.9], "worst": [2.2, 2.0, 2.1, 2.1]}, # Target GC ~ 2.0 (Risky/Expensive leg)
    (4, 5): {"best": [0.3, 0.4, 0.5, 0.4], "worst": [0.5, 0.6, 0.6, 0.6]}, # Target GC ~ 0.5
    
    (0, 2): {"best": [1.8, 1.7, 2.1, 1.9], "worst": [2.1, 2.1, 2.1, 2.1]}, # Target GC ~ 2.0
    (2, 3): {"best": [0.8, 0.9, 1.0, 0.9], "worst": [1.1, 1.1, 1.1, 1.2]}, # Target GC ~ 1.0
    (3, 5): {"best": [0.9, 0.8, 1.0, 0.9], "worst": [1.1, 1.2, 1.1, 1.1]}, # Target GC ~ 1.0
}

edges = {}
alpha = 0.5 # Hurwicz optimism-pessimism parameter

# Process all possible pairs to construct the full graph network
for i in range(len(cities)):
    for j in range(len(cities)):
        if i == j:
            continue
            
        dist = euclidean_distance(cities[i], cities[j])
        
        if (i, j) in edge_scenarios:
            scenarios = edge_scenarios[(i, j)]
            # Calculate the weighted cost under best-case and worst-case scenarios
            min_cost_ij = np.sum(weights * np.array(scenarios["best"]))
            max_cost_ij = np.sum(weights * np.array(scenarios["worst"]))
            # Apply alpha-Hurwicz equation to find the risk-adjusted Generalized Cost (GC)
            gc_ij = alpha * min_cost_ij + (1 - alpha) * max_cost_ij
        else:
            # Non-existent direct connections get a massive penalty cost
            gc_ij = 100.0
            
        edges[(i, j)] = {"generalized_cost": gc_ij, "distance": dist}

# =============================================================================
# STAGE 3: Gini Coefficient & Balancing Function
# =============================================================================
def gini(values: List[float]) -> float:
    """Calculates the Gini Coefficient to measure statistical dispersion/inequality."""
    n = len(values)
    if n < 2:
        return 0.0
    sorted_vals = sorted(values)
    total = sum(sorted_vals)
    if total == 0:
        return 0.0
    sum_abs = 0.0
    for i, v in enumerate(sorted_vals):
        sum_abs += v * (2 * i - n + 1)
    return (2 * sum_abs) / (n * total)

# =============================================================================
# STAGE 4: Equitable A* Pathfinding
# =============================================================================
def equitable_astar(lambda_penalty: float):
    # Standard admissible heuristic modifier: minimum generalized cost per unit of distance
    min_cost_per_dist = min(data["generalized_cost"] / data["distance"] for data in edges.values())
    
    def heuristic(city: City) -> float:
        return min_cost_per_dist * euclidean_distance(city, goal)

    # Priority queue format: (f_score, current_node, [edge_costs_so_far], cumulative_cost, [path_node_ids])
    open_set = [(heuristic(start), start.id, [], 0.0, [start.id])]
    
    best_sum_cost = {}
    best_goal_f = float('inf')
    best_path = None
    best_total = None
    best_gini = None

    while open_set:
        f, node_id, edge_costs_list, sum_cost, path_ids = heapq.heappop(open_set)
        
        if f >= best_goal_f:
            continue
        if node_id in best_sum_cost and sum_cost >= best_sum_cost[node_id]:
            continue
        best_sum_cost[node_id] = sum_cost

        # Goal verification
        if node_id == goal.id:
            g_coef = gini(edge_costs_list)
            obj = sum_cost + lambda_penalty * g_coef
            if obj < best_goal_f:
                best_goal_f = obj
                best_path = path_ids
                best_total = sum_cost
                best_gini = g_coef
            continue

        # Neighbor exploration
        for nxt in cities:
            nid = nxt.id
            if nid in path_ids:
                continue
                
            ekey = (node_id, nid)
            if ekey not in edges:
                continue
                
            gc = edges[ekey]["generalized_cost"]
            new_edges = edge_costs_list + [gc]
            new_sum = sum_cost + gc
            
            # Inject Gini penalty into g(n) for the path traveled so far
            new_g = gini(new_edges)
            g_equitable = new_sum + lambda_penalty * new_g
            
            h_val = heuristic(nxt)
            f_equitable = g_equitable + h_val
            
            heapq.heappush(open_set, (f_equitable, nid, new_edges, new_sum, path_ids + [nid]))

    if best_path is None:
        return None, None, None
    return [cities[i] for i in best_path], best_total, best_gini

# =============================================================================
# EXPERIMENTS & EVALUATION
# =============================================================================
print("STAGE 4: EQUITABLE A* EXPERIMENTS")
print(f"{'λ (Penalty)':<12} {'Selected Optimal Path':<25} {'Total Cost':<12} {'Gini Index':<8}")
print("-"*60)

lambda_vals = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0]
results = {}
for lam in lambda_vals:
    path, cost, g_idx = equitable_astar(lam)
    if path:
        path_str = " -> ".join([c.name for c in path])
        results[lam] = (path, cost, g_idx)
        print(f"{lam:<12} {path_str:<25} {cost:<12.4f} {g_idx:<8.4f}")
    else:
        print(f"{lam:<12} No path found")

# =============================================================================
# VISUALIZATION
# =============================================================================
plt.figure(figsize=(10, 7))
for (i, j), data in edges.items():
    if data["generalized_cost"] < 50:
        c1, c2 = cities[i], cities[j]
        plt.plot([c1.x, c2.x], [c1.y, c2.y], 'lightgray', linewidth=1, zorder=1)

for c in cities:
    plt.scatter(c.x, c.y, c='darkblue', s=120, zorder=2)
    plt.text(c.x + 0.1, c.y + 0.1, c.name, fontsize=12, fontweight='bold')

seen = {}
colors = ['crimson', 'forestgreen', 'royalblue', 'darkorange', 'purple']
for idx, (lam, (path, cost, g_idx)) in enumerate(results.items()):
    key = tuple(c.id for c in path)
    if key not in seen:
        seen[key] = (lam, path, cost, g_idx)
        xs = [c.x for c in path]
        ys = [c.y for c in path]
        plt.plot(xs, ys, color=colors[len(seen) % len(colors)], linewidth=3, 
                 marker='o', label=f"λ={lam:.1f} (Cost={cost:.2f}, Gini={g_idx:.3f})", zorder=3)

plt.title("Multi-Criteria Equitable A*: Total Cost vs. Inequality Trade-off", fontsize=14, pad=15)
plt.xlabel("X Coordinate Space", fontsize=11)
plt.ylabel("Y Coordinate Space", fontsize=11)
plt.legend(loc='upper left', frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("equitable_astar_final.png", dpi=300)
plt.show()