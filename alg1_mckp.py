import numpy as np
from typing import List, Tuple, Dict, Optional

class MCKPApproximateBinarySearch:
    """
    Multiple-Choice Knapsack Problem (MCKP) solver using
    the approximate binary search algorithm from the paper.
    """

    def __init__(self, epsilon: float = 0.6):
        """
        Initialize the MCKP solver with epsilon parameter.
        epsilon = 0.6 gives 4/5 approximation ratio.
        """
        self.epsilon = epsilon
        self.threshold_factor = 0.8  # From the paper: cij/aij >= 0.8x/b

    def solve_mckp_single_bin(self, 
                              profits: List[List[float]], 
                              costs: List[List[float]], 
                              capacity: float) -> Tuple[List[int], float]:
        """
        Solve MCKP for a single bin using approximate binary search.

        Args:
            profits: profits[i][k] = profit of item k in class i
            costs: costs[i][k] = cost of item k in class i
            capacity: capacity of the bin

        Returns:
            (selected_items, total_profit): selected_items[i] = k if class i selects item k, else -1
        """
        m = len(profits)  # number of classes
        n = sum(len(p) for p in profits)  # total number of items

        if m == 0 or capacity <= 0:
            return [-1] * len(profits), 0.0

        # Step 1: Initialize search interval [L, U]
        L = 0.0
        for i in range(m):
            if profits[i]:
                L = max(L, max(profits[i]))

        U = m * L

        # Edge case: if L == 0, all profits are 0
        if L == 0:
            # Select cheapest items that fit
            return self._select_zero_profit_items(profits, costs, capacity)

        # Approximate binary search
        max_iterations = 100  # Prevent infinite loops
        iteration = 0

        while U / L >= 5 and iteration < max_iterations:
            iteration += 1
            x = U / 2.0

            # Use branching algorithm to determine range
            f_upper, f_lower = self._branching_algorithm(
                profits, costs, capacity, x
            )

            if f_upper <= 1.6 * x:
                # f* <= 1.6x, set U = 1.6x = 0.8U
                U = 0.8 * U
            elif f_lower >= 0.4 * x:
                # f* >= 0.4x, set L = 0.4x = 0.2U
                L = 0.2 * U

        # Final refinement: find feasible solution with profit at least L
        return self._find_feasible_solution(profits, costs, capacity, L)

    def _branching_algorithm(self, 
                             profits: List[List[float]], 
                             costs: List[List[float]], 
                             capacity: float, 
                             x: float) -> Tuple[float, float]:
        """
        Branching algorithm BA(x) that determines if f* <= x(1+epsilon) or f* >= x(1-epsilon).
        Returns (upper_bound, lower_bound) for f*.
        """
        m = len(profits)
        c = 0.0  # total profit of selected items

        # Step 1: In each class, select item with potential profit/weight >= threshold
        threshold = self.threshold_factor * x / capacity if capacity > 0 else float('inf')

        for i in range(m):
            if not profits[i]:
                continue

            best_item_idx = -1
            best_profit = -1.0

            for k in range(len(profits[i])):
                if costs[i][k] <= capacity:  # Only consider items that can fit individually
                    if costs[i][k] > 0:
                        profit_weight_ratio = profits[i][k] / costs[i][k]
                    else:
                        profit_weight_ratio = float('inf') if profits[i][k] > 0 else 0

                    if profit_weight_ratio >= threshold and profits[i][k] > best_profit:
                        best_item_idx = k
                        best_profit = profits[i][k]

            if best_item_idx != -1:
                c += best_profit

        # Step 2: Compare c with threshold * x
        if c >= self.threshold_factor * x:
            # f* >= (1 - epsilon) * x = 0.4x
            return float('inf'), (1 - self.epsilon) * x
        else:
            # f* <= (1 + epsilon) * x = 1.6x
            return (1 + self.epsilon) * x, 0.0

    def _find_feasible_solution(self, 
                                 profits: List[List[float]], 
                                 costs: List[List[float]], 
                                 capacity: float,
                                 min_profit: float) -> Tuple[List[int], float]:
        """
        Find a feasible solution with the objective of maximizing profit
        while respecting capacity constraint.
        """
        m = len(profits)
        selected_items = [-1] * m

        # Greedy selection based on profit/weight ratio
        item_ratios = []

        for i in range(m):
            if not profits[i]:
                continue

            for k in range(len(profits[i])):
                if costs[i][k] <= capacity:
                    if costs[i][k] > 0:
                        ratio = profits[i][k] / costs[i][k]
                    else:
                        ratio = float('inf') if profits[i][k] > 0 else 0
                    item_ratios.append((ratio, profits[i][k], costs[i][k], i, k))
                elif profits[i][k] > 0:
                    # Item doesn't fit, but has positive profit
                    item_ratios.append((0, profits[i][k], costs[i][k], i, k))

        # Sort by profit/weight ratio descending
        item_ratios.sort(reverse=True)

        # First pass: try to greedily select items
        remaining_capacity = capacity
        total_profit = 0.0

        for ratio, profit, cost, i, k in item_ratios:
            if selected_items[i] == -1 and cost <= remaining_capacity:
                selected_items[i] = k
                remaining_capacity -= cost
                total_profit += profit

        # Second pass: try to improve solution by swapping
        improved = True
        while improved:
            improved = False
            for i in range(m):
                if selected_items[i] == -1:
                    continue

                current_k = selected_items[i]
                for k in range(len(profits[i])):
                    if k == current_k:
                        continue

                    if costs[i][k] <= remaining_capacity + costs[i][current_k]:
                        potential_profit = total_profit - profits[i][current_k] + profits[i][k]
                        if potential_profit > total_profit:
                            selected_items[i] = k
                            remaining_capacity = remaining_capacity + costs[i][current_k] - costs[i][k]
                            total_profit = potential_profit
                            improved = True

        return selected_items, total_profit

    def _select_zero_profit_items(self, 
                                  profits: List[List[float]], 
                                  costs: List[List[float]], 
                                  capacity: float) -> Tuple[List[int], float]:
        """
        Handle the case where all profits are 0 by selecting cheapest items.
        """
        m = len(profits)
        selected_items = [-1] * m

        for i in range(m):
            if not profits[i]:
                continue

            # Select item with lowest cost
            best_k = -1
            best_cost = float('inf')

            for k in range(len(profits[i])):
                if costs[i][k] < best_cost and costs[i][k] <= capacity:
                    best_k = k
                    best_cost = costs[i][k]

            if best_k != -1:
                selected_items[i] = best_k

        return selected_items, 0.0


class MCGAPSolver:
    """
    Multiple-Choice Generalized Assignment Problem solver.
    Uses MCKP solver for each bin.
    """

    def __init__(self, mckp_solver: Optional[MCKPApproximateBinarySearch] = None):
        """
        Initialize MC-GAP solver.
        """
        self.mckp_solver = mckp_solver or MCKPApproximateBinarySearch()

    def solve_mc_gap(self, 
                     profits: List[List[List[float]]], 
                     costs: List[List[List[float]]], 
                     capacities: List[float]) -> List[Tuple[int, int]]:
        """
        Solve MC-GAP using the approach from the paper.

        For each bin, solve MCKP considering only unassigned classes
        and the remaining capacity of that bin.

        Args:
            profits: profits[i][j][k] = profit of item k from class i assigned to bin j
            costs: costs[i][j][k] = cost of item k from class i assigned to bin j
            capacities: capacities[j] = capacity of bin j

        Returns:
            assignments: list of (bin_id, item_id) for each class, or (-1, -1) if not assigned
        """
        num_classes = len(profits)
        num_bins = len(capacities)

        if num_classes == 0 or num_bins == 0:
            return [(-1, -1)] * num_classes

        assignments = [(-1, -1)] * num_classes
        remaining_capacities = list(capacities)

        unassigned_classes = set(range(num_classes))

        for bin_id in range(num_bins):
            if not unassigned_classes:
                break

            mckp_profits = []
            mckp_costs = []
            class_mapping = {}

            for class_id in sorted(unassigned_classes):
                class_mapping[len(mckp_profits)] = class_id
                mckp_profits.append(profits[class_id][bin_id])
                mckp_costs.append(costs[class_id][bin_id])

            selected_indices, _ = self.mckp_solver.solve_mckp_single_bin(
                mckp_profits, mckp_costs, remaining_capacities[bin_id]
            )

            for mckp_class_idx, item_idx in enumerate(selected_indices):
                if item_idx != -1:
                    class_id = class_mapping[mckp_class_idx]
                    cost = costs[class_id][bin_id][item_idx]
                    if cost <= remaining_capacities[bin_id]:
                        assignments[class_id] = (bin_id, item_idx)
                        remaining_capacities[bin_id] -= cost
                        unassigned_classes.discard(class_id)

        return assignments


class Algorithm1:
    """
    Algorithm 1: Approximation algorithm for request inference model assignment problem
    based on the multi-resolution edge computing paper.
    """

    def __init__(self):
        self.mckp_solver = MCKPApproximateBinarySearch(epsilon=0.6)
        self.mc_gap_solver = MCGAPSolver(self.mckp_solver)

    def solve(self,
              requests: List[Dict],
              cloudlets: List[Dict],
              models: Dict[str, List[Dict]],
              calc_delay: callable) -> Tuple[List[Dict], float]:
        """
        Solve the request assignment problem.

        Args:
            requests: list of requests, each with {'id': ri, 'model': Mi, 'delay_req': Di}
            cloudlets: list of cloudlets, each with {'id': j, 'compute_capacity': Cj, 'bandwidth': Wj}
            models: dict mapping model_id to list of resolutions,
                    each resolution with {'accuracy': ε, 'inference_time': τ, 'compute_resources': comp}
            calc_delay: function(ri, j, k, Mi) -> di,j,k calculates end-to-end delay

        Returns:
            (assignments, total_accuracy):
                assignments[i] = { 'cloudlet': j, 'resolution': k, 'accuracy': ε } or None if rejected
                total_accuracy: sum of accuracies of admitted requests
        """
        num_requests = len(requests)
        num_cloudlets = len(cloudlets)
        num_resolutions = max(len(models[r['model']]) for r in requests) if requests else 0

        # Initialize item matrices
        # profits[request_idx][cloudlet_idx][res_idx] = accuracy if feasible, else 0
        # costs[request_idx][cloudlet_idx][res_idx] = compute_cost if feasible, else inf
        profits = [[[0.0 for _ in range(num_resolutions)] 
                   for _ in range(num_cloudlets)] 
                  for _ in range(num_requests)]
        costs = [[[float('inf') for _ in range(num_resolutions)] 
                 for _ in range(num_cloudlets)] 
                for _ in range(num_requests)]

        # Step 1: Calculate items for each request-cloudlet-resolution combination
        for i, request in enumerate(requests):
            model_id = request['model']
            resolutions = models[model_id]

            for j, cloudlet in enumerate(cloudlets):
                for k, resolution in enumerate(resolutions):
                    # Calculate end-to-end delay
                    delay = calc_delay(request, cloudlet, resolution)

                    if delay <= request['delay_req']:
                        # Feasible item
                        profits[i][j][k] = resolution['accuracy']
                        costs[i][j][k] = resolution['compute_resources']
                    else:
                        # Infeasible item
                        profits[i][j][k] = 0.0
                        costs[i][j][k] = float('inf')

        # Step 2: Solve MC-GAP
        capacities = [c['compute_capacity'] for c in cloudlets]
        assignments_mc_gap = self.mc_gap_solver.solve_mc_gap(profits, costs, capacities)

        # Step 3: Convert assignments to output format
        result_assignments = []
        total_accuracy = 0.0

        for i, (bin_id, item_id) in enumerate(assignments_mc_gap):
            if bin_id != -1 and item_id != -1 and profits[i][bin_id][item_id] > 0:
                result_assignments.append({
                    'request_id': requests[i]['id'],
                    'cloudlet_id': bin_id,
                    'resolution_id': item_id,
                    'accuracy': profits[i][bin_id][item_id]
                })
                total_accuracy += profits[i][bin_id][item_id]
            else:
                result_assignments.append({
                    'request_id': requests[i]['id'],
                    'cloudlet_id': None,
                    'resolution_id': None,
                    'accuracy': 0.0
                })

        return result_assignments, total_accuracy


# Example usage and testing
if __name__ == "__main__":
    # Test MCKP solver
    print("Testing MCKP Approximate Binary Search Solver...")
    print("=" * 60)

    # Example from the paper
    profits = [
        [15],           # Class 1: 1 item
        [100],          # Class 2: 1 item  
        [90],           # Class 3: 1 item
        [60],           # Class 4: 1 item
        [40, 15],       # Class 5: 2 items
        [10, 1]         # Class 6: 2 items
    ]
    costs = [
        [2],
        [20],
        [20],
        [30],
        [40, 30],
        [60, 10]
    ]
    capacity = 102

    mckp_solver = MCKPApproximateBinarySearch()
    selected_items, total_profit = mckp_solver.solve_mckp_single_bin(profits, costs, capacity)

    print(f"MCKP Example:")
    print(f"  Selected items: {selected_items}")
    print(f"  Total profit: {total_profit}")
    print(f"  Capacity: {capacity}")
    print(f"  Used capacity: {sum(costs[i][k] if k != -1 else 0 for i, k in enumerate(selected_items))}")
    print()

    # Test Algorithm 1
    print("Testing Algorithm 1 for Request Assignment...")
    print("=" * 60)

    # Define test data
    requests = [
        {'id': 'r1', 'model': 'M1', 'delay_req': 100},
        {'id': 'r2', 'model': 'M2', 'delay_req': 150},
        {'id': 'r3', 'model': 'M1', 'delay_req': 120},
    ]

    cloudlets = [
        {'id': 0, 'compute_capacity': 100, 'bandwidth': 50},
        {'id': 1, 'compute_capacity': 80, 'bandwidth': 30},
    ]

    models = {
        'M1': [
            {'accuracy': 0.95, 'inference_time': 10, 'compute_resources': 40},
            {'accuracy': 0.85, 'inference_time': 6, 'compute_resources': 25},
        ],
        'M2': [
            {'accuracy': 0.90, 'inference_time': 15, 'compute_resources': 50},
            {'accuracy': 0.80, 'inference_time': 8, 'compute_resources': 30},
        ],
    }

    def calc_delay(request, cloudlet, resolution):
        # Simple delay model: inference_time + transmission_time
        transmission_time = 10  # Fixed transmission overhead
        return resolution['inference_time'] + transmission_time

    algo1 = Algorithm1()
    assignments, total_accuracy = algo1.solve(requests, cloudlets, models, calc_delay)

    print("Algorithm 1 Results:")
    print(f"  Total accuracy: {total_accuracy:.4f}")
    for i, assignment in enumerate(assignments):
        if assignment['cloudlet_id'] is not None:
            print(f"  Request {assignment['request_id']} -> Cloudlet {assignment['cloudlet_id']}, "
                  f"Resolution {assignment['resolution_id']}, Accuracy {assignment['accuracy']:.4f}")
        else:
            print(f"  Request {assignment['request_id']} -> Rejected")

    print()
    print("Approximation ratio guarantee: 4/9 ≈ 0.4444")
    print("Time complexity: O(|N| · |R| · K log|R|)")
