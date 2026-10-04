import numpy as np
import time
import random
from typing import List, Dict, Tuple
import csv


class ExperimentConfig:
    MODEL_ACCURACIES = {
        'R-FCN': [0.102, 0.315, 0.512, 0.817],
        'SSD': [0.103, 0.252, 0.491, 0.818],
        'YOLOv2': [0.111, 0.265, 0.512, 0.765],
    }
    INFERENCE_TIMES = [40.6, 49.3, 65.8, 94.0]
    COMPUTE_RESOURCES = [0.5, 0.5, 1.0, 1.0]
    RESOLUTIONS = ['240p', '360p', '480p', '720p']
    IMAGE_SIZES = [(426, 240), (640, 360), (854, 480), (1280, 720)]

    @staticmethod
    def generate_cloudlets(num_cloudlets, seed):
        rng = random.Random(seed)
        return [{'id': j, 'compute_capacity': rng.uniform(2, 4), 'bandwidth': rng.uniform(50, 200)} for j in range(num_cloudlets)]

    @staticmethod
    def generate_requests(num_requests, model_names, seed):
        rng = random.Random(seed)
        return [{'id': i, 'model': rng.choice(model_names), 'delay_req': rng.uniform(80, 180)} for i in range(num_requests)]

    @staticmethod
    def get_models_info():
        models = {}
        for name, accs in ExperimentConfig.MODEL_ACCURACIES.items():
            models[name] = [{'id': k, 'accuracy': accs[k], 'inference_time': ExperimentConfig.INFERENCE_TIMES[k], 'compute_resources': ExperimentConfig.COMPUTE_RESOURCES[k]} for k in range(4)]
        return models

    @staticmethod
    def precompute_delay_data(requests, cloudlets, models, seed):
        num_requests = len(requests)
        num_cloudlets = len(cloudlets)
        num_res = 4
        rng = random.Random(seed)
        delays = np.full((num_requests, num_cloudlets, num_res), np.inf)
        accs = np.zeros((num_requests, num_cloudlets, num_res))
        costs = np.zeros((num_requests, num_cloudlets, num_res))
        for i, req in enumerate(requests):
            for j, cl in enumerate(cloudlets):
                for k, res in enumerate(models[req['model']]):
                    td = rng.uniform(0.00002, 0.00005) * (ExperimentConfig.IMAGE_SIZES[k][0] * ExperimentConfig.IMAGE_SIZES[k][1])
                    total_delay = td + res['inference_time']
                    if total_delay <= req['delay_req']:
                        delays[i, j, k] = total_delay
                        accs[i, j, k] = res['accuracy']
                        costs[i, j, k] = res['compute_resources']
        return delays, accs, costs


class Algorithm2:
    """
    Online primal-dual algorithm from the multi-resolution paper.
    Requests arrive one by one, must be admitted or rejected immediately.
    Uses dual variables alpha_j (cloudlet capacity) and gamma_i (request admission).
    """

    def solve(self, accs, costs, cloudlets, models, capacity_check=False):
        """
        Args:
            capacity_check: if True, reject requests that would exceed cloudlet capacity
        """
        start_time = time.time()
        num_requests, num_cloudlets, num_res = accs.shape

        phi1 = 0.0
        phi2 = float('inf')
        Rmax = 0.0
        for j in range(num_cloudlets):
            cap = cloudlets[j]['compute_capacity']
            for i in range(num_requests):
                for k in range(num_res):
                    if accs[i, j, k] > 0 and costs[i, j, k] > 0:
                        ratio = accs[i, j, k] / costs[i, j, k]
                        phi1 = max(phi1, ratio)
                        phi2 = min(phi2, ratio)
                        Rmax = max(Rmax, costs[i, j, k] / cap)

        if phi2 == float('inf'):
            phi2 = phi1
        if phi2 == 0:
            phi2 = 1e-10

        a = (1 + Rmax) ** (1.0 / Rmax) if Rmax > 0 else 2.718

        alpha = {j: 0.0 for j in range(num_cloudlets)}
        total_accuracy = 0.0
        assignments = {}
        usage = {j: 0.0 for j in range(num_cloudlets)}

        for i in range(num_requests):
            if capacity_check:
                # Feasible version: always admit if any feasible (j,k) exists,
                # use primal-dual score to decide WHERE to place
                best_val_feas = -float('inf')
                best_j_feas, best_k_feas = -1, -1
                best_acc_feas = -1.0
                best_j_acc, best_k_acc = -1, -1

                for j in range(num_cloudlets):
                    remaining = cloudlets[j]['compute_capacity'] - usage[j]
                    for k in range(num_res):
                        if accs[i, j, k] > 0 and costs[i, j, k] <= remaining:
                            val = accs[i, j, k] - costs[i, j, k] * alpha[j]
                            if val > best_val_feas:
                                best_val_feas = val
                                best_j_feas, best_k_feas = j, k
                            if accs[i, j, k] > best_acc_feas:
                                best_acc_feas = accs[i, j, k]
                                best_j_acc, best_k_acc = j, k

                if best_j_feas >= 0:
                    # Hybrid: use primal-dual for placement, but always admit
                    # If primal-dual suggests rejection (val < 0), fall back to best accuracy
                    if best_val_feas > 0:
                        j_star, k_star = best_j_feas, best_k_feas
                    else:
                        # Fallback: place where accuracy is highest among feasible options
                        j_star, k_star = best_j_acc, best_k_acc

                    total_accuracy += accs[i, j_star, k_star]
                    assignments[i] = (j_star, k_star)
                    usage[j_star] += costs[i, j_star, k_star]
                    cap_star = cloudlets[j_star]['compute_capacity']
                    comp_star = costs[i, j_star, k_star]
                    alpha[j_star] = alpha[j_star] * (1 + comp_star / cap_star) + (phi1 / (a - 1)) * (comp_star / cap_star)
            else:
                # Original version: use gamma_i > 0 for admission
                best_val = -float('inf')
                best_j, best_k = -1, -1

                for j in range(num_cloudlets):
                    for k in range(num_res):
                        if accs[i, j, k] > 0:
                            val = accs[i, j, k] - costs[i, j, k] * alpha[j]
                            if val > best_val:
                                best_val = val
                                best_j, best_k = j, k

                gamma_i = best_val
                if gamma_i > 0 and best_j >= 0:
                    j_star, k_star = best_j, best_k
                    total_accuracy += accs[i, j_star, k_star]
                    assignments[i] = (j_star, k_star)
                    usage[j_star] += costs[i, j_star, k_star]
                    cap_star = cloudlets[j_star]['compute_capacity']
                    comp_star = costs[i, j_star, k_star]
                    alpha[j_star] = alpha[j_star] * (1 + comp_star / cap_star) + (phi1 / (a - 1)) * (comp_star / cap_star)

        solve_time = time.time() - start_time

        violations = {}
        for j in range(num_cloudlets):
            cap = cloudlets[j]['compute_capacity']
            violations[j] = (usage[j] / cap) if cap > 0 else 0.0

        return total_accuracy, solve_time, violations, assignments


class LPOnlineBaseline:
    """
    Online LP relaxation baseline: solve LP with all requests seen so far at each arrival.
    Too slow for large problems, so we use a simplified version:
    Solve LP once with all requests (offline LP upper bound), then apply random rounding.
    """

    def solve(self, accs, costs, cloudlets, seed=None):
        from pulp import LpProblem, LpVariable, LpMaximize, lpSum, LpStatusOptimal
        import pulp

        start_time = time.time()
        num_requests, num_cloudlets, num_res = accs.shape

        prob = LpProblem("LP_Online", LpMaximize)
        x = {}
        for i in range(num_requests):
            for j in range(num_cloudlets):
                for k in range(num_res):
                    if accs[i, j, k] > 0:
                        x[(i, j, k)] = LpVariable(f"x_{i}_{j}_{k}", lowBound=0, upBound=1)

        prob += lpSum(accs[i, j, k] * x[(i, j, k)] for (i, j, k) in x)

        for i in range(num_requests):
            prob += lpSum(x[(i, j, k)] for j in range(num_cloudlets) for k in range(num_res) if (i, j, k) in x) <= 1

        for j in range(num_cloudlets):
            prob += lpSum(costs[i, j, k] * x[(i, j, k)] for i in range(num_requests) for k in range(num_res) if (i, j, k) in x) <= cloudlets[j]['compute_capacity']

        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        solve_time = time.time() - start_time
        success = prob.status == LpStatusOptimal

        if not success:
            return 0.0, solve_time, {}, 0.0

        lp_objective = prob.objective.value()

        # Random rounding
        rng = random.Random(seed)
        total_accuracy = 0.0
        usage = {j: 0.0 for j in range(num_cloudlets)}
        order = list(range(num_requests))
        rng.shuffle(order)

        for i in order:
            options, probs = [], []
            for j in range(num_cloudlets):
                for k in range(num_res):
                    if (i, j, k) in x:
                        val = x[(i, j, k)].value()
                        if val > 1e-8:
                            options.append((j, k))
                            probs.append(val)
            if not options:
                continue
            total_prob = sum(probs)
            probs = [p / total_prob for p in probs]

            if rng.random() < total_prob:
                chosen_idx = 0
                cum = 0.0
                rv = rng.random()
                for idx, p in enumerate(probs):
                    cum += p
                    if rv <= cum:
                        chosen_idx = idx
                        break
                j, k = options[chosen_idx]
                total_accuracy += accs[i, j, k]
                usage[j] += costs[i, j, k]

        violations = {}
        for j in range(num_cloudlets):
            cap = cloudlets[j]['compute_capacity']
            violations[j] = (usage[j] / cap) if cap > 0 else 0.0

        return total_accuracy, solve_time, violations, lp_objective


class EdgeAdaptorBaseline:
    """
    EdgeAdaptor-inspired online baseline:
    Greedily assigns each arriving request to the best (j,k) that maximizes
    accuracy per unit resource, while considering resource reservation
    through a penalty on resource usage (similar to regularization).
    
    Key idea from EdgeAdaptor: use a convex penalty on resource usage
    to balance current accuracy vs future resource availability.
    """

    def solve(self, accs, costs, cloudlets, penalty_coef=1.0):
        start_time = time.time()
        num_requests, num_cloudlets, num_res = accs.shape
        usage = {j: 0.0 for j in range(num_cloudlets)}
        total_accuracy = 0.0
        assignments = {}

        for i in range(num_requests):
            best_score = -float('inf')
            best_j, best_k = -1, -1

            for j in range(num_cloudlets):
                cap = cloudlets[j]['compute_capacity']
                remaining = cap - usage[j]
                for k in range(num_res):
                    if accs[i, j, k] > 0 and costs[i, j, k] <= remaining:
                        # Score = accuracy - penalty * (resource_usage / remaining_capacity)
                        # This reserves resources for future requests
                        resource_ratio = costs[i, j, k] / cap
                        penalty = penalty_coef * resource_ratio * (usage[j] / cap)
                        score = accs[i, j, k] - penalty
                        if score > best_score:
                            best_score = score
                            best_j, best_k = j, k

            if best_j >= 0:
                total_accuracy += accs[i, best_j, best_k]
                usage[best_j] += costs[i, best_j, best_k]
                assignments[i] = (best_j, best_k)

        solve_time = time.time() - start_time
        violations = {}
        for j in range(num_cloudlets):
            cap = cloudlets[j]['compute_capacity']
            violations[j] = (usage[j] / cap) if cap > 0 else 0.0
        return total_accuracy, solve_time, violations


class RandomOnlineBaseline:
    """
    Random online baseline: randomly assign each request to a feasible (j,k) pair.
    """

    def solve(self, accs, costs, cloudlets, seed=None):
        start_time = time.time()
        rng = random.Random(seed)
        num_requests, num_cloudlets, num_res = accs.shape
        usage = {j: 0.0 for j in range(num_cloudlets)}
        total_accuracy = 0.0

        for i in range(num_requests):
            feasible = []
            for j in range(num_cloudlets):
                cap = cloudlets[j]['compute_capacity']
                remaining = cap - usage[j]
                for k in range(num_res):
                    if accs[i, j, k] > 0 and costs[i, j, k] <= remaining:
                        feasible.append((j, k, accs[i, j, k], costs[i, j, k]))

            if feasible:
                j, k, acc, cost = rng.choice(feasible)
                total_accuracy += acc
                usage[j] += cost

        solve_time = time.time() - start_time
        violations = {}
        for j in range(num_cloudlets):
            cap = cloudlets[j]['compute_capacity']
            violations[j] = (usage[j] / cap) if cap > 0 else 0.0
        return total_accuracy, solve_time, violations


def run_experiment(num_cloudlets, num_requests_list, time_horizon_list, num_instances=5):
    models = ExperimentConfig.get_models_info()
    model_names = list(ExperimentConfig.MODEL_ACCURACIES.keys())
    rows = []

    for n_req, T in zip(num_requests_list, time_horizon_list):
        print(f"\n  |N|={num_cloudlets}, |R|={n_req}, T={T}", end="", flush=True)

        for inst in range(num_instances):
            seed = inst * 10000 + num_cloudlets * 100 + n_req
            cloudlets = ExperimentConfig.generate_cloudlets(num_cloudlets, seed)
            requests = ExperimentConfig.generate_requests(n_req, model_names, seed)
            _, accs, costs = ExperimentConfig.precompute_delay_data(requests, cloudlets, models, seed + 999)

            # Algorithm 2 (Online Primal-Dual, original)
            alg2_acc, alg2_time, alg2_viol, _ = Algorithm2().solve(accs, costs, cloudlets, models)

            # Algorithm 2-Feasible (no violation, reject when cloudlet full)
            alg2f_acc, alg2f_time, alg2f_viol, _ = Algorithm2().solve(accs, costs, cloudlets, models, capacity_check=True)

            # LP Online Baseline
            lp_acc, lp_time, lp_viol, lp_upper = LPOnlineBaseline().solve(accs, costs, cloudlets, seed=seed+5000)

            # EdgeAdaptor Baseline
            ea_acc, ea_time, ea_viol = EdgeAdaptorBaseline().solve(accs, costs, cloudlets, penalty_coef=0.3)

            # Random Baseline
            rnd_acc, rnd_time, rnd_viol = RandomOnlineBaseline().solve(accs, costs, cloudlets, seed=seed+6000)

            for algo, acc, t, viol in [
                ('Alg2', alg2_acc, alg2_time, alg2_viol),
                ('Alg2-Feasible', alg2f_acc, alg2f_time, alg2f_viol),
                ('LP', lp_acc, lp_time, lp_viol),
                ('EdgeAdaptor', ea_acc, ea_time, ea_viol),
                ('Random', rnd_acc, rnd_time, rnd_viol),
            ]:
                ratios = [viol[j] for j in range(num_cloudlets)]
                rows.append({
                    'Num_Cloudlets': num_cloudlets,
                    'Num_Requests': n_req,
                    'Time_Horizon': T,
                    'Algorithm': algo,
                    'Cumulative_Accuracy': acc,
                    'Solve_Time': t,
                    'Cloudlet_Usage_Ratios': ratios,
                    'LP_Upper_Bound': lp_upper if algo == 'LP' else None,
                })

            print(".", end="", flush=True)

        print(" done")

    return rows


def print_summary(rows):
    print(f"\n{'='*90}")
    print("  Cumulative Accuracy (mean +/- std)")
    print(f"{'='*90}")
    for n_req in sorted(set(r['Num_Requests'] for r in rows)):
        row = f"|R|={n_req:<6}"
        for algo in ['Alg2', 'Alg2-Feasible', 'LP', 'EdgeAdaptor', 'Random']:
            vals = [r['Cumulative_Accuracy'] for r in rows if r['Num_Requests'] == n_req and r['Algorithm'] == algo and r['Cumulative_Accuracy'] is not None]
            if vals:
                row += f"{algo}: {np.mean(vals):.2f}+/-{np.std(vals):.2f}   "
        lp_ub = [r['LP_Upper_Bound'] for r in rows if r['Num_Requests'] == n_req and r['Algorithm'] == 'LP' and r['LP_Upper_Bound'] is not None]
        if lp_ub:
            row += f"LP_UB: {np.mean(lp_ub):.2f}"
        print(row)

    print(f"\n{'='*90}")
    print("  Solve Time in seconds (mean +/- std)")
    print(f"{'='*90}")
    for n_req in sorted(set(r['Num_Requests'] for r in rows)):
        row = f"|R|={n_req:<6}"
        for algo in ['Alg2', 'Alg2-Feasible', 'LP', 'EdgeAdaptor', 'Random']:
            vals = [r['Solve_Time'] for r in rows if r['Num_Requests'] == n_req and r['Algorithm'] == algo]
            if vals:
                row += f"{algo}: {np.mean(vals):.4f}+/-{np.std(vals):.4f}   "
        print(row)

    print(f"\n{'='*90}")
    print("  Resource Violation (Usage_Ratio > 1.0 = violation)")
    print(f"{'='*90}")
    for n_req in sorted(set(r['Num_Requests'] for r in rows)):
        row = f"|R|={n_req:<6}"
        for algo in ['Alg2', 'Alg2-Feasible', 'LP', 'EdgeAdaptor', 'Random']:
            ratios_all = []
            viol_count = 0
            for r in rows:
                if r['Num_Requests'] == n_req and r['Algorithm'] == algo and r['Cloudlet_Usage_Ratios'] is not None:
                    ratios_all.extend(r['Cloudlet_Usage_Ratios'])
                    viol_count += sum(1 for v in r['Cloudlet_Usage_Ratios'] if v > 1.0)
            if ratios_all:
                row += f"{algo}: mean={np.mean(ratios_all):.3f}, max={np.max(ratios_all):.3f}, viol={viol_count}   "
        print(row)


def save_to_csv(rows, filename='experiment_results_alg2.csv'):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Num_Cloudlets', 'Num_Requests', 'Time_Horizon', 'Algorithm', 'Cumulative_Accuracy', 'Solve_Time', 'Usage_Ratios', 'LP_Upper_Bound'])
        for r in rows:
            ratios_str = ' '.join(f"{v:.6f}" for v in r['Cloudlet_Usage_Ratios']) if r['Cloudlet_Usage_Ratios'] is not None else ''
            lp_ub = f"{r['LP_Upper_Bound']:.6f}" if r.get('LP_Upper_Bound') is not None else ''
            writer.writerow([r['Num_Cloudlets'], r['Num_Requests'], r['Time_Horizon'], r['Algorithm'], r['Cumulative_Accuracy'], f"{r['Solve_Time']:.6f}", ratios_str, lp_ub])
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    print("Algorithm 2 Performance Evaluation (vs LP, EdgeAdaptor, Random)")
    print("=" * 90)
    print("  |N|=100  |  |R|: 400-1000, step 100")
    print("  Online setting: requests arrive one by one")
    print()

    results = run_experiment(
        num_cloudlets=100,
        num_requests_list=list(range(400, 1001, 100)),
        time_horizon_list=list(range(400, 1001, 100)),
        num_instances=5,
    )

    print_summary(results)
    save_to_csv(results)
