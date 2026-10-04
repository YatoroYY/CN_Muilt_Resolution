import numpy as np
import time
import random
from typing import List, Dict, Tuple
from algorithm1_mckp import Algorithm1
from pulp import LpProblem, LpVariable, LpMaximize, LpBinary, lpSum, LpStatusOptimal
import pulp
import signal
import sys


class ExperimentConfiguration:
    MODEL_ACCURACIES = {
        'R-FCN': [0.102, 0.315, 0.512, 0.817],
        'SSD': [0.103, 0.252, 0.491, 0.818],
        'YOLOv2': [0.111, 0.265, 0.512, 0.765],
    }
    INFERENCE_TIMES = [40.6, 49.3, 65.8, 94.0]
    COMPUTE_RESOURCES = [0.5, 0.5, 1.0, 1.0]
    RESOLUTIONS = ['240p', '360p', '480p', '720p']
    
    # Image sizes for different resolutions (width x height)
    IMAGE_SIZES = [(426, 240), (640, 360), (854, 480), (1280, 720)]

    @staticmethod
    def generate_cloudlets(num_cloudlets: int, seed: int) -> List[Dict]:
        rng = random.Random(seed)
        return [
            {
                'id': j,
                'compute_capacity': rng.uniform(2, 4),
                'bandwidth': rng.uniform(50, 200),
            }
            for j in range(num_cloudlets)
        ]

    @staticmethod
    def generate_requests(num_requests: int, model_names: List[str], seed: int) -> List[Dict]:
        rng = random.Random(seed)
        return [
            {
                'id': f'r{i}',
                'model': rng.choice(model_names),
                'delay_req': rng.uniform(80, 180),
            }
            for i in range(num_requests)
        ]

    @staticmethod
    def get_models_info() -> Dict[str, List[Dict]]:
        models = {}
        for model_name, accuracies in ExperimentConfiguration.MODEL_ACCURACIES.items():
            models[model_name] = [
                {
                    'id': k,
                    'label': ExperimentConfiguration.RESOLUTIONS[k],
                    'accuracy': accuracies[k],
                    'inference_time': ExperimentConfiguration.INFERENCE_TIMES[k],
                    'compute_resources': ExperimentConfiguration.COMPUTE_RESOURCES[k],
                }
                for k in range(4)
            ]
        return models

    @staticmethod
    def precompute_delay_data(requests, cloudlets, models, seed):
        """Precompute all delay/accuracy/cost data deterministically."""
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
                    # Calculate transmission delay based on image size
                    # Use per-pixel delay scaled to be in reasonable range (~10-80ms depending on resolution)
                    transmission_delay = rng.uniform(0.00002, 0.00005) * (ExperimentConfiguration.IMAGE_SIZES[k][0] * ExperimentConfiguration.IMAGE_SIZES[k][1])
                    
                    # Total delay = transmission + inference
                    total_delay = transmission_delay + res['inference_time']
                    
                    if total_delay <= req['delay_req']:
                        delays[i, j, k] = total_delay
                        accs[i, j, k] = res['accuracy']
                        costs[i, j, k] = res['compute_resources']
        return delays, accs, costs


class ILPSolver:
    @staticmethod
    def solve(accs, costs, requests, cloudlets, time_limit=30):
        start_time = time.time()
        num_requests, num_cloudlets, num_res = accs.shape

        prob = LpProblem("ILP", LpMaximize)
        x = {}
        for i in range(num_requests):
            for j in range(num_cloudlets):
                for k in range(num_res):
                    if accs[i, j, k] > 0:
                        x[(i, j, k)] = LpVariable(f"x_{i}_{j}_{k}", cat=LpBinary)

        prob += lpSum(accs[i, j, k] * x[(i, j, k)] for (i, j, k) in x)

        for i in range(num_requests):
            prob += lpSum(
                x[(i, j, k)]
                for j in range(num_cloudlets)
                for k in range(num_res)
                if (i, j, k) in x
            ) <= 1

        for j in range(num_cloudlets):
            prob += lpSum(
                costs[i, j, k] * x[(i, j, k)]
                for i in range(num_requests)
                for k in range(num_res)
                if (i, j, k) in x
            ) <= cloudlets[j]['compute_capacity']

        prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0))
        solve_time = time.time() - start_time
        success = prob.status == LpStatusOptimal
        total_accuracy = prob.objective.value() if success else 0.0
        return total_accuracy, solve_time, success


class LPSolver:
    @staticmethod
    def solve(accs, costs, requests, cloudlets, time_limit=30, seed=None):
        start_time = time.time()
        num_requests, num_cloudlets, num_res = accs.shape

        prob = LpProblem("LP", LpMaximize)
        x = {}
        for i in range(num_requests):
            for j in range(num_cloudlets):
                for k in range(num_res):
                    if accs[i, j, k] > 0:
                        x[(i, j, k)] = LpVariable(f"x_{i}_{j}_{k}", lowBound=0, upBound=1)

        prob += lpSum(accs[i, j, k] * x[(i, j, k)] for (i, j, k) in x)

        for i in range(num_requests):
            prob += lpSum(
                x[(i, j, k)]
                for j in range(num_cloudlets)
                for k in range(num_res)
                if (i, j, k) in x
            ) <= 1

        for j in range(num_cloudlets):
            prob += lpSum(
                costs[i, j, k] * x[(i, j, k)]
                for i in range(num_requests)
                for k in range(num_res)
                if (i, j, k) in x
            ) <= cloudlets[j]['compute_capacity']

        prob.solve(pulp.PULP_CBC_CMD(timeLimit=time_limit, msg=0))
        solve_time = time.time() - start_time
        success = prob.status == LpStatusOptimal
        
        if not success:
            return 0.0, solve_time, False, 0.0, {}
        
        lp_objective = prob.objective.value()
        
        rng = random.Random(seed)
        total_accuracy = 0.0
        usage = {j: 0.0 for j in range(num_cloudlets)}
        
        order = list(range(num_requests))
        rng.shuffle(order)
        
        for i in order:
            options = []
            probs = []
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
            
            r = rng.random()
            if r < total_prob:
                chosen_idx = 0
                cum = 0.0
                rand_val = rng.random()
                for idx, p in enumerate(probs):
                    cum += p
                    if rand_val <= cum:
                        chosen_idx = idx
                        break
                
                j, k = options[chosen_idx]
                total_accuracy += accs[i, j, k]
                usage[j] += costs[i, j, k]
        
        violations = {}
        for j in range(num_cloudlets):
            cap = cloudlets[j]['compute_capacity']
            violations[j] = (usage[j] / cap) if cap > 0 else 0.0
        
        return total_accuracy, solve_time, True, violations, lp_objective


class HEUSolver:
    @staticmethod
    def solve(accs, costs, requests, cloudlets, models):
        start_time = time.time()
        num_requests, num_cloudlets, num_res = accs.shape
        remaining = {j: cloudlets[j]['compute_capacity'] for j in range(num_cloudlets)}
        total_accuracy = 0.0
        assignments = {}

        for i in range(num_requests):
            best_acc = -1.0
            best_j, best_k = -1, -1
            for j in range(num_cloudlets):
                for k in range(num_res):
                    if accs[i, j, k] > 0 and costs[i, j, k] <= remaining[j]:
                        if accs[i, j, k] > best_acc:
                            best_acc = accs[i, j, k]
                            best_j, best_k = j, k
            if best_j >= 0:
                total_accuracy += best_acc
                remaining[best_j] -= costs[i, best_j, best_k]
                assignments[i] = (best_j, best_k)

        solve_time = time.time() - start_time
        
        # Calculate violations
        violations = {}
        for j in range(num_cloudlets):
            used = cloudlets[j]['compute_capacity'] - remaining[j]
            cap = cloudlets[j]['compute_capacity']
            violations[j] = (used / cap) if cap > 0 else 0.0
        
        return total_accuracy, solve_time, violations


class Alg1Solver:
    @staticmethod
    def solve(accs, costs, requests, cloudlets, models):
        start_time = time.time()
        algo1 = Algorithm1()

        class DelayCalc:
            def __init__(self, accs, costs):
                self.accs = accs
                self.costs = costs

            def __call__(self, request, cloudlet, resolution):
                i = int(request['id'][1:])
                j = cloudlet['id']
                k = resolution['id']
                if self.accs[i, j, k] > 0:
                    return 0
                return float('inf')

        calc = DelayCalc(accs, costs)
        assignments, total_accuracy = algo1.solve(requests, cloudlets, models, calc)
        solve_time = time.time() - start_time
        
        # Calculate resource usage
        usage = {j: 0.0 for j in range(len(cloudlets))}
        for a in assignments:
            if a['cloudlet_id'] is not None:
                i = int(a['request_id'][1:])
                j = a['cloudlet_id']
                k = a['resolution_id']
                usage[j] += costs[i, j, k]
        
        violations = {}
        for j in range(len(cloudlets)):
            cap = cloudlets[j]['compute_capacity']
            violations[j] = (usage[j] / cap) if cap > 0 else 0.0
        
        return total_accuracy, solve_time, violations


def run_experiment(num_cloudlets, num_requests_list, num_instances=5, ilp_time_limit=30):
    models = ExperimentConfiguration.get_models_info()
    model_names = list(ExperimentConfiguration.MODEL_ACCURACIES.keys())

    rows = []

    for n_req in num_requests_list:
        print(f"\n  |N|={num_cloudlets}, |R|={n_req}", end="", flush=True)

        for inst in range(num_instances):
            seed = inst * 10000 + num_cloudlets * 100 + n_req
            cloudlets = ExperimentConfiguration.generate_cloudlets(num_cloudlets, seed)
            requests = ExperimentConfiguration.generate_requests(n_req, model_names, seed)
            _, accs, costs = ExperimentConfiguration.precompute_delay_data(
                requests, cloudlets, models, seed + 999
            )

            a1_acc, a1_time, a1_viol = Alg1Solver.solve(accs, costs, requests, cloudlets, models)
            h_acc, h_time, h_viol = HEUSolver.solve(accs, costs, requests, cloudlets, models)
            lp_acc, lp_time, lp_ok, lp_viol, lp_upper = LPSolver.solve(
                accs, costs, requests, cloudlets, time_limit=ilp_time_limit, seed=seed+5000
            )

            for algo, acc, t, viol in [
                ('Alg1', a1_acc, a1_time, a1_viol),
                ('HEU', h_acc, h_time, h_viol),
                ('LP', lp_acc if lp_ok else None, lp_time, lp_viol if lp_ok else None),
            ]:
                ratios = [viol[j] if viol is not None else None for j in range(num_cloudlets)]
                rows.append({
                    'Num_Cloudlets': num_cloudlets,
                    'Num_Requests': n_req,
                    'Algorithm': algo,
                    'Cumulative_Accuracy': acc,
                    'Solve_Time': t,
                    'Cloudlet_Usage_Ratios': ratios,
                })

            print(".", end="", flush=True)

        print(" done")

    return rows


def print_summary(rows):
    print(f"\n{'='*80}")
    print("  Cumulative Accuracy (mean +/- std)")
    print(f"{'='*80}")
    for n_req in sorted(set(r['Num_Requests'] for r in rows)):
        row = f"|R|={n_req:<6}"
        for algo in ['Alg1', 'LP', 'HEU']:
            vals = [r['Cumulative_Accuracy'] for r in rows if r['Num_Requests'] == n_req and r['Algorithm'] == algo and r['Cumulative_Accuracy'] is not None]
            if vals:
                row += f"{algo}: {np.mean(vals):.2f}+/-{np.std(vals):.2f}   "
        print(row)

    print(f"\n{'='*80}")
    print("  Solve Time in seconds (mean +/- std)")
    print(f"{'='*80}")
    for n_req in sorted(set(r['Num_Requests'] for r in rows)):
        row = f"|R|={n_req:<6}"
        for algo in ['Alg1', 'LP', 'HEU']:
            vals = [r['Solve_Time'] for r in rows if r['Num_Requests'] == n_req and r['Algorithm'] == algo and r['Solve_Time'] is not None]
            if vals:
                row += f"{algo}: {np.mean(vals):.4f}+/-{np.std(vals):.4f}   "
        print(row)

    print(f"\n{'='*80}")
    print("  Resource Violation (Usage_Ratio > 1.0 = violation)")
    print(f"{'='*80}")
    for n_req in sorted(set(r['Num_Requests'] for r in rows)):
        row = f"|R|={n_req:<6}"
        for algo in ['Alg1', 'LP', 'HEU']:
            ratios_all = []
            viol_count = 0
            for r in rows:
                if r['Num_Requests'] == n_req and r['Algorithm'] == algo and r['Cloudlet_Usage_Ratios'] is not None:
                    ratios_all.extend(r['Cloudlet_Usage_Ratios'])
                    viol_count += sum(1 for v in r['Cloudlet_Usage_Ratios'] if v > 1.0)
            if ratios_all:
                row += f"{algo}: mean={np.mean(ratios_all):.3f}, max={np.max(ratios_all):.3f}, viol={viol_count}   "
        print(row)


def save_to_csv(rows, filename='experiment_results.csv'):
    import csv
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Num_Cloudlets', 'Num_Requests', 'Algorithm', 'Cumulative_Accuracy', 'Solve_Time', 'Usage_Ratios'])
        for r in rows:
            if r['Cloudlet_Usage_Ratios'] is not None:
                ratios_str = ' '.join(f"{v:.6f}" for v in r['Cloudlet_Usage_Ratios'])
            else:
                ratios_str = ''
            writer.writerow([r['Num_Cloudlets'], r['Num_Requests'], r['Algorithm'], r['Cumulative_Accuracy'], f"{r['Solve_Time']:.6f}", ratios_str])
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    print("Algorithm 1 Performance Evaluation (vs LP, HEU)")
    print("=" * 80)
    print("  |N|=100  |  |R|: 400-1000, step 100")
    print("  Cloudlet compute: 2-4 cores  |  Delay req: 80-180ms")
    print()

    results = run_experiment(
        num_cloudlets=100,
        num_requests_list=list(range(400, 1001, 100)),
        num_instances=5,
        ilp_time_limit=60,
    )

    print_summary(results)
    save_to_csv(results, 'experiment_results.csv')
