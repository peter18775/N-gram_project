import os
import csv
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.smoothing import LinearInterpolation, StupidBackoff
from src.evaluate import evaluate_model

def sample_lambdas(num_samples=1000):
    """Generate random λ1–λ4 that sum to 1."""
    samples = []
    for _ in range(num_samples):
        vals = [random.random() for _ in range(4)]
        s = sum(vals)
        samples.append([v / s for v in vals])
    return samples

def refine_lambdas(best_lambdas, delta=0.05, num_samples=200):
    """Small random perturbations around current best."""
    refined = []
    for _ in range(num_samples):
        perturbed = [max(0, min(1, lam + random.uniform(-delta, delta))) for lam in best_lambdas]
        s = sum(perturbed)
        refined.append([v / s for v in perturbed])
    return refined

def ensure_results_dir():
    os.makedirs("results", exist_ok=True)
    log_path = os.path.join("results", "lambda_tuning_log.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Round", "Lambda1", "Lambda2", "Lambda3", "Lambda4", "Perplexity"])
    return log_path

def tune_lambdas_4gram(models, dev, num_samples=800, refine_rounds=2, max_workers=8):
    """
    Fast randomized + refinement fine-tuning for λ1–λ4 with CSV logging.
    """
    log_path = ensure_results_dir()
    best_pp = float("inf")
    best_lambdas = None

    def eval_lambdas(lambdas, round_id):
        interp = LinearInterpolation(models, lambdas)
        pp = evaluate_model(interp, dev)
        # Log each evaluation
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([round_id, *[round(l, 4) for l in lambdas], round(pp, 4)])
        return lambdas, pp

    def parallel_eval(lambda_sets, round_id):
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(eval_lambdas, lam, round_id) for lam in lambda_sets]
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"[ERROR] Thread failed: {e}")
        return results

    # === Randomized search ===
    print(f"[INFO] Randomized search over {num_samples} λ sets using {max_workers} threads...")
    random_sets = sample_lambdas(num_samples)
    results = parallel_eval(random_sets, round_id=1)
    for lam, pp in results:
        if pp < best_pp:
            best_pp, best_lambdas = pp, lam
    print(f"[ROUND 1] Best λ={best_lambdas} → PP={best_pp:.2f}")

    # === Coordinate-descent refinement ===
    for round_idx in range(refine_rounds):
        refined_sets = refine_lambdas(best_lambdas, delta=0.05)
        results = parallel_eval(refined_sets, round_id=round_idx + 2)
        for lam, pp in results:
            if pp < best_pp:
                best_pp, best_lambdas = pp, lam
        print(f"[ROUND {round_idx+2}] Refined best λ={best_lambdas} → PP={best_pp:.2f}")

    print(f"[FINAL RESULT] Tuned λs={best_lambdas}, Dev PP={best_pp:.2f}")
    print(f"[LOGGED] All evaluations saved to {log_path}")
    return best_lambdas, best_pp
def ensure_results_dir():
    os.makedirs("results", exist_ok=True)
    return os.path.join("results", "alpha_tuning_log.csv")

def tune_alpha_4gram(models, dev, alpha_values=None, max_workers=6):
    """
    Parallel α tuning for 4-gram Stupid Backoff with CSV logging.
    models: [four, tri, bi, uni]
    dev: development set
    alpha_values: list of α values to try
    """
    if alpha_values is None:
        alpha_values = [round(a, 2) for a in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]]

    log_path = ensure_results_dir()
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Alpha", "Dev_Perplexity"])

    best_alpha, best_pp = None, float("inf")

    def eval_alpha(alpha):
        backoff = StupidBackoff(models, alpha=alpha)
        pp = backoff.perplexity(dev)
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([alpha, round(pp, 4)])
        return alpha, pp

    print(f"[INFO] Evaluating {len(alpha_values)} α values using {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(eval_alpha, a) for a in alpha_values]
        for future in as_completed(futures):
            try:
                alpha, pp = future.result()
                print(f"[α={alpha:.2f}] Dev PP={pp:.2f}")
                if pp < best_pp:
                    best_alpha, best_pp = alpha, pp
            except Exception as e:
                print(f"[ERROR] α evaluation failed: {e}")

    print(f"[RESULT] Best α={best_alpha:.2f}, Dev Perplexity={best_pp:.2f}")
    print(f"[LOGGED] All α results saved to {log_path}")
    return best_alpha, best_pp