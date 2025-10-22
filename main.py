import os
import csv
import pickle
import random
import sys, io
from src.preprocess import load_data, build_vocab
from src.ngram_model import NGramModel
from src.smoothing import AddOneSmoothing, LinearInterpolation, StupidBackoff
from src.evaluate import evaluate_model
from src.fine_tuning import tune_lambdas_4gram, tune_alpha_4gram
from src.generate import generate_text

# ----------------------------------------------------
# UTF-8 Safe Console Output for Windows
# ----------------------------------------------------
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ----------------------------------------------------
# Utility Helpers
# ----------------------------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_model(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"[SAVED] {os.path.basename(path)}")

def load_model(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    print(f"[LOADED] {os.path.basename(path)}")
    return obj

def log_result(model_name, setting, perplexity):
    """Append a row of results to results/summary.csv"""
    ensure_dir("results")
    log_path = os.path.join("results", "summary.csv")
    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["Model", "Setting", "Test_Perplexity"])
        writer.writerow([model_name, setting, round(perplexity, 4)])
    print(f"[LOGGED] {model_name} ({setting}) → {perplexity:.2f}")



# ----------------------------------------------------
# Data Loading & Training
# ----------------------------------------------------
def load_datasets():
    print("[INFO] Loading datasets...")
    train = load_data("data/ptb.train.txt")
    dev   = load_data("data/ptb.valid.txt")
    test  = load_data("data/ptb.test.txt")
    train, vocab = build_vocab(train)
    dev, _ = build_vocab(dev)
    test, _ = build_vocab(test)
    print("[INFO] Data successfully loaded.")
    return train, dev, test, vocab

def train_ngram_models(train):
    print("[INFO] Training N-gram models...")
    uni = NGramModel(1); uni.train(train)
    bi  = NGramModel(2); bi.train(train)
    tri = NGramModel(3); tri.train(train)
    tetra = NGramModel(4); tetra.train(train)
    print("[INFO] Training complete.")
    return uni, bi, tri, tetra

def save_base_models(models):
    names = ["uni", "bi", "tri", "tetra"]
    for m, name in zip(models, names):
        save_model(m, f"models/{name}.pkl")

# ----------------------------------------------------
# Evaluation
# ----------------------------------------------------
def evaluate_unsmoothed(models, test):
    print("\n[INFO] Evaluating unsmoothed models...")
    for m in models:
        pp = m.perplexity(test)
        print(f"{m.n}-gram MLE perplexity: {pp:.2f}")
        log_result(f"{m.n}-gram", "MLE", pp)

def evaluate_add1(models, test):
    print("\n[INFO] Evaluating Add-1 (Laplace) smoothing...")
    for m in models:
        add1 = AddOneSmoothing(m)
        pp = evaluate_model(add1, test)
        print(f"{m.n}-gram Add-1 perplexity: {pp:.2f}")
        log_result(f"{m.n}-gram", "Add-1", pp)

# ----------------------------------------------------
# Interpolation + Backoff
# ----------------------------------------------------
def build_interpolation_model(models, dev, test):
    ensure_dir("models")
    path = "models/interp_best.pkl"
    if os.path.exists(path):
        interp_best = load_model(path)
    else:
        print("[INFO] Tuning λ₁–λ₄ using validation set...")
        best_lambdas = tune_lambdas_4gram(models, dev, num_samples=500, refine_rounds=2, max_workers=8)
        interp_best = LinearInterpolation(models, best_lambdas)
        save_model(interp_best, path)
        print(f"[INFO] Best λs: {best_lambdas}")
    pp_interp = evaluate_model(interp_best, test)
    print(f"Final Test Perplexity (Interpolation): {pp_interp:.2f}")
    log_result("Interpolation", f"Tuned λ₁–λ₄ {tuple(round(l,3) for l in getattr(interp_best,'lambdas',[0,0,0,0]))}", pp_interp)
    return interp_best

def build_backoff_model(models, dev, test):
    ensure_dir("models")
    path = "models/backoff_best.pkl"
    if os.path.exists(path):
        backoff_best = load_model(path)
    else:
        print("[INFO] Tuning α for Stupid Backoff...")
        best_alpha, _ = tune_alpha_4gram(models, dev, alpha_values=[0.2,0.3,0.4,0.5,0.6], max_workers=6)
        backoff_best = StupidBackoff(models, alpha=best_alpha)
        save_model(backoff_best, path)
        print(f"[INFO] Best α: {best_alpha}")
    pp_backoff = backoff_best.perplexity(test)
    print(f"Final Test Perplexity (Stupid Backoff): {pp_backoff:.2f}")
    log_result("Stupid Backoff", f"α={getattr(backoff_best,'alpha','?')}", pp_backoff)
    return backoff_best

# ----------------------------------------------------
# Text Generation & Save
# ----------------------------------------------------
def generate_and_save(model, name, num=15, max_len=20):
    ensure_dir("results")
    out_path = f"results/generated_{name.lower().replace(' ', '_')}.txt"
    print(f"\n[INFO] Generating text using {name} model...")
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(num):
            text = generate_text(model, max_len=max_len)
            print(f"{i+1}. {text}")
            f.write(f"{i+1}. {text}\n")
    print(f"[SAVED] Generated text written to {out_path}")

# ----------------------------------------------------
# Main Pipeline
# ----------------------------------------------------
def main():
    train, dev, test, vocab = load_datasets()

    # Load or Train Base Models
    paths = [f"models/{n}.pkl" for n in ["uni","bi","tri","tetra"]]
    if all(os.path.exists(p) for p in paths):
        print("[INFO] Base models found — loading instead of retraining.")
        uni, bi, tri, tetra = [load_model(p) for p in paths]
    else:
        uni, bi, tri, tetra = train_ngram_models(train)
        save_base_models([uni, bi, tri, tetra])

    models = [uni, bi, tri, tetra]

    evaluate_unsmoothed(models, test)
    evaluate_add1(models, test)

    interp_best = build_interpolation_model(models, dev, test)
    backoff_best = build_backoff_model([tetra, tri, bi, uni], dev, test)

    generate_and_save(interp_best, "linear_Interpolation")
    generate_and_save(backoff_best, "Stupid_Backoff")

    print("\n✅ All evaluations and generations complete. Results logged in results/summary.csv")

# ----------------------------------------------------
if __name__ == "__main__":
    main()
