import random

def generate_text(model, max_len=15, temperature=1.0):
    """
    Generate text from any model (MLE, LinearInterpolation, or StupidBackoff).
    Supports models with either `.counts` or `.prob(context, word)` methods.
    """
    sentence = ["<s>"] * (getattr(model, "n", 1) - 1)
    output = []

    for _ in range(max_len):
        context = tuple(sentence[-(getattr(model, "n", 1) - 1):])

        # --- Case 1: Raw NGramModel (has .counts) ---
        if hasattr(model, "counts"):
            candidates = model.counts.get(context, {})
            if not candidates:
                break
            words, probs = zip(*candidates.items())

        # --- Case 2: Smoothed / interpolated / backoff models ---
        else:
            vocab = None
            # Try to extract vocab from sub-models if available
            if hasattr(model, "models"):
                vocab = model.models[-1].vocab
            elif hasattr(model, "model"):
                vocab = model.model.vocab
            elif hasattr(model, "vocab"):
                vocab = model.vocab
            else:
                raise ValueError("No accessible vocabulary in model.")

            # Compute probability distribution over vocab
            probs = [max(model.prob(context, w), 1e-12) for w in vocab]
            words = list(vocab)

        # --- Normalize and apply temperature ---
        probs = [p ** (1.0 / temperature) for p in probs]
        total = sum(probs)
        if total == 0:
            break
        probs = [p / total for p in probs]

        # --- Sample next word ---
        next_word = random.choices(words, weights=probs, k=1)[0]
        sentence.append(next_word)
        if next_word == "</s>":
            break

    # Clean up start tokens
    return " ".join(sentence[(getattr(model, "n", 1) - 1):])
