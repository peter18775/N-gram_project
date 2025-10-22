import math

def evaluate_model(model, data):
    log_prob, token_count = 0, 0
    for sentence in data:
        padded = ["<s>"] * (model.n - 1) + sentence
        for i in range(model.n - 1, len(padded)):
            context = tuple(padded[i-model.n+1:i])
            word = padded[i]
            p = model.prob(context, word)
            if p <= 0:
                return float("inf")
            log_prob += math.log2(p)
            token_count += 1
    return math.pow(2, -log_prob / token_count)