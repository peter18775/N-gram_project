import math
from collections import Counter, defaultdict
# from src.ngram_model import NGramModel
class AddOneSmoothing:
    def __init__(self, model):
        self.model = model
        self.n = model.n
    def prob(self, context, word):
        V = len(self.model.vocab)
        count = self.model.counts[context][word]
        total = self.model.context_counts[context]
        return (count + 1) / (total + V)

class LinearInterpolation:
    def __init__(self, models, lambdas:list, n:int=3):
        # assert abs(sum(lambdas) - 1.0) == 0
        self.models = models
        self.lambdas = lambdas
        self.n = n
        
    def prob(self, context, word):
        prob = 0
        for model, lam in zip(self.models, self.lambdas):
            n = model.n
            sub_context = tuple(context[-(n-1):]) if n > 1 else ()
            prob += lam * model.prob(sub_context, word)
        return prob

class StupidBackoff:
    def __init__(self, models, alpha=0.4):
        self.models = models  # ordered: highest n to lowest
        self.alpha = alpha
        self.n = models[0].n  # Use highest order n-gram for context length

    def prob(self, context, word):
        score = 1.0
        for i, model in enumerate(self.models):
            n = model.n
            sub_context = tuple(context[-(n-1):]) if n > 1 else ()
            
            count = model.counts[sub_context][word]
            total = model.context_counts[sub_context]
            
            if total > 0:
                if count > 0:
                    return score * (count / total)
                score *= self.alpha
            else:
                score *= self.alpha
                continue
        
        # If we've backed off through all models and still have no counts,
        # return a very small probability
        return score * (1.0 / len(self.models[0].vocab))  # Uniform over vocabulary

    def perplexity(self, data):
        log_prob = 0.0
        token_count = 0
        
        for sentence in data:
            padded = ["<s>"] * (self.n - 1) + sentence
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i-self.n+1:i])
                word = padded[i]
                p = self.prob(context, word)
                
                if p <= 0:
                    return float("inf")
                    
                log_prob += math.log2(p)
                token_count += 1
                
        if token_count == 0:
            return float("inf")
            
        return math.pow(2, -log_prob / token_count)
