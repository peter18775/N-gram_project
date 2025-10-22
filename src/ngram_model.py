from collections import defaultdict,Counter
import math
def counter_defaultdict():
    return defaultdict(int)
class NGramModel:
    def __init__(self,n):
        self.n = n
        self.counts = defaultdict(counter_defaultdict)
        self.context_counts = Counter()
        self.vocab = set()

    def train(self, data):
        for sentence in data:
            padded = ["<s>"] * (self.n - 1) + sentence
            for i in range(len(padded) - self.n + 1):
                ngram = tuple(padded[i:i+self.n])
                context, word = ngram[:-1], ngram[-1]
                self.counts[context][word] += 1
                self.context_counts[context] += 1
                self.vocab.add(word)
    
    def prob(self, context, word):
        if context not in self.counts or self.context_counts[context] == 0:
            return 0.0
        return self.counts[context][word] / self.context_counts[context]

    def perplexity(self,data):
        log_prob, token_count = 0, 0
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
        return math.pow(2, -log_prob / token_count)