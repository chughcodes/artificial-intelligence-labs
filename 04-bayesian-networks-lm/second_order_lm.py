"""
Second-order autoregressive language model (trigram model)
==========================================================

Bayesian network: each token has TWO parents, the two preceding tokens

        X_{t-2} --> X_t <-- X_{t-1}

    P(x1, ..., xT) = P(x1) P(x2 | x1) prod_{t=3..T} P(xt | x_{t-2}, x_{t-1})

The sentence is padded with two <START> tokens, so that
    P(x1)      is estimated as  P(x1 | <START>, <START>)
    P(x2 | x1) is estimated as  P(x2 | <START>, x1)
and every factor has exactly the same form P(Xt | Xt-2, Xt-1).

What changes compared with the first-order model
------------------------------------------------
* the context is a PAIR (x_{t-2}, x_{t-1}) instead of a single token;
* counts are of observed TRIPLES  C(u, v, w);
* the CPT has one row per pair-context:
      P(w | u, v) = C(u, v, w) / sum_k C(u, v, k)
* sampling / greedy choice / <END> handling are unchanged.

Run:
    python second_order_lm.py
"""

import random
from collections import defaultdict, Counter

from data import START, END, training_data


class SecondOrderLM:
    def __init__(self):
        # TRIPLE COUNTS:  counts[(u, v)][w] = C(u, v, w)
        self.counts = defaultdict(Counter)
        # CPT:  probs[(u, v)][w] = P(w | u, v)
        self.probs = {}

    def train(self, sentences):
        for tokens in sentences:
            padded = [START, START] + list(tokens) + [END]
            for u, v, w in zip(padded, padded[1:], padded[2:]):
                self.counts[(u, v)][w] += 1
        self.probs = {
            ctx: {w: c / sum(row.values()) for w, c in row.items()}
            for ctx, row in self.counts.items()
        }
        return self

    @property
    def vocabulary(self):
        return sorted({w for row in self.counts.values() for w in row})

    def distribution(self, context):
        """P(. | u, v) for context = (u, v).  Empty if never observed."""
        return dict(self.probs.get(tuple(context), {}))

    def show(self, context):
        context = tuple(context)
        dist = self.distribution(context)
        total = sum(self.counts[context].values()) if context in self.counts else 0
        lines = [f"P(next | {context[0]}, {context[1]})   [count = {total}]"]
        if not dist:
            lines.append("   (context never observed: distribution undefined)")
        for w, p in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"   {w:<8} {self.counts[context][w]}/{total} = {p:.3f}")
        return "\n".join(lines)

    def predict(self, context):
        dist = self.distribution(context)
        if not dist:
            return None, 0.0, []
        best = max(dist.values())
        tied = sorted(w for w, p in dist.items() if p == best)
        return tied[0], best, tied

    def next_token(self, context, mode="sample", rng=random):
        dist = self.distribution(context)
        if not dist:
            return None
        if mode == "greedy":
            return self.predict(context)[0]
        words = list(dist)
        return rng.choices(words, weights=[dist[w] for w in words], k=1)[0]

    def generate(self, mode="sample", rng=random, max_tokens=30):
        u, v, words = START, START, []
        while len(words) < max_tokens:
            nxt = self.next_token((u, v), mode, rng)
            if nxt is None:
                return words, "unseen"
            if nxt == END:
                return words, "end"
            words.append(nxt)
            u, v = v, nxt                       # slide the two-token window
        return words, "max_tokens"

    def sentence_probability(self, tokens):
        padded = [START, START] + list(tokens) + [END]
        p = 1.0
        for u, v, w in zip(padded, padded[1:], padded[2:]):
            p *= self.probs.get((u, v), {}).get(w, 0.0)
        return p

    def size_report(self):
        words = [w for w in self.vocabulary if w != END]
        outcomes = words + [END]
        # possible contexts: (<START>,<START>), (<START>, w), (w, w')
        contexts = [(START, START)] + [(START, w) for w in words] + \
                   [(a, b) for a in words for b in words]
        observed = [c for c in contexts if c in self.counts]
        nonzero = sum(len(r) for r in self.counts.values())
        return {
            "vocabulary (words)": len(words),
            "possible contexts": len(contexts),
            "observed contexts": len(observed),
            "unobserved (zero-probability) contexts": len(contexts) - len(observed),
            "full CPT entries (contexts x outcomes)": len(contexts) * len(outcomes),
            "free parameters (rows sum to 1)": len(contexts) * (len(outcomes) - 1),
            "non-zero probabilities (distinct triples seen)": nonzero,
            "zero entries in observed rows": len(observed) * len(outcomes) - nonzero,
        }


if __name__ == "__main__":
    model = SecondOrderLM().train(training_data())
    for ctx in [(START, START), (START, "the"), ("the", "cat"), ("on", "the"),
                ("to", "the"), ("cat", "ran")]:
        print(model.show(ctx), "\n")
    rng = random.Random(0)
    print("Five sampled sentences:")
    for _ in range(5):
        words, status = model.generate("sample", rng)
        print("  ", " ".join(words), f"[{status}]")
    words, status = model.generate("greedy")
    print("\nGreedy:", " ".join(words), f"[{status}]")
