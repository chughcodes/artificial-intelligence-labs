"""
First-order autoregressive language model (bigram model)
=========================================================

Bayesian network:   <START> -> X1 -> X2 -> ... -> XT -> <END>

    P(x1, ..., xT) = P(x1 | <START>) * prod_{t=2..T} P(xt | x_{t-1}) * P(<END> | xT)

Each conditional probability table (CPT) is estimated by counting:

    P(wj | wi) = C(wi, wj) / sum_k C(wi, wk)

where C(wi, wj) is the number of times wj immediately follows wi.

No machine-learning library or pretrained model is used: only dictionaries,
Counters and random sampling.

Run:
    python first_order_lm.py
"""

import random
from collections import defaultdict, Counter

from data import START, END, training_data


class FirstOrderLM:
    def __init__(self):
        # TRANSITION COUNTS:  counts[previous][next] = C(previous, next)
        self.counts = defaultdict(Counter)
        # CPT:  probs[previous][next] = P(next | previous)
        self.probs = {}

    # ------------------------------------------------------------------
    # 1-3. Train: count transitions, then normalise each row into P(Xt|Xt-1)
    # ------------------------------------------------------------------
    def train(self, sentences):
        for tokens in sentences:
            padded = [START] + list(tokens) + [END]
            for prev, nxt in zip(padded, padded[1:]):
                self.counts[prev][nxt] += 1
        self.probs = {
            prev: {nxt: c / sum(row.values()) for nxt, c in row.items()}
            for prev, row in self.counts.items()
        }
        return self

    @property
    def vocabulary(self):
        """All tokens that can be generated (words and <END>)."""
        return sorted({w for row in self.counts.values() for w in row})

    # ------------------------------------------------------------------
    # 4. Display the probabilities for a specified previous token
    # ------------------------------------------------------------------
    def distribution(self, prev):
        """P(. | prev) as a dict.  Empty dict if prev was never observed."""
        return dict(self.probs.get(prev, {}))

    def show(self, prev):
        dist = self.distribution(prev)
        total = sum(self.counts[prev].values()) if prev in self.counts else 0
        lines = [f"P(next | {prev})   [C({prev}, *) = {total}]"]
        if not dist:
            lines.append("   (no transitions observed from this token)")
        for w, p in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])):
            lines.append(f"   {w:<8} {self.counts[prev][w]}/{total} = {p:.3f}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 5. Most probable next token: argmax_w P(w | prev)
    # ------------------------------------------------------------------
    def predict(self, prev):
        """Return (best_token, probability, tied_tokens) or (None, 0, [])."""
        dist = self.distribution(prev)
        if not dist:
            return None, 0.0, []
        best_p = max(dist.values())
        tied = sorted(w for w, p in dist.items() if p == best_p)
        return tied[0], best_p, tied      # ties broken alphabetically

    # ------------------------------------------------------------------
    # 6-7. Generation: sample (or argmax) repeatedly until <END>
    # ------------------------------------------------------------------
    def next_token(self, prev, mode="sample", rng=random):
        dist = self.distribution(prev)
        if not dist:
            return None                      # unseen context: cannot continue
        if mode == "greedy":
            return self.predict(prev)[0]
        words = list(dist)
        return rng.choices(words, weights=[dist[w] for w in words], k=1)[0]

    def generate(self, mode="sample", rng=random, max_tokens=30):
        """
        Returns (list_of_words, status) where status is
            'end'        - <END> was generated (normal termination)
            'max_tokens' - stopped by the length guard (no <END> reached)
            'unseen'     - reached a token with no observed successor
        """
        prev, words = START, []
        while len(words) < max_tokens:
            nxt = self.next_token(prev, mode, rng)
            if nxt is None:
                return words, "unseen"
            if nxt == END:
                return words, "end"
            words.append(nxt)
            prev = nxt
        return words, "max_tokens"

    # ------------------------------------------------------------------
    # Probability of a whole sentence (chain rule with the Markov assumption)
    # ------------------------------------------------------------------
    def sentence_probability(self, tokens):
        padded = [START] + list(tokens) + [END]
        p = 1.0
        for prev, nxt in zip(padded, padded[1:]):
            p *= self.probs.get(prev, {}).get(nxt, 0.0)
        return p

    # ------------------------------------------------------------------
    # Model size
    # ------------------------------------------------------------------
    def size_report(self):
        words = [w for w in self.vocabulary if w != END]        # 10 words
        contexts = [START] + words                                # possible X_{t-1}
        outcomes = words + [END]                                  # possible X_t
        full_entries = len(contexts) * len(outcomes)
        free_params = len(contexts) * (len(outcomes) - 1)
        nonzero = sum(len(r) for r in self.counts.values())
        unseen_ctx = [c for c in contexts if c not in self.counts]
        return {
            "vocabulary (words)": len(words),
            "possible contexts": len(contexts),
            "observed contexts": len(contexts) - len(unseen_ctx),
            "unobserved (zero-probability) contexts": len(unseen_ctx),
            "full CPT entries (contexts x outcomes)": full_entries,
            "free parameters (rows sum to 1)": free_params,
            "non-zero probabilities (distinct bigrams seen)": nonzero,
            "zero entries in observed rows":
                (len(contexts) - len(unseen_ctx)) * len(outcomes) - nonzero,
        }


if __name__ == "__main__":
    model = FirstOrderLM().train(training_data())
    for w in [START, "the", "cat", "dog", "sat", "ran"]:
        print(model.show(w), "\n")
    print("Most probable next word after 'the':", model.predict("the"))
    rng = random.Random(0)
    print("\nFive sampled sentences:")
    for _ in range(5):
        words, status = model.generate("sample", rng)
        print("  ", " ".join(words), f"[{status}]")
    words, status = model.generate("greedy")
    print("\nGreedy:", " ".join(words), f"[{status}]")
