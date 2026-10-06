"""
Produces every deliverable of the laboratory and writes them to results/.

    results/cpt_tables.md                       CPTs for selected contexts (Q3)
    results/normalisation_test.txt              sum_v P(v|w) for every w   (Part VII)
    results/predictions.md                      next-word distributions + argmax (Part VIII)
    results/generated_first_order.txt           25 sampled sentences       (Part IX)
    results/generated_second_order.txt          25 sampled sentences       (Part XII)
    results/greedy_vs_sampling.md               5 + 5 sentences per model  (Part X)
    results/model_comparison.md                 parameters, zero contexts,
                                                diversity, coherence       (Part XIII)

Run:
    python experiments.py
"""

import os
import random
from collections import Counter

from data import START, END, training_data
from first_order_lm import FirstOrderLM
from second_order_lm import SecondOrderLM

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
SEED = 2026


def write(name, text):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        f.write(text.rstrip() + "\n")
    print(f"--- {name} ---\n{text}\n")


def cpt_markdown(model, contexts, label):
    vocab = model.vocabulary
    head = "| " + label + " | " + " | ".join(vocab) + " | total |"
    sep = "|" + "---|" * (len(vocab) + 2)
    rows = [head, sep]
    for ctx in contexts:
        dist = model.distribution(ctx)
        cnt = model.counts.get(ctx if not isinstance(ctx, list) else tuple(ctx), {})
        total = sum(cnt.values())
        cells = []
        for w in vocab:
            c = cnt.get(w, 0)
            cells.append(f"{c}/{total}" if c else "0")
        name = ctx if isinstance(ctx, str) else ", ".join(ctx)
        rows.append(f"| **{name}** | " + " | ".join(cells) + f" | {sum(dist.values()):.2f} |")
    return "\n".join(rows)


def main():
    data = training_data()
    m1 = FirstOrderLM().train(data)
    m2 = SecondOrderLM().train(data)
    train_set = {" ".join(s) for s in data}

    # ---------------- Q3: CPTs -----------------------------------------
    words = [w for w in m1.vocabulary if w != END]
    ctx1 = [START, "the", "cat", "dog", "sat", "ran", "on", "to", "mat", "rug", "park"]
    md = ["# Conditional probability tables\n",
          "## First-order model: P(next word | current word)\n",
          "Entries are C(current, next) / C(current, *). Rows are contexts, columns the next token.\n",
          cpt_markdown(m1, ctx1, "current \\ next"), ""]
    md.append("### Zero-probability transitions (first-order)\n")
    for ctx in ctx1:
        zeros = [w for w in m1.vocabulary if w not in m1.counts[ctx]]
        md.append(f"- **{ctx}**: P = 0 for {', '.join(zeros)}")
    unseen = [c for c in [START] + words if c not in m1.counts]
    md.append("\nContexts with no observations at all (whole row undefined): "
              + (", ".join(unseen) if unseen else "none: every word was observed "
                 "as a context at least once. The only undefined context is any word "
                 "outside the vocabulary (e.g. 'elephant')."))
    ctx2 = sorted(m2.counts, key=lambda c: (c[0] != START, c))
    md += ["\n## Second-order model: P(next word | previous two words)\n",
           "Only the observed contexts are shown; every other pair-context has no data.\n",
           cpt_markdown(m2, ctx2, "context \\ next")]
    write("cpt_tables.md", "\n".join(md))

    # ---------------- Part VII: normalisation ---------------------------
    lines = ["First-order model:  sum_v P(v | w)"]
    for w in m1.probs:
        lines.append(f"  {w:<10} {sum(m1.probs[w].values()):.6f}")
    lines.append("\nSecond-order model:  sum_v P(v | u, w)")
    for c in m2.probs:
        lines.append(f"  ({c[0]}, {c[1]})".ljust(24) + f"{sum(m2.probs[c].values()):.6f}")
    ok = all(abs(sum(d.values()) - 1) < 1e-9
             for d in list(m1.probs.values()) + list(m2.probs.values()))
    lines.append(f"\nAll rows sum to 1: {ok}")
    write("normalisation_test.txt", "\n".join(lines))

    # ---------------- Part VIII: predictions -----------------------------
    md = ["# Next-word prediction (first-order model)\n",
          "| Previous word | Distribution P(X_t+1 \\| X_t) | argmax | P(argmax) | ties |",
          "|---|---|---|---|---|"]
    for w in [START, "the", "cat", "dog", "sat", "ran", "on", "to", "mat", "elephant"]:
        dist = m1.distribution(w)
        best, p, tied = m1.predict(w)
        d = ", ".join(f"{k}: {v:.3f}" for k, v in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))) \
            or "(never observed: no distribution)"
        md.append(f"| {w} | {d} | {best or '-'} | {p:.3f} | "
                  f"{', '.join(tied) if len(tied) > 1 else '-'} |")
    md += ["\n# Next-word prediction (second-order model)\n",
           "| Previous two words | Distribution | argmax | P(argmax) |", "|---|---|---|---|"]
    for c in [(START, "the"), ("the", "cat"), ("sat", "on"), ("on", "the"), ("to", "the"),
              ("the", "mat")]:
        dist = m2.distribution(c)
        best, p, _ = m2.predict(c)
        d = ", ".join(f"{k}: {v:.3f}" for k, v in sorted(dist.items(), key=lambda kv: (-kv[1], kv[0])))
        md.append(f"| {c[0]}, {c[1]} | {d} | {best} | {p:.3f} |")
    write("predictions.md", "\n".join(md))

    # ---------------- Part IX: generation --------------------------------
    for name, m in [("generated_first_order.txt", m1), ("generated_second_order.txt", m2)]:
        rng = random.Random(SEED)
        lines = []
        for i in range(1, 26):
            ws, status = m.generate("sample", rng)
            s = " ".join(ws)
            tag = "in training data" if s in train_set else "NEW"
            lines.append(f"{i:2d}. {s}   [{tag}; P = {m.sentence_probability(ws):.3g}]")
        write(name, f"# 25 sentences sampled from the {name.split('_')[1]}-order model "
                    f"(seed {SEED})\n" + "\n".join(lines))

    # ---------------- Part X: greedy vs sampling --------------------------
    md = ["# Greedy vs sampling\n"]
    for title, m in [("First-order model", m1), ("Second-order model", m2)]:
        rng = random.Random(SEED)
        md.append(f"## {title}\n\n**Mode A, greedy** (argmax, ties broken alphabetically, "
                  "max 30 tokens):\n")
        for i in range(5):
            ws, status = m.generate("greedy")
            md.append(f"{i+1}. {' '.join(ws)}   *[{status}]*")
        md.append("\n**Mode B, sampling:**\n")
        for i in range(5):
            ws, status = m.generate("sample", rng)
            md.append(f"{i+1}. {' '.join(ws)}   *[{status}]*")
        md.append("")
    write("greedy_vs_sampling.md", "\n".join(md))

    # ---------------- Part XIII: comparison -------------------------------
    N = 10000
    md = ["# Comparison of first- and second-order models\n",
          "## Size of the conditional probability tables\n",
          "| Measure | First-order | Second-order |", "|---|---|---|"]
    r1, r2 = m1.size_report(), m2.size_report()
    keys2 = list(r2)
    for k1, k2 in zip(r1, keys2):
        md.append(f"| {k1.replace('bigrams', 'n-grams')} | {r1[k1]} | {r2[k2]} |")

    md += [f"\n## Diversity ({N} sampled sentences each, seed {SEED})\n",
           "| Measure | First-order | Second-order |", "|---|---|---|"]
    stats = {}
    for key, m in [("1", m1), ("2", m2)]:
        rng = random.Random(SEED)
        sents = [" ".join(m.generate("sample", rng, max_tokens=200)[0]) for _ in range(N)]
        c = Counter(sents)
        lens = [len(s.split()) for s in sents]
        stats[key] = dict(
            distinct=len(c),
            novel_types=sum(1 for s in c if s not in train_set),
            novel_frac=sum(n for s, n in c.items() if s not in train_set) / N,
            mean_len=sum(lens) / N, min_len=min(lens), max_len=max(lens),
            top=c.most_common(3),
            counter=c,
            examples=[s for s, _ in c.most_common() if s not in train_set][:8],
            shortest=sorted(c, key=len)[:3])
    s1, s2 = stats["1"], stats["2"]
    md += [f"| Distinct sentences generated | {s1['distinct']} | {s2['distinct']} |",
           f"| Distinct sentences NOT in the training data | {s1['novel_types']} | {s2['novel_types']} |",
           f"| Fraction of samples that are new sentences | {s1['novel_frac']:.1%} | {s2['novel_frac']:.1%} |",
           f"| Mean length (words) | {s1['mean_len']:.2f} | {s2['mean_len']:.2f} |",
           f"| Shortest / longest (words) | {s1['min_len']} / {s1['max_len']} | {s2['min_len']} / {s2['max_len']} |",
           "\n## Qualitative coherence: examples\n",
           "**First-order, new sentences (most frequent first):**\n"]
    md += [f"- {s}" for s in s1["examples"]]
    md += ["\n**First-order, shortest outputs:**\n"] + [f"- {s}" for s in s1["shortest"]]
    md += ["\n**Second-order, all distinct outputs:**\n"] + \
          [f"- {s}  ({n} times)" for s, n in s2["counter"].most_common()]
    two_word = sum(n for s, n in s1["counter"].items() if len(s.split()) == 2) / N
    md.append(f"\nFraction of first-order samples that are only two words "
              f"('the mat' / 'the rug' / 'the park'): {two_word:.1%}")
    md += ["\n## Probability of selected sentences\n",
           "| Sentence | In training? | P first-order | P second-order |", "|---|---|---|---|"]
    for s in ["the cat sat on the mat", "the dog ran to the park",
              "the cat ran to the mat", "the dog sat on the park", "the rug",
              "the cat sat on the dog ran to the park"]:
        ws = s.split()
        md.append(f"| {s} | {'yes' if s in train_set else 'no'} | "
                  f"{m1.sentence_probability(ws):.4g} | {m2.sentence_probability(ws):.4g} |")
    write("model_comparison.md", "\n".join(md))


if __name__ == "__main__":
    main()
