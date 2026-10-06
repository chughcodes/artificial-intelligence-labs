# Comparison of first- and second-order models

## Size of the conditional probability tables

| Measure | First-order | Second-order |
|---|---|---|
| vocabulary (words) | 10 | 10 |
| possible contexts | 11 | 111 |
| observed contexts | 11 | 15 |
| unobserved (zero-probability) contexts | 0 | 96 |
| full CPT entries (contexts x outcomes) | 121 | 1221 |
| free parameters (rows sum to 1) | 110 | 1110 |
| non-zero probabilities (distinct n-grams seen) | 17 | 19 |
| zero entries in observed rows | 104 | 146 |

## Diversity (10000 sampled sentences each, seed 2026)

| Measure | First-order | Second-order |
|---|---|---|
| Distinct sentences generated | 787 | 6 |
| Distinct sentences NOT in the training data | 781 | 0 |
| Fraction of samples that are new sentences | 85.9% | 0.0% |
| Mean length (words) | 6.07 | 6.00 |
| Shortest / longest (words) | 2 / 50 | 6 / 6 |

## Qualitative coherence: examples

**First-order, new sentences (most frequent first):**

- the mat
- the rug
- the park
- the dog sat on the park
- the cat sat on the park
- the dog ran to the mat
- the dog ran to the rug
- the cat ran to the mat

**First-order, shortest outputs:**

- the rug
- the mat
- the park

**Second-order, all distinct outputs:**

- the cat sat on the rug  (1694 times)
- the cat ran to the park  (1690 times)
- the cat sat on the mat  (1675 times)
- the dog sat on the mat  (1665 times)
- the dog ran to the park  (1642 times)
- the dog sat on the rug  (1634 times)

Fraction of first-order samples that are only two words ('the mat' / 'the rug' / 'the park'): 49.3%

## Probability of selected sentences

| Sentence | In training? | P first-order | P second-order |
|---|---|---|---|
| the cat sat on the mat | yes | 0.02778 | 0.1667 |
| the dog ran to the park | yes | 0.01389 | 0.1667 |
| the cat ran to the mat | no | 0.01389 | 0 |
| the dog sat on the park | no | 0.02778 | 0 |
| the rug | no | 0.1667 | 0 |
| the cat sat on the dog ran to the park | no | 0.002315 | 0 |
