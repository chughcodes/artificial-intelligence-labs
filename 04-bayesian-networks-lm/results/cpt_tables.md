# Conditional probability tables

## First-order model: P(next word | current word)

Entries are C(current, next) / C(current, *). Rows are contexts, columns the next token.

| current \ next | <END> | cat | dog | mat | on | park | ran | rug | sat | the | to | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **<START>** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6/6 | 0 | 1.00 |
| **the** | 0 | 3/12 | 3/12 | 2/12 | 0 | 2/12 | 0 | 2/12 | 0 | 0 | 0 | 1.00 |
| **cat** | 0 | 0 | 0 | 0 | 0 | 0 | 1/3 | 0 | 2/3 | 0 | 0 | 1.00 |
| **dog** | 0 | 0 | 0 | 0 | 0 | 0 | 1/3 | 0 | 2/3 | 0 | 0 | 1.00 |
| **sat** | 0 | 0 | 0 | 0 | 4/4 | 0 | 0 | 0 | 0 | 0 | 0 | 1.00 |
| **ran** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2/2 | 1.00 |
| **on** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4/4 | 0 | 1.00 |
| **to** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2/2 | 0 | 1.00 |
| **mat** | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.00 |
| **rug** | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.00 |
| **park** | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.00 |

### Zero-probability transitions (first-order)

- **<START>**: P = 0 for <END>, cat, dog, mat, on, park, ran, rug, sat, to
- **the**: P = 0 for <END>, on, ran, sat, the, to
- **cat**: P = 0 for <END>, cat, dog, mat, on, park, rug, the, to
- **dog**: P = 0 for <END>, cat, dog, mat, on, park, rug, the, to
- **sat**: P = 0 for <END>, cat, dog, mat, park, ran, rug, sat, the, to
- **ran**: P = 0 for <END>, cat, dog, mat, on, park, ran, rug, sat, the
- **on**: P = 0 for <END>, cat, dog, mat, on, park, ran, rug, sat, to
- **to**: P = 0 for <END>, cat, dog, mat, on, park, ran, rug, sat, to
- **mat**: P = 0 for cat, dog, mat, on, park, ran, rug, sat, the, to
- **rug**: P = 0 for cat, dog, mat, on, park, ran, rug, sat, the, to
- **park**: P = 0 for cat, dog, mat, on, park, ran, rug, sat, the, to

Contexts with no observations at all (whole row undefined): none: every word was observed as a context at least once. The only undefined context is any word outside the vocabulary (e.g. 'elephant').

## Second-order model: P(next word | previous two words)

Only the observed contexts are shown; every other pair-context has no data.

| context \ next | <END> | cat | dog | mat | on | park | ran | rug | sat | the | to | total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **<START>, <START>** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 6/6 | 0 | 1.00 |
| **<START>, the** | 0 | 3/6 | 3/6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.00 |
| **cat, ran** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1/1 | 1.00 |
| **cat, sat** | 0 | 0 | 0 | 0 | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 | 1.00 |
| **dog, ran** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1/1 | 1.00 |
| **dog, sat** | 0 | 0 | 0 | 0 | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 | 1.00 |
| **on, the** | 0 | 0 | 0 | 2/4 | 0 | 0 | 0 | 2/4 | 0 | 0 | 0 | 1.00 |
| **ran, to** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2/2 | 0 | 1.00 |
| **sat, on** | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4/4 | 0 | 1.00 |
| **the, cat** | 0 | 0 | 0 | 0 | 0 | 0 | 1/3 | 0 | 2/3 | 0 | 0 | 1.00 |
| **the, dog** | 0 | 0 | 0 | 0 | 0 | 0 | 1/3 | 0 | 2/3 | 0 | 0 | 1.00 |
| **the, mat** | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.00 |
| **the, park** | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.00 |
| **the, rug** | 2/2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.00 |
| **to, the** | 0 | 0 | 0 | 0 | 0 | 2/2 | 0 | 0 | 0 | 0 | 0 | 1.00 |
