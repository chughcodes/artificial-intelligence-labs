# Next-word prediction (first-order model)

| Previous word | Distribution P(X_t+1 \| X_t) | argmax | P(argmax) | ties |
|---|---|---|---|---|
| <START> | the: 1.000 | the | 1.000 | - |
| the | cat: 0.250, dog: 0.250, mat: 0.167, park: 0.167, rug: 0.167 | cat | 0.250 | cat, dog |
| cat | sat: 0.667, ran: 0.333 | sat | 0.667 | - |
| dog | sat: 0.667, ran: 0.333 | sat | 0.667 | - |
| sat | on: 1.000 | on | 1.000 | - |
| ran | to: 1.000 | to | 1.000 | - |
| on | the: 1.000 | the | 1.000 | - |
| to | the: 1.000 | the | 1.000 | - |
| mat | <END>: 1.000 | <END> | 1.000 | - |
| elephant | (never observed: no distribution) | - | 0.000 | - |

# Next-word prediction (second-order model)

| Previous two words | Distribution | argmax | P(argmax) |
|---|---|---|---|
| <START>, the | cat: 0.500, dog: 0.500 | cat | 0.500 |
| the, cat | sat: 0.667, ran: 0.333 | sat | 0.667 |
| sat, on | the: 1.000 | the | 1.000 |
| on, the | mat: 0.500, rug: 0.500 | mat | 0.500 |
| to, the | park: 1.000 | park | 1.000 |
| the, mat | <END>: 1.000 | <END> | 1.000 |
