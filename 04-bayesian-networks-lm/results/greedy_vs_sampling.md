# Greedy vs sampling

## First-order model

**Mode A, greedy** (argmax, ties broken alphabetically, max 30 tokens):

1. the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat   *[max_tokens]*
2. the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat   *[max_tokens]*
3. the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat   *[max_tokens]*
4. the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat   *[max_tokens]*
5. the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat sat on the cat   *[max_tokens]*

**Mode B, sampling:**

1. the rug   *[end]*
2. the cat sat on the dog sat on the dog sat on the cat ran to the dog ran to the park   *[end]*
3. the park   *[end]*
4. the mat   *[end]*
5. the rug   *[end]*

## Second-order model

**Mode A, greedy** (argmax, ties broken alphabetically, max 30 tokens):

1. the cat sat on the mat   *[end]*
2. the cat sat on the mat   *[end]*
3. the cat sat on the mat   *[end]*
4. the cat sat on the mat   *[end]*
5. the cat sat on the mat   *[end]*

**Mode B, sampling:**

1. the dog sat on the mat   *[end]*
2. the dog sat on the rug   *[end]*
3. the dog sat on the rug   *[end]*
4. the cat sat on the rug   *[end]*
5. the cat sat on the mat   *[end]*
