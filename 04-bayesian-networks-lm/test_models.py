"""
Tests that the implementations really implement the intended probabilistic
models (Part VII and beyond).

Run:
    python -m unittest test_models -v
"""

import math
import random
import unittest
from collections import Counter, defaultdict

from data import START, END, training_data
from first_order_lm import FirstOrderLM
from second_order_lm import SecondOrderLM


class TestNormalisation(unittest.TestCase):
    """For every context w:  sum_v P(v | w) = 1."""

    def check(self, model):
        for ctx, dist in model.probs.items():
            self.assertAlmostEqual(sum(dist.values()), 1.0, places=12, msg=str(ctx))
            for p in dist.values():
                self.assertTrue(0.0 < p <= 1.0)

    def test_first_order_rows_sum_to_one(self):
        self.check(FirstOrderLM().train(training_data()))

    def test_second_order_rows_sum_to_one(self):
        self.check(SecondOrderLM().train(training_data()))

    def test_detects_a_buggy_implementation(self):
        """Negative control: a typical bug (add-one smoothing applied to the
        numerators of the OBSERVED words only, but with the denominator
        enlarged by the vocabulary size) must FAIL the normalisation test."""
        m = FirstOrderLM().train(training_data())
        V = len(m.vocabulary)
        buggy = {prev: {w: (c + 1) / (sum(row.values()) + V)
                        for w, c in row.items()}
                 for prev, row in m.counts.items()}
        totals = {prev: sum(d.values()) for prev, d in buggy.items()}
        self.assertTrue(any(abs(t - 1) > 1e-6 for t in totals.values()))


class TestFirstOrderCPT(unittest.TestCase):
    def setUp(self):
        self.m = FirstOrderLM().train(training_data())

    def test_counts_by_hand(self):
        self.assertEqual(self.m.counts["the"],
                         Counter({"cat": 3, "dog": 3, "mat": 2, "rug": 2, "park": 2}))
        self.assertEqual(self.m.counts["cat"], Counter({"sat": 2, "ran": 1}))
        self.assertEqual(self.m.counts["sat"], Counter({"on": 4}))
        self.assertEqual(self.m.counts["ran"], Counter({"to": 2}))
        self.assertEqual(self.m.counts[START], Counter({"the": 6}))

    def test_probabilities_by_hand(self):
        self.assertAlmostEqual(self.m.probs["the"]["cat"], 3 / 12)
        self.assertAlmostEqual(self.m.probs["the"]["mat"], 2 / 12)
        self.assertAlmostEqual(self.m.probs["dog"]["sat"], 2 / 3)
        self.assertAlmostEqual(self.m.probs["mat"][END], 1.0)

    def test_zero_probability_transition(self):
        self.assertEqual(self.m.distribution("the").get("sat", 0.0), 0.0)
        self.assertEqual(self.m.sentence_probability(["the", "sat"]), 0.0)

    def test_unseen_word_has_no_distribution(self):
        self.assertEqual(self.m.distribution("elephant"), {})
        self.assertEqual(self.m.predict("elephant"), (None, 0.0, []))
        self.assertIsNone(self.m.next_token("elephant"))

    def test_end_is_never_a_context(self):
        self.assertNotIn(END, self.m.probs)

    def test_sentence_probability_chain_rule(self):
        # P = P(the|S)P(cat|the)P(sat|cat)P(on|sat)P(the|on)P(mat|the)P(END|mat)
        expected = 1 * (3/12) * (2/3) * 1 * 1 * (2/12) * 1
        self.assertAlmostEqual(
            self.m.sentence_probability("the cat sat on the mat".split()), expected)

    def test_joint_distribution_sums_to_one_over_short_sentences(self):
        """The model defines a proper distribution over sentences: summing
        P(sentence) over ALL sentences up to length L approaches 1."""
        m = self.m
        total, frontier = 0.0, [((START,), 1.0)]
        for _ in range(40):
            new = []
            for seq, p in frontier:
                for w, q in m.probs[seq[-1]].items():
                    if w == END:
                        total += p * q
                    else:
                        new.append((seq + (w,), p * q))
            frontier = new
        self.assertGreater(total, 0.999)


class TestSampling(unittest.TestCase):
    def test_sampling_frequencies_match_cpt(self):
        m = FirstOrderLM().train(training_data())
        rng = random.Random(123)
        n = 60000
        freq = Counter(m.next_token("the", "sample", rng) for _ in range(n))
        for w, p in m.probs["the"].items():
            sd = math.sqrt(p * (1 - p) / n)
            self.assertLess(abs(freq[w] / n - p), 5 * sd, w)
        self.assertEqual(set(freq), set(m.probs["the"]))   # never invents words

    def test_greedy_is_deterministic(self):
        m = FirstOrderLM().train(training_data())
        outs = {tuple(m.generate("greedy")[0]) for _ in range(5)}
        self.assertEqual(len(outs), 1)

    def test_first_order_greedy_never_reaches_end(self):
        """the -> cat -> sat -> on -> the ... is a cycle of argmax choices."""
        m = FirstOrderLM().train(training_data())
        words, status = m.generate("greedy", max_tokens=100)
        self.assertEqual(status, "max_tokens")

    def test_sampled_sentences_terminate_and_are_valid(self):
        m = FirstOrderLM().train(training_data())
        rng = random.Random(7)
        for _ in range(500):
            words, status = m.generate("sample", rng, max_tokens=200)
            self.assertEqual(status, "end")
            self.assertGreater(m.sentence_probability(words), 0.0)


class TestSecondOrder(unittest.TestCase):
    def setUp(self):
        self.m = SecondOrderLM().train(training_data())

    def test_triple_counts_by_hand(self):
        self.assertEqual(self.m.counts[("on", "the")], Counter({"mat": 2, "rug": 2}))
        self.assertEqual(self.m.counts[("to", "the")], Counter({"park": 2}))
        self.assertEqual(self.m.counts[(START, START)], Counter({"the": 6}))

    def test_two_tokens_of_context_are_used(self):
        """First-order: P(.|the) is one distribution.  Second-order: it depends
        on the word before 'the'."""
        self.assertNotEqual(self.m.distribution(("on", "the")),
                            self.m.distribution(("to", "the")))
        self.assertEqual(self.m.distribution(("to", "the")), {"park": 1.0})

    def test_greedy_terminates(self):
        words, status = self.m.generate("greedy")
        self.assertEqual((" ".join(words), status), ("the cat sat on the mat", "end"))

    def test_unseen_context(self):
        self.assertEqual(self.m.distribution(("cat", "cat")), {})
        self.assertEqual(self.m.sentence_probability("the cat ran to the mat".split()), 0.0)

    def test_sentence_probability(self):
        # P(the|S,S) P(cat|S,the) P(sat|the,cat) P(on|cat,sat) P(the|sat,on)
        # P(mat|on,the) P(END|the,mat)
        expected = 1 * 0.5 * (2 / 3) * 1 * 1 * 0.5 * 1
        self.assertAlmostEqual(
            self.m.sentence_probability("the cat sat on the mat".split()), expected)

    def test_distribution_over_sentences_is_exactly_the_training_set(self):
        """With this tiny dataset the trigram model puts all its probability
        on the six training sentences."""
        rng = random.Random(1)
        seen = {" ".join(self.m.generate("sample", rng)[0]) for _ in range(2000)}
        self.assertEqual(seen, {" ".join(s) for s in training_data()})


if __name__ == "__main__":
    unittest.main(verbosity=2)
