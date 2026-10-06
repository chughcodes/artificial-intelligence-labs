"""
Verification tests for the neural-models laboratory.

Run:
    python -m unittest test_neural -v
"""

import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

import xor_experiments as xe
import three_class as tc

torch.set_num_threads(1)


class TestXOR(unittest.TestCase):

    def test_affine_only_models_cannot_fit_xor(self):
        torch.manual_seed(0)
        r = xe.train(nn.Sequential(nn.Linear(2, 1)))
        self.assertLess(r["correct"], 4)
        self.assertAlmostEqual(r["final_loss"], 0.693147, places=3)   # ln 2
        r2 = xe.train(xe.make_net("identity (no nonlinearity)"))
        self.assertLess(r2["correct"], 4)

    def test_tanh_network_learns_xor(self):
        r = xe.train(xe.make_net("tanh"))
        self.assertEqual(r["correct"], 4)
        self.assertLess(r["final_loss"], 1e-3)
        self.assertLess(r["final_loss"], r["init_loss"])

    def test_backward_matches_hand_derivation_and_finite_differences(self):
        model = xe.make_net("tanh")
        model.zero_grad()
        nn.BCEWithLogitsLoss()(model(xe.X), xe.Y).backward()
        g = model[0].weight.grad.clone()
        self.assertTrue(torch.allclose(g, xe.manual_grad_W1(model), atol=1e-7))
        self.assertTrue(torch.allclose(g, xe.finite_difference_grad_W1(model), atol=1e-6))
        self.assertGreater(g.norm().item(), 0)

    def test_mean_loss_gradient_is_mean_of_example_gradients(self):
        model = xe.make_net("tanh")
        model.zero_grad()
        nn.BCEWithLogitsLoss()(model(xe.X), xe.Y).backward()
        full = model[0].weight.grad.clone()
        per = []
        for i in range(4):
            model.zero_grad()
            nn.BCEWithLogitsLoss()(model(xe.X[i:i + 1]), xe.Y[i:i + 1]).backward()
            per.append(model[0].weight.grad.clone())
        self.assertTrue(torch.allclose(full, torch.stack(per).mean(0), atol=1e-8))

    def test_zero_initialisation_gives_zero_gradient(self):
        m = xe.zero_init(xe.make_net("sigmoid"))
        m.zero_grad()
        nn.BCEWithLogitsLoss()(m(xe.X), xe.Y).backward()
        for p in m.parameters():
            self.assertTrue(torch.all(p.grad == 0))

    def test_identical_initialisation_keeps_rows_identical(self):
        m = xe.zero_init(xe.make_net("sigmoid"), value=0.3)
        r = xe.train(m, steps=300, record_rows_at=(0, 10, 100, 300))
        for W in r["rows"].values():
            self.assertTrue(torch.equal(W[0], W[1]))
        self.assertFalse(torch.equal(r["rows"][300], r["rows"][0]))  # they DO change
        self.assertLess(r["correct"], 4)


class TestThreeClass(unittest.TestCase):

    def test_output_shapes(self):
        m = tc.make_net()
        self.assertEqual(tuple(m[2].weight.shape), (3, 2))
        self.assertEqual(tuple(m(tc.X).shape), (4, 3))

    def test_softmax_sums_to_one_and_is_shift_invariant(self):
        z = torch.randn(10, 3)
        p = F.softmax(z, dim=1)
        self.assertTrue(torch.allclose(p.sum(1), torch.ones(10)))
        self.assertTrue(torch.allclose(F.softmax(z + 100, dim=1), p, atol=1e-6))

    def test_logit_gradient_is_p_minus_y(self):
        z = torch.randn(4, 3, requires_grad=True)
        F.cross_entropy(z, tc.Y3, reduction="sum").backward()
        expected = F.softmax(z.detach(), 1) - F.one_hot(tc.Y3, 3).float()
        self.assertTrue(torch.allclose(z.grad, expected, atol=1e-6))

    def test_three_class_model_learns(self):
        m = tc.make_net()
        tc.train(m)
        with torch.no_grad():
            self.assertTrue(torch.equal(m(tc.X).argmax(1), tc.Y3))


if __name__ == "__main__":
    unittest.main(verbosity=2)
