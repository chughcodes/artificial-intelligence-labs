"""
Task 5: three-class sensor decision with softmax + cross-entropy
================================================================

    class 0 : both sensors inactive   (0, 0)
    class 1 : sensors disagree        (0, 1) or (1, 0)
    class 2 : both sensors active     (1, 1)

Only the output/loss part of the binary model changes:
    hidden : unchanged  - Linear(2, 2) + tanh
    output : Linear(2, 1) -> Linear(2, 3)       (three logits per example)
    loss   : BCEWithLogitsLoss -> CrossEntropyLoss (softmax + NLL, combined)
    labels : float column (4, 1) -> integer class indices (4,)

Predictions made BEFORE running (see README.md):
    1. final weight matrix shape  = (3, 2)   [out_features x in_features]
    2. logits per example         = 3
    3. softmax sums to one        : sum_k exp(z_k) / sum_j exp(z_j) = 1
    4. dL/dz = p - y              : derivative of -log softmax_y(z)

Run:
    python three_class.py            (output saved to results/three_class_output.txt)
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y3 = torch.tensor([0, 1, 1, 2])                   # class indices
STEPS, LR, SEED = 3000, 0.05, 0


def make_net(seed=SEED):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 2), nn.Tanh(),   # unchanged hidden layer
                         nn.Linear(2, 3))              # CHANGED: 3 logits


def train(model):
    loss_fn = nn.CrossEntropyLoss()                    # CHANGED loss
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    with torch.no_grad():
        init = loss_fn(model(X), Y3).item()
    for _ in range(STEPS):
        opt.zero_grad()
        loss = loss_fn(model(X), Y3)
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = loss_fn(model(X), Y3).item()
    return init, final


def main():
    torch.set_printoptions(precision=6, sci_mode=False)
    model = make_net()

    print("=" * 70)
    print("Checking the predictions about the output design")
    print("=" * 70)
    W_out = model[2].weight
    logits = model(X)
    print(f"  final weight matrix shape : {tuple(W_out.shape)}   (predicted (3, 2))")
    print(f"  final bias shape          : {tuple(model[2].bias.shape)}")
    print(f"  logits tensor shape       : {tuple(logits.shape)}   -> "
          f"{logits.shape[1]} logits per example (predicted 3)")

    init, final = train(model)
    print("\n" + "=" * 70)
    print(f"Training: seed {SEED}, Adam lr={LR}, {STEPS} full-batch steps")
    print("=" * 70)
    print(f"  initial loss {init:.4f}  (ln 3 = {torch.log(torch.tensor(3.)).item():.4f})")
    print(f"  final loss   {final:.6f}")

    with torch.no_grad():
        logits = model(X)
        probs = F.softmax(logits, dim=1)
    print("\n  Predicted class probabilities")
    print("  x       target   P(class 0)  P(class 1)  P(class 2)   predicted   sum")
    for xi, t, p in zip(X.int().tolist(), Y3.tolist(), probs):
        print(f"  {xi}  {t:^6}   {p[0]:.6f}    {p[1]:.6f}    {p[2]:.6f}    "
              f"{int(p.argmax()):^9}   {p.sum().item():.8f}")
    correct = int((probs.argmax(1) == Y3).sum())
    print(f"  correct: {correct}/4")

    # ---- one example in detail ----------------------------------------
    i = 1                                                  # x = (0, 1)
    z = logits[i]
    p = F.softmax(z, dim=0)
    print("\n" + "=" * 70)
    print(f"Softmax check for one example: x = {X[i].int().tolist()}")
    print("=" * 70)
    print(f"  logits z          = {z.tolist()}")
    print(f"  softmax(z)        = {p.tolist()}")
    print(f"  sum of components = {p.sum().item():.10f}")
    shifted = F.softmax(z + 100, dim=0)
    print(f"  softmax(z + 100)  = {shifted.tolist()}")
    print(f"  max |difference|  = {(shifted - p).abs().max().item():.2e}  "
          f"(unchanged apart from round-off)")

    # ---- numerical stability ------------------------------------------
    print("\n" + "=" * 70)
    print("Why stable softmax subtracts the maximum logit")
    print("=" * 70)
    big = (z + 1000).double()
    naive = torch.exp(big) / torch.exp(big).sum()
    stable = torch.exp(big - big.max()) / torch.exp(big - big.max()).sum()
    print(f"  logits + 1000                 = {big.tolist()}")
    print(f"  exp(logits + 1000)            = {torch.exp(big).tolist()}")
    print(f"  naive  exp(z)/sum exp(z)      = {naive.tolist()}")
    print(f"  stable exp(z-max)/sum(...)    = {stable.tolist()}")
    print(f"  torch.softmax(logits + 1000)  = {F.softmax(big, dim=0).tolist()}")

    # ---- gradient p - y -----------------------------------------------
    print("\n" + "=" * 70)
    print("Checking dL/dz = p - y  (cross-entropy w.r.t. the logits)")
    print("=" * 70)
    y_onehot = F.one_hot(Y3, 3).float()
    for label, net in [("UNTRAINED network (large errors)", make_net()),
                       ("TRAINED network", model)]:
        z_all = net(X).detach().requires_grad_(True)
        loss = F.cross_entropy(z_all, Y3, reduction="sum")   # sum: no 1/N factor
        loss.backward()
        p_all = F.softmax(z_all.detach(), dim=1)
        print(f"  {label}")
        print("  autograd dL/dz:\n", z_all.grad)
        print("  p - y:\n", p_all - y_onehot)
        print(f"  max |difference| = "
              f"{(z_all.grad - (p_all - y_onehot)).abs().max().item():.2e}\n")
    print("  (with the default reduction='mean' the gradient is (p - y) / 4)")

    # ---- a linear softmax also works here ------------------------------
    torch.manual_seed(SEED)
    lin = nn.Sequential(nn.Linear(2, 3))
    opt = torch.optim.Adam(lin.parameters(), lr=LR)
    for _ in range(STEPS):
        opt.zero_grad()
        F.cross_entropy(lin(X), Y3).backward()
        opt.step()
    with torch.no_grad():
        lin_correct = int((lin(X).argmax(1) == Y3).sum())
        lin_loss = F.cross_entropy(lin(X), Y3).item()
    print("\n" + "=" * 70)
    print("Extra: a model with NO hidden layer (single affine map + softmax)")
    print("=" * 70)
    print(f"  final loss {lin_loss:.4f}, correct {lin_correct}/4")
    print("  (0/1/2 = number of active sensors, so the three classes are linearly")
    print("   separable; but class 1 by itself is XOR and is not.  See README.)")


class Tee:
    def __init__(self, *s):
        self.s = s

    def write(self, x):
        for st in self.s:
            st.write(x)

    def flush(self):
        for st in self.s:
            st.flush()


if __name__ == "__main__":
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "three_class_output.txt"), "w") as f:
        sys.stdout = Tee(sys.__stdout__, f)
        main()
        sys.stdout = sys.__stdout__
