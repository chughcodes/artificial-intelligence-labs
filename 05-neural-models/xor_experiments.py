"""
Neural models laboratory: binary XOR experiments (Tasks 1 and 4)
================================================================

    Task 1 check : affine-only models cannot learn XOR
                   (a) single affine layer + sigmoid
                   (b) 2-2-1 network WITHOUT a hidden nonlinearity (affine depth)
    Part A       : 2-2-1 tanh network - initial/final loss, probabilities, labels
    Part B       : dL/dW1 from backward() checked against (i) a hand-derived
                   backprop formula, (ii) central finite differences, and
                   (iii) the average of the four per-example gradients
    Part C       : symmetry - all weights initialised to zero (and a
                   non-zero-but-identical control)
    Part D       : sigmoid vs tanh vs ReLU from the SAME random initialisation,
                   plus repeated-run success rates over 50 seeds
    Diagnostics  : pre-activations/activations to tell a saturated sigmoid
                   from a dead ReLU

All numbers in README.md come from this script.  Output is also saved to
results/xor_experiments_output.txt and results/loss_curves.png.

Run:
    python xor_experiments.py
"""

import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

X = torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
Y = torch.tensor([[0.], [1.], [1.], [0.]])

STEPS = 3000
LR = 0.05
SEED = 0
ACTIVATIONS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU,
               "identity (no nonlinearity)": nn.Identity}


# ---------------------------------------------------------------------------
def make_net(activation="tanh", seed=SEED, hidden=2):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, hidden), ACTIVATIONS[activation](),
                         nn.Linear(hidden, 1))


def train(model, steps=STEPS, lr=LR, record_rows_at=(), x=X, y=Y):
    """Full-batch Adam on BCEWithLogitsLoss.  Returns a dict of evidence."""
    loss_fn = nn.BCEWithLogitsLoss()                    # mean over examples
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    with torch.no_grad():
        init_loss = loss_fn(model(x), y).item()
    losses, early_grad, rows = [], None, {}
    for step in range(steps):
        if step in record_rows_at:
            rows[step] = model[0].weight.detach().clone()
        opt.zero_grad()
        loss = loss_fn(model(x), y)                     # forward + scalar loss
        loss.backward()                                 # backprop
        if step == 0:
            early_grad = model[0].weight.grad.detach().clone()
        losses.append(loss.item())
        opt.step()                                      # parameter update
    if steps in record_rows_at:
        rows[steps] = model[0].weight.detach().clone()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        final_loss = loss_fn(logits, y).item()
    labels = (probs > 0.5).float()
    return dict(init_loss=init_loss, final_loss=final_loss, losses=losses,
                probs=probs.squeeze(1), labels=labels.squeeze(1),
                correct=int((labels == y).sum()), early_grad=early_grad,
                early_grad_norm=early_grad.norm().item(), rows=rows)


def show_predictions(r):
    for xi, p, lab, t in zip(X.tolist(), r["probs"].tolist(),
                             r["labels"].tolist(), Y.squeeze(1).tolist()):
        print(f"    x={[int(v) for v in xi]}  P(y=1)={p:.4f}  "
              f"label={int(lab)}  target={int(t)}  {'ok' if lab == t else 'WRONG'}")


# ---------------------------------------------------------------------------
def task1_linear_baselines():
    print("=" * 74)
    print("TASK 1 CHECK - models without a hidden nonlinearity")
    print("=" * 74)
    torch.manual_seed(SEED)
    single = nn.Sequential(nn.Linear(2, 1))
    r = train(single)
    print("(a) single affine layer + sigmoid output")
    print(f"    initial loss {r['init_loss']:.4f}  final loss {r['final_loss']:.4f}"
          f"  (ln 2 = {torch.log(torch.tensor(2.)).item():.4f})  correct {r['correct']}/4")
    show_predictions(r)
    r2 = train(make_net("identity (no nonlinearity)"))
    print("(b) 2-2-1 network with identity hidden activation (affine depth only)")
    print(f"    initial loss {r2['init_loss']:.4f}  final loss {r2['final_loss']:.4f}"
          f"  correct {r2['correct']}/4")
    show_predictions(r2)
    return r, r2


def part_a():
    print("\n" + "=" * 74)
    print(f"PART A - 2-2-1 tanh network, seed {SEED}, Adam lr={LR}, {STEPS} steps")
    print("=" * 74)
    r = train(make_net("tanh"))
    print(f"    initial loss {r['init_loss']:.4f}")
    print(f"    final loss   {r['final_loss']:.6f}")
    for s in (0, 100, 250, 500, 1000, 2000, 2999):
        print(f"    loss at step {s:4d}: {r['losses'][s]:.6f}")
    show_predictions(r)
    print(f"    all four correct: {r['correct'] == 4}")
    return r


# ---------------------------------------------------------------------------
def manual_grad_W1(model, x=X, y=Y):
    """Hand-derived backprop for tanh hidden layer + BCE-with-logits (mean)."""
    W1, b1 = model[0].weight.detach(), model[0].bias.detach()
    W2, b2 = model[2].weight.detach(), model[2].bias.detach()
    a1 = x @ W1.T + b1                    # (4,2) hidden pre-activations
    h1 = torch.tanh(a1)                   # (4,2)
    z = h1 @ W2.T + b2                    # (4,1) output logits
    p = torch.sigmoid(z)
    n = x.shape[0]
    delta2 = (p - y) / n                  # dL/dz : (p - y)/N for mean BCE
    delta1 = (delta2 @ W2) * (1 - h1 ** 2)  # dL/da1 = dL/dz * W2 * tanh'(a1)
    return delta1.T @ x                   # dL/dW1 = sum_i delta1_i x_i^T


def finite_difference_grad_W1(model, eps=1e-6):
    m = model.double()
    loss_fn = nn.BCEWithLogitsLoss()
    W = m[0].weight
    g = torch.zeros_like(W)
    with torch.no_grad():
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                old = W[i, j].item()
                W[i, j] = old + eps
                lp = loss_fn(m(X.double()), Y.double()).item()
                W[i, j] = old - eps
                lm = loss_fn(m(X.double()), Y.double()).item()
                W[i, j] = old
                g[i, j] = (lp - lm) / (2 * eps)
    m.float()
    return g.float()


def part_b():
    print("\n" + "=" * 74)
    print("PART B - backpropagation check (fresh tanh network, before training)")
    print("=" * 74)
    model = make_net("tanh")
    loss = nn.BCEWithLogitsLoss()(model(X), Y)
    model.zero_grad()
    loss.backward()
    autograd = model[0].weight.grad.clone()
    manual = manual_grad_W1(model)
    fd = finite_difference_grad_W1(model)
    # per-example gradients
    per_example = []
    for i in range(4):
        model.zero_grad()
        nn.BCEWithLogitsLoss()(model(X[i:i + 1]), Y[i:i + 1]).backward()
        per_example.append(model[0].weight.grad.clone())
    mean_of_examples = torch.stack(per_example).mean(0)

    print("    W1 =\n", model[0].weight.data)
    print("    dL/dW1 from loss.backward() (W1.grad):\n", autograd)
    print("    dL/dW1 from hand-derived backprop:\n", manual)
    print("    dL/dW1 from central finite differences (float64):\n", fd)
    for i, g in enumerate(per_example):
        print(f"    per-example gradient for x={X[i].int().tolist()}:\n", g)
    print("    mean of the four per-example gradients:\n", mean_of_examples)
    errs = {"autograd vs hand-derived": (autograd - manual).abs().max().item(),
            "autograd vs finite differences": (autograd - fd).abs().max().item(),
            "autograd vs mean of per-example": (autograd - mean_of_examples).abs().max().item()}
    for k, v in errs.items():
        print(f"    max |difference| {k:<34}: {v:.2e}")
    return autograd, manual, fd, per_example, mean_of_examples, errs


# ---------------------------------------------------------------------------
def zero_init(model, value=0.0):
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(value)
    return model


def part_c():
    print("\n" + "=" * 74)
    print("PART C - symmetry experiment: all weights and biases set to zero")
    print("=" * 74)
    checkpoints = (0, 1, 2, 5, 10, 100, 1000, STEPS)
    out = {}
    for act in ("sigmoid", "tanh"):
        model = zero_init(make_net(act))
        r = train(model, record_rows_at=checkpoints)
        print(f"\n  hidden activation = {act}")
        for s in checkpoints:
            W = r["rows"][s]
            print(f"    step {s:4d}: row1 = {W[0].tolist()}  row2 = {W[1].tolist()}"
                  f"  identical: {torch.equal(W[0], W[1])}")
        print(f"    final W2 = {model[2].weight.data.tolist()}")
        print(f"    final loss {r['final_loss']:.4f}, correct {r['correct']}/4")
        show_predictions(r)
        out[act] = r
    # Control: non-zero but identical weights -> still symmetric
    model = zero_init(make_net("sigmoid"), value=0.3)
    r = train(model, record_rows_at=checkpoints)
    print("\n  control: every weight and bias = 0.3 (non-zero but identical), sigmoid")
    for s in checkpoints:
        W = r["rows"][s]
        print(f"    step {s:4d}: row1 = {[round(v, 4) for v in W[0].tolist()]}  "
              f"row2 = {[round(v, 4) for v in W[1].tolist()]}  identical: {torch.equal(W[0], W[1])}")
    print(f"    final W2 = {[round(v, 4) for v in model[2].weight.data[0].tolist()]}")
    print(f"    final loss {r['final_loss']:.4f}, correct {r['correct']}/4")
    show_predictions(r)
    # Why nothing moves from exactly zero: the gradient is exactly zero
    m0 = zero_init(make_net("sigmoid"))
    m0.zero_grad()
    nn.BCEWithLogitsLoss()(m0(X), Y).backward()
    print("\n  gradients at the all-zero initialisation (sigmoid hidden layer):")
    for name, prm in m0.named_parameters():
        print(f"    d L / d {name:<9} = {prm.grad.flatten().tolist()}")
    out["control_0.3"] = r
    return out


# ---------------------------------------------------------------------------
def hidden_diagnostics(model, act):
    with torch.no_grad():
        a = model[0](X)
        h = model[1](a)
        if act == "sigmoid":
            d = h * (1 - h)
        elif act == "tanh":
            d = 1 - h ** 2
        else:
            d = (a > 0).float()
    print(f"    {act}: pre-activations a (rows = inputs, cols = hidden units)")
    for xi, ai, hi, di in zip(X.int().tolist(), a.tolist(), h.tolist(), d.tolist()):
        print(f"      x={xi}  a={[round(v, 3) for v in ai]}  h={[round(v, 3) for v in hi]}"
              f"  f'(a)={[round(v, 4) for v in di]}")


def part_d():
    print("\n" + "=" * 74)
    print(f"PART D - activation experiment (same random initialisation, seed {SEED})")
    print("=" * 74)
    table, curves = [], {}
    for act in ("sigmoid", "tanh", "relu"):
        model = make_net(act)
        print(f"\n  {act}: hidden units BEFORE training")
        hidden_diagnostics(model, act)
        r = train(model)
        curves[act] = r["losses"]
        table.append((act, r["final_loss"], r["correct"], r["early_grad_norm"]))
        print(f"\n  {act}: initial loss {r['init_loss']:.4f}  final loss "
              f"{r['final_loss']:.6f}  correct {r['correct']}/4  "
              f"||dL/dW1|| at step 0 = {r['early_grad_norm']:.4f}")
        show_predictions(r)
        print(f"  {act}: hidden units AFTER training")
        hidden_diagnostics(model, act)

    print("\n  Result table")
    print(f"  {'activation':<10} {'final loss':>12} {'4/4?':>6} {'early ||grad W1||':>18}")
    for act, fl, c, g in table:
        print(f"  {act:<10} {fl:>12.6f} {('yes' if c == 4 else f'no ({c}/4)'):>6} {g:>18.4f}")

    # Repeated-run behaviour
    print("\n  Repeated runs: 50 seeds per activation (same settings)")
    rates = {}
    for act in ("sigmoid", "tanh", "relu"):
        ok, norms, losses = 0, [], []
        for seed in range(50):
            r = train(make_net(act, seed=seed))
            ok += r["correct"] == 4
            norms.append(r["early_grad_norm"])
            losses.append(r["final_loss"])
        norms_t = torch.tensor(norms)
        rates[act] = (ok, norms_t.median().item(), norms_t.mean().item())
        print(f"    {act:<8} solved {ok}/50   early ||grad W1||: median "
              f"{norms_t.median():.4f}, mean {norms_t.mean():.4f}")
    return table, curves, rates


def plot_curves(curves, linear_losses):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, ls in curves.items():
        ax.plot(ls, label=f"2-2-1, {name}" + (" (on top of the affine line)" if name == "relu" else ""))
    ax.plot(linear_losses, "--", label="single affine layer")
    ax.axhline(0.6931, color="grey", lw=0.8, ls=":",
               label="ln 2 = 0.693 (always predicting 0.5)")
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel("BCE loss (log scale)")
    ax.set_title(f"XOR training loss (seed {SEED}, Adam lr={LR})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "loss_curves.png"), dpi=130)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


if __name__ == "__main__":
    torch.set_num_threads(1)          # tiny tensors: threading only adds overhead
    os.makedirs(RESULTS, exist_ok=True)
    torch.set_printoptions(precision=6, sci_mode=False)
    with open(os.path.join(RESULTS, "xor_experiments_output.txt"), "w") as f:
        sys.stdout = Tee(sys.__stdout__, f)
        lin, _ = task1_linear_baselines()
        part_a()
        part_b()
        part_c()
        _, curves, _ = part_d()
        plot_curves(curves, lin["losses"])
        sys.stdout = sys.__stdout__
