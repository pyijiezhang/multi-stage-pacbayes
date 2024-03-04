import torch
from pbb.bounds import inv_kl
import numpy as np
from scipy.stats import binom


def get_l_01(net, input, target, sample=True, clamping=True, pmin=1e-5):
    """Compute 0-1 loss for h given S."""
    net.eval()
    with torch.no_grad():
        outputs = net(input, sample=sample, clamping=clamping, pmin=pmin)
        pred = outputs.max(1)[1]
        l_01 = (pred != target).long()
    return l_01


def get_delta_j(l_01_S, l_01_S1):
    """Compute delta_j^{\hat}."""
    delta_js = []
    for j in [0, 1]:
        delta_j = ((l_01_S - l_01_S1) < j).float().mean()
        delta_js.append(delta_j)
    return torch.tensor(delta_js)


def mcsampling_delta(
    net,
    net0,
    mc_samples,
    mc_samples0,
    input,
    target,
    clamping=True,
    pmin=1e-5,
):
    """Compute expectation of delta_j^{\hat} using MC sampling."""

    delta_js = torch.zeros(2)
    # sample from prior
    # approach 2
    if mc_samples0 == 1:
        sample = False
        # approach 3
    else:
        sample = True
    for _ in range(mc_samples0):
        l_01_S1 = get_l_01(
            net0, input, target, sample=sample, clamping=clamping, pmin=pmin
        )
        # sample from posterior
        for _ in range(mc_samples):
            l_01_S = get_l_01(
                net, input, target, sample=True, clamping=clamping, pmin=pmin
            )
            delta_js_mc = get_delta_j(l_01_S, l_01_S1)
            delta_js += delta_js_mc
    return delta_js / (mc_samples * mc_samples0)


def get_binominal_inv(n, k, delta):
    for p in np.linspace(0, 1, 100001):
        if binom.pmf(k, n, p) >= delta:
            return p


def compute_final_stats_risk_delta(
    net,
    net0,
    mc_samples,
    mc_samples0,
    input,
    target,
    clamping=True,
    pmin=1e-5,
    delta_test=0.01,
):

    kl = net.compute_kl()
    delta_js = mcsampling_delta(net, net0, mc_samples, mc_samples0, input, target)

    n_bound = input.shape[0]

    # approach 2
    if mc_samples0 == 1:
        inv_2 = 0
        for i in range(2):
            inv_1 = inv_kl(
                delta_js[i], np.log(2 / delta_test) / (mc_samples * mc_samples0)
            )
            # clamp for numerical issue
            if inv_1 > 0.9999:
                inv_1 = 0.9999
            inv_2 += inv_kl(
                inv_1,
                (kl + np.log((6 * np.sqrt(n_bound)) / delta_test)) / n_bound,
            )
        wrong_net0 = (
            get_l_01(net0, input, target, sample=False, clamping=clamping, pmin=pmin)
            .sum()
            .item()
        )
        binominal_inv = get_binominal_inv(n_bound, wrong_net0, delta_test / 3)
        risk_delta = -1 + inv_2 + binominal_inv
    # approach 3
    else:
        pass

    return risk_delta.item()


def get_kl_q_p(mu_q, sigma_q, mu_p, sigma_p):
    q = torch.distributions.normal.Normal(mu_q, sigma_q)
    p = torch.distributions.normal.Normal(mu_p, sigma_p)
    return torch.distributions.kl.kl_divergence(q, p).mean().item()
