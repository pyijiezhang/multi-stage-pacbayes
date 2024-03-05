import torch
import numpy as np
from scipy.stats import binom
from scipy import optimize
from math import log
from tqdm import trange


def get_loss_01(pi, input, target, sample=True, clamping=True, pmin=1e-5):
    """Compute 0-1 loss of h~pi(h) on S_2.

    Args:
        pi: A probabilistic net. Set 'sample' to get stochastic or deterministic output.
    Returns:
        A tensor with shape (n_2), where n_2 is the size of the S_2. Each entry is 0 (predict correctly)
        or 1 (predict wrong).
    """
    outputs = pi(input, sample=sample, clamping=clamping, pmin=pmin)
    pred = outputs.max(1)[1]
    loss_01 = (pred != target).long()
    return loss_01


def get_delta_j(loss_01_pi_S, loss_01_pi_S_1):
    """Compute the excess loss delta_j^{\hat} over the data S_2.

    Args:
        loss_01_pi_S: A tensor with shape (n_2). 0-1 loss of pi_S.
        loss_01_pi_S_1: A tensor with shape (n_2). 0-1 loss of pi_S_1.
    Returns:
        A tensor with shape (2). The excess loss delta_j^{\hat} over the data S_2, where the
        first entry is for j=0 and the second is for j=1.
    """
    delta_js = []
    for j in [0, 1]:
        # compute the indicator function and then average
        delta_j = ((loss_01_pi_S - loss_01_pi_S_1) >= j).float().mean()
        delta_js.append(delta_j)
    return torch.tensor(delta_js)


def mcsampling_delta(
    pi_S,
    pi_S_1,
    mc_samples_pi_S,
    mc_samples_pi_S_1,
    input,
    target,
    clamping=True,
    pmin=1e-5,
):
    """Compute expectation of delta_j^{\hat} using MC sampling.

    Returns:
        A tensor with shape (2). The expected excess loss delta_j^{\hat} over the data S_2, where the
        first entry is for j=0 and the second is for j=1.
    """

    delta_js = torch.zeros(2)
    if mc_samples_pi_S_1 == 1:  # approach 2
        sample = False
    else:  # approach 3
        sample = True
    # sample from prior
    for _ in trange(mc_samples_pi_S_1):
        loss_01_pi_S_1 = get_loss_01(
            pi_S_1, input, target, sample=sample, clamping=clamping, pmin=pmin
        )
        # sample from posterior
        for _ in trange(mc_samples_pi_S):
            loss_01_pi_S = get_loss_01(
                pi_S, input, target, sample=True, clamping=clamping, pmin=pmin
            )
            delta_js_mc = get_delta_j(loss_01_pi_S, loss_01_pi_S_1)
            delta_js += delta_js_mc
    return delta_js.numpy() / (mc_samples_pi_S * mc_samples_pi_S_1)


def get_binominal_inv(n, k, delta):
    for p in np.linspace(1, 0, 100001):
        if binom.pmf(k, n, p) >= delta:
            return p


def compute_final_stats_risk_delta(
    pi_S,
    pi_S_1,
    mc_samples_pi_S,
    mc_samples_pi_S_1,
    input,
    target,
    n,  # size of S
    kl_approach3,
    mc_samples_pi_S_1_approach3,
    clamping=True,
    pmin=1e-5,
    delta_test=0.01,
):

    pi_S.eval()
    pi_S_1.eval()

    n_2 = input.shape[0]

    with torch.no_grad():
        kl = pi_S.compute_kl().detach().item()
        delta_js_expected = mcsampling_delta(
            pi_S, pi_S_1, mc_samples_pi_S, mc_samples_pi_S_1, input, target
        )

        inv_2 = 0
        for i in range(2):
            inv_1 = solve_kl_sup(
                delta_js_expected[i],
                np.log(2 / delta_test) / (mc_samples_pi_S * mc_samples_pi_S_1),
            )
            inv_2 += solve_kl_sup(
                inv_1,
                (kl + np.log((6 * np.sqrt(n_2)) / delta_test)) / n_2,
            )

        # approach 2
        if kl_approach3 == None:
            wrong_net0 = (
                get_loss_01(
                    pi_S_1, input, target, sample=False, clamping=clamping, pmin=pmin
                )
                .sum()
                .item()
            )
            binominal_inv = get_binominal_inv(n_2, wrong_net0, delta_test / 3)
            risk_delta = -1 + inv_2 + binominal_inv
        else:
            # approach 3
            loss_01_pi_0 = 0
            for _ in trange(mc_samples_pi_S_1_approach3):
                loss_01_pi_0 += (
                    get_loss_01(
                        pi_S_1, input, target, sample=True, clamping=True, pmin=1e-5
                    )
                    .float()
                    .mean()
                    .item()
                )
            loss_01_pi_0 /= mc_samples_pi_S_1_approach3
            inv_3 = solve_kl_sup(
                loss_01_pi_0, (kl_approach3 + np.log((6 * np.sqrt(n)) / delta_test)) / n
            )
            risk_delta = -1 + inv_2 + inv_3
    return risk_delta


def get_kl_q_p(mu_q, sigma_q, mu_p, sigma_p):
    q = torch.distributions.normal.Normal(mu_q, sigma_q)
    p = torch.distributions.normal.Normal(mu_p, sigma_p)
    return torch.distributions.kl.kl_divergence(q, p).sum().item()


def KL(Q, P):
    """
    Compute Kullback-Leibler (KL) divergence between distributions Q and P.
    """
    return sum([q * log(q / p) if q > 0.0 else 0.0 for q, p in zip(Q, P)])


def KL_binomial(q, p):
    """
    Compute the KL-divergence between two Bernoulli distributions of probability
    of success q and p. That is, Q=(q,1-q), P=(p,1-p).
    """
    return KL([q, 1.0 - q], [p, 1.0 - p])


def solve_kl_sup(q, right_hand_side):
    """
    find x such that:
        kl( q || x ) = right_hand_side
        x > q
    """
    f = lambda x: KL_binomial(q, x) - right_hand_side

    if f(1.0 - 1e-9) <= 0.0:
        return 1.0 - 1e-9
    else:
        return optimize.brentq(f, q, 1.0 - 1e-11)
