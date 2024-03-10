import math
import numpy as np
import torch
import torch.distributions as td
from tqdm import tqdm, trange
import torch.nn.functional as F
from pbb.eval import get_loss_01


class PBBobj:
    """Class including all functionalities needed to train a NN with a PAC-Bayes inspired
    training objective and evaluate the risk certificate at the end of training.

    Parameters
    ----------
    objective : string
        training objective to be optimised (choices are fquad, flamb, fclassic or fbbb)

    pmin : float
        minimum probability to clamp to have a loss in [0,1]

    classes : int
        number of classes in the learning problem

    train_size : int
        n (number of training examples)

    delta : float
        confidence value for the training objective

    delta_test : float
        confidence value for the chernoff bound (used when computing the risk)

    mc_samples : int
        number of Monte Carlo samples when estimating the risk

    kl_penalty : float
        penalty for the kl coefficient in the training objective

    device : string
        Device the code will run in (e.g. 'cuda')

    """

    def __init__(
        self,
        objective="fquad",
        pmin=1e-4,
        classes=10,
        delta=0.025,
        delta_test=0.01,
        mc_samples=1000,
        kl_penalty=1,
        device="cuda",
        n_posterior=30000,
        n_bound=30000,
        use_delta_bound=False,
        use_approach2=False,
        use_approach3=False,
    ):
        super().__init__()
        self.objective = objective
        self.pmin = pmin
        self.classes = classes
        self.device = device
        self.delta = delta
        self.delta_test = delta_test
        self.mc_samples = mc_samples
        self.kl_penalty = kl_penalty
        self.n_posterior = n_posterior
        self.n_bound = n_bound
        self.use_delta_bound = use_delta_bound
        self.use_approach2 = use_approach2
        self.use_approach3 = use_approach3

    def compute_empirical_risk(self, outputs, targets, bounded=True):
        # compute negative log likelihood loss and bound it with pmin (if applicable)
        empirical_risk = F.nll_loss(outputs, targets)
        if bounded == True:
            empirical_risk = (1.0 / (np.log(1.0 / self.pmin))) * empirical_risk
        return empirical_risk

    def compute_losses(
        self,
        net,
        data,
        target,
        clamping=True,
        net0=None,
        c1=1,
        c2=1,
        js=[-1 / 2, 1 / 2],
    ):
        # compute both cross entropy and 01 loss
        # returns outputs of the network as well

        outputs = net(data, sample=True, clamping=clamping, pmin=self.pmin)
        loss_ce = self.compute_empirical_risk(outputs, target, clamping)
        pred = outputs.max(1, keepdim=True)[1]
        correct = pred.eq(target.view_as(pred)).sum().item()
        total = target.size(0)
        loss_01 = 1 - (correct / total)

        if net0 and self.use_delta_bound:
            net0.eval()
            if self.use_approach2:
                sample_net0 = False
            elif self.use_approach3:
                sample_net0 = True
            loss_01_net0 = get_loss_01(
                net0,
                data,
                target,
                sample=sample_net0,
                clamping=clamping,
                pmin=self.pmin,
            )
            loss_ce_delta = F.nll_loss(outputs * c1, target, reduce=False)
            loss_ce_delta -= loss_01_net0

            loss_delta = []
            for j in js:
                loss_delta.append(F.sigmoid(c2 * (loss_ce_delta - j)).mean())
            # print(loss_delta)
        else:
            loss_delta = None
        return loss_ce, loss_01, outputs, loss_delta

    def bound(self, empirical_risk, kl, train_size, lambda_var=None):
        # compute training objectives
        if not self.use_delta_bound:
            if self.objective == "fquad":
                kl = kl * self.kl_penalty
                repeated_kl_ratio = torch.div(
                    kl + np.log((2 * np.sqrt(train_size)) / self.delta), 2 * train_size
                )
                first_term = torch.sqrt(empirical_risk + repeated_kl_ratio)
                second_term = torch.sqrt(repeated_kl_ratio)
                train_obj = torch.pow(first_term + second_term, 2)
            elif self.objective == "flamb":
                kl = kl * self.kl_penalty
                lamb = lambda_var.lamb_scaled
                kl_term = torch.div(
                    kl + np.log((2 * np.sqrt(train_size)) / self.delta),
                    train_size * lamb * (1 - lamb / 2),
                )
                first_term = torch.div(empirical_risk, 1 - lamb / 2)
                train_obj = first_term + kl_term
            elif self.objective == "fclassic":
                kl = kl * self.kl_penalty
                kl_ratio = torch.div(
                    kl + np.log((2 * np.sqrt(train_size)) / self.delta), 2 * train_size
                )
                train_obj = empirical_risk + torch.sqrt(kl_ratio)
            elif self.objective == "bbb":
                # ipdb.set_trace()
                train_obj = empirical_risk + self.kl_penalty * (kl / train_size)
            else:
                raise RuntimeError(f"Wrong objective {self.objective}")
        else:
            train_obj_total = 0
            # print(empirical_risk)
            for risk_term in empirical_risk:
                if self.objective == "fquad":
                    kl = kl * self.kl_penalty
                    repeated_kl_ratio = torch.div(
                        kl + np.log((2 * np.sqrt(train_size)) / self.delta),
                        2 * train_size,
                    )
                    first_term = torch.sqrt(risk_term + repeated_kl_ratio)
                    second_term = torch.sqrt(repeated_kl_ratio)
                    train_obj = torch.pow(first_term + second_term, 2)
                elif self.objective == "flamb":
                    kl = kl * self.kl_penalty
                    lamb = lambda_var.lamb_scaled
                    kl_term = torch.div(
                        kl + np.log((2 * np.sqrt(train_size)) / self.delta),
                        train_size * lamb * (1 - lamb / 2),
                    )
                    first_term = torch.div(risk_term, 1 - lamb / 2)
                    train_obj = first_term + kl_term
                elif self.objective == "fclassic":
                    kl = kl * self.kl_penalty
                    kl_ratio = torch.div(
                        kl + np.log((2 * np.sqrt(train_size)) / self.delta),
                        2 * train_size,
                    )
                    train_obj = risk_term + torch.sqrt(kl_ratio)
                elif self.objective == "bbb":
                    # ipdb.set_trace()
                    train_obj = risk_term + self.kl_penalty * (kl / train_size)
                else:
                    raise RuntimeError(f"Wrong objective {self.objective}")
                train_obj_total += train_obj
            train_obj = -1 + train_obj_total
        return train_obj

    def mcsampling(
        self, net, input, target, batches=True, clamping=True, data_loader=None
    ):
        # compute empirical risk with Monte Carlo sampling
        error = 0.0
        cross_entropy = 0.0
        if batches:
            for batch_id, (data_batch, target_batch) in enumerate(tqdm(data_loader)):
                data_batch, target_batch = data_batch.to(self.device), target_batch.to(
                    self.device
                )
                cross_entropy_mc = 0.0
                error_mc = 0.0
                for i in range(self.mc_samples):
                    loss_ce, loss_01, _, _ = self.compute_losses(
                        net, data_batch, target_batch, clamping
                    )
                    cross_entropy_mc += loss_ce
                    error_mc += loss_01
                # we average cross-entropy and 0-1 error over all MC samples
                cross_entropy += cross_entropy_mc / self.mc_samples
                error += error_mc / self.mc_samples
            # we average cross-entropy and 0-1 error over all batches
            cross_entropy /= batch_id
            error /= batch_id
        else:
            cross_entropy_mc = 0.0
            error_mc = 0.0
            for i in range(self.mc_samples):
                loss_ce, loss_01, _, _ = self.compute_losses(
                    net, input, target, clamping
                )
                cross_entropy_mc += loss_ce
                error_mc += loss_01
                # we average cross-entropy and 0-1 error over all MC samples
            cross_entropy += cross_entropy_mc / self.mc_samples
            error += error_mc / self.mc_samples
        return cross_entropy, error

    def train_obj(self, net, input, target, clamping=True, lambda_var=None, net0=None):
        # compute train objective and return all metrics
        outputs = torch.zeros(target.size(0), self.classes).to(self.device)
        kl = net.compute_kl()
        loss_ce, loss_01, outputs, loss_delta = self.compute_losses(
            net, input, target, clamping, net0=net0
        )
        # if self.use_delta_bound:
        #     loss_ce = torch.tensor(loss_ce).sum()
        if net0 and self.use_delta_bound:
            train_obj = self.bound(loss_delta, kl, self.n_posterior, lambda_var)
        else:
            train_obj = self.bound(loss_ce, kl, self.n_posterior, lambda_var)
        return train_obj, kl / self.n_posterior, outputs, loss_ce, loss_01

    def compute_final_stats_risk(
        self,
        net,
        input=None,
        target=None,
        data_loader=None,
        clamping=True,
        lambda_var=None,
    ):
        # compute all final stats and risk certificates

        kl = net.compute_kl()
        if data_loader:
            error_ce, error_01 = self.mcsampling(
                net, input, target, batches=True, clamping=True, data_loader=data_loader
            )
        else:
            error_ce, error_01 = self.mcsampling(
                net, input, target, batches=False, clamping=True
            )

        empirical_risk_ce = inv_kl(
            error_ce.item(), np.log(2 / self.delta_test) / self.mc_samples
        )
        empirical_risk_01 = inv_kl(
            error_01, np.log(2 / self.delta_test) / self.mc_samples
        )

        train_obj = self.bound(empirical_risk_ce, kl, self.n_posterior, lambda_var)

        risk_ce = inv_kl(
            empirical_risk_ce,
            (kl + np.log((2 * np.sqrt(self.n_bound)) / self.delta_test)) / self.n_bound,
        )
        risk_01 = inv_kl(
            empirical_risk_01,
            (kl + np.log((2 * np.sqrt(self.n_bound)) / self.delta_test)) / self.n_bound,
        )
        return (
            train_obj.item(),
            kl.item() / self.n_bound,
            empirical_risk_ce,
            empirical_risk_01,
            risk_ce,
            risk_01,
        )


def inv_kl(qs, ks):
    """Inversion of the binary kl

    Parameters
    ----------
    qs : float
        Empirical risk

    ks : float
        second term for the binary kl inversion

    """
    # computation of the inversion of the binary KL
    qd = 0
    ikl = 0
    izq = qs
    dch = 1 - 1e-10
    while (dch - izq) / dch >= 1e-5:
        p = (izq + dch) * 0.5
        if qs == 0:
            ikl = ks - (0 + (1 - qs) * math.log((1 - qs) / (1 - p)))
        elif qs == 1:
            ikl = ks - (qs * math.log(qs / p) + 0)
        else:
            ikl = ks - (qs * math.log(qs / p) + (1 - qs) * math.log((1 - qs) / (1 - p)))
        if ikl < 0:
            dch = p
        else:
            izq = p
        qd = p
    return qd
