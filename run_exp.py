import os
import math
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions as td
from torchvision import datasets, transforms
from tqdm import tqdm, trange
from pbb.models import (
    NNet4l,
    CNNet4l,
    ProbNNet4l,
    ProbCNNet4l,
    ProbCNNet9l,
    CNNet9l,
    CNNet13l,
    ProbCNNet13l,
    ProbCNNet15l,
    CNNet15l,
    trainNNet,
    testNNet,
    Lambda_var,
    trainPNNet,
)
from pbb.bounds import PBBobj
from pbb import data

# TODOS: 1. make a train prior function (bbb, erm)
#        2. make train posterior function
#        3. rename partitions of data (prior_data, posterior_data, eval_data)
#        4. implement early stopping with validation set & speed
#        5. add data augmentation (maria)
#        6. better way of logging


def main(
    name_data,
    objective,
    prior_type,
    model,
    sigma_prior=0.03,
    pmin=1e-5,
    learning_rate=0.005,
    momentum=0.95,
    learning_rate_prior=0.005,
    momentum_prior=0.95,
    delta=0.025,
    layers=9,
    delta_test=0.01,
    mc_samples=1000,
    kl_penalty=1,
    initial_lamb=1.0,
    train_epochs=100,
    prior_dist="gaussian",
    verbose=False,
    device="cuda",
    prior_epochs=100,
    dropout_prob=0.2,
    perc_train=1.0,
    perc_prior=0.5,
    batch_size=250,
):
    """Run an experiment with PAC-Bayes inspired training objectives

    Parameters
    ----------
    name_data : string
        name of the dataset to use (check data file for more info)

    objective : string
        training objective to use

    prior_type : string
        could be rand or learnt depending on whether the prior
        is data-free or data-dependent

    model : string
        could be cnn or fcn

    sigma_prior : float
        scale hyperparameter for the prior

    pmin : float
        minimum probability to clamp the output of the cross entropy loss

    learning_rate : float
        learning rate hyperparameter used for the optimiser

    momentum : float
        momentum hyperparameter used for the optimiser

    learning_rate_prior : float
        learning rate used in the optimiser for learning the prior (only
        applicable if prior is learnt)

    momentum_prior : float
        momentum used in the optimiser for learning the prior (only
        applicable if prior is learnt)

    delta : float
        confidence parameter for the risk certificate

    layers : int
        integer indicating the number of layers (applicable for CIFAR-10,
        to choose between 9, 13 and 15)

    delta_test : float
        confidence parameter for chernoff bound

    mc_samples : int
        number of monte carlo samples for estimating the risk certificate
        (set to 1000 by default as it is more computationally efficient,
        although larger values lead to tighter risk certificates)

    samples_ensemble : int
        number of members for the ensemble predictor

    kl_penalty : float
        penalty for the kl coefficient in the training objective

    initial_lamb : float
        initial value for the lambda variable used in flamb objective
        (scaled later)

    train_epochs : int
        numer of training epochs for training

    prior_dist : string
        type of prior and posterior distribution (can be gaussian or laplace)

    verbose : bool
        whether to print metrics during training

    device : string
        device the code will run in (e.g. 'cuda')

    prior_epochs : int
        number of epochs used for learning the prior (not applicable if prior is rand)

    dropout_prob : float
        probability of an element to be zeroed.

    perc_train : float
        percentage of train data to use for the entire experiment (can be used to run
        experiments with reduced datasets to test small data scenarios)

    verbose_test : bool
        whether to print test and risk certificate stats during training epochs

    perc_prior : float
        percentage of data to be used to learn the prior

    batch_size : int
        batch size for experiments
    """

    exp_settings = f"{name_data}_{objective}_{prior_type}_{kl_penalty}.pt"

    # this makes the initialised prior the same for all bounds
    torch.manual_seed(7)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loader_kargs = (
        {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}
    )

    train, test = data.loaddataset(name_data)
    classes = len(train.classes)
    rho_prior = math.log(math.exp(sigma_prior) - 1.0)

    ##############################################################################################
    ### initialise prior net0 and data-dependent prior pnet1
    ##############################################################################################

    if prior_type == "rand":
        dropout_prob = 0.0

    # initialise net0 and pnet1
    if model == "cnn":
        if name_data == "cifar10":
            if layers == 9:
                net0 = CNNet9l(dropout_prob=dropout_prob).to(device)
                pnet1 = ProbCNNet9l(
                    rho_prior, prior_dist=prior_dist, device=device, init_net=net0
                ).to(device)
            elif layers == 13:
                net0 = CNNet13l(dropout_prob=dropout_prob).to(device)
                pnet1 = ProbCNNet13l(
                    rho_prior, prior_dist=prior_dist, device=device, init_net=net0
                ).to(device)
            elif layers == 15:
                net0 = CNNet15l(dropout_prob=dropout_prob).to(device)
                pnet1 = ProbCNNet15l(
                    rho_prior, prior_dist=prior_dist, device=device, init_net=net0
                ).to(device)
            else:
                raise RuntimeError(f"Wrong number of layers {layers}")
        else:
            net0 = CNNet4l(dropout_prob=dropout_prob).to(device)
            pnet1 = ProbCNNet4l(
                rho_prior, prior_dist=prior_dist, device=device, init_net=net0
            ).to(device)
    elif model == "fcn":
        if name_data == "cifar10":
            raise RuntimeError(f"Cifar10 not supported with given architecture {model}")
        elif name_data == "mnist":
            net0 = NNet4l(dropout_prob=dropout_prob).to(device)
            pnet1 = ProbNNet4l(
                rho_prior, prior_dist=prior_dist, device=device, init_net=net0
            ).to(device)
    else:
        raise RuntimeError(f"Architecture {model} not supported")

    # save net0
    if not os.path.exists("./saved_models"):
        os.makedirs("./saved_models")
    dir_net0 = f"./saved_models/net0_" + exp_settings
    torch.save(net0, dir_net0)

    ##############################################################################################
    ### train data-dependent prior pnet1
    ##############################################################################################
    if prior_type == "rand":
        # prepare data
        (
            train_loader,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = data.loadbatches(
            train,
            test,
            loader_kargs,
            batch_size,
            prior=False,
            perc_train=perc_train,
            perc_prior=perc_prior,
        )

        # save pnet1
        dir_pnet1 = f"./saved_models/pnet1_" + exp_settings
        torch.save(pnet1, dir_pnet1)
    elif prior_type == "learnt":
        # prepare data
        (
            train_loader,
            _,
            train1_loader,
            _,
            _,
            _,
            _,
            _,
        ) = data.loadbatches(
            train,
            test,
            loader_kargs,
            batch_size,
            prior=True,
            perc_train=perc_train,
            perc_prior=perc_prior,
        )

        n1 = len(train1_loader.dataset)
        bound = PBBobj(
            objective,
            pmin,
            classes,
            delta,
            delta_test,
            mc_samples,
            kl_penalty,
            device,
            n_posterior=n1,
            n_bound=1,  # set to 1, not related to training
        )

        if objective == "flamb":
            lambda_var = Lambda_var(initial_lamb, n1).to(device)
            optimizer_lambda = optim.SGD(
                lambda_var.parameters(), lr=learning_rate, momentum=momentum
            )
        else:
            optimizer_lambda = None
            lambda_var = None

        optimizer = optim.SGD(
            pnet1.parameters(), lr=learning_rate_prior, momentum=momentum_prior
        )

        for epoch in trange(prior_epochs):
            trainPNNet(
                pnet1,
                optimizer,
                bound,
                epoch,
                train1_loader,
                lambda_var,
                optimizer_lambda,
                verbose,
            )

        # save pnet1
        dir_pnet1 = f"./saved_models/pnet1_" + exp_settings
        torch.save(pnet1, dir_pnet1)

    ##############################################################################################
    ### train posterior
    ##############################################################################################

    n = len(train_loader.dataset)

    if model == "cnn":
        if name_data == "cifar10":
            if layers == 9:
                pnet2 = ProbCNNet9l(
                    rho_prior, prior_dist=prior_dist, device=device, init_pnet=pnet1
                ).to(device)
            elif layers == 13:
                pnet2 = ProbCNNet13l(
                    rho_prior, prior_dist=prior_dist, device=device, init_pnet=pnet1
                ).to(device)
            elif layers == 15:
                pnet2 = ProbCNNet15l(
                    rho_prior, prior_dist=prior_dist, device=device, init_pnet=pnet1
                ).to(device)
            else:
                raise RuntimeError(f"Wrong number of layers {layers}")
        else:
            pnet2 = ProbCNNet4l(
                rho_prior, prior_dist=prior_dist, device=device, init_pnet=pnet1
            ).to(device)
    elif model == "fcn":
        if name_data == "cifar10":
            raise RuntimeError(f"Cifar10 not supported with given architecture {model}")
        elif name_data == "mnist":
            pnet2 = ProbNNet4l(
                rho_prior, prior_dist=prior_dist, device=device, init_pnet=pnet1
            ).to(device)
    else:
        raise RuntimeError(f"Architecture {model} not supported")

    # pnet2 = torch.load(
    #     dir_pnet1,
    #     map_location=torch.device(device),
    # )

    bound = PBBobj(
        objective,
        pmin,
        classes,
        delta,
        delta_test,
        mc_samples,
        kl_penalty,
        device,
        n_posterior=n,
        n_bound=1,  # set to 1, not related to training
    )

    if objective == "flamb":
        lambda_var = Lambda_var(initial_lamb, n).to(device)
        optimizer_lambda = optim.SGD(
            lambda_var.parameters(), lr=learning_rate, momentum=momentum
        )
    else:
        optimizer_lambda = None
        lambda_var = None

    optimizer = optim.SGD(pnet2.parameters(), lr=learning_rate, momentum=momentum)

    for epoch in trange(train_epochs):
        trainPNNet(
            pnet2,
            optimizer,
            bound,
            epoch,
            train_loader,
            lambda_var,
            optimizer_lambda,
            verbose,
        )

    dir_pnet2 = f"./saved_models/net2_" + exp_settings
    torch.save(pnet2, dir_pnet2)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
