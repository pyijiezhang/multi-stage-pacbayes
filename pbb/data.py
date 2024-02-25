import torch
import numpy as np
import torch.optim as optim
import torch.distributions as td
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


def loaddataset(name):
    """Function to load the datasets (mnist and cifar10)

    Parameters
    ----------
    name : string
        name of the dataset ('mnist' or 'cifar10')

    """
    torch.manual_seed(7)

    if name == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        train = datasets.MNIST(
            "mnist-data/", train=True, download=True, transform=transform
        )
        test = datasets.MNIST(
            "mnist-data/", train=False, download=True, transform=transform
        )
    elif name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        train = datasets.CIFAR10(
            "./data", train=True, download=True, transform=transform
        )
        test = datasets.CIFAR10(
            "./data", train=False, download=True, transform=transform
        )
    else:
        raise RuntimeError(f"Wrong dataset chosen {name}")

    return train, test


def loadbatches(
    train, test, loader_kargs, batch_size, prior=False, perc_train=1.0, perc_prior=0.2
):
    """Function to load the batches for the dataset

    Parameters
    ----------
    train : torch dataset object
        train split

    test : torch dataset object
        test split

    loader_kargs : dictionary
        loader arguments

    batch_size : int
        size of the batch

    prior : bool
        boolean indicating the use of a learnt prior (e.g. this would be False for a random prior)

    perc_train : float
        percentage of data used for training (set to 1.0 if not intending to do data scarcity experiments)

    perc_prior : float
        percentage of data to use for building the prior (1-perc_prior is used to estimate the risk)

    """

    ntrain = len(train.data)
    ntest = len(test.data)

    if prior == False:
        indices = list(range(ntrain))
        split = int(np.round((perc_train) * ntrain))
        random_seed = 10
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        train_idx = indices[:split]
        train_sampler = SubsetRandomSampler(train_idx)

        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, sampler=train_sampler, **loader_kargs
        )
        train1_loader = None
        train2_loader = train_loader
        train2_1batch_loader = torch.utils.data.DataLoader(
            train, batch_size=len(train_idx), sampler=train_sampler, **loader_kargs
        )
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=batch_size, shuffle=True, **loader_kargs
        )
        test_1batch_loader = torch.utils.data.DataLoader(
            test, batch_size=ntest, shuffle=True, **loader_kargs
        )

    else:
        # reduce training data if needed
        new_num_train = int(np.round((perc_train) * ntrain))
        indices = list(range(new_num_train))
        split = int(np.round((perc_prior) * new_num_train))
        random_seed = 10
        np.random.seed(random_seed)
        np.random.shuffle(indices)

        all_train_sampler = SubsetRandomSampler(indices)
        train2_idx, train1_idx = indices[split:], indices[:split]
        train2_sampler = SubsetRandomSampler(train2_idx)
        train1_sampler = SubsetRandomSampler(train1_idx)

        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, sampler=all_train_sampler, shuffle=False
        )
        train1_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, sampler=train1_sampler, shuffle=False
        )
        train2_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, sampler=train2_sampler, shuffle=False
        )
        train2_1batch_loader = torch.utils.data.DataLoader(
            train, batch_size=len(train2_idx), sampler=train2_sampler, **loader_kargs
        )
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=batch_size, shuffle=True, **loader_kargs
        )
        test_1batch_loader = torch.utils.data.DataLoader(
            test, batch_size=ntest, shuffle=True, **loader_kargs
        )

    # train_loader comprises all the data used in training and train1_loader the data used to build
    # the prior
    # train2_1batch_loader and set_bound are the set of data points used to evaluate the bound.
    # the only difference between these two is that onf of them is splitted in multiple batches
    # while the 1batch one is only one batch. This is for computational efficiency with some
    # of the large architectures used.
    # The same is done for test_1batch_loader
    return (
        train_loader,
        test_loader,
        train1_loader,
        train2_1batch_loader,
        test_1batch_loader,
        train2_loader,
    )
