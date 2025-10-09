import math

import torch
import torchvision
from torch_geometric.datasets import NELL, Planetoid
from torch_geometric.transforms import BaseTransform

from . import mnist_dataset
from . import uci_datasets


class MinMaxNormalize(BaseTransform):
    """Normalize the node features to be between 0 and 1."""

    def __call__(self, data):
        x = data.x  # Node features
        min_vals = x.min(dim=0, keepdim=True)[0]  # Min for each feature
        max_vals = x.max(dim=0, keepdim=True)[0]  # Max for each feature
        data.x = (x - min_vals) / (
            max_vals - min_vals + 1e-10
        )  # Avoid division by zero
        return data


BITS_TO_TORCH_FLOATING_POINT_TYPE = {
    16: torch.float16,
    32: torch.float32,
    64: torch.float64,
}

IMPL_TO_DEVICE = {"cuda": "cuda", "python": "cpu"}
    

class BooleanTransform:
    def __call__(self, sample):
        return sample.bool().float()


def load_dataset(args):
    """Load a public dataset."""
    validation_loader = None
    if args.dataset == "adult":
        train_set = uci_datasets.AdultDataset(
            "./data-uci", split="train", download=True, with_val=False
        )
        test_set = uci_datasets.AdultDataset("./data-uci", split="test", with_val=False)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=int(1e6), shuffle=False
        )
    elif args.dataset == "breast_cancer":
        train_set = uci_datasets.BreastCancerDataset(
            "./data-uci", split="train", download=True, with_val=False
        )
        test_set = uci_datasets.BreastCancerDataset(
            "./data-uci", split="test", with_val=False
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=int(1e6), shuffle=False
        )
    elif args.dataset.startswith("monk"):
        style = int(args.dataset[4])
        train_set = uci_datasets.MONKsDataset(
            "./data-uci", style, split="train", download=True, with_val=False
        )
        test_set = uci_datasets.MONKsDataset(
            "./data-uci", style, split="test", with_val=False
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=int(1e6), shuffle=False
        )
    elif args.dataset in ["mnist", "mnist20x20"]:
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                BooleanTransform(),  # Binarize the input
            ]
        )        
        train_set = mnist_dataset.MNIST(
            "./data-mnist",
            train=True,
            download=True,
            remove_border=args.dataset == "mnist20x20",
            transform=transforms
        )
        test_set = mnist_dataset.MNIST(
            "./data-mnist", train=False, remove_border=args.dataset == "mnist20x20", transform=transforms
        )

        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(
            train_set, [train_set_size, valid_set_size]
        )

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size // 10,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
    elif "cifar-10" in args.dataset:
        transform = {
            "cifar-10-3-thresholds": lambda x: torch.cat(
                [(x > (i + 1) / 4).float() for i in range(3)], dim=0
            ),
            "cifar-10-31-thresholds": lambda x: torch.cat(
                [(x > (i + 1) / 32).float() for i in range(31)], dim=0
            ),
        }[args.dataset]
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(transform),
            ]
        )
        train_set = torchvision.datasets.CIFAR10(
            "./data-cifar", train=True, download=True, transform=transforms
        )
        test_set = torchvision.datasets.CIFAR10(
            "./data-cifar", train=False, transform=transforms
        )
        train_set_size = math.ceil((1 - args.valid_set_size) * len(train_set))
        valid_set_size = len(train_set) - train_set_size
        train_set, validation_set = torch.utils.data.random_split(
            train_set, [train_set_size, valid_set_size]
        )

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=4,
        )
        validation_loader = torch.utils.data.DataLoader(
            validation_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=True,
        )
    elif args.dataset in ["cora", "citeseer", "pubmed"]:
        print("ARGS.DATASET", args.dataset)
        dataset = Planetoid(
            root=f"./data-{args.dataset}",
            name=args.dataset,
            split="public",
            transform=MinMaxNormalize(),
        )
        data = dataset[0]
        return data, None, None
    elif "nell" in args.dataset:
        dataset = NELL(root="data/nell")
        data = dataset[0]  # Get the first (and only) graph object
        # needs to be dense for indexing in logic layer
        data.x = data.x.to_dense()
        return data, None, None
    else:
        raise NotImplementedError(f"The data set {args.dataset} is not supported!")

    return train_loader, validation_loader, test_loader


def load_n(loader, n):
    i = 0
    while i < n:
        for x in loader:
            yield x
            i += 1
            if i == n:
                break


def input_dim_of_dataset(dataset):
    return {
        "adult": 116,
        "breast_cancer": 51,
        "monk1": 17,
        "monk2": 17,
        "monk3": 17,
        "mnist": 784,
        "mnist20x20": 400,
        "cicada": 720,
        "cifar-10-3-thresholds": 3 * 32 * 32 * 3,
        "cifar-10-31-thresholds": 3 * 32 * 32 * 31,
        "cora": 1433,
        "pubmed": 500,
        "citeseer": 3703,
        "nell": 5414,
    }[dataset]


def num_classes_of_dataset(dataset):
    return {
        "adult": 2,
        "breast_cancer": 2,
        "monk1": 2,
        "monk2": 2,
        "monk3": 2,
        "mnist": 10,
        "mnist20x20": 10,
        "cicada": 1,
        "cifar-10-3-thresholds": 10,
        "cifar-10-31-thresholds": 10,
        "cora": 7,
        "pubmed": 3,
        "citeseer": 6,
        "nell": 210,
    }[dataset]
