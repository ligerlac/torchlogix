"""Models module for neurodifflogic."""

from neurodifflogic.models.baseline_nn import FullyConnectedNN
from neurodifflogic.models.difflog_layers.conv import LogicConv2d, LogicConv3d, OrPoolingLayer
from neurodifflogic.models.difflog_layers.gat import LogicGATConv
from neurodifflogic.models.difflog_layers.gcn import CustomMessagePassing, LogicGCNConv
from neurodifflogic.models.difflog_layers.linear import GroupSum, LogicLayer
from neurodifflogic.models.gcn_conv import GCN
from neurodifflogic.models.conv import CNN
from neurodifflogic.models.nn import RandomlyConnectedNN

__all__ = [
    "FullyConnectedNN",
    "LogicConv2d",
    "LogicConv3d",
    "OrPoolingLayer",
    "LogicGATConv",
    "LogicGCNConv",
    "CustomMessagePassing",
    "LogicLayer",
    "GroupSum",
    "GATDifflog",
    "GCN",
    "CNN",
    "RandomlyConnectedNN",
]
