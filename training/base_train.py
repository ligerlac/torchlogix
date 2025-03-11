import torch
import numpy as np
from models.difflog_layers.linear import PackBitsTensor

def train(model, x, y, loss_fn, optimizer):
    model.train()
    x = model(x)
    loss = loss_fn(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def eval(model, loader, mode, device='cuda'):
    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=mode)
        res = np.mean(
            [
                (model(x.to(device).round()).argmax(-1) == y.to(device)).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()


def packbits_eval(model, loader, device='cuda'):
    orig_mode = model.training
    with torch.no_grad():
        model.eval()
        res = np.mean(
            [
                (model(PackBitsTensor(x.to(device).reshape(x.shape[0], -1).round().bool())).argmax(-1) == y.to(
                    device)).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()