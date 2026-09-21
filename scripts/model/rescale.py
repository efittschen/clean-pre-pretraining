"""Body reparametrization W = c * W' (CONFIG.RESCALE_STD): c = std(W) / RESCALE_STD
fixed, W' trained at init scale."""
import torch
from torch.nn.utils import parametrize


class FixedScale(torch.nn.Module):

    def __init__(self, c):
        super().__init__()
        self.register_buffer("c", torch.tensor(float(c)))

    def forward(self, x):
        return self.c.to(x.dtype) * x


def rescaled_matrices(model):
    return [(name, module) for name, module in model.model.layers.named_modules()
            if isinstance(module, torch.nn.Linear)]


def apply_rescale(model, target_std):
    constants = {}
    for name, module in rescaled_matrices(model):
        with torch.no_grad():
            c = (module.weight.float().std() / target_std).item()
            module.weight.div_(c)
        parametrize.register_parametrization(module, "weight", FixedScale(c))
        constants[f"model.layers.{name}.weight"] = c
    return constants


def bake_rescale(model):
    for _name, module in rescaled_matrices(model):
        if parametrize.is_parametrized(module, "weight"):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
