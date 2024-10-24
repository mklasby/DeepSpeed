from typing import Callable
from math import prod
import logging

import torch

_logger = logging.getLogger(__name__)

def sparse_flops(sparse_fn: Callable)-> Callable:
    def sparse_flops_wrapper(orig_fn: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            for a in args:
                if isinstance(a, torch.Tensor) and hasattr(a, "mask"):
                    return sparse_fn(*args, **kwargs)
            for v in kwargs.values():
                if isinstance(v, torch.Tensor) and hasattr(v, "mask"):
                    return sparse_fn(*args, **kwargs)
            # no sparsity, call original
            return orig_fn(*args, **kwargs)
        return wrapper
    return sparse_flops_wrapper


def _calculate_sparsity(t):
    s = 0
    if hasattr(t, "mask"):
        s = ((t.mask == 0).sum() / t.numel()).item()
    return s

def sparse_matmul_flops_compute(input, other, *, out=None):
    """
    # Count flops for the matmul operation.
    """
    input_sparsity = _calculate_sparsity(input)
    other_sparsity = _calculate_sparsity(other)
    _logger.info("Tracing sparse matmul...")
    _logger.info(
        f"Input sparsity {input_sparsity}, other sparsity {other_sparsity}"
    )
    _logger.info(f"Input shape {input.shape}, other shape {other.shape}")
    macs = (
        prod(input.shape)
        * other.shape[-1]
        * (1 - other_sparsity)
        * (1 - input_sparsity)
    )
    return 2 * macs, macs


def sparse_linear_flops_compute(input, weight, bias=None):
    out_features = weight.shape[0]
    input_sparsity = _calculate_sparsity(input)
    weight_sparsity = _calculate_sparsity(weight)
    _logger.info("Tracing sparse linear:")
    _logger.info(
        f"Input sparsity {input_sparsity}, weight sparsity {weight_sparsity}"
    )
    _logger.info(f"Input shape {input.shape}, weight shape {weight.shape}")
    macs = (
        input.numel()
        * out_features
        * (1 - weight_sparsity)
        * (1 - input_sparsity)
    )
    return 2 * macs, macs