import torch
import torch.nn as nn
import torch.nn.functional as F

def tensor_to_one_hot(tensor, vocabulary_size):
    
    pad_mask = tensor == -100
    one_hot = torch.zeros(tensor.size(0), tensor.size(1), vocabulary_size)
    one_hot = one_hot.to('cuda')
    tensor_ignored = tensor.clone()
    tensor_ignored[pad_mask] = 0  # Set pad tokens to 0 to avoid scatter_ issues
    one_hot.scatter_(2, tensor_ignored.unsqueeze(2), 1)
    pad_mask = pad_mask.unsqueeze(2)  # Add an extra dimension to the mask
    one_hot[pad_mask.expand_as(one_hot)] = 0
    return one_hot

def compute_el2n(probs, target, log_probs=False, ignore_indices=None):
    
    """
    Computes the 2 norm of the error vector probs - target_one_hot, ignoring indices where target = 1.

    Args:
    - probs: torch.Tensor, shape of (batch size, length, vocab size)
    - target: torch.Tensor, shape of (batch size, length)
    - log_probs: boolean, indicating whether the input probs tensor is log probs or raw probs
    - ignore_indices: list of indices in the batch to ignore, e.g. [0, 1, 3, 5] ignores the first, second, fourth, and sixth data points in the batch

    Returns:
    - el2n: torch.Tensor, shape of (batch size, length)
    """

    if log_probs:
        probs = torch.exp(probs)

    vocab_size = probs.size(-1)
    target_one_hot = tensor_to_one_hot(target, vocab_size)
    
    # Masking out the <pad> tokens and setting their probabilities to zero
    mask = target == -100 
    target_one_hot[mask, :] = 0
    probs[mask, :] = 0

    # Masking out the indices to ignore
    if ignore_indices != None and len(ignore_indices) > 0:
        mask = torch.zeros_like(target_one_hot, dtype=torch.bool)
        mask[ignore_indices, :] = 1
        target_one_hot[mask] = 0
        probs[mask] = 0
    
    el2n = torch.linalg.norm(probs - target_one_hot, dim=-1)
    return el2n


def unit_test_el2n():
    tensor1 = torch.tensor([[1, 2, -100, 3], [0, 4, -100, -100]]) 
    vocabulary_size1 = 5
    one_hot1 = tensor_to_one_hot(tensor1, vocabulary_size1) 

