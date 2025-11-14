import torch
import numpy as np
from torch.linalg import svd
import tqdm
from typing import List, Tuple, Dict, Optional, Union

def svd_steering(
    grad_matrix: torch.Tensor,
    # target_layer: int,
    low_dim: int = 512,
    rank: int = 2,
    beta: float = 1.0,
) -> np.ndarray:
    """
    Generate a steering vector using SVD on projected gradients.

    Args:
        grad_matrix: Tensor of shape [num_pairs, hidden_dim] containing gradients
        low_dim: (Unused) Kept for API compatibility.

    Returns:
        The steering vector as a numpy array.
    """
    # 1. Random projection is not necessary and adds noise. SVD can run on the full matrix.
    # 2. Reconstructing and averaging the vectors is not the standard way to get the principal direction.
    #    The principal direction is simply the first right singular vector (Vh[0]).

    # Center the gradients
    grad_matrix = grad_matrix - grad_matrix.mean(dim=0, keepdim=True)

    # Apply SVD to the centered gradient matrix (full_matrices=False is faster for extracting 1st PC)
    _U, _S, Vh = svd(grad_matrix, full_matrices=False)

    # The first right singular vector is the direction of highest variance
    steering_vector = Vh[0]

    return steering_vector

