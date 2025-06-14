import numpy as np


def symmetrize(predictions):
    """
    Given an array of shape (N_samples, 9) where each row represents
    a flattened 3x3 tensor,
    this function returns the symmetric part of each tensor,
    also flattened to (N_samples, 9).

    Parameters:
    predictions (np.ndarray): Array of shape (N_samples, 9)

    Returns:
    np.ndarray: Array of shape (N_samples, 9) containing the symmetric parts
    of the tensors.
    """
    # Reshape to (N_samples, 3, 3)
    tensors = predictions.reshape(-1, 3, 3)
    # Compute the symmetric part: (A + A^T)/2 for each tensor
    symmetric_tensors = (tensors + np.transpose(tensors, axes=(0, 2, 1))) / 2
    # Return flattened back to (N_samples, 9)
    return symmetric_tensors.reshape(-1, 9)


# define the transformation matrix for cartesian to spherical tensorial components
T_sym_np = np.array(
    [
        [-1 / np.sqrt(3), 0, 0, 0, -1 / np.sqrt(3), 0, 0, 0, -1 / np.sqrt(3)],  # l_0_0
        [0, 0, -1 / 2 * (-np.sqrt(2)), 0, 0, 0, 1 / 2 * (-np.sqrt(2)), 0, 0],  # Y_1_-1
        [0, 1 / 2 * (-np.sqrt(2)), 0, -1 / 2 * (-np.sqrt(2)), 0, 0, 0, 0, 0],  # Y_1_0
        [0, 0, 0, 0, 0, 1 / 2 * (-np.sqrt(2)), 0, -1 / 2 * (-np.sqrt(2)), 0],  # Y_1_1
        [0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0, 0, 0, 0, 0],  # -2
        [0, 0, 0, 0, 0, 1 / np.sqrt(2), 0, 1 / np.sqrt(2), 0],
        [-1 / np.sqrt(6), 0, 0, 0, -1 / np.sqrt(6), 0, 0, 0, 2 / np.sqrt(6)],
        [0, 0, 1 / np.sqrt(2), 0, 0, 0, 1 / np.sqrt(2), 0, 0],
        [1 / np.sqrt(2), 0, 0, 0, -1 / np.sqrt(2), 0, 0, 0, 0],
    ]
)

T_sym_np_inv = np.linalg.inv(T_sym_np)
