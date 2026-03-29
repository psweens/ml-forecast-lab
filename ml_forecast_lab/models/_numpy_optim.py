"""
Pure NumPy optimisation utilities for neural network backends.

Provides activation functions, weight initialisation, and the Adam optimiser
implementation for LSTM and CNN models. All operations use NumPy only, with
numerical stability safeguards.
"""

from typing import Dict, Tuple
import numpy as np


# ============================================================================
# Activation Functions
# ============================================================================


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Compute sigmoid activation with numerical stability.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Sigmoid output in [0, 1].

    Notes
    -----
    Clips input to [-500, 500] to prevent overflow in exp().
    """
    x_clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def tanh(x: np.ndarray) -> np.ndarray:
    """
    Compute hyperbolic tangent activation.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        Tanh output in [-1, 1].
    """
    return np.tanh(x)


def relu(x: np.ndarray) -> np.ndarray:
    """
    Compute ReLU (Rectified Linear Unit) activation.

    Parameters
    ----------
    x : np.ndarray
        Input array.

    Returns
    -------
    np.ndarray
        ReLU output (max(0, x)).
    """
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """
    Compute ReLU gradient.

    Parameters
    ----------
    x : np.ndarray
        Input array to ReLU.

    Returns
    -------
    np.ndarray
        Gradient (1 where x > 0, else 0).
    """
    return (x > 0).astype(np.float32)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute softmax with numerical stability.

    Parameters
    ----------
    x : np.ndarray
        Input array.
    axis : int, optional
        Axis along which to normalise. Default is -1 (last axis).

    Returns
    -------
    np.ndarray
        Softmax probabilities.

    Notes
    -----
    Subtracts max along axis to prevent overflow.
    """
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ============================================================================
# Weight Initialisation
# ============================================================================


def xavier_init(fan_in: int, fan_out: int, seed: int = None) -> np.ndarray:
    """
    Initialise weights using Xavier (Glorot) uniform distribution.

    Parameters
    ----------
    fan_in : int
        Number of input units.
    fan_out : int
        Number of output units.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Initialised weight matrix of shape (fan_in, fan_out).

    Notes
    -----
    Samples uniformly from [-limit, limit] where limit = sqrt(6 / (fan_in + fan_out)).
    Suitable for tanh and sigmoid activations.
    """
    if seed is not None:
        np.random.seed(seed)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)


def he_init(fan_in: int, seed: int = None) -> np.ndarray:
    """
    Initialise weights using He (Kaiming) normal distribution.

    Parameters
    ----------
    fan_in : int
        Number of input units.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Initialised weight matrix of shape (fan_in, fan_in).

    Notes
    -----
    Samples from normal distribution with std = sqrt(2 / fan_in).
    Suitable for ReLU activations.
    """
    if seed is not None:
        np.random.seed(seed)
    std = np.sqrt(2.0 / fan_in)
    return np.random.normal(0, std, size=(fan_in, fan_in)).astype(np.float32)


def zeros_init(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Initialise array with zeros.

    Parameters
    ----------
    shape : tuple of int
        Shape of output array.

    Returns
    -------
    np.ndarray
        Zero-initialised array.
    """
    return np.zeros(shape, dtype=np.float32)


def ones_init(shape: Tuple[int, ...]) -> np.ndarray:
    """
    Initialise array with ones.

    Parameters
    ----------
    shape : tuple of int
        Shape of output array.

    Returns
    -------
    np.ndarray
        One-initialised array.
    """
    return np.ones(shape, dtype=np.float32)


# ============================================================================
# Gradient Utilities
# ============================================================================


def gradient_clip(grads: Dict[str, np.ndarray], max_norm: float = 5.0) -> Dict[str, np.ndarray]:
    """
    Clip gradients to prevent exploding gradients.

    Parameters
    ----------
    grads : dict[str, np.ndarray]
        Dictionary mapping parameter names to gradient arrays.
    max_norm : float, optional
        Maximum norm threshold. Default is 5.0.

    Returns
    -------
    dict[str, np.ndarray]
        Clipped gradients.

    Notes
    -----
    Computes global norm across all gradients and scales if it exceeds max_norm.
    """
    total_norm = 0.0
    for grad in grads.values():
        if grad is not None:
            total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-8)
        clipped = {}
        for name, grad in grads.items():
            clipped[name] = grad * scale if grad is not None else None
        return clipped
    return grads


# ============================================================================
# Adam Optimiser
# ============================================================================


class Adam:
    """
    Adam optimiser implementation in pure NumPy.

    Maintains exponential moving averages of gradients (m) and squared
    gradients (v) for each parameter, with bias correction.

    Parameters
    ----------
    learning_rate : float, optional
        Step size. Default is 0.001.
    beta1 : float, optional
        Exponential decay rate for 1st moment estimates. Default is 0.9.
    beta2 : float, optional
        Exponential decay rate for 2nd moment estimates. Default is 0.999.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-8.

    Attributes
    ----------
    learning_rate : float
        Current learning rate.
    beta1 : float
        Decay rate for first moment.
    beta2 : float
        Decay rate for second moment.
    epsilon : float
        Numerical stability constant.
    t : int
        Timestep counter (incremented each update).
    m : dict[str, np.ndarray]
        First moment estimates (mean of gradients).
    v : dict[str, np.ndarray]
        Second moment estimates (mean of squared gradients).
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialise Adam optimiser."""
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

    def step(self, params: Dict[str, np.ndarray], grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Perform one optimisation step.

        Parameters
        ----------
        params : dict[str, np.ndarray]
            Current parameter values.
        grads : dict[str, np.ndarray]
            Gradients of loss w.r.t. parameters.

        Returns
        -------
        dict[str, np.ndarray]
            Updated parameters.

        Notes
        -----
        Implements bias correction for m and v estimates.
        """
        self.t += 1
        updated_params = {}

        for name, param in params.items():
            if name not in grads or grads[name] is None:
                updated_params[name] = param
                continue

            grad = grads[name]

            # Initialise moments on first call for this parameter
            if name not in self.m:
                self.m[name] = np.zeros_like(param)
                self.v[name] = np.zeros_like(param)

            # Update biased first moment estimate
            self.m[name] = self.beta1 * self.m[name] + (1 - self.beta1) * grad

            # Update biased second moment estimate
            self.v[name] = self.beta2 * self.v[name] + (1 - self.beta2) * (grad ** 2)

            # Bias-corrected first moment estimate
            m_corrected = self.m[name] / (1 - self.beta1 ** self.t)

            # Bias-corrected second moment estimate
            v_corrected = self.v[name] / (1 - self.beta2 ** self.t)

            # Update parameters
            updated_params[name] = param - self.learning_rate * m_corrected / (
                np.sqrt(v_corrected) + self.epsilon
            )

        return updated_params

    def reset(self) -> None:
        """Reset optimiser state (timestep, moments)."""
        self.t = 0
        self.m = {}
        self.v = {}
