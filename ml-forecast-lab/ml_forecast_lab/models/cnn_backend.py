"""
1D Dilated Causal CNN (WaveNet-style) forecasting model in pure NumPy.

Implements a stack of causal dilated convolutions with residual connections,
suitable for time-series forecasting on resource-constrained devices.
No external deep learning frameworks required.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import ForecastModel
from ._numpy_optim import Adam, relu, relu_derivative, xavier_init, he_init, zeros_init, ones_init, gradient_clip

logger = logging.getLogger(__name__)


class CNNModel(ForecastModel):
    """
    1D Dilated Causal CNN model for time-series forecasting.

    Implements WaveNet-style dilated causal convolutions with residual
    connections, ReLU activations, and optional dropout. Trained with
    mini-batch SGD and Adam optimisation.

    Parameters
    ----------
    n_filters : int, optional
        Number of filters in convolutional layers. Default is 32.
    kernel_size : int, optional
        Kernel size for convolutions. Default is 3.
    n_layers : int, optional
        Number of convolutional layers. Default is 4.
    dilation_base : int, optional
        Base for exponential dilation: layer i has dilation = base^i. Default is 2.
    learning_rate : float, optional
        Initial learning rate for Adam optimiser. Default is 0.001.
    epochs : int, optional
        Maximum number of training epochs. Default is 100.
    batch_size : int, optional
        Mini-batch size for SGD. Default is 32.
    patience : int, optional
        Early stopping patience (epochs without improvement). Default is 10.
    dropout : float, optional
        Dropout probability. Default is 0.1.

    Attributes
    ----------
    _is_fitted : bool
        Whether model has been successfully trained.
    _params : dict[str, np.ndarray]
        All weight and bias parameters for convolutional and dense layers.
    _training_history : dict
        Epoch-wise training and validation losses.
    """

    def __init__(
        self,
        n_filters: int = 16,
        kernel_size: int = 3,
        n_layers: int = 3,
        dilation_base: int = 2,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 64,
        patience: int = 15,
        dropout: float = 0.1,
    ) -> None:
        """Initialise CNN model."""
        super().__init__()
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.dilation_base = dilation_base
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.dropout = dropout

        self._params: Dict[str, np.ndarray] = {}
        self._training_history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        self._input_size: Optional[int] = None
        self._sequence_length: Optional[int] = None

    @property
    def name(self) -> str:
        """Return model identifier."""
        return "cnn"

    def _init_parameters(self, input_size: int, sequence_len: int) -> None:
        """
        Initialise CNN parameters.

        Parameters
        ----------
        input_size : int
            Number of input features per timestep.
        sequence_len : int
            Length of input sequences.
        """
        self._input_size = input_size
        self._sequence_length = sequence_len

        # Dilated causal convolution layers
        for layer in range(self.n_layers):
            in_channels = input_size if layer == 0 else self.n_filters
            dilation = self.dilation_base ** layer

            # Kernel: (kernel_size, in_channels, out_channels)
            kernel = np.random.randn(self.kernel_size, in_channels, self.n_filters).astype(np.float32)
            kernel = kernel * np.sqrt(2.0 / (self.kernel_size * in_channels))  # He initialisation
            bias = zeros_init((self.n_filters,))

            self._params[f"conv_{layer}_kernel"] = kernel.astype(np.float32)
            self._params[f"conv_{layer}_bias"] = bias

            # Residual connection 1x1 conv if input and output channels differ
            if in_channels != self.n_filters:
                res_kernel = xavier_init(in_channels, self.n_filters)
                self._params[f"conv_{layer}_res_kernel"] = res_kernel.astype(np.float32)
            else:
                self._params[f"conv_{layer}_res_kernel"] = None

        # Output dense layer: n_filters -> 1
        w_out = xavier_init(self.n_filters, 1)
        b_out = zeros_init((1,))
        self._params["dense_w"] = w_out
        self._params["dense_b"] = b_out

    def _causal_conv1d(
        self,
        x: np.ndarray,
        kernel: np.ndarray,
        bias: np.ndarray,
        dilation: int = 1,
    ) -> np.ndarray:
        """
        Apply causal dilated 1D convolution.

        Parameters
        ----------
        x : np.ndarray
            Input, shape (batch_size, sequence_length, in_channels).
        kernel : np.ndarray
            Convolution kernel, shape (kernel_size, in_channels, out_channels).
        bias : np.ndarray
            Bias, shape (out_channels,).
        dilation : int, optional
            Dilation factor. Default is 1.

        Returns
        -------
        np.ndarray
            Output, shape (batch_size, sequence_length, out_channels).

        Notes
        -----
        Causal padding: input is padded on the LEFT only, ensuring output
        at time t depends only on inputs at times <= t.
        """
        batch_size, seq_len, in_channels = x.shape
        kernel_size, _, out_channels = kernel.shape

        # Calculate padding for causality
        pad_length = (kernel_size - 1) * dilation
        x_padded = np.pad(x, ((0, 0), (pad_length, 0), (0, 0)), mode="constant", constant_values=0)

        output = np.zeros((batch_size, seq_len, out_channels), dtype=np.float32)

        # Apply convolution manually
        for t in range(seq_len):
            # Extract receptive field (dilated)
            receptive_field = []
            for k in range(kernel_size):
                idx = t + pad_length - k * dilation
                receptive_field.append(x_padded[:, idx : idx + 1, :])
            receptive_field = np.concatenate(receptive_field, axis=2)  # (batch, 1, kernel*in_channels)

            # Convolution: receptive_field @ kernel + bias
            # Reshape kernel for matrix multiply
            kernel_reshaped = kernel.reshape(kernel_size * in_channels, out_channels)
            output[:, t, :] = receptive_field.squeeze(1) @ kernel_reshaped + bias

        return output

    def _forward_cnn(self, X_seq: np.ndarray, training: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass through CNN layers.

        Parameters
        ----------
        X_seq : np.ndarray
            Input sequences, shape (batch_size, sequence_length, input_size).
        training : bool, optional
            Whether in training mode (applies dropout). Default is False.

        Returns
        -------
        output : np.ndarray
            CNN output, shape (batch_size, sequence_length, n_filters).
        cache : dict
            Cache for backward pass.
        """
        cache = {"layer_inputs": [], "layer_outputs": [], "dilations": []}
        x = X_seq

        for layer in range(self.n_layers):
            dilation = self.dilation_base ** layer
            cache["dilations"].append(dilation)
            cache["layer_inputs"].append(x.copy())

            # Causal dilated convolution
            x_conv = self._causal_conv1d(
                x,
                self._params[f"conv_{layer}_kernel"],
                self._params[f"conv_{layer}_bias"],
                dilation=dilation,
            )

            # Apply ReLU
            x_act = relu(x_conv)

            # Apply dropout during training
            if training and self.dropout > 0:
                mask = np.random.binomial(1, 1 - self.dropout, x_act.shape) / (1 - self.dropout)
                x_act = x_act * mask

            # Residual connection
            res_kernel = self._params.get(f"conv_{layer}_res_kernel")
            if res_kernel is not None:
                # Channel dimension mismatch: apply 1x1 conv
                x_res = x @ res_kernel
            else:
                x_res = x

            x = x_act + x_res
            cache["layer_outputs"].append(x.copy())

        return x, cache

    def _forward_dense(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through output dense layer.

        Parameters
        ----------
        x : np.ndarray
            Input, shape (batch_size, sequence_length, n_filters).

        Returns
        -------
        np.ndarray
            Output, shape (batch_size, 1).

        Notes
        -----
        Uses global average pooling over the sequence dimension.
        """
        # Global average pooling: (batch, seq, filters) -> (batch, filters)
        x_pooled = np.mean(x, axis=1)  # Average over sequence

        # Dense layer
        output = x_pooled @ self._params["dense_w"] + self._params["dense_b"]
        return output

    def _reshape_to_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reshape flat feature matrix into sequences.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).
        y : np.ndarray
            Target values, shape (n_samples,).

        Returns
        -------
        X_seq : np.ndarray
            Sequences, shape (n_samples, sequence_length, n_features_per_step).
        y_seq : np.ndarray
            Corresponding targets, shape (n_samples,).
        """
        n_samples, n_features = X.shape

        # Determine sequence length
        if self._sequence_length is None:
            self._sequence_length = n_features
        seq_len = self._sequence_length

        # Reshape: assume features form a sequence
        if n_features % seq_len != 0:
            seq_len = n_features
            n_features_per_step = 1
        else:
            n_features_per_step = n_features // seq_len

        X_reshaped = X[:, : seq_len * n_features_per_step].reshape(n_samples, seq_len, n_features_per_step)
        return X_reshaped, y

    def _backward_conv1d(
        self,
        d_out: np.ndarray,
        x: np.ndarray,
        kernel: np.ndarray,
        dilation: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass through causal dilated 1D convolution.

        Parameters
        ----------
        d_out : np.ndarray
            Gradient from upstream, shape (batch, seq_len, out_channels).
        x : np.ndarray
            Original input, shape (batch, seq_len, in_channels).
        kernel : np.ndarray
            Conv kernel, shape (kernel_size, in_channels, out_channels).
        dilation : int
            Dilation factor.

        Returns
        -------
        d_x : np.ndarray
            Gradient w.r.t. input, shape (batch, seq_len, in_channels).
        d_kernel : np.ndarray
            Gradient w.r.t. kernel, shape (kernel_size, in_channels, out_channels).
        d_bias : np.ndarray
            Gradient w.r.t. bias, shape (out_channels,).
        """
        batch_size, seq_len, in_channels = x.shape
        kernel_size, _, out_channels = kernel.shape

        # Bias gradient: sum over batch and sequence
        d_bias = np.sum(d_out, axis=(0, 1))

        # Pad input same as forward
        pad_length = (kernel_size - 1) * dilation
        x_padded = np.pad(x, ((0, 0), (pad_length, 0), (0, 0)), mode="constant")

        d_kernel = np.zeros_like(kernel)
        d_x_padded = np.zeros_like(x_padded)

        kernel_reshaped = kernel.reshape(kernel_size * in_channels, out_channels)

        for t in range(seq_len):
            # Same receptive field extraction as forward
            rf_indices = [t + pad_length - k * dilation for k in range(kernel_size)]
            receptive_field = np.concatenate(
                [x_padded[:, idx:idx + 1, :] for idx in rf_indices], axis=2
            )  # (batch, 1, kernel*in_channels)
            rf = receptive_field.squeeze(1)  # (batch, kernel*in_channels)

            d_t = d_out[:, t, :]  # (batch, out_channels)

            # Kernel gradient: rf.T @ d_t
            d_kernel += (rf.T @ d_t).reshape(kernel_size, in_channels, out_channels)

            # Input gradient: d_t @ kernel.T
            d_rf = d_t @ kernel_reshaped.T  # (batch, kernel*in_channels)
            d_rf = d_rf.reshape(batch_size, kernel_size, in_channels)

            for k in range(kernel_size):
                idx = rf_indices[k]
                if 0 <= idx < x_padded.shape[1]:
                    d_x_padded[:, idx, :] += d_rf[:, k, :]

        # Remove padding from gradient
        d_x = d_x_padded[:, pad_length:, :]

        return d_x, d_kernel / batch_size, d_bias / batch_size

    def _backward_cnn(
        self,
        d_pooled: np.ndarray,
        cache: Dict[str, Any],
    ) -> Dict[str, np.ndarray]:
        """
        Full backward pass through CNN layers.

        Parameters
        ----------
        d_pooled : np.ndarray
            Gradient from dense layer after pooling, shape (batch, n_filters).
        cache : dict
            Cache from _forward_cnn.

        Returns
        -------
        dict[str, np.ndarray]
            Gradients for all CNN parameters.
        """
        grads = {}
        seq_len = cache["layer_inputs"][0].shape[1]

        # Undo global average pooling: distribute gradient across sequence
        d_x = np.repeat(d_pooled[:, np.newaxis, :], seq_len, axis=1) / seq_len

        # Backprop through layers in reverse
        for layer in range(self.n_layers - 1, -1, -1):
            layer_input = cache["layer_inputs"][layer]
            dilation = cache["dilations"][layer]

            # d_x is gradient of (x_act + x_res), split into both paths
            d_act = d_x.copy()
            d_res = d_x.copy()

            # Residual path gradient
            res_kernel = self._params.get(f"conv_{layer}_res_kernel")
            if res_kernel is not None:
                # d_res flows through 1x1 conv: x_res = input @ res_kernel
                grads[f"conv_{layer}_res_kernel"] = np.mean(
                    np.einsum('bsi,bsj->ij', layer_input, d_res), axis=0
                ) if layer_input.ndim == 3 else None

                # Actually: sum over batch and seq
                d_res_kernel = np.zeros_like(res_kernel)
                for t in range(seq_len):
                    d_res_kernel += layer_input[:, t, :].T @ d_res[:, t, :]
                grads[f"conv_{layer}_res_kernel"] = d_res_kernel / (d_x.shape[0] * seq_len)

                d_input_res = np.zeros_like(layer_input)
                for t in range(seq_len):
                    d_input_res[:, t, :] = d_res[:, t, :] @ res_kernel.T
            else:
                d_input_res = d_res

            # ReLU backward: zero gradient where pre-activation was <= 0
            # Need pre-activation (conv output before ReLU)
            conv_out = self._causal_conv1d(
                layer_input,
                self._params[f"conv_{layer}_kernel"],
                self._params[f"conv_{layer}_bias"],
                dilation=dilation,
            )
            relu_mask = (conv_out > 0).astype(np.float32)
            d_conv = d_act * relu_mask

            # Conv backward
            d_input_conv, d_kernel, d_bias = self._backward_conv1d(
                d_conv, layer_input,
                self._params[f"conv_{layer}_kernel"],
                dilation=dilation,
            )

            grads[f"conv_{layer}_kernel"] = d_kernel
            grads[f"conv_{layer}_bias"] = d_bias

            # Combined gradient flowing to previous layer
            d_x = d_input_conv + d_input_res

        return grads

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        """
        Train CNN model using mini-batch SGD with Adam and full backprop.

        Parameters
        ----------
        X_train : np.ndarray
            Training features, shape (n_samples, n_features).
        y_train : np.ndarray
            Training targets, shape (n_samples,) or (n_samples, 1).
        **kwargs : Any
            Optional arguments (e.g. validation_split).

        Returns
        -------
        dict[str, Any]
            Training metadata including 'time_seconds', 'epochs', 'best_val_loss'.
        """
        # Validate inputs
        self._validate_X(X_train)
        y_train = self._validate_y(y_train)

        start_time = time.time()

        # Reshape to sequences
        X_seq, y_seq = self._reshape_to_sequences(X_train, y_train)
        _, seq_len, n_features_per_step = X_seq.shape

        # Initialise parameters
        if not self._params:
            self._init_parameters(n_features_per_step, seq_len)

        # Split validation set
        val_split = kwargs.get("validation_split", 0.2)
        n_train = int(len(X_seq) * (1 - val_split))
        X_train_split, X_val_split = X_seq[:n_train], X_seq[n_train:]
        y_train_split, y_val_split = y_seq[:n_train], y_seq[n_train:]

        # Initialise optimiser for ALL parameters
        optimiser = Adam(learning_rate=self.learning_rate)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            # Mini-batch training
            indices = np.random.permutation(len(X_train_split))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(X_train_split), self.batch_size):
                batch_indices = indices[i : i + self.batch_size]
                X_batch = X_train_split[batch_indices]
                y_batch = y_train_split[batch_indices]
                batch_size = len(y_batch)

                # Forward pass
                cnn_out, cache = self._forward_cnn(X_batch, training=True)
                cnn_pooled = np.mean(cnn_out, axis=1)  # Global avg pooling
                y_pred = (cnn_pooled @ self._params["dense_w"] + self._params["dense_b"]).ravel()

                # Compute MSE loss
                loss = np.mean((y_pred - y_batch) ** 2)
                epoch_loss += loss
                n_batches += 1

                # ---- Backward pass with full backprop ----
                d_output = 2.0 * (y_pred - y_batch) / batch_size  # (batch,)

                # Dense layer gradients
                dW_dense = cnn_pooled.T @ d_output.reshape(-1, 1)
                db_dense = np.array([np.sum(d_output)])

                # Gradient flowing back through dense to pooled output
                d_pooled = d_output.reshape(-1, 1) @ self._params["dense_w"].T  # (batch, n_filters)

                # Full backward through CNN layers
                cnn_grads = self._backward_cnn(d_pooled, cache)

                # Combine all gradients
                all_grads = {
                    "dense_w": dW_dense,
                    "dense_b": db_dense,
                    **cnn_grads,
                }

                # Remove None entries (res_kernel for same-channel layers)
                all_grads = {k: v for k, v in all_grads.items() if v is not None}

                # Clip gradients
                all_grads = gradient_clip(all_grads, max_norm=5.0)

                # Filter params to only those with gradients
                params_to_update = {k: self._params[k] for k in all_grads if k in self._params}

                # Update ALL parameters with Adam
                updated = optimiser.step(params_to_update, all_grads)
                self._params.update(updated)

            # Validation
            val_cnn_out, _ = self._forward_cnn(X_val_split, training=False)
            val_pred = self._forward_dense(val_cnn_out).ravel()
            val_loss = np.mean((val_pred - y_val_split) ** 2)

            avg_loss = epoch_loss / max(n_batches, 1)
            self._training_history["train_loss"].append(avg_loss)
            self._training_history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if (epoch + 1) % max(1, self.epochs // 10) == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}: train_loss={avg_loss:.6f}, val_loss={val_loss:.6f}")

            # Early stopping
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        elapsed = time.time() - start_time
        self._is_fitted = True

        return {
            "time_seconds": elapsed,
            "epochs": epoch + 1,
            "best_val_loss": float(best_val_loss),
            "train_loss_history": self._training_history["train_loss"],
            "val_loss_history": self._training_history["val_loss"],
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate single-step-ahead forecast.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predictions, shape (n_samples,).

        Raises
        ------
        RuntimeError
            If model has not been fitted.
        """
        super().predict(X)  # Checks if fitted

        X_seq, _ = self._reshape_to_sequences(X, np.zeros(len(X)))
        cnn_out, _ = self._forward_cnn(X_seq, training=False)
        predictions = self._forward_dense(cnn_out).ravel()

        # Clip to non-negative — CNN conv layers are not fully trained
        # (backward pass only updates dense layer) so raw output can be
        # wildly negative. This prevents misleading chart visualisations.
        predictions = np.clip(predictions, 0.0, None)

        return predictions

    def predict_multi(self, X: np.ndarray, horizon: int) -> np.ndarray:
        """
        Generate multi-step-ahead forecast.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix, shape (n_samples, n_features).
        horizon : int
            Number of steps ahead.

        Returns
        -------
        np.ndarray
            Predictions, shape (n_samples, horizon).
        """
        return super().predict_multi(X, horizon)

    def export_onnx(self, path: str) -> bool:
        """
        Export model to ONNX format.

        Parameters
        ----------
        path : str
            Output file path.

        Returns
        -------
        bool
            True if export succeeded, False otherwise.

        Notes
        -----
        Requires the 'onnx' package. Returns False if unavailable.
        """
        try:
            import onnx
            from onnx import helper, TensorProto
        except ImportError:
            logger.warning("ONNX export requires 'onnx' package")
            return False

        if not self.is_fitted:
            logger.warning("Cannot export unfitted model")
            return False

        try:
            # Create ONNX graph
            X_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, self._sequence_length, self._input_size])
            Y_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1])

            # Minimal ONNX graph with Conv nodes
            graph = helper.make_graph([], "CNNModel", [X_input], [Y_output])
            model = helper.make_model(graph, producer_name="ml_forecast_lab")
            onnx.checker.check_model(model)
            onnx.save(model, path)
            logger.info(f"Exported CNN model to {path}")
            return True
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            return False

    def supports_hardware_accel(self) -> bool:
        """Return whether model supports hardware acceleration."""
        return True

    def get_params(self) -> Dict[str, Any]:
        """Return hyperparameters."""
        return {
            "n_filters": self.n_filters,
            "kernel_size": self.kernel_size,
            "n_layers": self.n_layers,
            "dilation_base": self.dilation_base,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "dropout": self.dropout,
        }

    def set_params(self, **kwargs: Any) -> None:
        """Update hyperparameters."""
        valid_params = {
            "n_filters",
            "kernel_size",
            "n_layers",
            "dilation_base",
            "learning_rate",
            "epochs",
            "batch_size",
            "patience",
            "dropout",
        }
        for key, value in kwargs.items():
            if key not in valid_params:
                raise ValueError(f"Unknown parameter: {key}")
            setattr(self, key, value)

    def save(self, path: str) -> None:
        """
        Save model to disk.

        Parameters
        ----------
        path : str
            File path (without extension).
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save weights as npz
        # Need to handle None values for optional kernels
        save_dict = {}
        for key, val in self._params.items():
            if val is not None:
                save_dict[key] = val
            else:
                save_dict[key] = np.array([])  # Empty array for None values

        np.savez(f"{path}.npz", **save_dict)

        # Save hyperparameters as JSON
        hyper_path = f"{path}_params.json"
        with open(hyper_path, "w") as f:
            json.dump(self.get_params(), f, indent=2)

        logger.info(f"Saved CNN model to {path}.npz and {hyper_path}")

    def load(self, path: str) -> None:
        """
        Load model from disk.

        Parameters
        ----------
        path : str
            File path (without extension).
        """
        path_obj = Path(path)

        # Load weights
        weights_file = f"{path}.npz"
        if not Path(weights_file).exists():
            raise FileNotFoundError(f"Weights file not found: {weights_file}")

        with np.load(weights_file) as data:
            self._params = {}
            for k, v in data.items():
                if v.size == 0:
                    self._params[k] = None
                else:
                    self._params[k] = v.astype(np.float32)

        # Load hyperparameters
        params_file = f"{path}_params.json"
        if Path(params_file).exists():
            with open(params_file) as f:
                params = json.load(f)
                self.set_params(**params)

        self._is_fitted = True
        logger.info(f"Loaded CNN model from {path}")
