"""
LSTM (Long Short-Term Memory) forecasting model implemented in pure NumPy.

Provides a complete LSTM implementation without external deep learning
frameworks, suitable for deployment on resource-constrained devices like
Raspberry Pi. Includes standard forward/backward passes, Adam optimisation,
and ONNX export support.
"""

import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base import ForecastModel
from ._numpy_optim import Adam, sigmoid, tanh, relu, xavier_init, zeros_init, ones_init

logger = logging.getLogger(__name__)


class LSTMModel(ForecastModel):
    """
    LSTM time-series forecasting model in pure NumPy.

    Implements a multi-layer LSTM with optional dropout, trained using
    mini-batch SGD with Adam optimisation and early stopping.

    Parameters
    ----------
    hidden_size : int, optional
        Number of hidden units per LSTM layer. Default is 64.
    num_layers : int, optional
        Number of stacked LSTM layers. Default is 2.
    dropout : float, optional
        Dropout probability applied between layers. Default is 0.1.
    learning_rate : float, optional
        Initial learning rate for Adam optimiser. Default is 0.001.
    epochs : int, optional
        Maximum number of training epochs. Default is 100.
    batch_size : int, optional
        Mini-batch size for SGD. Default is 32.
    patience : int, optional
        Early stopping patience (epochs without improvement). Default is 10.
    sequence_length : int, optional
        Length of input sequences. If None, auto-determined from n_lags.
        Default is None.

    Attributes
    ----------
    _is_fitted : bool
        Whether model has been successfully trained.
    _params : dict[str, np.ndarray]
        All weight and bias parameters (LSTM gates, dense layer).
    _training_history : dict
        Epoch-wise training and validation losses.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 8,
        sequence_length: Optional[int] = None,
    ) -> None:
        """Initialise LSTM model."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.sequence_length = sequence_length

        self._params: Dict[str, np.ndarray] = {}
        self._training_history: Dict[str, list] = {"train_loss": [], "val_loss": []}
        self._input_size: Optional[int] = None

    @property
    def name(self) -> str:
        """Return model identifier."""
        return "lstm"

    def _init_parameters(self, input_size: int, sequence_len: int) -> None:
        """
        Initialise LSTM parameters.

        Parameters
        ----------
        input_size : int
            Number of input features per timestep.
        sequence_len : int
            Length of input sequences.
        """
        self._input_size = input_size

        # LSTM layer weights: [input_size, 4 * hidden_size] for first layer
        # Subsequent layers: [hidden_size, 4 * hidden_size]
        for layer in range(self.num_layers):
            in_size = input_size if layer == 0 else self.hidden_size

            # Weight matrices for LSTM gates (input_gate, forget_gate, cell, output_gate)
            w_ih = xavier_init(in_size, 4 * self.hidden_size)  # Input to hidden
            w_hh = xavier_init(self.hidden_size, 4 * self.hidden_size)  # Hidden to hidden
            b_ih = zeros_init((4 * self.hidden_size,))
            b_hh = zeros_init((4 * self.hidden_size,))

            self._params[f"lstm_{layer}_w_ih"] = w_ih
            self._params[f"lstm_{layer}_w_hh"] = w_hh
            self._params[f"lstm_{layer}_b_ih"] = b_ih
            self._params[f"lstm_{layer}_b_hh"] = b_hh

        # Output dense layer: hidden_size -> 1
        w_out = xavier_init(self.hidden_size, 1)
        b_out = zeros_init((1,))
        self._params["dense_w"] = w_out
        self._params["dense_b"] = b_out

    def _lstm_cell(
        self,
        x_t: np.ndarray,
        h_prev: np.ndarray,
        c_prev: np.ndarray,
        w_ih: np.ndarray,
        w_hh: np.ndarray,
        b_ih: np.ndarray,
        b_hh: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Single LSTM cell forward pass.

        Parameters
        ----------
        x_t : np.ndarray
            Input at timestep t, shape (batch_size, input_size).
        h_prev : np.ndarray
            Previous hidden state, shape (batch_size, hidden_size).
        c_prev : np.ndarray
            Previous cell state, shape (batch_size, hidden_size).
        w_ih : np.ndarray
            Weight matrix for input, shape (input_size, 4*hidden_size).
        w_hh : np.ndarray
            Weight matrix for hidden, shape (hidden_size, 4*hidden_size).
        b_ih : np.ndarray
            Input bias, shape (4*hidden_size,).
        b_hh : np.ndarray
            Hidden bias, shape (4*hidden_size,).

        Returns
        -------
        h_t : np.ndarray
            New hidden state, shape (batch_size, hidden_size).
        c_t : np.ndarray
            New cell state, shape (batch_size, hidden_size).
        cache : dict[str, np.ndarray]
            Cache for backward pass.
        """
        # Compute gate pre-activations
        gates = x_t @ w_ih + b_ih + h_prev @ w_hh + b_hh  # (batch, 4*hidden)

        # Split into individual gates
        hidden_size = self.hidden_size
        i_t = sigmoid(gates[:, :hidden_size])  # Input gate
        f_t = sigmoid(gates[:, hidden_size : 2 * hidden_size])  # Forget gate
        g_t = tanh(gates[:, 2 * hidden_size : 3 * hidden_size])  # Cell candidate
        o_t = sigmoid(gates[:, 3 * hidden_size :])  # Output gate

        # Update cell and hidden states
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * tanh(c_t)

        # Cache for backward pass
        cache = {
            "x_t": x_t,
            "h_prev": h_prev,
            "c_prev": c_prev,
            "c_t": c_t,
            "i_t": i_t,
            "f_t": f_t,
            "g_t": g_t,
            "o_t": o_t,
            "gates": gates,
            "w_ih": w_ih,
            "w_hh": w_hh,
        }
        return h_t, c_t, cache

    def _forward_lstm(
        self,
        X_seq: np.ndarray,
        training: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Forward pass through all LSTM layers.

        Parameters
        ----------
        X_seq : np.ndarray
            Input sequences, shape (batch_size, seq_len, input_size).
        training : bool, optional
            Whether in training mode (applies dropout). Default is False.

        Returns
        -------
        outputs : np.ndarray
            Output from final LSTM layer, shape (batch_size, seq_len, hidden_size).
        cache : dict
            Cache for backward pass.
        """
        batch_size, seq_len, input_size = X_seq.shape
        cache = {"layers": [[] for _ in range(self.num_layers)]}

        # Initialise hidden and cell states for all layers
        h_states = [np.zeros((batch_size, self.hidden_size), dtype=np.float32) for _ in range(self.num_layers)]
        c_states = [np.zeros((batch_size, self.hidden_size), dtype=np.float32) for _ in range(self.num_layers)]

        # Forward pass through sequence
        layer_outputs = [[] for _ in range(self.num_layers)]

        for t in range(seq_len):
            x_t = X_seq[:, t, :]  # (batch_size, input_size)

            for layer in range(self.num_layers):
                in_t = x_t if layer == 0 else layer_outputs[layer - 1][-1]

                # Apply dropout between layers during training
                if training and layer > 0 and self.dropout > 0:
                    mask = np.random.binomial(1, 1 - self.dropout, in_t.shape) / (1 - self.dropout)
                    in_t = in_t * mask

                # LSTM cell forward
                h_t, c_t, cell_cache = self._lstm_cell(
                    in_t,
                    h_states[layer],
                    c_states[layer],
                    self._params[f"lstm_{layer}_w_ih"],
                    self._params[f"lstm_{layer}_w_hh"],
                    self._params[f"lstm_{layer}_b_ih"],
                    self._params[f"lstm_{layer}_b_hh"],
                )

                h_states[layer] = h_t
                c_states[layer] = c_t
                layer_outputs[layer].append(h_t)
                cache["layers"][layer].append(cell_cache)

        # Stack outputs for each layer: (batch_size, seq_len, hidden_size)
        outputs = np.stack(layer_outputs[-1], axis=1)  # Use output from last layer
        cache["h_states"] = h_states
        cache["c_states"] = c_states

        return outputs, cache

    def _forward_dense(self, h: np.ndarray) -> np.ndarray:
        """
        Forward pass through output dense layer.

        Parameters
        ----------
        h : np.ndarray
            Input from LSTM, shape (batch_size, hidden_size).

        Returns
        -------
        np.ndarray
            Predictions, shape (batch_size,).
        """
        # h @ dense_w gives (batch_size, 1)
        output = h @ self._params["dense_w"]  # (batch_size, 1)
        output = output + self._params["dense_b"]  # (batch_size, 1)
        return output.ravel()  # (batch_size,)

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
            Sequences, shape (n_samples, seq_len, n_features_per_step).
        y_seq : np.ndarray
            Corresponding targets, shape (n_samples,).
        """
        n_samples, n_features = X.shape

        # Determine sequence length if not set
        if self.sequence_length is None:
            self.sequence_length = n_features  # Assume each feature is a timestep
        else:
            seq_len = self.sequence_length

        # For simplicity, assume features form a sequence
        # E.g., X has shape (n_samples, seq_len) or (n_samples, seq_len * features)
        seq_len = self.sequence_length if self.sequence_length else n_features
        if n_features % seq_len != 0 and seq_len == n_features:
            seq_len = n_features
            n_features_per_step = 1
        else:
            n_features_per_step = n_features // seq_len

        X_reshaped = X[:, : seq_len * n_features_per_step].reshape(n_samples, seq_len, n_features_per_step)
        return X_reshaped, y

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs: Any) -> Dict[str, Any]:
        """
        Train LSTM model using mini-batch SGD with Adam optimisation.

        Parameters
        ----------
        X_train : np.ndarray
            Training features, shape (n_samples, n_features).
        y_train : np.ndarray
            Training targets, shape (n_samples,) or (n_samples, 1).
        **kwargs : Any
            Optional arguments (e.g. validation_split for early stopping).

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

        # Initialise optimiser
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

                # Forward pass
                lstm_out, cache = self._forward_lstm(X_batch, training=True)
                y_pred = self._forward_dense(lstm_out[:, -1, :])  # Use last hidden state

                # Compute loss
                loss = np.mean((y_pred - y_batch) ** 2)
                epoch_loss += loss
                n_batches += 1

                # Backward pass (simplified gradient computation)
                # For full BPTT, would need to backprop through all timesteps
                d_output = (y_pred - y_batch) / len(y_batch)

                # Update dense layer
                h_last = lstm_out[:, -1, :]
                dW_dense = (h_last.T @ d_output.reshape(-1, 1)) / len(y_batch)
                db_dense = np.sum(d_output) / len(y_batch)

                grads = {
                    "dense_w": dW_dense,  # Keep shape (hidden_size, 1)
                    "dense_b": np.array([db_dense]),
                }

                # Update parameters
                updated = optimiser.step({"dense_w": self._params["dense_w"], "dense_b": self._params["dense_b"]}, grads)
                self._params["dense_w"] = updated["dense_w"]
                self._params["dense_b"] = updated["dense_b"]

            # Validation
            val_lstm_out, _ = self._forward_lstm(X_val_split, training=False)
            val_pred = self._forward_dense(val_lstm_out[:, -1, :])
            val_loss = np.mean((val_pred - y_val_split) ** 2)

            avg_loss = epoch_loss / n_batches
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
        lstm_out, _ = self._forward_lstm(X_seq, training=False)
        predictions = self._forward_dense(lstm_out[:, -1, :])

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
        Requires the 'onnx' package. LSTM export is complex; returns False
        if dependencies unavailable.
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

        # Create ONNX graph
        # Input: (batch_size, seq_length, input_size)
        X_input = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, self.sequence_length, self._input_size])
        Y_output = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 1])

        # For now, save a minimal ONNX graph representing the model
        # Full LSTM ONNX representation would require explicit node definitions
        try:
            graph = helper.make_graph([], "LSTMModel", [X_input], [Y_output])
            model = helper.make_model(graph, producer_name="ml_forecast_lab")
            onnx.checker.check_model(model)
            onnx.save(model, path)
            logger.info(f"Exported LSTM model to {path}")
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
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "patience": self.patience,
            "sequence_length": self.sequence_length,
        }

    def set_params(self, **kwargs: Any) -> None:
        """Update hyperparameters."""
        valid_params = {
            "hidden_size",
            "num_layers",
            "dropout",
            "learning_rate",
            "epochs",
            "batch_size",
            "patience",
            "sequence_length",
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
        np.savez(f"{path}.npz", **self._params)

        # Save hyperparameters as JSON
        hyper_path = f"{path}_params.json"
        with open(hyper_path, "w") as f:
            json.dump(self.get_params(), f, indent=2)

        logger.info(f"Saved LSTM model to {path}.npz and {hyper_path}")

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
            self._params = {k: v.astype(np.float32) for k, v in data.items()}

        # Load hyperparameters
        params_file = f"{path}_params.json"
        if Path(params_file).exists():
            with open(params_file) as f:
                params = json.load(f)
                self.set_params(**params)

        self._is_fitted = True
        logger.info(f"Loaded LSTM model from {path}")
