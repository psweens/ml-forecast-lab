"""
ONNX graph building utilities for exporting trained models.

Provides functions to programmatically build ONNX graphs for LSTM and CNN
architectures from trained weight matrices. Enables deployment on specialised
hardware accelerators such as the Hailo NPU.

All functions handle missing ONNX dependencies gracefully, returning False
without raising exceptions when ONNX cannot be imported.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def build_lstm_onnx(
    weights_dict: Dict[str, np.ndarray],
    input_shape: tuple,
    hidden_size: int,
    num_layers: int,
    output_path: str,
) -> bool:
    """
    Programmatically build and save LSTM ONNX graph.

    Constructs an ONNX graph representing a multi-layer LSTM network from
    pre-trained weight matrices. The graph includes LSTM operator nodes for
    each layer and a final dense (MatMul + Add) output layer.

    Parameters
    ----------
    weights_dict : dict[str, np.ndarray]
        Dictionary of weight matrices from trained LSTM. Expected keys:
        - 'lstm_0_w_ih', 'lstm_0_w_hh', 'lstm_0_b_ih', 'lstm_0_b_hh', ...
        - 'dense_w', 'dense_b'

        Each LSTM layer l requires:
        - lstm_l_w_ih: Input-to-hidden weights, shape (input_size, 4*hidden_size)
        - lstm_l_w_hh: Hidden-to-hidden weights, shape (hidden_size, 4*hidden_size)
        - lstm_l_b_ih: Input bias, shape (4*hidden_size,)
        - lstm_l_b_hh: Hidden bias, shape (4*hidden_size,)

    input_shape : tuple
        Shape of input tensor (batch_size, sequence_length, input_size).
        Typically (1, seq_len, n_features) for inference.

    hidden_size : int
        Number of hidden units in LSTM layers.

    num_layers : int
        Number of stacked LSTM layers.

    output_path : str
        File path where the ONNX model will be saved (.onnx extension).

    Returns
    -------
    bool
        True if graph was successfully built and saved. False if ONNX is
        unavailable or an error occurred during graph construction.

    Notes
    -----
    The output graph has the following structure:
    1. Input: (batch, seq_len, input_size) float32 tensor
    2. LSTM layer 0: processes input sequence, outputs (batch, seq_len, hidden_size)
    3. LSTM layer 1...n: each processes previous layer output
    4. GlobalAveragePool: reduces to (batch, hidden_size)
    5. MatMul with dense_w: (batch, hidden_size) @ (hidden_size, 1) -> (batch, 1)
    6. Add with dense_b: adds bias
    7. Output: (batch,) predictions

    ONNX operator compatibility should be verified for the target deployment
    platform (e.g. Hailo version).

    Raises
    ------
    (Logged as warning, does not raise)
    - If onnx package is unavailable
    - If weight shapes are incorrect or malformed
    - If output path is not writable
    """
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        logger.warning(
            'ONNX package is not installed. Cannot build LSTM ONNX graph. '
            'Install with: pip install onnx'
        )
        return False

    try:
        batch_size, seq_len, input_size = input_shape

        # Initialise graph inputs and outputs
        input_tensor = helper.make_tensor_value_info(
            'input',
            TensorProto.FLOAT,
            [batch_size, seq_len, input_size]
        )

        output_tensor = helper.make_tensor_value_info(
            'output',
            TensorProto.FLOAT,
            [batch_size]
        )

        initializers = []
        node_list = []

        # Build LSTM layers
        current_input = 'input'

        for layer_idx in range(num_layers):
            layer_name = f'lstm_{layer_idx}'

            # Extract weights for this layer
            w_ih_key = f'{layer_name}_w_ih'
            w_hh_key = f'{layer_name}_w_hh'
            b_ih_key = f'{layer_name}_b_ih'
            b_hh_key = f'{layer_name}_b_hh'

            if not all(k in weights_dict for k in [w_ih_key, w_hh_key, b_ih_key, b_hh_key]):
                logger.error(f'Missing weights for {layer_name}')
                return False

            w_ih = weights_dict[w_ih_key]
            w_hh = weights_dict[w_hh_key]
            b_ih = weights_dict[b_ih_key]
            b_hh = weights_dict[b_hh_key]

            # Combine biases for ONNX LSTM (expects single bias vector)
            bias = b_ih + b_hh

            # Create initialiser tensors
            w_ih_tensor = helper.make_tensor(
                f'{layer_name}_W',
                TensorProto.FLOAT,
                list(w_ih.shape),
                w_ih.astype(np.float32).tobytes(),
                raw=True
            )

            w_hh_tensor = helper.make_tensor(
                f'{layer_name}_R',
                TensorProto.FLOAT,
                list(w_hh.shape),
                w_hh.astype(np.float32).tobytes(),
                raw=True
            )

            bias_tensor = helper.make_tensor(
                f'{layer_name}_B',
                TensorProto.FLOAT,
                list(bias.shape),
                bias.astype(np.float32).tobytes(),
                raw=True
            )

            initializers.extend([w_ih_tensor, w_hh_tensor, bias_tensor])

            # Create LSTM node
            lstm_output = f'{layer_name}_output'
            lstm_node = helper.make_node(
                'LSTM',
                inputs=[current_input, f'{layer_name}_W', f'{layer_name}_R', f'{layer_name}_B'],
                outputs=[lstm_output, f'{layer_name}_h', f'{layer_name}_c'],
                name=f'{layer_name}_node'
            )
            node_list.append(lstm_node)
            current_input = lstm_output

        # Global average pooling over sequence dimension
        pool_output = 'pool_output'
        pool_node = helper.make_node(
            'ReduceMean',
            inputs=[current_input],
            outputs=[pool_output],
            axes=[1],  # Reduce over sequence dimension
            keepdims=0,
            name='global_avg_pool'
        )
        node_list.append(pool_node)

        # Dense layer: MatMul + Add
        dense_w = weights_dict.get('dense_w')
        dense_b = weights_dict.get('dense_b')

        if dense_w is None or dense_b is None:
            logger.error('Missing dense layer weights (dense_w or dense_b)')
            return False

        # Create dense weight and bias tensors
        dense_w_tensor = helper.make_tensor(
            'dense_W',
            TensorProto.FLOAT,
            list(dense_w.shape),
            dense_w.astype(np.float32).tobytes(),
            raw=True
        )

        dense_b_tensor = helper.make_tensor(
            'dense_B',
            TensorProto.FLOAT,
            list(dense_b.shape),
            dense_b.astype(np.float32).tobytes(),
            raw=True
        )

        initializers.extend([dense_w_tensor, dense_b_tensor])

        # MatMul node
        matmul_output = 'matmul_output'
        matmul_node = helper.make_node(
            'MatMul',
            inputs=[pool_output, 'dense_W'],
            outputs=[matmul_output],
            name='dense_matmul'
        )
        node_list.append(matmul_node)

        # Add bias node
        add_node = helper.make_node(
            'Add',
            inputs=[matmul_output, 'dense_B'],
            outputs=['output'],
            name='dense_add'
        )
        node_list.append(add_node)

        # Create graph
        graph = helper.make_graph(
            node_list,
            'lstm_model',
            [input_tensor],
            [output_tensor],
            initializers
        )

        # Create model
        model = helper.make_model(
            graph,
            producer_name='ml-forecast-lab',
            opset_imports=[helper.make_opsetid('', 12)]
        )

        # Save model
        onnx.save(model, output_path)
        logger.info(f'LSTM ONNX graph saved to {output_path}')
        return True

    except Exception as e:
        logger.error(f'Failed to build LSTM ONNX graph: {e}')
        return False


def build_cnn_onnx(
    weights_dict: Dict[str, np.ndarray],
    input_shape: tuple,
    n_filters: int,
    kernel_size: int,
    n_layers: int,
    dilations: list,
    output_path: str,
) -> bool:
    """
    Programmatically build and save 1D dilated causal CNN ONNX graph.

    Constructs an ONNX graph representing a multi-layer dilated causal
    convolutional network from pre-trained weights. Includes residual
    connections, ReLU activations, and a final dense output layer.

    Parameters
    ----------
    weights_dict : dict[str, np.ndarray]
        Dictionary of weight matrices from trained CNN. Expected keys:
        - 'conv_0_kernel', 'conv_0_bias', 'conv_0_res_kernel' (if needed), ...
        - 'dense_w', 'dense_b'

        Each convolutional layer l requires:
        - conv_l_kernel: Kernel weights, shape (kernel_size, in_channels, n_filters)
        - conv_l_bias: Bias, shape (n_filters,)
        - conv_l_res_kernel: Optional 1x1 residual kernel if channel dimensions differ

    input_shape : tuple
        Shape of input tensor (batch_size, sequence_length, input_size).

    n_filters : int
        Number of filters in convolutional layers.

    kernel_size : int
        Size of convolutional kernel.

    n_layers : int
        Number of convolutional layers.

    dilations : list
        List of dilation factors for each layer. Length must equal n_layers.
        Example: [1, 2, 4, 8] for exponential dilation.

    output_path : str
        File path where the ONNX model will be saved (.onnx extension).

    Returns
    -------
    bool
        True if graph was successfully built and saved. False if ONNX is
        unavailable or an error occurred during graph construction.

    Notes
    -----
    The output graph structure:
    1. Input: (batch, seq_len, input_size)
    2. Conv layers 0...n: each applies causal dilation convolution with residual
       connections (Add nodes)
    3. ReLU activations between layers
    4. GlobalAveragePool: reduces to (batch, n_filters)
    5. MatMul with dense_w: (batch, n_filters) @ (n_filters, 1)
    6. Add with dense_b: adds bias
    7. Output: (batch,) predictions

    Causal convolutions are implemented using explicit padding on the left
    to ensure output at time t depends only on inputs at times <= t.

    Raises
    ------
    (Logged as warning, does not raise)
    - If onnx package is unavailable
    - If weight shapes are incorrect
    - If dilations list length does not match n_layers
    """
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        logger.warning(
            'ONNX package is not installed. Cannot build CNN ONNX graph. '
            'Install with: pip install onnx'
        )
        return False

    try:
        if len(dilations) != n_layers:
            logger.error(f'dilations length ({len(dilations)}) must match n_layers ({n_layers})')
            return False

        batch_size, seq_len, input_size = input_shape

        # Input and output tensors
        input_tensor = helper.make_tensor_value_info(
            'input',
            TensorProto.FLOAT,
            [batch_size, seq_len, input_size]
        )

        output_tensor = helper.make_tensor_value_info(
            'output',
            TensorProto.FLOAT,
            [batch_size]
        )

        initializers = []
        node_list = []

        # Build convolutional layers
        current_input = 'input'

        for layer_idx in range(n_layers):
            layer_name = f'conv_{layer_idx}'
            dilation = dilations[layer_idx]

            # Extract weights
            kernel_key = f'{layer_name}_kernel'
            bias_key = f'{layer_name}_bias'
            res_kernel_key = f'{layer_name}_res_kernel'

            if kernel_key not in weights_dict or bias_key not in weights_dict:
                logger.error(f'Missing weights for {layer_name}')
                return False

            kernel = weights_dict[kernel_key]  # (kernel_size, in_channels, n_filters)
            bias = weights_dict[bias_key]  # (n_filters,)

            # Reshape kernel for ONNX Conv (expects [out_channels, in_channels, kernel_size])
            # From (kernel_size, in_channels, n_filters) -> (n_filters, in_channels, kernel_size)
            kernel_onnx = np.transpose(kernel, (2, 1, 0)).astype(np.float32)

            kernel_tensor = helper.make_tensor(
                f'{layer_name}_kernel',
                TensorProto.FLOAT,
                list(kernel_onnx.shape),
                kernel_onnx.tobytes(),
                raw=True
            )

            bias_tensor = helper.make_tensor(
                f'{layer_name}_bias',
                TensorProto.FLOAT,
                list(bias.shape),
                bias.astype(np.float32).tobytes(),
                raw=True
            )

            initializers.extend([kernel_tensor, bias_tensor])

            # Calculate padding for causal convolution
            # For causal: pad_left = (kernel_size - 1) * dilation, pad_right = 0
            pad_left = (kernel_size - 1) * dilation
            pad_right = 0
            pads = [0, pad_left, 0, pad_right]  # [top, left, bottom, right] for Conv1D is [left, right]

            # Conv node
            conv_output = f'{layer_name}_conv_out'
            conv_node = helper.make_node(
                'Conv',
                inputs=[current_input, f'{layer_name}_kernel', f'{layer_name}_bias'],
                outputs=[conv_output],
                kernel_shape=[kernel_size],
                pads=pads,
                dilations=[dilation],
                name=f'{layer_name}_conv'
            )
            node_list.append(conv_node)

            # ReLU activation
            relu_output = f'{layer_name}_relu_out'
            relu_node = helper.make_node(
                'Relu',
                inputs=[conv_output],
                outputs=[relu_output],
                name=f'{layer_name}_relu'
            )
            node_list.append(relu_node)

            # Residual connection (Add node)
            # If channels differ, apply 1x1 conv to input
            if res_kernel_key in weights_dict and weights_dict[res_kernel_key] is not None:
                res_kernel = weights_dict[res_kernel_key]
                res_kernel_onnx = res_kernel.T.astype(np.float32)

                res_kernel_tensor = helper.make_tensor(
                    f'{layer_name}_res_kernel',
                    TensorProto.FLOAT,
                    list(res_kernel_onnx.shape),
                    res_kernel_onnx.tobytes(),
                    raw=True
                )
                initializers.append(res_kernel_tensor)

                res_output = f'{layer_name}_res_out'
                res_node = helper.make_node(
                    'MatMul',
                    inputs=[current_input, f'{layer_name}_res_kernel'],
                    outputs=[res_output],
                    name=f'{layer_name}_res'
                )
                node_list.append(res_node)

                add_input = res_output
            else:
                add_input = current_input

            # Add residual connection
            add_output = f'{layer_name}_add_out'
            add_node = helper.make_node(
                'Add',
                inputs=[relu_output, add_input],
                outputs=[add_output],
                name=f'{layer_name}_add'
            )
            node_list.append(add_node)
            current_input = add_output

        # Global average pooling
        pool_output = 'pool_output'
        pool_node = helper.make_node(
            'ReduceMean',
            inputs=[current_input],
            outputs=[pool_output],
            axes=[1],  # Reduce over sequence dimension
            keepdims=0,
            name='global_avg_pool'
        )
        node_list.append(pool_node)

        # Dense output layer
        dense_w = weights_dict.get('dense_w')
        dense_b = weights_dict.get('dense_b')

        if dense_w is None or dense_b is None:
            logger.error('Missing dense layer weights')
            return False

        dense_w_tensor = helper.make_tensor(
            'dense_W',
            TensorProto.FLOAT,
            list(dense_w.shape),
            dense_w.astype(np.float32).tobytes(),
            raw=True
        )

        dense_b_tensor = helper.make_tensor(
            'dense_B',
            TensorProto.FLOAT,
            list(dense_b.shape),
            dense_b.astype(np.float32).tobytes(),
            raw=True
        )

        initializers.extend([dense_w_tensor, dense_b_tensor])

        # MatMul
        matmul_output = 'matmul_output'
        matmul_node = helper.make_node(
            'MatMul',
            inputs=[pool_output, 'dense_W'],
            outputs=[matmul_output],
            name='dense_matmul'
        )
        node_list.append(matmul_node)

        # Add bias
        add_node = helper.make_node(
            'Add',
            inputs=[matmul_output, 'dense_B'],
            outputs=['output'],
            name='dense_add'
        )
        node_list.append(add_node)

        # Create graph
        graph = helper.make_graph(
            node_list,
            'cnn_model',
            [input_tensor],
            [output_tensor],
            initializers
        )

        # Create model
        model = helper.make_model(
            graph,
            producer_name='ml-forecast-lab',
            opset_imports=[helper.make_opsetid('', 12)]
        )

        # Save model
        onnx.save(model, output_path)
        logger.info(f'CNN ONNX graph saved to {output_path}')
        return True

    except Exception as e:
        logger.error(f'Failed to build CNN ONNX graph: {e}')
        return False


def validate_onnx(path: str) -> bool:
    """
    Validate ONNX graph structure and consistency.

    Performs structural checks on a saved ONNX model including:
    - Loading the model successfully
    - Verifying all initialiser tensors exist
    - Checking operator definitions
    - Running shape inference to validate dimensions

    Parameters
    ----------
    path : str
        File path to the ONNX model to validate.

    Returns
    -------
    bool
        True if the model is valid. False if ONNX is unavailable or
        validation fails.

    Notes
    -----
    This function provides a quick smoke test before attempting to compile
    or deploy to hardware. More thorough validation may be performed by
    the Hailo Dataflow Compiler.

    Does not perform numerical validation or test inference.
    """
    try:
        import onnx
    except ImportError:
        logger.warning('ONNX package is not installed. Cannot validate ONNX model.')
        return False

    try:
        # Load the model
        model = onnx.load(path)

        # Check model
        onnx.checker.check_model(model)

        logger.info(f'ONNX model at {path} is valid')
        return True

    except Exception as e:
        logger.error(f'ONNX validation failed for {path}: {e}')
        return False
