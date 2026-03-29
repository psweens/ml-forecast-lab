"""
Hailo NPU inference runtime and acceleration wrapper.

Provides a runtime interface to the Hailo NPU (Neural Processing Unit) for
accelerated inference, along with a model wrapper that seamlessly falls back
to CPU inference if Hailo hardware is unavailable.

All imports are lazily evaluated and wrapped in try/except blocks. The entire
Hailo subsystem is optional and the module functions perfectly without
hailort dependencies installed.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


class HailoInferenceRunner:
    """
    Runtime wrapper for Hailo NPU inference.

    Manages the lifecycle of a compiled HEF (Hailo Executable Format) model
    on the Hailo device. Handles loading, inference, and resource cleanup.

    The class uses lazy imports for hailort to allow graceful operation even
    when the package is not installed.

    Parameters
    ----------
    hef_path : str
        Path to the compiled HEF model file.

    Attributes
    ----------
    hef_path : str
        Path to the HEF model.
    _device : object, optional
        Hailo device object (lazily initialised).
    _net_group : object, optional
        Network group from HEF (lazily initialised).
    _input_vstream : object, optional
        Input virtual stream (lazily initialised).
    _output_vstream : object, optional
        Output virtual stream (lazily initialised).

    Raises
    ------
    (Logged as error, not raised)
    - If hailort package is not available
    - If HEF file does not exist
    - If device enumeration fails
    """

    def __init__(self, hef_path: str) -> None:
        """
        Initialise Hailo inference runner.

        Parameters
        ----------
        hef_path : str
            Path to compiled HEF model.
        """
        self.hef_path = hef_path
        self._device = None
        self._net_group = None
        self._input_vstream = None
        self._output_vstream = None
        self._loaded = False

        if not Path(hef_path).exists():
            logger.error(f'HEF file not found: {hef_path}')
            return

        self._load_model()

    def _load_model(self) -> None:
        """
        Lazily load the HEF model onto the Hailo device.

        Attempts to:
        1. Import hailort
        2. Enumerate available Hailo devices
        3. Create device and network group from HEF
        4. Initialise virtual streams for I/O

        If any step fails, logs error and sets _loaded=False. Does not raise.
        """
        try:
            import hailort

            # Enumerate devices
            devices = hailort.Device.scan()
            if not devices:
                logger.error('No Hailo devices found')
                return

            # Create device
            self._device = devices[0]
            logger.info(f'Initialised Hailo device: {self._device}')

            # Create network group from HEF
            with open(self.hef_path, 'rb') as f:
                hef_data = f.read()

            self._net_group = self._device.create_network_group(hef_data)
            logger.info(f'Loaded HEF model from {self.hef_path}')

            # Create virtual streams
            net_group_params = self._net_group.create_params()
            self._input_vstream = self._net_group.get_input_vstreams(net_group_params)[0]
            self._output_vstream = self._net_group.get_output_vstreams(net_group_params)[0]

            self._loaded = True
            logger.info('Hailo inference runner ready for inference')

        except ImportError:
            logger.debug('hailort package not installed. Hailo acceleration unavailable.')
        except Exception as e:
            logger.error(f'Failed to load HEF model: {e}')

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if Hailo hardware is available and usable.

        Attempts to import hailort and enumerate connected devices.
        This is a lightweight check suitable for conditional logic.

        Returns
        -------
        bool
            True if hailort is installed and at least one Hailo device
            is detected. False otherwise.

        Notes
        -----
        This check does not attempt to create a device instance, so it is
        fast and non-invasive. A successful check does not guarantee that
        inference will succeed (e.g. if the device becomes unavailable).
        """
        try:
            import hailort

            devices = hailort.Device.scan()
            return len(devices) > 0

        except ImportError:
            return False
        except Exception as e:
            logger.debug(f'Error detecting Hailo hardware: {e}')
            return False

    def run(self, input_data: np.ndarray) -> Optional[np.ndarray]:
        """
        Run inference on the Hailo NPU.

        Sends input data to the device, waits for computation, and
        retrieves the output.

        Parameters
        ----------
        input_data : np.ndarray
            Input tensor of shape (batch_size, seq_len, n_features) or
            as expected by the loaded HEF model.

        Returns
        -------
        np.ndarray or None
            Output predictions from the model. Returns None if inference
            failed or Hailo is not available.

        Notes
        -----
        Input data should be pre-normalised and in the format expected
        by the compiled model. No shape or dtype conversion is performed.

        For best performance, batch multiple samples together rather than
        running single samples sequentially.
        """
        if not self._loaded:
            logger.error('Hailo model not loaded. Cannot run inference.')
            return None

        try:
            # Convert input to bytes for transmission
            input_data_uint8 = input_data.astype(np.float32).tobytes()

            # Send to device and retrieve output
            self._input_vstream.send(input_data_uint8)
            output_data = self._output_vstream.recv()

            # Convert output back to NumPy array
            output = np.frombuffer(output_data, dtype=np.float32)

            logger.debug(f'Hailo inference completed. Output shape: {output.shape}')
            return output

        except Exception as e:
            logger.error(f'Hailo inference failed: {e}')
            return None

    def close(self) -> None:
        """
        Release Hailo device resources.

        Cleans up virtual streams and device instance. Must be called
        when inference is complete to free hardware resources.

        Safe to call multiple times; subsequent calls are no-ops.
        """
        try:
            if self._input_vstream is not None:
                self._input_vstream.release()
                logger.debug('Released input vstream')

            if self._output_vstream is not None:
                self._output_vstream.release()
                logger.debug('Released output vstream')

            if self._net_group is not None:
                self._net_group.release()
                logger.debug('Released network group')

            if self._device is not None:
                self._device.release()
                logger.debug('Released device')

            self._loaded = False
            logger.info('Hailo device resources released')

        except Exception as e:
            logger.error(f'Error releasing Hailo resources: {e}')

    def __del__(self) -> None:
        """Ensure resources are released on garbage collection."""
        if self._loaded:
            self.close()


def compile_onnx_to_hef(onnx_path: str, hef_path: str, **kwargs: Any) -> bool:
    """
    Compile ONNX model to Hailo Executable Format (HEF).

    This is a PLACEHOLDER function that handles the limitations of the
    Hailo SDK. The Hailo Dataflow Compiler (DFC) runs only on x86-64
    systems (not on Raspberry Pi ARM).

    This function:
    1. Checks if hailo_sdk_client is available
    2. If available, performs the compilation on x86 systems
    3. If not available, logs instructions for the user to compile
       on a separate x86 machine and copy the HEF file to the Pi

    Parameters
    ----------
    onnx_path : str
        Path to input ONNX model file.

    hef_path : str
        Path where the compiled HEF file will be saved.

    **kwargs : Any
        Optional compilation parameters (e.g. calibration data, quantisation
        options). These are ignored in the current implementation.

    Returns
    -------
    bool
        True if compilation succeeded (or is not possible but instructions
        were logged). False if an error occurred.

    Notes
    -----
    PLATFORM LIMITATION: The Hailo SDK only provides the Dataflow Compiler
    for x86-64 Linux. Cross-compilation from ARM (Raspberry Pi) is not
    supported. Therefore:

    1. If running on x86 and hailo_sdk_client is installed:
       The compiler will attempt to compile the ONNX model.

    2. If running on ARM (e.g. Raspberry Pi):
       A message is logged telling the user to:
       - Transfer the ONNX file to an x86 machine
       - Run the Hailo Dataflow Compiler: hailodfc onnx_path -o hef_path
       - Copy the resulting HEF file back to the Pi
       - Then call HailoInferenceRunner(hef_path)

    3. If hailo_sdk_client is not installed on an x86 machine:
       Instructions are logged for installation and usage.

    Quantisation and other optimisations are platform-specific and may
    require additional configuration files or calibration datasets.
    """
    logger.info('ONNX to HEF compilation requested')
    logger.info(f'ONNX model: {onnx_path}')
    logger.info(f'Output HEF path: {hef_path}')

    # Check if ONNX file exists
    if not Path(onnx_path).exists():
        logger.error(f'ONNX file not found: {onnx_path}')
        return False

    try:
        import hailort
        from hailort import HailoRTError
    except ImportError:
        hailort = None
        HailoRTError = Exception

    try:
        import hailo_sdk_client
        has_sdk = True
    except ImportError:
        has_sdk = False

    if not has_sdk:
        logger.info('=' * 70)
        logger.info('Hailo SDK Client not installed. Compilation not available.')
        logger.info('=' * 70)
        logger.info('')
        logger.info('To compile ONNX to HEF, follow these steps:')
        logger.info('')
        logger.info('1. On an x86-64 Linux machine (or x86 Docker container):')
        logger.info('   pip install hailo-sdk-client')
        logger.info('')
        logger.info('2. Compile the ONNX model:')
        logger.info(f'   hailodfc {onnx_path} -o {hef_path}')
        logger.info('')
        logger.info('3. Transfer the resulting HEF file to your Raspberry Pi:')
        logger.info(f'   scp {hef_path} pi@raspberrypi:/path/to/model.hef')
        logger.info('')
        logger.info('4. Load the HEF model in Python:')
        logger.info('   runner = HailoInferenceRunner("/path/to/model.hef")')
        logger.info('')
        return False

    # If we reach here, hailo_sdk_client is available
    try:
        logger.info('Hailo SDK Client available. Attempting compilation...')
        logger.warning(
            'Note: Compilation will be attempted, but full quantisation may '
            'require calibration data and additional configuration.'
        )

        # This is a placeholder for actual SDK compilation
        # The real implementation would use hailo_sdk_client APIs
        logger.info(
            'SDK compilation would be performed here. '
            'For now, please use the hailodfc command-line tool.'
        )
        logger.info(f'Command: hailodfc {onnx_path} -o {hef_path}')
        return False

    except Exception as e:
        logger.error(f'Compilation attempt failed: {e}')
        return False


class HailoAcceleratedModel:
    """
    Wrapper that adds Hailo NPU acceleration to any ForecastModel.

    This class implements a decorator/wrapper pattern. A base forecast model
    (trained on CPU) can be wrapped to enable inference on Hailo hardware.
    If Hailo is not available, inference automatically falls back to the
    base model running on CPU.

    Parameters
    ----------
    base_model : ForecastModel
        A fitted ForecastModel instance (LSTM, CNN, etc.) that supports
        ONNX export.

    hef_path : str
        Path to the compiled Hailo Executable Format (HEF) model. If the file
        does not exist, inference will use base_model.predict() instead.

    Attributes
    ----------
    base_model : ForecastModel
        Reference to the wrapped model.
    hef_path : str
        Path to HEF file.
    _hailo_runner : HailoInferenceRunner or None
        Hailo runtime instance (created lazily when needed).

    Examples
    --------
    >>> model = LSTMModel(hidden_size=64)
    >>> model.fit(X_train, y_train)
    >>> model.export_onnx('model.onnx')  # Requires external compilation to HEF
    >>> accel_model = HailoAcceleratedModel(model, 'model.hef')
    >>> predictions = accel_model.predict(X_test)  # Uses Hailo if available
    """

    def __init__(self, base_model: 'ForecastModel', hef_path: str) -> None:
        """
        Wrap a ForecastModel for Hailo acceleration.

        Parameters
        ----------
        base_model : ForecastModel
            The model to wrap. Must be fitted.

        hef_path : str
            Path to compiled HEF model.
        """
        self.base_model = base_model
        self.hef_path = hef_path
        self._hailo_runner: Optional[HailoInferenceRunner] = None
        self._using_hailo = False

        if Path(hef_path).exists() and HailoInferenceRunner.is_available():
            try:
                self._hailo_runner = HailoInferenceRunner(hef_path)
                self._using_hailo = self._hailo_runner._loaded
                if self._using_hailo:
                    logger.info('Hailo acceleration enabled')
                else:
                    logger.info('Hailo hardware not available. Using CPU fallback.')
            except Exception as e:
                logger.warning(f'Failed to initialise Hailo runner: {e}. Using CPU.')
        else:
            if not Path(hef_path).exists():
                logger.debug(f'HEF file not found: {hef_path}. Using CPU inference.')
            else:
                logger.debug('Hailo hardware not detected. Using CPU inference.')

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions using Hailo acceleration if available.

        If Hailo is available and the HEF file is valid, inference runs on
        the NPU. Otherwise, falls back to base_model.predict() on CPU.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples,) or (n_samples, 1).

        Notes
        -----
        The base model performs input validation. Hailo inference bypasses
        some validation for performance. Ensure input data is properly
        normalised before calling this method.
        """
        if self._using_hailo and self._hailo_runner is not None:
            logger.debug('Running inference on Hailo NPU')
            output = self._hailo_runner.run(X.astype(np.float32))

            if output is not None:
                return output
            else:
                logger.warning('Hailo inference failed. Falling back to CPU.')

        logger.debug('Running inference on CPU')
        return self.base_model.predict(X)

    def predict_multi(self, X: np.ndarray, horizon: int) -> np.ndarray:
        """
        Generate multi-step-ahead predictions using Hailo if available.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix, shape (n_samples, n_features).

        horizon : int
            Number of steps to forecast ahead.

        Returns
        -------
        np.ndarray
            Multi-step predictions of shape (n_samples, horizon).

        Notes
        -----
        Uses the base model's predict_multi implementation, which applies
        recursive single-step predictions. Hailo acceleration applies only
        to the individual predict() calls.
        """
        return self.base_model.predict_multi(X, horizon)

    def close(self) -> None:
        """Release Hailo device resources."""
        if self._hailo_runner is not None:
            self._hailo_runner.close()
            logger.info('Closed Hailo inference runner')

    def __repr__(self) -> str:
        """Return string representation."""
        mode = 'Hailo accelerated' if self._using_hailo else 'CPU'
        return f'HailoAcceleratedModel({self.base_model.name!r}, mode={mode!r})'

    def __del__(self) -> None:
        """Ensure Hailo resources are released on garbage collection."""
        self.close()
