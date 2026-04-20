"""
Model registry and factory for creating forecast model instances.

Provides a centralised registry for all available forecast models,
with factory methods for instantiation and discovery.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from .base import ForecastModel

logger = logging.getLogger(__name__)

# Global registry instance
_default_registry: Optional['ModelRegistry'] = None


class ModelRegistry:
    """
    Registry and factory for forecast models.

    Manages registration of model classes and provides methods to:
    - Register new model implementations
    - Create instances of registered models
    - List available models
    - Create multiple models in batch

    The registry uses lazy instantiation: models are registered by class,
    not by instance, allowing flexible hyperparameter configuration.

    Attributes
    ----------
    _models : dict[str, Type[ForecastModel]]
        Mapping of model names to their class definitions.
    """

    def __init__(self) -> None:
        """Initialise empty registry."""
        self._models: Dict[str, Type[ForecastModel]] = {}
        logger.debug('ModelRegistry initialised')

    def register(self, name: str, model_class: Type[ForecastModel]) -> None:
        """
        Register a model class in the registry.

        Parameters
        ----------
        name : str
            Unique identifier for the model (e.g. 'lightgbm', 'lstm').
            Should be lowercase and hyphenated (e.g. 'random-forest').
        model_class : Type[ForecastModel]
            Class that inherits from ForecastModel. Must implement all
            abstract methods.

        Raises
        ------
        TypeError
            If model_class does not inherit from ForecastModel.
        ValueError
            If name is empty or model_class is not a class.

        Notes
        -----
        If a model with the same name is already registered, it will be
        overwritten. A warning will be logged in this case.

        Examples
        --------
        >>> from ml_forecast_lab.models import ModelRegistry, ForecastModel
        >>> registry = ModelRegistry()
        >>> class MyModel(ForecastModel):
        ...     @property
        ...     def name(self) -> str:
        ...         return 'my_model'
        ...     # ... implement other abstract methods
        >>> registry.register('my_model', MyModel)
        """
        if not name:
            raise ValueError('name cannot be empty')
        if not isinstance(model_class, type):
            raise ValueError(f'model_class must be a class, got {type(model_class)}')
        if not issubclass(model_class, ForecastModel):
            raise TypeError(
                f'model_class must inherit from ForecastModel, '
                f'got {model_class.__name__}'
            )

        if name in self._models:
            logger.warning(
                f'Overwriting existing model registration for {name!r}; '
                f'old class: {self._models[name].__name__}, '
                f'new class: {model_class.__name__}'
            )

        self._models[name] = model_class
        logger.info(f'Registered model {name!r} ({model_class.__name__})')

    def create(self, name: str, **kwargs: Any) -> ForecastModel:
        """
        Instantiate a registered model with optional hyperparameters.

        Parameters
        ----------
        name : str
            Name of the registered model.
        **kwargs : Any
            Hyperparameters to pass to the model's __init__() and set_params().
            Parameters are first passed to __init__, then any remaining
            parameters are passed to set_params().

        Returns
        -------
        ForecastModel
            Fully initialised and configured model instance.

        Raises
        ------
        KeyError
            If the model name is not registered.
        TypeError
            If kwargs contains invalid hyperparameters.

        Examples
        --------
        >>> registry = get_registry()
        >>> model = registry.create('lightgbm', num_leaves=31, learning_rate=0.1)
        >>> model.name
        'lightgbm'
        """
        if name not in self._models:
            available = self.list_available()
            raise KeyError(
                f'Model {name!r} not registered. '
                f'Available models: {available}'
            )

        model_class = self._models[name]
        logger.debug(f'Creating instance of {name!r}')

        # Instantiate with no args, then configure
        model = model_class()
        if kwargs:
            model.set_params(**kwargs)
            logger.debug(f'Set hyperparameters for {name!r}: {kwargs}')

        return model

    def list_available(self) -> List[str]:
        """
        Return list of all registered model names.

        Returns
        -------
        list[str]
            Sorted list of model identifiers.

        Examples
        --------
        >>> registry = get_registry()
        >>> models = registry.list_available()
        >>> print(models)
        ['lightgbm', 'lstm', 'xgboost', ...]
        """
        return sorted(self._models.keys())

    def create_all(
        self,
        names: List[str],
        **kwargs: Any,
    ) -> Dict[str, ForecastModel]:
        """
        Create multiple models at once.

        Useful for comparing multiple architectures with the same
        hyperparameters.

        Parameters
        ----------
        names : list[str]
            List of model names to instantiate.
        **kwargs : Any
            Hyperparameters to apply to all models.

        Returns
        -------
        dict[str, ForecastModel]
            Mapping of model names to their instances.

        Raises
        ------
        KeyError
            If any name is not registered.

        Examples
        --------
        >>> registry = get_registry()
        >>> models = registry.create_all(
        ...     ['lightgbm', 'xgboost'],
        ...     learning_rate=0.05
        ... )
        >>> for name, model in models.items():
        ...     print(f'{name}: {model}')
        """
        results = {}
        failed = []

        for name in names:
            try:
                results[name] = self.create(name, **kwargs)
                logger.debug(f'Successfully created {name!r}')
            except Exception as e:
                logger.error(f'Failed to create {name!r}: {e}', exc_info=True)
                failed.append((name, str(e)))

        if failed:
            logger.warning(
                f'Failed to create {len(failed)} model(s): {failed}'
            )

        return results

    def is_registered(self, name: str) -> bool:
        """
        Check whether a model is registered.

        Parameters
        ----------
        name : str
            Model name to check.

        Returns
        -------
        bool
            True if model is registered, False otherwise.
        """
        return name in self._models

    def __repr__(self) -> str:
        """Return string representation of registry."""
        count = len(self._models)
        return f'ModelRegistry({count} models: {self.list_available()})'


def get_registry() -> ModelRegistry:
    """
    Get the global default model registry.

    Returns
    -------
    ModelRegistry
        The shared registry instance used throughout the application.

    Notes
    -----
    The default registry is initialised on first call with built-in
    models pre-registered. Subsequent calls return the same instance.

    Examples
    --------
    >>> registry = get_registry()
    >>> model = registry.create('lightgbm')
    """
    global _default_registry

    if _default_registry is None:
        _default_registry = ModelRegistry()
        _initialise_default_registry(_default_registry)

    return _default_registry


def _initialise_default_registry(registry: ModelRegistry) -> None:
    """
    Populate the default registry with built-in models.

    This function is called once when the global registry is first accessed.
    Built-in models are those defined in the ml_forecast_lab.models package.

    Parameters
    ----------
    registry : ModelRegistry
        The registry instance to populate.

    Notes
    -----
    Currently, this is a no-op placeholder. Concrete model implementations
    (LightGBM, LSTM, etc.) should be registered here as they are created.

    Examples of registration:
    >>> from ml_forecast_lab.models.lightgbm_model import LightGBMModel
    >>> registry.register('lightgbm', LightGBMModel)
    """
    logger.info('Initialising default model registry with built-in models')

    # Models will be registered here as they are implemented
    # Example:
    # from .lightgbm_model import LightGBMModel
    # registry.register('lightgbm', LightGBMModel)
    #
    # from .lstm_model import LSTMModel
    # registry.register('lstm', LSTMModel)

    logger.debug(f'Default registry initialised with {len(registry.list_available())} models')
