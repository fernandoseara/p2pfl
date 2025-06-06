"""Custom Keras Model with de-biased gradient updates."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Optional

import numpy as np
import tensorflow as tf

from p2pfl.learning.frameworks.p2pfl_model import P2PFLModelDecorator
from p2pfl.management.logger import logger

if TYPE_CHECKING:
    from p2pfl.learning.frameworks.p2pfl_model import P2PFLModel


@tf.keras.utils.register_keras_serializable(package="p2pfl")
class DeBiasedAsyDFLKerasModel(tf.keras.Model):
    """
    Custom Keras Model with de-biased gradient updates.

    Args:
        base_model: The base model.
        de_biased_model: The de-biased model.

    """

    def __init__(self, model: tf.keras.Model, push_sum_weight: float = 1.0, last_training_loss: Optional[float] = None, **kwargs):
        """Initialize the model."""
        super().__init__(**kwargs)
        self.model = model
        self.push_sum_weight = tf.Variable(tf.constant(push_sum_weight, dtype=tf.float32))
        self.last_training_loss = last_training_loss

        self.loss = self.model.loss

    @property
    def optimizer(self):
        """
        Get the optimizer of the model.

        Returns:
            The optimizer.

        """
        return self.model.optimizer

    @optimizer.setter
    def optimizer(self, value):
        """
        Set the optimizer of the model.

        Args:
            value: The optimizer.

        """
        self.model.optimizer = value

    @tf.function
    def train_step(self, data):
        """
        Apply a de-biasing adjustment in a custom training step.

        Args:
            data: The training data.

        Returns:
            Dict: The training metrics.

        """
        # Unpack the data
        x, y = data[0], data[1]

        # Scale trainable variables by μ_t before the forward pass
        for var in self.model.trainable_variables:
            # Apply scaling before computing loss
            var.assign(var / self.push_sum_weight)  # Scale ω_t by μ_t

        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)  # Forward pass
            # Compute the loss value
            loss = self.compute_loss(y=y, y_pred=y_pred, training=True)

        # Compute gradients
        trainable_vars = self.model.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Restore trainable variables by multiplying them back by μ_t
        for var in self.model.trainable_variables:
            var.assign(var * self.push_sum_weight)  # Restore ω_t by multiplying with μ_t

        # Update weights using the computed gradients
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update metrics (includes the metric that tracks the loss)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=None) -> tf.Tensor:
        """
        Call the model with the given inputs.

        Args:
            inputs: The input data.
            training: Whether the model is in training mode.
            *args: Positional arguments for the model.
            **kwargs: Keyword arguments for the model.

        Returns:
            The model output.

        """
        return self.model(inputs, training=training)

    def get_config(self) -> dict:
        """
        Return the configuration of the model for saving.

        Returns:
            The configuration of the model.

        """
        config = super().get_config()
        config.update({
            "model": tf.keras.utils.serialize_keras_object(self.model),
            "push_sum_weight": self.push_sum_weight.numpy(),
            "last_training_loss": self.last_training_loss
        })
        return config

    @classmethod
    def from_config(cls, config: dict) -> "DeBiasedAsyDFLKerasModel":
        """
        Create an instance from the configuration dictionary.

        Args:
            config: The configuration dictionary.

        Returns:
            The model instance.

        """
        def load_model_from_config(model_config: dict) -> tf.keras.Model:
            """Dynamically load a model from its config."""
            module_name = model_config["module"]
            class_name = model_config["class_name"]

            # Dynamically import the module
            module = importlib.import_module(module_name)

            # Get the model class
            model_class = getattr(module, class_name)

            # Deserialize model instance
            return model_class.from_config(model_config["config"])

        # Load base_model and de_biased_model dynamically
        config["model"] = load_model_from_config(config["model"])
        return cls(**config)

    def get_weights(self) -> list[np.ndarray]:
        """Get the weights of the model."""
        return self.model.get_weights()

    def set_weights(self, weights: list[np.ndarray]) -> None:
        """
        Set the weights of the model.

        Args:
            weights: The weights.

        """
        self.model.set_weights(weights)


class AsyDFLKerasP2PFLModel(P2PFLModelDecorator):
    """
    Keras P2PFL Model with de-biased gradient updates.

    Args:
        model: The Keras model.
        push_sum_weight: The push sum weight for de-biasing.
        last_training_loss: The last training loss.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    """

    def __init__(self,
        wrapped_model: P2PFLModel,
        push_sum_weight: float = 1.0,
        last_training_loss: float | None = None,
        ) -> None:
        """Initialize the model."""
        if not isinstance(wrapped_model, DeBiasedAsyDFLKerasModel):
            # If the wrapped model is not already a DeBiasedAsyDFLKerasModel, wrap it
            debiased_model = DeBiasedAsyDFLKerasModel(
                wrapped_model.get_model(),
                push_sum_weight,
                last_training_loss,
            )
            wrapped_model.model = debiased_model
        super().__init__(wrapped_model)

    def get_push_sum_weight(self) -> float:
        """
        Get the push sum weight.

        Returns:
            The push sum weight.

        """
        return self.get_model().push_sum_weight.numpy()

    def set_push_sum_weight(self, weight: float) -> None:
        """
        Set the push sum weight.

        Args:
            weight: The push sum weight.

        """
        if not isinstance(weight, (float, int)):
            raise ValueError("Push sum weight must be a float or int.")
        self.get_model().push_sum_weight.assign(tf.constant(weight, dtype=tf.float32))

    def get_last_training_loss(self) -> float | None:
        """
        Get the last training loss.

        Returns:
            The last training loss or None if not set.

        """
        return self.get_model().last_training_loss

    def build_copy(self, **kwargs) -> "AsyDFLKerasP2PFLModel":
        """
        Build a copy of the model with the same configuration.

        Args:
            **kwargs: Additional keyword arguments for the copy.

        Returns:
            A new instance of AsyDFLKerasP2PFLModel with the same configuration.

        """
        copied_model = self._wrapped_model.build_copy(**kwargs)

        # Recover push_sum_weight and last_training_loss from the model if needed
        if isinstance(copied_model.model, DeBiasedAsyDFLKerasModel):
            push_sum_weight = copied_model.model.push_sum_weight.numpy()
            last_training_loss = copied_model.model.last_training_loss
        else:
            push_sum_weight = 1.0
            last_training_loss = None

        return AsyDFLKerasP2PFLModel(
            wrapped_model=copied_model,
            push_sum_weight=push_sum_weight,
            last_training_loss=last_training_loss
        )
