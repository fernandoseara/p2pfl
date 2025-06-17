#
# This file is part of the federated_learning_p2p (p2pfl) distribution
# (see https://github.com/pguijas/p2pfl).
# Copyright (c) 2025 Pedro Guijas Bravo.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#

"""Simple MLP on Tensorflow Keras for MNIST."""

import tensorflow as tf  # type: ignore
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Resizing  # type: ignore
from tensorflow.keras.losses import SparseCategoricalCrossentropy  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

from p2pfl.learning.frameworks.tensorflow.keras_model import KerasP2PFLModel
from p2pfl.settings import Settings
from p2pfl.utils.seed import set_seed


####
# Example VGG16
####
@tf.keras.utils.register_keras_serializable("p2pfl")
class VGG16(tf.keras.Model):
    """VGG16-like CNN model for image classification using Keras."""

    def __init__(self, input_shape=(224, 224, 3), out_channels=10, lr_rate=0.001, **kwargs):
        """
        Initialize the VGG16-like CNN.

        Args:
            input_shape (tuple): Shape of the input images.
            out_channels (int): Number of output classes.
            lr_rate (float): Learning rate for the Adam optimizer.
            kwargs: Additional keyword arguments.

        """
        super().__init__(**kwargs)
        set_seed(Settings.general.SEED, "tensorflow")

        self.resize = Resizing(224, 224)

        # Block 1
        self.conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool1 = MaxPooling2D((2, 2))

        # Block 2
        self.conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.pool2 = MaxPooling2D((2, 2))

        # Block 3
        self.conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.pool3 = MaxPooling2D((2, 2))

        # Block 4
        self.conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.pool4 = MaxPooling2D((2, 2))

        # Block 5
        self.conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same')
        self.pool5 = MaxPooling2D((2, 2))

        # Fully connected layers
        self.flatten = Flatten()
        self.fc1 = Dense(4096, activation='relu')
        self.fc2 = Dense(4096, activation='relu')
        self.output_layer = Dense(out_channels)

        # Compile config
        self.loss = SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = Adam(learning_rate=lr_rate)

        # Force the model to be built
        self(tf.zeros((1, *input_shape)))

    def call(self, inputs):
        """Forward pass of the VGG16 model."""
        if tf.rank(inputs) == 3:
            inputs = tf.expand_dims(inputs, axis=0)  # Add batch dimension
        x = self.resize(inputs)

        x = self.conv1_1(x)
        x = self.conv1_2(x)
        x = self.pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.pool4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.pool5(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return self.output_layer(x)

def model_build_fn(*args, **kwargs) -> KerasP2PFLModel:
    """Export the model build function."""
    compression = kwargs.pop("compression", None)
    return KerasP2PFLModel(VGG16(*args, **kwargs), compression=compression)
