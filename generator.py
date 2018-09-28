import tensorflow as tf
from densenet.upsample_layer import *
from densenet.dense_block import *
l2 = tf.keras.regularizers.l2

class Generator(tf.keras.Model):
    def __init__(self, k, units_per_block, weight_decay=1e-4, num_classes=10, theta=1.0, momentum=0.99,
                 epsilon=0.001):
        super(Generator, self).__init__()

        self.linear = tf.keras.layers.Dense(units=4 * 4 * 512, activation=None)
        self.number_of_blocks = len(units_per_block)
        self.dense_blocks = self._add_cells(
            [DenseBlock(k=k, number_of_units=units_per_block[i], weight_decay=weight_decay,
                        momentum=momentum, epsilon=epsilon) for i in range(self.number_of_blocks)])

        self.transition_layers = self._add_cells(
            [UpsampleLayer(theta=theta, depth=k * units_per_block[i], weight_decay=weight_decay,
                             momentum=momentum, epsilon=epsilon) for i in range(self.number_of_blocks)])

        self.bn = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
        self.conv = tf.keras.layers.Conv2D(filters=num_classes, use_bias=False, kernel_size=3, activation=None,
                                            kernel_initializer="he_normal", kernel_regularizer=l2(weight_decay),
                                            strides=1, padding="same")
        self.out = tf.keras.layers.Activation(activation='tanh')

    def _add_cells(self, cells):
        # "Magic" required for keras.Model classes to track all the variables in
        # a list of layers.Layer objects.
        for i, c in enumerate(cells):
            setattr(self, "cell-%d" % i, c)
        return cells

    @tf.contrib.eager.defun
    def call(self, input, training):
        """Run the model."""
        net = self.linear(input)
        net = tf.reshape(net, (-1, 4, 4, 512))

        for block, transition in zip(self.dense_blocks, self.transition_layers):
            net = block(net, training=training)
            net = transition(net, training=training)

        net = self.bn(net, training=training)
        net = tf.nn.relu(net)
        net = self.conv(net)

        return self.out(net)