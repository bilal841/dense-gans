import tensorflow as tf
from densenet.transition_layer import *
from densenet.dense_block import *
l2 = tf.keras.regularizers.l2

class Discriminator(tf.keras.Model):
    def __init__(self, k, units_per_block, weight_decay=1e-4, num_classes=10, theta=1.0, momentum=0.99,
                 epsilon=0.001):
        super(Discriminator, self).__init__()

        # The initial convolution layer comprises 2k convolutions of size 7×7 with stride 2;
        self.conv1 = tf.keras.layers.Conv2D(filters=2 * k, kernel_size=7, strides=2, padding="same",
                                            kernel_initializer="he_normal",
                                            kernel_regularizer=l2(weight_decay))

        self.number_of_blocks = len(units_per_block)
        self.dense_blocks = self._add_cells(
            [DenseBlock(k=k, number_of_units=units_per_block[i], weight_decay=weight_decay,
                        momentum=momentum, epsilon=epsilon) for i in range(self.number_of_blocks)])

        self.transition_layers = self._add_cells(
            [TransitionLayer(theta=theta, depth=k * units_per_block[i], weight_decay=weight_decay,
                             momentum=momentum, epsilon=epsilon) for i in range(self.number_of_blocks - 1)])

        self.logits = tf.keras.layers.Dense(units=num_classes)

    def _add_cells(self, cells):
        # "Magic" required for keras.Model classes to track all the variables in
        # a list of layers.Layer objects.
        for i, c in enumerate(cells):
            setattr(self, "cell-%d" % i, c)
        return cells

    @tf.contrib.eager.defun
    def call(self, input, training):
        """Run the model."""
        net = self.conv1(input)
        print(net.shape)

        for block, transition in zip(self.dense_blocks[:-1], self.transition_layers):
            net = block(net, training=training)
            net = transition(net, training=training)
            print(net.shape)

        net = self.dense_blocks[-1](net, training=training)
        print(net.shape)

        net = tf.nn.relu(net)
        net = tf.reduce_sum(net, axis=[1,2])
        return self.logits(net)