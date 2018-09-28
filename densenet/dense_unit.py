import tensorflow as tf
l2 = tf.keras.regularizers.l2

class DenseUnit(tf.keras.Model):
    def __init__(self, k, weight_decay, momentum=0.99, epsilon=0.001):
        super(DenseUnit, self).__init__()

        self.bn1 = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
        self.conv1 = tf.keras.layers.Conv2D(filters=4 * k, use_bias=False, kernel_size=1, strides=1,
                                            padding="same", name="unit_conv", activation=None,
                                            kernel_initializer="he_normal", kernel_regularizer=l2(weight_decay))

        self.bn2 = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
        self.conv2 = tf.keras.layers.Conv2D(filters=k, use_bias=False, kernel_size=3, strides=1, padding="same",
                                            activation=None, kernel_initializer="he_normal",
                                            kernel_regularizer=l2(weight_decay))

    def call(self, inputs, training):
        net = self.bn1(inputs, training=training)
        net = tf.nn.relu(net)
        net = self.conv1(net)
        net = self.bn2(net, training=training)
        net = tf.nn.relu(net)
        net = self.conv2(net)
        return net