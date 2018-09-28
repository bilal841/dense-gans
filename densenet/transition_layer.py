import tensorflow as tf
l2 = tf.keras.regularizers.l2

class TransitionLayer(tf.keras.Model):
    def __init__(self, theta, depth, weight_decay, momentum=0.99, epsilon=0.001):
        super(TransitionLayer, self).__init__()
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon)
        self.conv1 = tf.keras.layers.Conv2D(filters=int(theta * depth), use_bias=False, kernel_size=1, activation=None,
                                            kernel_initializer="he_normal", kernel_regularizer=l2(weight_decay),
                                            strides=1, padding="same")
        self.pool1 = tf.keras.layers.AveragePooling2D(pool_size=2, strides=2, padding="same")


    def call(self, inputs, training):
        net = self.bn1(inputs, training=training)
        net = tf.nn.relu(net)
        net = self.conv1(net)
        net = self.pool1(net)
        return net