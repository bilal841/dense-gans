import tensorflow as tf
from densenet.dense_unit import DenseUnit
l2 = tf.keras.regularizers.l2

class DenseBlock(tf.keras.Model):
    def __init__(self, k, weight_decay, number_of_units, momentum=0.99, epsilon=0.001):
        super(DenseBlock, self).__init__()
        self.number_of_units = number_of_units
        self.units = self._add_cells([DenseUnit(k, weight_decay=weight_decay, momentum=momentum, epsilon=epsilon) for i in range(number_of_units)])

    def _add_cells(self, cells):
        # "Magic" required for keras.Model classes to track all the variables in
        # a list of layers.Layer objects.
        for i, c in enumerate(cells):
            setattr(self, "cell-%d" % i, c)
        return cells

    def call(self, x, training):
        x = self.units[0](x, training=training)
        for i in range(1, int(self.number_of_units)):
            output = self.units[i](x, training=training)
            x = tf.concat([x, output], axis=3)

        return x