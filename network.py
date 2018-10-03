import tensorflow as tf

class Network:
    def __init__(self, n_layers, input_placeholder, output_size, scope, size=64):
        with tf.variable_scope(scope):
            y = input_placeholder
            for i in range(n_layers):
                y = tf.layers.dense(y, size, activation=tf.tanh, use_bias=True)

            self.logits = tf.layers.dense(y, output_size, use_bias=True)
            self.scope = tf.get_variable_scope().name


    def get_vars(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
