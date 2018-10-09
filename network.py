import tensorflow as tf

def build_mlp(n_layers, input_placeholder, output_size, scope, size=64):

    with tf.variable_scope(scope):
        y = input_placeholder
        for i in range(n_layers):
            y = tf.layers.dense(y, size, activation=tf.tanh, use_bias=True)

        y = tf.layers.dense(y, output_size, use_bias=True)

        scope = tf.get_variable_scope().name
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, pi_scope)

    return y, variables
