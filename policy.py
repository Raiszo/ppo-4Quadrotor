import tensorflow as tf

def build_mlp(
        input_placeholder,
        output_size,
        scope,
        n_layers=2,
        size=64,
        activation=tf.tanh
        ):
    with tf.variable_scope(scope):
        y = input_placeholder
        
        for i in range(n_layers):
            y = tf.layers.dense(y, size, activation=activation, use_bias=True)

        y = tf.layers.dense(y,output_size, use_bias=True)
        return y
