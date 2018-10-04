import tensorflow as tf

def build_mlp(n_layers, input_placeholder, output_size, size=64):
    y = input_placeholder
    for i in range(n_layers):
        y = tf.layers.dense(y, size, activation=tf.tanh, use_bias=True)

    return tf.layers.dense(y, output_size, use_bias=True)
