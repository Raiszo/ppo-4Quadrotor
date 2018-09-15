import tensorflow as tf

class Policy(
        input_placeholder,
        output_size,
        scope,
        n_layers=2,
        size=64,
        activation=tf.tanh
        ):
    def __init__(self, name, state):
        with tf.variable_scope(name):
            state = input_placeholder
            
            with tf.variable_scope('vf'):
                y = state
                for i in range(n_layers):
                    y = tf.layers.dense(y, size, activation=activation, use_bias=True)

                self.vpred = tf.layers.dense(y,output_size, use_bias=True)

            with tf.variable_scope('pol'):
                y = state
                for i in range(n_layers):
                    y = tf.layers.dense(y, size, activation=activation, use_bias=True)

                self.vpred = tf.layers.dense(y, 1, use_bias=True)

    def act(self):
        
