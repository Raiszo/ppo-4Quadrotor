import tensorflow as tf

def build_mlp(n_layers, input_placeholder, output_size):
    y = input_placeholder
    for i in range(n_layers):
        y = tf.layers.dense(y, size, activation=tf.tanh, use_bias=True)

    return tf.layers.dense(y, output_size, use_bias=True)

class Policy:
    def __init__(self, name, state_placeholder, action_dim, n_layers, continuos=True):
        self.state = state_placeholder
        activation = tf.tanh
        
        with tf.variable_scope(name):
            
            with tf.variable_scope('vf'):
                y = self.state
                for i in range(n_layers):
                    y = tf.layers.dense(y, 64, activation=activation, use_bias=True)

                self.vpred = tf.layers.dense(y, 1, use_bias=True)

            with tf.variable_scope('pol'):
                y = self.state
                for i in range(n_layers):
                    y = tf.layers.dense(y, 64, activation=activation, use_bias=True)

                self.action = tf.layers.dense(y, action_dim, use_bias=True)

    def act(self, sess, obs):
        ac, v = sess.run([self.action, self.vpred], feed_dict={self.state: obs[None]})
        return ac[0], v[0]


class Random_policy:
    def __init__(self, env):
        self.action = env.action_space.sample

    def act(self):
        return self.action(), np.random.randn()

