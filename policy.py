import tensorflow as tf

def build_mlp(n_layers, input_placeholder, output_size, size=64):
    y = input_placeholder
    for i in range(n_layers):
        y = tf.layers.dense(y, size, activation=tf.tanh, use_bias=True)

    return tf.layers.dense(y, output_size, use_bias=True)

class Policy:
    def __init__(self, name, state_placeholder, action_dim, n_layers, continuos=True):
        self.state = state_placeholder
        activation = tf.tanh
        
        with tf.variable_scope(name):

            self.vpred = build_mlp(2, self.state, 1)
            self.action = build_mlp(2, self.state, action_dim)

    def act(self, sess, obs):
        ac, v = sess.run([self.action, self.vpred], feed_dict={self.state: obs[None]})
        return ac[0], v[0]

class Random_policy:
    def __init__(self, env):
        self.action = env.action_space.sample

    def act(self):
        return self.action(), np.random.randn()

