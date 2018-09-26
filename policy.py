import tensorflow as tf
from ppo import multi_normal

def build_mlp(n_layers, input_placeholder, output_size, scope, size=64):
    y = input_placeholder
    for i in range(n_layers):
        y = tf.layers.dense(y, size, activation=tf.tanh, use_bias=True)

    return tf.layers.dense(y, output_size, use_bias=True)

class Policy:
    def __init__(self, name, state_placeholder, action_dim, continuous, n_layers, sigma=0.5):
        self.state = state_placeholder
        self.std = tf.constant(0.2)
        activation = tf.tanh
        
        with tf.variable_scope(name):

            self.vpred = build_mlp(n_layers, self.state, 1, scope='value')
            self.logits = build_mlp(n_layers, self.state, action_dim, scope='policy')
            # use the batch_size as the size of values sampled from the normal distribution

            self.sample_action = multi_normal(self.logits, sigma) \
                if continuous \
                   else tf.multinomial(self.logits - tf.reduce_max(self.logits, 1, keepdims=True), 1)

            
    def act(self, sess, obs):
        ac, v = sess.run([self.sample_action, self.vpred], feed_dict={self.state: obs[None]})
        return ac[0], v[0]

class Random_policy:
    def __init__(self, env):
        self.action = env.action_space.sample

    def act(self):
        return self.action(), np.random.randn()

