import tensorflow as tf
from ppo import multi_normal

def build_mlp(n_layers, input_placeholder, output_size, scope, size=64):
    with tf.variable_scope(scope):
        y = input_placeholder
        for i in range(n_layers):
            y = tf.layers.dense(y, size, activation=tf.tanh, use_bias=True)

    output = tf.layers.dense(y, output_size, use_bias=True)
    var_names = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)
    
    return output, var_names

class Policy:
    def __init__(self, name, state_placeholder, action_dim, continuous, n_layers, sigma=0.5):
        self.continuous = continuous
        self.state = state_placeholder
        self.std = tf.constant(0.2)
        activation = tf.tanh
        
        with tf.variable_scope(name):

            self.vpred = build_mlp(n_layers, self.state, 1, scope='value')
            self.logits, self.policy_vars = build_mlp(n_layers, self.state, action_dim, scope='policy')
            self.old_logits, self.old_policy_vars = build_mlp(n_layers, self.state, action_dim, scope='old_policy')


            self.sample_action = multi_normal(self.logits, sigma) \
                if continuous \
                   else tf.multinomial(self.logits - tf.reduce_max(self.logits, axis=1, keepdims=True), 1)

        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(self.old_policy_vars, self.policy_vars):
                self.assign_ops.append(tf.assign(v_old, v))

        
            
    def act(self, sess, obs):
        ac, v = sess.run([self.sample_action, self.vpred], feed_dict={self.state: obs[None]})
        return(ac[0], v[0]) if self.continuous else (ac[0][0], v[0][0])

    def save_policy(self, sess):
        return sess.run(self.assign_ops)

class Random_policy:
    def __init__(self, env):
        self.action = env.action_space.sample

    def act(self):
        return self.action(), np.random.randn()

