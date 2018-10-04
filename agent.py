import tensorflow as tf
from network import Network
from ppo import multi_normal

class Agent:
    def __init__(self, name, state_placeholder, action_dim, continuous, n_layers, std=0.5):
        self.continuous = continuous
        self.state = state_placeholder
        self.std = tf.constant(std)


        self.pi = Network(2, state_placeholder, action_dim, 'policy')
        self.old_pi = Network(2, state_placeholder, action_dim, 'old_policy')
        self.v_func = Network(2, state_placeholder, action_dim, 'value_estimator')
        
        self.pi_vars = self.pi.get_vars()
        self.old_pi_vars = self.old_pi.get_vars()

        self.sample_action = multi_normal(self.pi.logits, sigma) \
                if continuous \
                   else tf.multinomial(self.pi.logits - tf.reduce_max(self.pi.logits, axis=1, keepdims=True), 1)
        self.vpred = self.v_func.logits

        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(self.old_pi_vars, self.pi_vars):
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

