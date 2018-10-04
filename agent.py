import tensorflow as tf
from network import build_mlp
from ppo import multi_normal

class Agent:
    def __init__(self, state_placeholder, action_dim, continuous, n_layers):
        self.continuous = continuous
        self.state = state_placeholder


        with tf.variable_scope('policy'):
            self.pi = build_mlp(2, state_placeholder, action_dim)
            logstd = tf.get_variable(name='logstd', shape=[1, action_dim],
                                     initializer=tf.zeros_initializer())
            self.std = tf.zeros_like(self.pi) + tf.exp(logstd)
            self.dist = tf.distributions.Normal(loc=self.pi, scale=self.std)
            pi_scope = tf.get_variable_scope().name
            pi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, pi_scope)

        self.vars = pi_vars
            
        with tf.variable_scope('old_policy'):
            self.old_pi = build_mlp(2, state_placeholder, action_dim)
            old_logstd = tf.get_variable(name='logstd', shape=[1, action_dim],
                                         initializer=tf.zeros_initializer())
            self.old_std = tf.zeros_like(self.old_pi) + tf.exp(old_logstd)
            self.old_dist = tf.distributions.Normal(loc=self.old_pi, scale=self.old_std)
            old_pi_scope = tf.get_variable_scope().name
            old_pi_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, old_pi_scope)

        with tf.variable_scope('value'):
            self.vpred = build_mlp(1, state_placeholder, action_dim)
            # vpred_vars = tf.get_variable_scope().name
        
        self.sample_action = self.dist.sample(1) \
                if continuous \
                   else tf.multinomial(self.pi - tf.reduce_max(self.pi, axis=1, keepdims=True), 1)

        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_vars, pi_vars):
                self.assign_ops.append(tf.assign(v_old, v))

        
        
    def act(self, sess, obs):
        ac, v = sess.run([self.sample_action, self.vpred], feed_dict={self.state: obs[None]})
        return(ac[0][0], v[0][0]) if self.continuous else (ac[0][0], v[0][0])

    def save_policy(self, sess):
        return sess.run(self.assign_ops)

class Random_policy:
    def __init__(self, env):
        self.action = env.action_space.sample

    def act(self):
        return self.action(), np.random.randn()

