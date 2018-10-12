import tensorflow as tf
from network import build_mlp

def dist_continuous(logits):
    logstd = tf.get_variable(name='logstd', shape=[1, logits.get_shape()[1]],
                             initializer=tf.zeros_initializer())
    std = tf.zeros_like(logits) + tf.exp(logstd)
    dist = tf.distributions.Normal(loc=logits, scale=std)

    return dist, dist.sample(), [logstd]

def dist_discrete(logits):
    dist = tf.distributions.Multinomial(total_count=1.0, logits=logits)

    sample = dist.sample()
    sample = tf.argmax(sample, axis=1)

    return dist, sample, []

def get_dist(logits, continuous, scope):
    with tf.variable_scope(scope):
        dist = dist_continuous(logits) if continuous else dist_discrete(logits)

    return dist


class Agent:
    def __init__(self, continuous, ob_dim, action_dim, n_layers):
        self.continuous = continuous
        self.state = tf.placeholder(shape=[None, ob_dim], name="observations", dtype=tf.float32)


        pi       = build_mlp(2, self.state, action_dim, 'policy')
        old_pi   = build_mlp(2, self.state, action_dim, 'old_policy')
        vpred    = build_mlp(2, self.state, 1, 'vale_pred')

        dist     = get_dist(pi[0], continuous, 'dist')
        old_dist = get_dist(old_pi[0], continuous, 'old_dist')

        self.pi = pi[0]
        self.old_pi = old_pi[0]
        self.vpred = tf.squeeze(vpred[0], axis=1)
        
        self.dist = dist[0]
        self.old_dist = old_dist[0]
        self.sample_action = dist[1]
        

        self.tvars = pi[1] + vpred[1] + dist[2]


        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi[1], pi[1]):
                self.assign_ops.append(tf.assign(v_old, v))

        
    def act(self, sess, obs):
        ac, v = sess.run([self.sample_action, self.vpred], feed_dict={self.state: obs[None]})
        return(ac[0], v[0]) if self.continuous else (ac[0][0], v[0][0])

    def save_policy(self, sess):
        return sess.run(self.assign_ops)

# class Random_policy:
#     def __init__(self, env):
#         self.action = env.action_space.sample

#     def act(self):
#         return self.action(), np.random.randn()

