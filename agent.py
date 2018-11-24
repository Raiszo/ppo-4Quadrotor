import tensorflow as tf
from network import build_mlp

def dist_continuous(logits):
    logstd = tf.get_variable(name='logstd', shape=[1, logits.get_shape()[1]],
                             # initializer=tf.constant_initializer(0))
                             initializer=tf.constant_initializer(-0.53))
                             # initializer=tf.zeros_initializer())
    std = tf.zeros_like(logits) + tf.exp(logstd)
    dist = tf.distributions.Normal(loc=logits, scale=std)
    sample = dist.sample()
    
    log_prob = dist.log_prob(sample)
    # log_prob = tf.squeeze(log_prob, axis=1)

    return dist, sample, log_prob

# def dist_discrete(logits):
#     dist = tf.distributions.Multinomial(total_count=1.0, logits=logits)

#     sample = dist.sample()
#     sample = tf.argmax(sample, axis=1)

#     return dist, sample, []

class Agent:
    def __init__(self, continuous, ob_dim, action_dim, n_layers):
        self.continuous = continuous
        self.state = tf.placeholder(shape=[None, ob_dim], name="observations", dtype=tf.float32)

        with tf.variable_scope('policy'):
            self.pi = pi = build_mlp(2, self.state, action_dim)
            self.dist, self.sample, self.log_prob = dist_continuous(pi)

        with tf.variable_scope('v_pred'):
            vpred = build_mlp(2, self.state, 1)
            self.vpred = tf.squeeze(vpred, axis=1)


    def act(self, obs, sess):
        ac, v, lp = sess.run([self.sample, self.vpred, self.log_prob], feed_dict={self.state: obs[None]})
        return(ac[0], v[0], lp[0]) if self.continuous else (ac[0][0], v[0][0])

    def act_deterministic(self, obs, sess):
        """
        Do not take the action stocastically, just use the logit
        """
        ac = sess.run(self.pi, feed_dict={self.state: obs[None]})

        return ac[0] if self.continuous else ac[0][0]
