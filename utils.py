import tensorflow as tf

def get_session():
    # tf_config = tf.ConfigProto(
    #     inter_op_parallelism_threads=1,
    #     intra_op_parallelism_threads=1
    # )

    return tf.Session()
    # return tf.Session(config=tf_config)
    
