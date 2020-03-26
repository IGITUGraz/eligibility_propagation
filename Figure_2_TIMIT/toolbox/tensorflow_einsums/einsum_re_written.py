import tensorflow as tf

def einsum_bi_ijk_to_bjk(a,b):
    batch_size = tf.shape(a)[0]
    shp_a = a.get_shape()
    shp_b = b.get_shape()

    b_ = tf.reshape(b,(int(shp_b[0]), int(shp_b[1]) * int(shp_b[2])))
    ab_ = tf.matmul(a,b_)
    ab = tf.reshape(ab_,(batch_size,int(shp_b[1]),int(shp_b[2])))
    return ab


def einsum_bi_bij_to_bj(a,b):
    with tf.name_scope('Einsum-Bi-Bij-Bj'):
        a_ = tf.expand_dims(a,axis=1)
        a_b = tf.matmul(a_,b)
        ab = a_b[:,0,:]
    return ab

def einsum_bi_bijk_to_bjk(a,b):
    with tf.name_scope('Einsum-Bi-Bijk-Bjk'):
        a_ = a[:,:,None,None]
        a_b = a_ * b
        return tf.reduce_sum(a_b,axis=1)

def einsum_bij_jk_to_bik(a,b):
    try:
        n_b = int(a.get_shape()[0])
    except:
        n_b = tf.shape(a)[0]

    try:
        n_i = int(a.get_shape()[1])
    except:
        n_i = tf.shape(a)[1]

    try:
        n_j = int(a.get_shape()[2])
    except:
        n_j = tf.shape(a)[2]

    try:
        n_k = int(b.get_shape()[1])
    except:
        n_k = tf.shape(b)[1]

    a_ = tf.reshape(a,(n_b * n_i,n_j))
    a_b = tf.matmul(a_,b)
    ab = tf.reshape(a_b,(n_b,n_i,n_k))
    return ab


def einsum_bij_ki_to_bkj(a,b):

    # Write them as b k i j
    a_ = tf.expand_dims(a,axis=1)
    b_ = tf.expand_dims(b,axis=0)
    b_ = tf.expand_dims(b_,axis=3)

    ab = tf.reduce_sum(a_ * b_,axis=[2])
    return ab
