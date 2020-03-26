import numpy.random as rd
import tensorflow as tf

from toolbox.tensorflow_einsums.einsum_re_written import  einsum_bij_jk_to_bik

a = rd.rand(2,3,4)
b = rd.rand(4,5)

tf_a = tf.constant(a,)
tf_b = tf.placeholder(shape=(4,None),dtype=tf.float64)

prod1 = tf.einsum('bij,jk->bik',tf_a,tf_b)
prod2 = einsum_bij_jk_to_bik(tf_a,tf_b)

print(tf_b.get_shape())
print(prod2.get_shape())

sess = tf.Session()
np_prod_1 = sess.run(prod1, feed_dict={tf_b:b})
np_prod_2 = sess.run(prod2, feed_dict={tf_b:b})
assert (np_prod_1 == np_prod_2).all(), 'Mistmatch'
print('Prod 1')
print(np_prod_1)
print('Prod 2')
print(np_prod_2)