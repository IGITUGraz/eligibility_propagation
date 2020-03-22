import numpy.random as rd
import tensorflow as tf

from toolbox.einsum_re_writter.einsum_re_written import  einsum_bi_bij_to_bj

a = rd.rand(2,3)
b = rd.rand(2,3,4)

tf_a = tf.constant(a)
tf_b = tf.constant(b)

prod1 = tf.einsum('bi,bij->bj',tf_a,tf_b)
prod2 = einsum_bi_bij_to_bj(tf_a,tf_b)

sess = tf.Session()
np_prod_1 = sess.run(prod1)
np_prod_2 = sess.run(prod2)
assert (np_prod_1 == np_prod_2).all(), 'Mistmatch'
print('Prod 1')
print(np_prod_1)
print('Prod 2')
print(np_prod_2)