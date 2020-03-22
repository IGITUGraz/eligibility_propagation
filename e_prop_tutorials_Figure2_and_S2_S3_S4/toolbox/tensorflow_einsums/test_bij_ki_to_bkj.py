import numpy as np
import numpy.random as rd
import tensorflow as tf

from toolbox.einsum_re_writter.einsum_re_written import einsum_bij_ki_to_bkj

b = 2
i,j,k = 3,4,5

a = rd.rand(b,i,j)
b = rd.rand(k,i)

tf_a = tf.constant(a)
tf_b = tf.constant(b)

prod2 = einsum_bij_ki_to_bkj(tf_a,tf_b)

sess = tf.Session()
np_prod_1 = np.einsum('bij,ki->bkj',a,b)
np_prod_2 = sess.run(prod2)
assert (np_prod_1 == np_prod_2).all(), 'Mistmatch'
print('Prod 1')
print(np_prod_1)
print('Prod 2')
print(np_prod_2)