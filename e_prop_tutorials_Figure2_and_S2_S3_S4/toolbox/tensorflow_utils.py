import tensorflow as tf
import numpy.random as rd
import numpy as np
#import matplotlib.pyplot as plt
from collections import OrderedDict

def reduce_variance(v,axis=None):
    m = tf.reduce_mean(v,axis=axis)
    if axis is not None:
        m = tf.expand_dims(m,axis=axis)

    return tf.reduce_mean((v - m)**2,axis=axis)

def boolean_count(var,axis=-1):
    v = tf.cast(var,dtype=tf.int32)
    return tf.reduce_sum(v,axis=axis)

def variable_summaries(var,name=''):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope(name + 'Summary'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def tf_repeat(tensor,num, axis):
    with tf.name_scope('Repeat'):

        dims = len(tensor.get_shape())
        dtype = tensor.dtype
        assert dtype in [tf.float32,tf.float16,tf.float64,tf.int32,tf.int64,tf.bool], 'Data type not understood: ' + dtype.__str__

        # Generate a new dimension with the
        tensor = tf.expand_dims(tensor,axis=dims)
        exp = tf.ones(shape=np.concatenate([np.ones(dims,dtype=int),[num]]),dtype=dtype)
        tensor_exp = tensor * exp

        # Split and stack in the right dimension
        splitted = tf.unstack(tensor_exp,axis=axis)
        concatenated = tf.concat(splitted,axis=dims-1)

        # permute to put back the axis where it should be
        axis_permutation = np.arange(dims-1)
        axis_permutation = np.insert(axis_permutation,axis,dims-1)

        transposed = tf.transpose(concatenated,perm=axis_permutation)

        return transposed

def tf_repeat_test():

  a = np.arange(12).reshape(3,4)
  axis_a = 1
  num_a = 5

  b = rd.randn(3)
  axis_b = 0
  num_b = 4

  c = rd.randn(4,5,7,3,2)
  axis_c = 1
  num_c = 11

  sess = tf.Session()
  for tensor, axis, num in zip([a,b,c], [axis_a,axis_b,axis_c], [num_a,num_b,num_c]):

    res_np = np.repeat(tensor, repeats=num, axis=axis)
    res_tf = sess.run(tf_repeat(tf.constant(value=tensor,dtype=tf.float32),axis=axis,num=num))
    assert np.mean((res_np - res_tf)**2) < 1e-6, 'Repeat mismatched between np and tf: \n np: {} \n tf: {}'.format(res_np,res_tf)


  print('tf_repeat_test -> success')

def tf_downsample(tensor,new_size,axis):

    with tf.name_scope('Downsample'):
        dims = len(tensor.get_shape())

        splitted = tf.split(tensor,num_or_size_splits=new_size,axis=axis)
        stacked = tf.stack(splitted,axis=dims)
        reduced = tf.reduce_mean(stacked,axis=axis)

        permutation = np.arange(dims-1)
        permutation = np.insert(permutation,axis,dims-1)
        transposed = tf.transpose(reduced,perm=permutation)

        return transposed

def tf_downsample_test():

    a = np.array([1,2,1,2,4,6,4,6])
    sol_a = np.array([1.5,5.])
    axis_a = 0
    num_a = 2

    sol_c = rd.randn(4, 5, 7, 3, 2)
    axis_c = 1
    num_c = 5
    c = np.repeat(sol_c,repeats=11,axis=axis_c)

    sess = tf.Session()

    for t_np,axis,num,sol in zip([a,c],[axis_a,axis_c],[num_a,num_c],[sol_a,sol_c]):
        t = tf.constant(t_np,dtype=tf.float32)
        t_ds = tf_downsample(t,new_size=num,axis=axis)

        t_ds_np = sess.run(t_ds)
        assert np.sum((t_ds_np - sol)**2) < 1e-6, 'Failed test: mistmatch between downsampled: \n arg: {} \n output: {} \n should be: {}'.format(t_np,t_ds_np,sol)

    print('tf_downsample_test -> success')

def tf_roll(buffer, new_last_element=None, axis=0):
    with tf.name_scope('roll'):
        l_shp = len(buffer.get_shape())

        # Permute the index to roll over the right index
        perm = np.concatenate([[axis],np.arange(axis),np.arange(start=axis+1,stop=l_shp)])
        buffer = tf.transpose(buffer, perm=perm)

        # Add an element at the end of the buffer if requested, otherwise, add zero
        if new_last_element is None:
            shp = tf.shape(buffer)
            new_last_element = tf.zeros(shape=shp[1:], dtype=buffer.dtype)
        new_last_element = tf.expand_dims(new_last_element, axis=0)
        new_buffer = tf.concat([buffer[1:], new_last_element], axis=0, name='rolled')

        # Revert the index permutation
        inv_perm = np.argsort(perm)
        new_buffer = tf.transpose(new_buffer,perm=inv_perm)

        new_buffer = tf.identity(new_buffer,name='Roll')
        return new_buffer


def tf_tuple_of_placeholder(shape_named_tuple,dtype,default_named_tuple=None, name='TupleOfPlaceholder'):
    with tf.name_scope(name):
        placeholder_dict = OrderedDict({})

        if not(default_named_tuple) is None:
            default_dict = default_named_tuple._asdict()
            for k,v in default_dict.items():
                placeholder_dict[k] = tf.placeholder_with_default(v,v.get_shape(),name=k,dtype=dtype)
        else:
            shape_dict = shape_named_tuple._asdict()
            for k,v in shape_dict.items():
                placeholder_dict[k] = tf.placeholder(shape=v,dtype=dtype,name=k)

        placeholder_tuple = default_named_tuple.__class__(**placeholder_dict)
        return placeholder_tuple


def tf_feeding_dict_of_placeholder_tuple(tuple_of_placeholder,tuple_of_values):
    dict = {}
    for k,v in tuple_of_placeholder.__asdict().items():
        dict[v] = tuple_of_values.__asdict()[k]

    return dict


def moving_sum(tensor,n_steps):
    with tf.name_scope('MovingSum'):
        # Try to get the shape if int
        try: n_batch = int(tensor.get_shape()[0])
        except: n_batch = tf.shape(tensor)[0]

        try: n_time = int(tensor.get_shape()[1])
        except: n_time = tf.shape(tensor)[1]

        try: n_neuron = int(tensor.get_shape()[2])
        except: n_neuron = tf.shape(tensor)[2]


        shp = tensor.get_shape()
        assert len(shp) == 3, 'Shape tuple for time filtering should be of length 3, found {}'.format(shp)

        t0 = tf.constant(0, dtype=tf.int32, name="time")
        out = tf.TensorArray(dtype=tensor.dtype, size=n_time, element_shape=(n_batch,n_neuron))
        buffer = tf.zeros(shape=(n_batch,n_steps,n_neuron),dtype=tensor.dtype)

        def body(out, buffer, t):
            x = tensor[:,t,:]

            buffer = tf_roll(buffer, new_last_element=x, axis=1)
            new_y = tf.reduce_sum(buffer,axis=1)
            out = out.write(t, new_y)
            return (out, buffer, t + 1)

        def condition(out, buffer, t):
            return t < n_time

        out, _, _ = tf.while_loop(cond=condition, body=body, loop_vars=[out, buffer, t0])
        out = out.stack()
        out = tf.transpose(out, perm=[1, 0, 2])

        return out

def exp_convolve(tensor, decay):
    with tf.name_scope('ExpConvolve'):
        assert tensor.dtype in [tf.float16,tf.float32,tf.float64]

        tensor_time_major = tf.transpose(tensor, perm=[1, 0, 2])
        initializer = tf.zeros_like(tensor_time_major[0])

        filtered_tensor = tf.scan(lambda a, x: a * decay + (1-decay) * x,tensor_time_major,initializer=initializer)
        filtered_tensor = tf.transpose(filtered_tensor,perm=[1,0,2])
    return filtered_tensor

def discounted_return(reward,discount,axis=-1,boundary_value=0):
    with tf.name_scope('DiscountedReturn'):

        l_shp = len(reward.get_shape())
        assert l_shp >= 1, 'Tensor must be rank 1 or higher'

        axis = np.mod(axis,l_shp)
        perm = np.arange(l_shp)

        perm[0] = axis
        perm[axis] = 0


        t = tf.transpose(reward, perm=perm)
        t = tf.reverse(tensor=t,axis=[0])

        initializer = tf.ones_like(t[0]) * boundary_value
        t = tf.scan(lambda a, x: a * discount + x,t,initializer=initializer)
        t = tf.reverse(t,axis=[0])

        t = tf.transpose(t,perm=perm)
    return t

def tf_moving_sum_test():
    sess = tf.Session()

    def moving_sum_numpy(tensor,n_steps):
        n_batch,n_time,n_neuron = tensor.shape

        def zz(d):
            z = np.zeros(shape=(n_batch,d,n_neuron),dtype=tensor.dtype)
            return z

        stacks = [np.concatenate([zz(d),tensor[:,:n_time-d,:]],axis=1) for d in range(n_steps)]
        stacks = np.array(stacks)
        return np.sum(np.array(stacks),axis=0)

    def assert_quantitative_error(arr1,arr2):

        err = np.mean((arr1 - arr2) ** 2)
        if err > 1e-6:
            plt.plot(arr1[0, :, :],color='blue')
            plt.plot(arr2[0, :, :],color='green')
            plt.show()
            raise ValueError('Mistmatch of the smoothing with error {}'.format(err))

    # quick test
    a = np.array([0,1,2,4,1,2]).reshape((1,6,1))
    n_a = 2
    sol_a = np.array([0,1,3,6,5,3]).reshape((1,6,1))

    # Check the numpy function
    summed_np = moving_sum_numpy(a,n_a)
    assert_quantitative_error(sol_a,summed_np)


    # Check the tf function
    summed_tf = sess.run(moving_sum(tf.constant(a),n_a))
    assert_quantitative_error(sol_a,summed_tf)

    T = 100
    n_neuron = 10
    n_batch=3
    n_delay = 5

    tensor = rd.randn(n_batch,T,n_neuron)

    summed_np = moving_sum_numpy(tensor,n_delay)
    summed_tf = sess.run(moving_sum(tf.constant(tensor,dtype=tf.float32),n_delay))
    assert_quantitative_error(summed_np,summed_tf)

    print('tf_moving_sum_test -> success')

def tf_exp_convolve_test():
    sess = tf.Session()

    def exp_convolve_numpy(tensor,decay):
        n_batch,n_time,n_neuron = tensor.shape

        out = np.zeros_like(tensor,dtype=float)
        running = np.zeros_like(tensor[:,0,:],dtype=float)
        for t in range(n_time):
            out[:,t,:] = decay * running + (1-decay) * tensor[:,t,:]
            running = out[:,t,:]

        return out

    def assert_quantitative_error(arr_np, arr_tf):

        err = np.mean((arr_np - arr_tf) ** 2)
        if err > 1e-6:
            plt.plot(arr_np[0, :, :], color='blue', label='np')
            plt.plot(arr_tf[0, :, :], color='green', label='tf')
            plt.legend()
            plt.show()
            raise ValueError('Mistmatch of the smoothing with error {}'.format(err))

    # quick test
    a = np.array([0,1,2,4,1,2]).reshape((1,6,1))
    decay_a = 0.5

    # Check the numpy function
    summed_np = exp_convolve_numpy(a,decay_a)
    summed_tf = sess.run(exp_convolve(tf.constant(a,dtype=tf.float32),decay_a))
    assert_quantitative_error(summed_np,summed_tf)

    T = 100
    n_neuron = 10
    n_batch= 3
    decay = .5

    tensor = rd.randn(n_batch,T,n_neuron)

    summed_np = exp_convolve_numpy(tensor,decay)
    summed_tf = sess.run(exp_convolve(tf.constant(tensor,dtype=tf.float32),decay))
    assert_quantitative_error(summed_np,summed_tf)

    print('tf_exp_convolve_test -> success')

def tf_discounted_reward_test():

    g = .8

    a = [.1, 0, 1.]
    a_return = [.1 + g**2,g,1.]

    b = np.array([[1,2,3], a])
    b_return = np.array([[1 + 2*g + 3*g**2, 2 + 3*g,3], a_return])

    c = rd.rand(3,4,2)
    c_return = np.zeros_like(c)
    n_b,T,n = c.shape
    for i in range(n_b):
        tmp = np.zeros(n)
        for t in range(T):
            tmp =  g * tmp + c[i,T-1-t]
            c_return[i,T-1-t] = tmp

    sess = tf.Session()

    for t,t_return,axis in zip([a,b,c],[a_return,b_return,c_return],[-1,-1,1]):
        tf_return = discounted_return(tf.constant(t,dtype=tf.float32),g,axis=axis)
        np_return = sess.run(tf_return)
        assert np.sum((np_return - np.array(t_return))**2) < 1e-6, 'Mismatch: \n tensor {} \n solution {} \n found {}'.format(t,t_return,np_return)

if __name__ == '__main__':
  tf_repeat_test()
  tf_downsample_test()
  tf_moving_sum_test()
  tf_exp_convolve_test()
  tf_discounted_reward_test()