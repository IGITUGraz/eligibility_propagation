import argparse
import subprocess as sp
import tensorflow as tf

cflags = " ".join(tf.sysconfig.get_compile_flags())
lflags = " ".join(tf.sysconfig.get_link_flags())
tf_inc = tf.sysconfig.get_include()
tf_lib = tf.sysconfig.get_lib()

parser = argparse.ArgumentParser()
parser.add_argument('ale_path', type=str, default='')
args = parser.parse_args()
ale_path = args.ale_path
if ale_path == '':
    print('[ ! must set ale_path ]')

cmd = f'g++ -std=c++11 -shared ale.cc -o tfaleop.so -fPIC -I {tf_inc} -O2 -D_GLIBCXX_USE_CXX11_ABI=1 -L{tf_lib} {cflags} {lflags} -I{ale_path}/include -L{ale_path}/lib -lale'
print(f'- compiling using command: {cmd}')
res = sp.check_call(cmd, shell=True)
if res == 0:
    print('[ sucessfully compiled ]')

