
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[5]:


a = tf.constant(6.5,name='constant_a')
b = tf.constant(3.4,name='constant_b')
c = tf.constant(3.0,name='constant_c')
d = tf.constant(100.2,name='constant_d')

square = tf.square(a,name="square_a")
power = tf.pow(b,c,name="power_b_c")
sqrt = tf.sqrt(d,name="sqrt_d")

final_sum  = tf.add_n([square,power,sqrt],name="final_sum")

sess  = tf.Session()

print("Square of a: ",sess.run(square))
print("power of b ^ c: ",sess.run(power))
print("sqrt root of d: ",sess.run(sqrt))

print("sum of square, power and sqrt root: ",sess.run(final_sum))


# In[8]:


write = tf.summary.FileWriter('./m2_example2',sess.graph)
write.close()
sess.close()

