
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[3]:


x = tf.constant([100,200,300],name='x')
y = tf.constant([1,2,3],name='y')

sum_x = tf.reduce_sum(x,name="sum_x")#100 + 200 + 300
prod_y = tf.reduce_prod(y,name="prod_y")#1*2*3

final_div = tf.div(sum_x,prod_y,name="final_div")
final_mean = tf.reduce_mean([sum_x,prod_y],name="final_mean")


# In[6]:


sess = tf.Session()

print("x: ",sess.run(x))
print("y: ",sess.run(y))
print("sum(x): ",sess.run(sum_x))
print("prod(y): ",sess.run(prod_y))
print("sum(x)/prod(y): ",sess.run(final_div))
print("mean(suma(x),prod(y)): ",sess.run(final_mean))


# In[8]:


write = tf.summary.FileWriter('./m2_example3',sess.graph)
write.close()
sess.close()

