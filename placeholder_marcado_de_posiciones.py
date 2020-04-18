
# coding: utf-8

# In[16]:


import tensorflow as tf


# In[17]:


x = tf.placeholder(tf.int32,shape=[3],name='x')
y = tf.placeholder(tf.int32, shape=[3],name='y')

sum_x = tf.reduce_sum(x,name="sum_x")
prod_y = tf.reduce_prod(y,name="prod_y")

final_div = tf.div(sum_x,prod_y,name="final_div")
final_mean = tf.reduce_mean([sum_x,prod_y],name="final_mean")


# In[18]:


sess = tf.Session()


# In[19]:


print("sum(x): ",sess.run(sum_x,feed_dict={x:[100,200,300]}))
print("prod(y): ",sess.run(prod_y,feed_dict={y:[1,2,3]}))

#esto necesita a ambos x e y placeholder valores para este calculo
print("sum(x)/prod(y): ",sess.run(final_div,feed_dict={x: [10,20,30],y: [1,2,3]}))
print("mean(sum(x),prod(y)): ",sess.run(final_mean,feed_dict={x: [1000,2000,3000],y: [10,20,30]}))


# In[20]:


writer = tf.summary.FileWriter('./m4_example1',sess.graph)

writer.close()
sess.close()

