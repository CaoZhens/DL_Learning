# Day10: 超参数调试、Batch正则化和程序框架 Video 9-11
> 今天的主要学习内容包括：Softmax分类器的训练；深度学习框架； tenserflow；  
## 训练一个softmax分类器
以$\mathbf{z}^{[l]} = [5, 2, -1, 3]$为例：  
临时变量$t = [e^5, e^2, e^{-1}, e^3]$  
输出$\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]}) = \left[
    \begin{matrix}
       e^5 / (e^5+e^2+e^{-1}+e^3)   \\  
       e^2 / (e^5+e^2+e^{-1}+e^3)   \\
       e^{-1} / (e^5+e^2+e^{-1}+e^3)   \\
       e^3 / (e^5+e^2+e^{-1}+e^3)   \\
    \end{matrix}
  \right] = \left[
    \begin{matrix}
       0.842\\  
       0.042\\
       0.002\\
       0.114\\
    \end{matrix}
  \right]$

Softmax这个名称的来源是与所谓hardmax对比:  
- hardmax函数会观察$\mathbf{z}$的元素，然后在$\mathbf{z}$中最大元素的位置放上1，其它位置放上0;会把向量$\mathbf{z}$变成这个向量$[1,0,0,0]$

## 深度学习框架
选择框架的基本标准
- 便于编程
- 运行速度
- 是否真的开放

常见的框架
- Caffe／Caffe2
- CNTK
- DL4J
- Keras
- Lasagne
- mxnet
- PaddlePaddle
- Tensorflow
- Theano
- Torch

## Tensorflow基础
```python
import numpy as np
import tensorflow as tf

w = tf.Variable(0,dtype = tf.float32)
# 定义参数w，在TensorFlow中用tf.Variable()来定义参数

# 然后定义损失函数：

cost = tf.add(tf.add(w**2,tf.multiply(- 10.,w)),25)

train = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
# 0.01的学习率

#最后下面的几行是惯用表达式:

init = tf.global_variables_initializer()

session = tf.Session()#这样就开启了一个TensorFlow session。

session.run(init)#来初始化全局变量。

#然后让TensorFlow评估一个变量，我们要用到:

session.run(w)

#上面的这一行将w初始化为0，并定义损失函数，我们定义train为学习算法，它用梯度下降法优化器使损失函数最小化，但实际上我们还没有运行学习算法，所以#上面的这一行将w初始化为0，并定义损失函数，我们定义train为学习算法，它用梯度下降法优化器使损失函数最小化，但实际上我们还没有运行学习算法，所以session.run(w)评估了w，让我：：

print(session.run(w))
```