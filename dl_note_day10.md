### Day10: Learning Note of Deep neural networks (Video 5-8)
> 备注：今天的内容主要包括：搭建神经网络块；参数与超参数

#### 搭建神经网络块
以第$l$层神经网络块为例：
**正向传播块**  
输入： 本层的输入即前一层的输出$a^{[l-1]}$  
输出： 本层的输出 $a^{[l]}$
参数： 本层的参数 $W^{[l]}, b^{[l]}$
计算过程： $z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}, a^{[l]} = g^{[l]}(z^{[l]})$
trick: 把$z^{[l]}$的值缓存起来，因为缓存的值对以后的正向反向传播的步骤非常有用。

**反向传播块**  
输入： $\mathbf{d}a^{[l]}$（trick： 输入在这里其实是$\mathbf{d}a^{[l]}$ 以及所缓存的$z^{[l]}$的值）
输出： $\mathbf{d}a^{[l-1]}$（trick： 输出除了$\mathbf{d}a^{[l-1]}$ 也需要输出$\mathbf{d}W^{[l]}$和$\mathbf{d}b^{[l]}$, 为了实现梯度下降）

将上述组合，就可以形成神经网络的正向传播和反向传播过程。

#### 运算过程（之前也讲过，此处再次复习巩固，加深印象）
**正向传播**  
单样本：  
$$z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}$$
​$$a^{[l]} = g^{[l]}(z^{[l]})$$
向量化：
$$Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$$
​$$A^{[l]} = g^{[l]}(Z^{[l]})$$
trick: 前向传播需要喂入$A^{[0]}$也就是$X$来初始化；初始化的是第一层的输入值。

**反向传播**  
单样本：  
$$\mathbf{d}z^{[l]} = \mathbf{d}a^{[l]} * {g^{[l]}}^{\prime}$$
$$\mathbf{d}w^{[l]} = \mathbf{d}z^{[l]} . a^{[l-1]}$$
$$\mathbf{d}b^{[l]} = \mathbf{d}z^{[l]}$$
$$\mathbf{d}a^{[l-1]} = {w^{[l]}}^{\prime}\mathbf{d}z^{[l]}$$

向量化： 
$$\mathbf{d}Z^{[l]} = \mathbf{d}A^{[l]} * {g^{[l]}}^{\prime}$$ 
$$\mathbf{d}W^{[l]} = \frac1m \mathbf{d}Z^{[l]} {A^{[l-1]}}^{\prime}$$  
$$\mathbf{d}b^{[l]} = \frac1mnp.sum(\mathbf{d}Z^{[l]}, axis=1, keeplims=True)$$
$$\mathbf{d}A^{[l-1]} = {W^{[l]}}^{\prime}\mathbf{d}Z^{[l]}$$

#### 参数与超参数
想要深度神经网络起很好的效果，还需要规划好参数以及超参数。  
什么是超参数？  
比如算法中的learning rate a（学习率）、iterations(梯度下降法循环的数量)、L（隐藏层数目）、n[l]（隐藏层单元数目）、choice of activation function（激活函数的选择）都需要你来设置，这些数字实际上控制了最后的参数W和b的值，所以它们被称作超参数。  
实际上深度学习有很多不同的超参数，之后我们也会介绍一些其他的超参数，如momentum、mini batch size、regularization parameters等等。