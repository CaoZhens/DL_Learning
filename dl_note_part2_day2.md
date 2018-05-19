# Day1: 深度学习的实用层面(Practical aspects of Deep Learning) Video4-5
> 今天的主要学习内容包括：正则化； 正则化防止过拟合的直观解释；

## 逻辑回归中的正则化
**L2正则化**  
$$J(\mathbf{w},b) = \frac1m \sum_{i=1}^{m}{L(\hat{y}^{(i)}, y^{(i)})} + \frac{\lambda}{2m}||\mathbf{w} ||_2^2$$
其中：  
$$||\mathbf{w} ||_2^2 = \sum_{j=1}^{n_x}w_j^2 = \mathbf{w}^T\mathbf{w}$$
**L1正则化**  
$$J(\mathbf{w},b) = \frac1m \sum_{i=1}^{m}{L(\hat{y}^{(i)}, y^{(i)})} + \frac{\lambda}{m}||\mathbf{w} ||_1$$
其中：  
$$||\mathbf{w} ||_1 = \sum_{j=1}^{n_x}|w_j|$$
- $\lambda$是正则化参数，是一个需要调整的超参数 
- 为了方便写代码，在Python中，$\lambda$是一个保留字段，编写代码时，我们删掉a，写成lambd，以免与Python中的保留字段冲突

## 在神经网络中应用L2正则化
$$J(\mathbf{W}^{[1]}, b^{[1]},\mathbf{W}^{[2]}, b^{[2]},...,\mathbf{W}^{[L]}, b^{[L]}) = \frac1m \sum_{i=1}^{m}{L(\hat{y}^{(i)}, y^{(i)})} + \frac{\lambda}{2m}\sum_{l=1}^{L}||\mathbf{W}^{[l]} ||_F^2$$
其中：  
$$||\mathbf{W}^{[l]} ||_F^2 = \sum_{i=1}^{n^{[l]}}\sum_{j=1}^{n^{[l-1]}}(W_{ij})^2$$
*备注：视频中的这个公式应该写错了！此处更正*  
**该矩阵范数被称作弗罗贝尼乌斯范数（Frobenius Norm），用下标F标注**  

**包含正则化项的梯度下降**  
$$ \mathbf{W}^{[l]} := \mathbf{W}^{[l]} - \alpha\mathbf{d}\mathbf{W}^{[l]}$$
其中：  
$$\mathbf{d}\mathbf{W}^{[l]} = (calc by backprop) + \frac{\lambda}{m}\mathbf{W}^{[l]}$$
即：  
$$ \mathbf{W}^{[l]} := \mathbf{W}^{[l]} - \frac{\alpha\lambda}{m}\mathbf{W}^{[l]} = \alpha(calc by backprop)$$
**该正则项说明，不论$\mathbf{W}^{[l]}$是什么，我们都试图让它变得更小**

## 正则化为什么可以防止过拟合的直观解释
直观解释1:  
如果正则化参数$\lambda$设置得足够大，权重矩阵$\mathbf{W}$将被设置为接近于0的值，直观理解就是把多个隐藏单元的权重设为0（直觉上认为大量隐藏单元被完全消除了，其实不然，实际上是该神经网络的所有隐藏单元依然存在，但是它们的影响变得更小了。神经网络变得更简单了），于是基本上消除了这些隐藏单元的许多影响。如果是这种情况，这个被大大简化了的神经网络会变成一个很小的网络，小到如同一个逻辑回归单元，可是深度却很大，它会使这个网络从过度拟合的状态更接近高偏差状态。但是$\lambda$会存在一个中间值，于是会有一个接近“Just Right”的中间状态。  

直观解释2:  
假设使用双曲线激活函数（tanh）；
如果正则化参数$\lambda$设置得足够大，权重矩阵$\mathbf{W}$将会变得较小，那么$z$也较小；当$z$接近0时，激活函数大致呈线性；整个网络将会退化为一个简单线性模型，因此不会过拟合。