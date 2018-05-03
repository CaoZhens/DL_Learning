### Day2: Learning Note of Neutral Networks Basics (Vedio 1-4)
> 备注：今天的内容主要包括两部分：逻辑回归、梯度下降法；这两个知识点在之前CS229学习过；今天的笔记重点在于复习回顾，并重点对比了两门课程在讲解过程中的不同之处。

#### 二分类问题的一个例子：识别图片是不是猫？
知识点1: 图片的表示  
将图片保存成R/G/B三通道的三个矩阵，每个矩阵的大小取决于图片的像素值（如64*64像素）；  
将三个矩阵合并成一个特征向量$x$，向量$x$的总维度将是(以64✖️64像素为例): 64✖️️64✖️3，即
```python
x.shape = (12288, 1)
```  
知识点2:二分类问题的数学表示  
训练一个分类器， 以图片的特征向量$x$为输入， 输出预测结果$y \in (0, 1)$；即：图片是否是猫？  

#### 课程中常用的符号约定
单样本$\{\mathbf{x}^{(i)}, y^{(i)}\}$  
记特征个数为$n$，即：$\mathbf{x} = (x_{0}, x_{1}, ... , x_{n})^T$
记样本个数为$m$。  

样本矩阵：
$$\mathbf{X} = \left[
    \begin{matrix}
      \vdots & \vdots &  &\vdots  \\  
      {\mathbf{x}^1} & {\mathbf{x}^2} & \cdots & {\mathbf{x}^m}  \\
      \vdots & \vdots & & \vdots \\
    \end{matrix}
  \right],
  \mathbf{y} = \left[
      y^1, y^2,\cdots,y^m
  \right]$$
**在神经网络中，使用上面一种符号约定更好！**  

> 回顾 CS229中的样本矩阵表示约定：
$$\mathbf{X} = \left[
    \begin{matrix}
      \cdots & {\mathbf{x}^1}^T & \cdots  \\  
      \cdots & {\mathbf{x}^2}^T & \cdots  \\ 
       & \vdots &                         \\
      \cdots & {\mathbf{x}^m}^T & \cdots
    \end{matrix}
  \right],
  \mathbf{y} = \left[
    \begin{matrix}
      y^1  \\
      y^2  \\
      \vdots       \\
      y^m
    \end{matrix}  
  \right]$$

#### 逻辑回归
sigmoid函数  
$$ g(z) = \frac{1}{1+e^{-z}} = \frac{e^z}{1+e^z}$$
逻辑回归建模  
令$y \in [0, 1]$， 逻辑回归模型如下：
基于单个样本：
$$p(y=1|\mathbf{x};w,b) = h_{w,b}(\mathbf{x}) = g(w^T\mathbf{x}+b) = \frac{1}{1+e^{-(w^T\mathbf{x}+b)}}$$

基于多个样本：
$$p(\mathbf{y}=\vec{1}|\mathbf{X};w,b) = h_{w,b}(\mathbf{X}) = g(w^T\mathbf{X}+b) = \frac{1}{1+e^{-(w^T\mathbf{X}+b)}}$$

**在神经网络中，模型参数用$w$,$b$而不是$\theta$！**

> 回顾 CS229中的表示：
基于单个样本：
$$p(y=1|\mathbf{x};\theta) = h_\theta(\mathbf{x}) = g(\theta^T\mathbf{x}) = \frac{1}{1+e^{-\theta^T\mathbf{x}}}$$
基于多个样本：
$$p(\mathbf{y}=\vec{1}|\mathbf{X};\theta) = h_\theta(\mathbf{X}) = g(\mathbf{X}\theta) = \frac{1}{1+e^{-\mathbf{X}\theta}}$$

#### 逻辑回归的损失函数
损失函数用来衡量预测值和实际值的近似程度。在线性回归中我们使用了预测值和实际值的平方差，但是通常在逻辑回归中我们不这么做，因为当我们在学习逻辑回归参数的时候，会发现我们的优化目标不是凸优化，只能找到多个局部最优值，梯度下降法很可能找不到全局最优值。
**虽然平方差是一个不错的损失函数，但是在逻辑回归模型中，我们会定义另外一个损失函数。**  
$$L(y^i, \hat{y}^i) = y^i\log{\hat{y}^i} + (1-y^i)\log{(1-\hat{y}^i)} $$
**损失函数是在单个训练样本中定义的，它衡量的是算法在单个训练样本中表现如何，为了衡量算法在全部训练样本上的表现如何，我们需要定义一个算法的代价函数**  
$$J(w,b) = \frac1m\sum_{i=1}^m\left [y^i\log{\hat{y}^i} + (1-y^i)\log{(1-\hat{y}^i)} \right ]$$

#### 梯度下降法
**直接用自己之前实现的一个gif，应该可以形象地展示该算法**
<img src="https://github.com/CaoZhens/ML_Learning/blob/master/study/6_LinearRegression/pic/LinearR_GD_LossFuncSurface.gif" alt="" data-canonical-src="" width="600" height="600" />