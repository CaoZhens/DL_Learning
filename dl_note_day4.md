### Day4: Learning Note of Neutral Networks Basics (Vedio 10-14)
> 备注：今天的内容主要包括：m个样本的逻辑回归的梯度下降算法、向量化、向量化后的逻辑回归正向求解与反向求解。

#### m个样本的逻辑回归的梯度下降算法
回顾单个样本的逻辑回归的梯度下降：  
$$a^i = \hat{y}^i = \sigma{(z^i)} = \sigma{(w^Tx^i+b)} $$
求$$\frac{\partial{L(a^i, y^i)}}{\partial{w_1^i}}, \frac{\partial{L(a^i, y^i)}}{\partial{w_2^i}}, \frac{\partial{L(a^i, y^i)}}{\partial{b}}$$

**m个样本的总代价函数相当于m个单样本损失函数的和平均**
$$J(w, b) = \frac1m\sum_{i=1}^m{L(a^i, y^i)}$$

具体算法流程
> init: $J = 0$, $\mathbf{d}w_1=0$, $\mathbf{d}w_2=0$  
for i = 1 to m:  
    $z^i = w^Tx^i+b$  
    $a^i = \sigma{(z^i)}$  
    $J += -(y^i\log{(a^i)} + (1-y^i)\log{(1-a^i)})$  
    $\mathbf{d}z^i = a^i - y^i$  
    $\mathbf{d}w_1 += x_1^i\mathbf{d}z^i$  
    $\mathbf{d}w_2 += x_2^i\mathbf{d}z^i$  
    $\mathbf{d}b += \mathbf{d}z^i$   
$J /= m$  
$\mathbf{d}w_1 /= m$  
$\mathbf{d}w_2 /= m$  
$\mathbf{d}b /= m$  

**备注：上面的算法流程存在两个缺点： 即需要进行两次for循环：外循环为m个样本；内循环为n个特征。**

#### 向量化
对$w$向量化后，算法流程可简化为：  
> init: $J = 0$, $\mathbf{d}w$ = np.zeros($n_x$, 1)  
for i = 1 to m:  
    $z^i = w^Tx^i+b$  
    $a^i = \sigma{(z^i)}$  
    $J += -(y^i\log{(a^i)} + (1-y^i)\log{(1-a^i)})$  
    $\mathbf{d}z^i = a^i - y^i$  
    $\mathbf{d}w += x^i\mathbf{d}z^i$  
    $\mathbf{d}b += \mathbf{d}z^i$   
$J /= m$  
$\mathbf{d}w /= m$  
$\mathbf{d}b /= m$  

#### 向量化后的逻辑回归正向求解
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
  
即 ``X.shape=(n, m)``  
$$Z = w^T\mathbf{X}+b$$
即 ``Z = np.dot(w.T, X) + b``  
$$A = \sigma{(Z)}$$

#### 向量化后的逻辑回归反向求解梯度
> init: $J = 0$, $\mathbf{d}w$ = np.zeros($n_x$, 1)   
$\mathbf{d}Z = A - y$  
$\mathbf{d}w = \frac1m\mathbf{X}\mathbf{d}Z^T$  
$\mathbf{d}b = \frac1mnp.sum(\mathbf{d}Z)$  