### Day8: Learning Note of Shallow neural networks (Video 9-11)
> 备注：今天的内容主要包括：神经网络的梯度下降、随机初始化

#### 总结回顾：神经网络的参数表示
神经网络的参数包括：
$$w^{[1]}, b^{[1]},w^{[2]}, b^{[2]}$$
其中，输入数据（输入层）的特征数量记为$n_x$(或者$n^{[0]}$);  
隐藏层节点个数记为$n^{[1]}$，输出层节点个数记为$n^{[2]}$(备注：目前我们只讨论了$n^{[2]}=1$的情况)。  
即：  
$$w^{[i]}.shape = (n^{[i]},n^{[i-1]})$$
$$b^{[i]}.shape = (n^{[i]},1)$$
神经网络的代价函数：  
$$J() = \frac1m\sum_{i=1}^{m}L(\hat{y}, y) = \frac1m\sum_{i=1}^{m}L(a^{[2]}, y)$$

#### 神经网络的梯度下降算法流程
Repeat {
    Compute prediction $\hat{y}^{(i)}$(i = 1 to m)
    Compute $\mathbf{d}w^{[1]}$, $\mathbf{d}b^{[1]}$, $\mathbf{d}w^{[2]}$, $\mathbf{d}b^{[2]}$
    $w^{[1]} := w^{[1]} - \alpha\mathbf{d}w^{[1]}$  
    $b^{[1]} := b^{[1]} - \alpha\mathbf{d}b^{[1]}$  
    $w^{[2]} := w^{[2]} - \alpha\mathbf{d}w^{[2]}$  
    $b^{[2]} := b^{[2]} - \alpha\mathbf{d}b^{[2]}$
}
**关键在于如何计算其中的导数项！**

#### 前向传播与后向传播
**Forward Propagation (4个公式)**  
$$\mathbf{z}^{[1]} = \mathbf{w}^{[1]T}x + \mathbf{b}^{[1]}, \mathbf{a}^{[1]} = \mathbf{g}^{[1]}{(\mathbf{z}^{[1]})}$$
$$Z^{[2]} = \mathbf{w}^{[2]T}x + \mathbf{b}^{[2]}, A^{[2]} = \mathbf{g}^{[2]}{(Z^{[2]})} = \sigma{(Z^{[2]})}$$

**Back Propagation (6个公式)**  
$$\mathbf{d}Z^{[2]} = A^{[2]} - Y$$ 
$$\mathbf{d}w^{[2]} = \frac1m\mathbf{d}Z^{[2]}{A^{[1]}}^T$$ 
$$\mathbf{d}b^{[2]} = \frac1m np.sum(\mathbf{d}Z^{[2]}, axis=1, keepdims=True)$$
$$\mathbf{d}Z^{[1]} = {w^{[2]}}^T\mathbf{d}Z^{[2]} * {g^{[1]}}^\prime(Z^{[1]})$$
$$\mathbf{d}w^{[1]} = \frac1m\mathbf{d}Z^{[1]}X^T$$
$$\mathbf{d}b^{[1]} = \frac1m np.sum(\mathbf{d}Z^{[1]}, axis=1, keepdims=True)$$

#### 随机初始化
**对于一个神经网络，如果把权重或者参数都初始化为0，那么梯度下降将不会起作用。**  
如果初始化成0，那么隐藏层将会失去它的意义；  

随机初始化：  
$$w^{[1]} = np.random.randn((2,2)) * 0.01$$
$$b^{[1]} = np.zeros((2,1))$$
为什么要乘0.01？  初始化一个非常小的数，避免激活函数的梯度接近零。