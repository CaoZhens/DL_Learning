## Day8: Learning Note of Shallow neural networks (Video 9-11)
> 备注：今天的内容主要包括：神经网络的梯度下降、随机初始化

### 回顾：神经网络的参数表示
仍以前面给出的二层神经网络为例：  
<img src="https://github.com/CaoZhens/DL_Learning/blob/master/pic/shallow_nn_picture_4.png" alt="" data-canonical-src="" width="400" height="200" />  
**神经网络的参数**  
$$\mathbf{W}^{[1]}, \mathbf{b}^{[1]},\mathbf{W}^{[2]}, \mathbf{b}^{[2]}$$
其中，输入数据（输入层）的特征数量记为$n_x$(或者$n^{[0]}$);  
隐藏层节点个数记为$n^{[1]}$，输出层节点个数记为$n^{[2]}$(备注：目前我们只讨论$n^{[2]}=1$的情况)。  
即：  
```python
W^{[i]}.shape = (n^{[i]},n^{[i-1]})
b^{[i]}.shape = (n^{[i]},1)
```
**神经网络的代价函数**  
备注：假设是二元分类问题  
$$J(\mathbf{W}^{[1]}, \mathbf{b}^{[1]},\mathbf{W}^{[2]}, \mathbf{b}^{[2]}) = \frac1m\sum_{i=1}^{m}L(\hat{y}^{(i)}, y^{(i)}) = \frac1m\sum_{i=1}^{m}L(a^{[2](i)}, y^{(i)})$$

### 神经网络的梯度下降算法流程
Repeat {
    Compute prediction $\hat{y}^{(i)}$(i = 1 to m)
    Compute $\mathbf{dW}^{[1]}=\frac{\mathbf{d}J}{\mathbf{dW}^{[1]}}$, $\mathbf{db}^{[1]}$, $\mathbf{dW}^{[2]}$, $\mathbf{db}^{[2]}$
    $\mathbf{W}^{[1]} := \mathbf{W}^{[1]} - \alpha\mathbf{dW}^{[1]}$  
    $\mathbf{b}^{[1]} := \mathbf{b}^{[1]} - \alpha\mathbf{db}^{[1]}$  
    $\mathbf{W}^{[2]} := \mathbf{W}^{[2]} - \alpha\mathbf{dW}^{[2]}$  
    $\mathbf{b}^{[2]} := \mathbf{b}^{[2]} - \alpha\mathbf{db}^{[2]}$
}
**关键在于如何计算其中的导数项！**

### 前向传播与后向传播
**Forward Propagation (回顾：4个公式)**  
备注：前向传播的具体推导过程可参考[dl_note_day6](dl_note_day6.md)
$$\mathbf{Z}^{[1]} = \mathbf{W}^{[1]}\mathbf{X} + \mathbf{b}^{[1]}$$
$$\mathbf{A}^{[1]} = {g}^{[1]}{(\mathbf{Z}^{[1]})}$$
$$\mathbf{Z}^{[2]} = \mathbf{W}^{[2]}\mathbf{A}^{[1]} + \mathbf{b}^{[2]}$$
$$\mathbf{A}^{[2]} = {g}^{[2]}{(\mathbf{Z}^{[2]})} = \sigma{(\mathbf{Z}^{[2]})}$$

**Back Propagation (6个公式)**  
$$\mathbf{dZ}^{[2]} = \mathbf{A}^{[2]} - \mathbf{Y}$$

(备注：$\mathbf{Y} = \left[y^{(1)}, y^{(2)}, \cdots, y^{(m)} \right]$)  

$$\mathbf{dW}^{[2]} = \frac1m\mathbf{dZ}^{[2]}{\mathbf{A}^{[1]}}^T$$

$$\mathbf{db}^{[2]} = \frac1m np.sum(\mathbf{dZ}^{[2]}, axis=1, keepdims=True)$$

$$\mathbf{dZ}^{[1]} = {\mathbf{W}^{[2]}}^T\mathbf{dZ}^{[2]} * {g^{[1]}}^\prime(\mathbf{Z}^{[1]})$$

$$\mathbf{dW}^{[1]} = \frac1m\mathbf{dZ}^{[1]}\mathbf{X}^T$$

$$\mathbf{db}^{[1]} = \frac1m np.sum(\mathbf{dZ}^{[1]}, axis=1, keepdims=True)$$


#### 反向传播的推导过程
以单层神经网络（单样本）为例：  
$$L(a, y) = -(y\log(a)+(1-y)\log(1-a))$$

$$\mathbf{d}a =\frac{\mathbf{d}L}{\mathbf{d}a} = -\frac{y}{a} + \frac{1-y}{1-a}$$

$$\mathbf{dz} = \frac{\mathbf{d}L}{\mathbf{d}a}\frac{\mathbf{d}a}{\mathbf{dz}} = \mathbf{d}a g^{\prime}{(\mathbf{z})}
= \left( -\frac{y}{a} + \frac{1-y}{1-a} \right) \left[ a(1-a) \right] = a - y$$

$$\mathbf{dw} = \frac{\mathbf{d}L}{\mathbf{dz}} \frac{\mathbf{dz}}{\mathbf{dw}} = \mathbf{dz} \mathbf{x} $$

$$\mathbf{db} = \frac{\mathbf{d}L}{\mathbf{dz}} \frac{\mathbf{dz}}{\mathbf{db}} = \mathbf{dz} $$

推广至二层神经网络： 
$$\mathbf{d}{\mathbf{z}}^{[2]} = {\mathbf{a}}^{[2]} - y$$

$$\mathbf{d}{\mathbf{W}}^{[2]} = \mathbf{d}{\mathbf{z}}^{[2]} {\mathbf{a}^{[1]}}^T$$
```python
# 维度检查
W^[2].shape = (n^[2], n^[1])
z^[2].shape = (n^[2], 1)
a^[1].shape = (n^[1], 1)
```
$$\mathbf{d}{\mathbf{b}}^{[2]} = \mathbf{d}{\mathbf{z}}^{[2]}$$

$$\mathbf{d}{\mathbf{z}}^{[1]} = {{\mathbf{W}}^{[2]}}^T\mathbf{d}{\mathbf{z}}^{[2]} * {g^{[1]}}^{\prime}{(\mathbf{z}^{[1]})}, * :element-wise$$
```python
# 维度检查
W^[2].shape = (n^[2], n^[1])
z^[2].shape = (n^[2], 1)
z^[1].shape = (n^[1], 1)
```
$$\mathbf{d}{\mathbf{W}}^{[1]} = \mathbf{d}{\mathbf{z}}^{[1]} {\mathbf{x}}^T$$

$$\mathbf{d}{\mathbf{b}}^{[1]} = \mathbf{d}{\mathbf{z}}^{[1]}$$

进一步，多样本向量化  
$$\mathbf{d}{\mathbf{Z}}^{[2]} = {\mathbf{A}}^{[2]} - Y$$

$$\mathbf{dW}^{[2]} = \frac1m\mathbf{dZ}^{[2]}{\mathbf{A}^{[1]}}^T$$

$$\mathbf{db}^{[2]} = \frac1m np.sum(\mathbf{dZ}^{[2]}, axis=1, keepdims=True)$$

$$\mathbf{dZ}^{[1]} = {\mathbf{W}^{[2]}}^T\mathbf{dZ}^{[2]} * {g^{[1]}}^\prime(\mathbf{Z}^{[1]})$$

$$\mathbf{dW}^{[1]} = \frac1m\mathbf{dZ}^{[1]}\mathbf{X}^T$$

$$\mathbf{db}^{[1]} = \frac1m np.sum(\mathbf{dZ}^{[1]}, axis=1, keepdims=True)$$

#### 随机初始化
**对于一个神经网络，如果把权重或者参数都初始化为0，那么梯度下降将不会起作用。**  
如果初始化成0，那么隐藏层将会失去它的意义；  

随机初始化：  
$$w^{[1]} = np.random.randn((2,2)) * 0.01$$
$$b^{[1]} = np.zeros((2,1))$$
为什么要乘0.01？  初始化一个非常小的数，避免激活函数的梯度接近零。