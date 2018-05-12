### Day11: Learning Note of Deep neural networks (Review Back Propagation)
> 备注：今天的内容主要包括：进一步复习并理解神经网络的反向传播

#### 回顾：反向传播
**Back Propagation (6个公式)**  
$$\mathbf{d}Z^{[2]} = A^{[2]} - Y$$
其中，
$$Y = \left[y^{(1)}, y^{(2)}, \cdots, y^{(m)} \right]$$
$$\mathbf{d}w^{[2]} = \frac1m\mathbf{d}Z^{[2]}{A^{[1]}}^T$$ 
$$\mathbf{d}b^{[2]} = \frac1m np.sum(\mathbf{d}Z^{[2]}, axis=1, keepdims=True)$$
$$\mathbf{d}Z^{[1]} = {w^{[2]}}^T\mathbf{d}Z^{[2]} * {g^{[1]}}^\prime(Z^{[1]})$$
$$\mathbf{d}w^{[1]} = \frac1m\mathbf{d}Z^{[1]}X^T$$
$$\mathbf{d}b^{[1]} = \frac1m np.sum(\mathbf{d}Z^{[1]}, axis=1, keepdims=True)$$

#### 推导过程
$$L(a, y) = -(y\log(a)+(1-y)\log(1-a))$$
$$\mathbf{d}a =\frac{\mathbf{d}L}{\mathbf{d}a} = -\frac{y}{a} + \frac{1-y}{1-a}$$
$$\mathbf{d}z =\frac{\mathbf{d}L}{\mathbf{d}z} = \frac{\mathbf{d}L}{\mathbf{d}a}\frac{\mathbf{d}a}{\mathbf{d}z} = \left( -\frac{y}{a} + \frac{1-y}{1-a} \right) \left( a(1-a) \right) = a - y$$