# Day5: 优化算法 (Optimization algorithms) Video1-4
> 今天的主要学习内容包括：mini-batch梯度下降及理解；指数加权平均及理解

## mini-batch梯度下降
**Batch Gradient Descent**  
每次迭代利用全部训练数据计算的方法，称为batch梯度下降；但是，即使使用向量化，如果训练样本数目$m$很大（比如超过500万甚至更大），梯度下降算法的处理速度是十分缓慢的。

**mini-batch Gradient Descent**  
一个自然的思路是，将训练集分隔成小一点的子集，这些子集被取名为mini-batch。  
假设训练集共5000万，每个mini-batch设置为1000个样本，也就是一共5000个mini-batch；记作$X^{\{t\}}$(对$Y$也做相同的处理)。  
```python
X^{{t}}.shape = (n_x, 1000)
Y^{{t}}.shape = (1, 1000)
```

**mini-batch Gradient Descent的算法过程**  
Repeat{
for t=1 to 5000:
**Forward Propagation on $X^{\{t\}}$**  
$\mathbf{Z}^{[1]} = \mathbf{W}^{[1]}X^{\{t\}} + \mathbf{b}^{[1]}$  
$\mathbf{A}^{[1]} = g^{[1]}(\mathbf{Z}^{[1]})$  
...
$\mathbf{A}^{[L]} = g^{[L]}(\mathbf{Z}^{[L]})$  

Compute the cost:  
$J^{\{t\}} = \frac1{1000}\sum_{i=1}^{l}{L(\hat{y}^{(i)}, y^{(i)})} + \frac{\lambda}{2*1000}\sum_{l}{\|\mathbf{W}^{l} \| } _F^2$  
（其中，$\hat{y}^{(i)}, y^{(i)}$ is from $X^{\{t\}}, Y^{\{t\}}$)  
**Backward Propagation**  
$\mathbf{W}^{[l]} = \mathbf{W}^{[l]} - \alpha\mathbf{dW}^{[l]}$  
$\mathbf{b}^{[l]} = \mathbf{b}^{[l]} - \alpha\mathbf{db}^{[l]}$  
}

上面用伪代码表示了mini-batch梯度下降法训练样本的一步，也可以称之为**1 epoch**。

**理解**  
Batch梯度下降法每次迭代都需要遍历整个训练集，可以预期每次迭代成本都会下降；所以如果成本函数J是迭代次数的一个函数，它应该会随着每次迭代而减少，如果J在某次迭代中增加了，那肯定出了问题，也许是因为学习率设置得太大导致的。

使用mini-batch梯度下降法，则并不是每次迭代都是下降的

**如果mini-batch大小为m，即Batch Gradient Descent**  
**如果mini-batch大小为1，即Stochastic Gradient Descent**  

mini-batch取值大小的指导原则  
- 如果训练集较小（小于2000个），直接使用batch梯度下降法
- 如果训练集较大，一般的mini-batch大小为64到1024，考虑到电脑内存设置和使用的方式，如果mini-batch大小是2的n次方最好

## 指数加权平均
$$v_t = \beta v_{t-1} + (1-\beta)\theta_t$$
直观理解：
- $\beta$越大，曲线越平坦（认为参与平均的计数更多，或者说平均了更多的值）

理解：  
- 每个数值分别与指数衰减函数相乘，然后求和


