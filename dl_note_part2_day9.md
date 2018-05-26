# Day9: 超参数调试、Batch正则化和程序框架 Video5-8
> 今天的主要学习内容包括：Batch归一化用于神经网络；深入理解Batch归一化； 
## batch归一化在神经网络中的使用
$$\mathbf{z}^{[1]} = W^{[1]}X + b^{[1]}$$

$${\mathbf{z}}_{norm}^{[1]} = \frac{\mathbf{z}^{[1]} - \mu}{\sqrt{\sigma^2 + \varepsilon}}$$

$$\tilde{\mathbf{z}}^{[1]} = \gamma z_{norm}^{(i)} + \beta$$

Batch归一化是发生在计算$z$和$a$之间的。直观理解就是：与其应用没有归一化的$z$值，不如用归一过的$\tilde{z}$

实践中，Batch归一化通常和训练集的mini-batch一起使用。  
应用Batch归一化的方式是:  
使用第一个mini-batch($X^{\{1\}}$)计算$\mathbf{z}^{\{1\}[1]}$，随后Batch归一化会对$\mathbf{z}^{\{1\}[1]}$减去均值，除以标准差，并由$\beta^{[1]}$和$\gamma^{[1]}$重新缩放，这样就得到了$\tilde{z}^{\{1\}[1]}$，而所有的这些都是在第一个mini-batch的基础上，再应用激活函数得到$a[1]$。

## 深入理解batch归一化
- 通过使输入层特征值以及隐藏层特征值的均值和方差规约，来加速学习
- 使权重比网络更滞后或者更深层
- 有轻微的正则化效果
## 测试时的batch归一化
为了将你的神经网络运用于测试，需要单独估算$μ$和$\sigma^2$；在典型的Batch归一化运用中，你需要用一个指数加权平均来估算。
如果使用的是某种深度学习框架，通常会有默认的估算$μ$和$\sigma^2$的方式，应该一样会起到比较好的效果。但在实践中，任何合理的估算你的隐藏单元z值的均值和方差的方式，在测试中应该都会有效。

## softmax回归
激活函数：  
$$\mathbf{z}^{[l]} = W^{[l]} \mathbf{a}^{[l-1]} + b^{[l]}$$
$$\mathbf{t} = e^{\mathbf{z}^{[l]}}$$
$$\mathbf{a}^{[l]} = \frac{e^{\mathbf{z}^{[l]}}}{\sum_i \mathbf{t}_i}$$


