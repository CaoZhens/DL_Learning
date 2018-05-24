# Day7: 优化算法 (Optimization algorithms) Video8-10
> 今天的主要学习内容包括：Adam优化算法；学习率衰减；局部最优问题

## Adam优化算法
Adam优化算法基本上就是将Momentum和RMSprop结合在一起。  
算法基本流程：  
$v_{dw}=0, v_{db}=0, S_{dw}=0, S_{db}=0$  
compute dW, db using mini-batch  
$v_{dw} = \beta_1 v_{dw} + (1-\beta_1)dW$ (Momentum)  
$v_{db} = \beta_1 v_{db} + (1-\beta_1)db$ (Momentum)  
$S_{dw} = \beta_2 S_{dw} + (1-\beta_2)(dW)^2$(RMSprop)
$S_{db} = \beta_2 S_{db} + (1-\beta_2)(db)^2$(RMSprop)

Momentum更新了超参数$\beta_1$，RMSprop更新了超参数$\beta_2$。  
一般使用Adam算法的时候，要计算偏差修正:  
$v_{dw}^{corr} = \frac{v_{dw}}{1-\beta_1^t}, v_{db}^{corr} = \frac{v_{db}}{1-\beta_1^t}$  
$S_{dw}^{corr} = \frac{S_{dw}}{1-\beta_2^t}, S_{db}^{corr} = \frac{S_{db}}{1-\beta_2^t}$  

最后更新权重： 
$$W := W - \alpha \frac{v_{dw}^{corr}}{\sqrt{S_{dw}^{corr} + \epsilon}}$$

$$b := b - \alpha \frac{v_{db}^{corr}}{\sqrt{S_{db}^{corr} + \epsilon}}$$ 

超参数推荐：  
$\beta_1 = 0.9, \beta_2 = 0.999, \epsilon=10^{-8}$

## 学习率衰减
随着时间逐渐减少学习率，可以加快算法的学习速度。称之为学习率衰减。

$$\alpha = \frac{1}{1 + decayrate * epochnum} \alpha_0$$

## 局部最优问题
在深度学习研究早期，人们总是担心优化算法会困在极差的局部最优，不过随着深度学习理论不断发展，我们对局部最优的理解也发生了改变。

事实上，如果创建一个神经网络，通常梯度为零的点并不是图中的局部最优点，实际上成本函数的零梯度点，通常是鞍点。

此时可能面临的问题是：在鞍点附近的平稳段，算法学习速率会非常缓慢；此时，类似前面介绍的Adam算法，能够加快速度，使算法尽早走出平稳段。
