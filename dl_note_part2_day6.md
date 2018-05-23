# Day6: 优化算法 (Optimization algorithms) Video5-7
> 今天的主要学习内容包括：指数加权平均的偏差修正（bias correction in exponentially weighted averages);动量梯度下降法（momentum）；RMSprop算法

## 指数加权平均的偏差修正
$$v_t = \beta v_{t-1} + (1-\beta)\theta_t$$
以$\beta = 0.98$为例，如果初始化$v_0 = 0$:  
$$v_1 = 0.98v_0 + 0.02\theta_1 = 0.02\theta_1$$

$$v_2 = 0.98v_1 + 0.02\theta_2 = 0.0196\theta_1 + 0.02\theta_2$$
因此，在这种情况下，指数加权平均对前面几个值的估测会远远小于实际值。  

有个办法可以修改这一估测，让估测变得更好更准确，特别是在估测初期，即不直接使用公式$v_t = \beta v_{t-1} + (1-\beta)\theta_t$；而是在其基础上去除一个偏差：
$$v_t = \frac{v_t}{1-\beta^t}$$
此时前两个估测值变成：
$$v_1 = \frac{v_1}{1-0.98^1} = \frac{0.02\theta_1}{0.02} = \theta_1$$

$$v_2 = \frac{v_2}{1-0.98^2} = \frac{0.0196\theta_1 + 0.02\theta_2}{0.0396}$$

随着$t$增加，$\beta^t$接近于0，所以当$t$很大的时候，偏差修正几乎没有作用。不过在开始阶段，偏差修正可以帮助你更好预测温度，偏差修正可以使结果估测更加准确。

## 动量梯度下降法
Momentum的运行速度几乎总是快于标准的梯度下降算法，该算法的基本思路就是计算梯度的指数加权平均数，并利用该梯度更新权重。

类比前面介绍的指数加权平均算法：
$$v_{\mathbf{d}W} = \beta v_{\mathbf{d}W} + (1-\beta)\mathbf{d}W$$

$$v_{\mathbf{d}b} = \beta v_{\mathbf{d}b} + (1-\beta)\mathbf{d}b$$

$$W := W - \alpha v_{\mathbf{d}W}$$

$$b := b - \alpha v_{\mathbf{d}b}$$

## RMSprop
RMSprop的算法的全称是root mean square prop算法，它也可以加速梯度下降。  

$$S_{\mathbf{d}W} = \beta S_{\mathbf{d}W} + (1-\beta)\mathbf{d}W^2$$

$$S_{\mathbf{d}b} = \beta S_{\mathbf{d}b} + (1-\beta)\mathbf{d}b^2$$

$$W := W - \alpha \frac{\mathbf{d}W}{\sqrt{S_{\mathbf{d}W}}}$$

$$b := b - \alpha \frac{\mathbf{d}b}{\sqrt{S_{\mathbf{d}b}}}$$