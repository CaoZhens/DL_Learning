### Day3: Learning Note of Neutral Networks Basics (Vedio 5-9)
> 备注：今天的学习内容主要包括：导数、计算图、逻辑回归的梯度下降算法(单样本)

#### 导数
导数的几何意义： 某点的导数即函数在该点处的切线斜率  
导数的数学意义：
$$y^\prime(x) = \lim_{\Delta{x} \to 0} \frac{f(x+\Delta{x})-f(x)}{\Delta{x}}$$
知识点整理：  
导数就是斜率，而函数的斜率，在不同的点是不同的。
如果你想知道一个函数的导数，可参考你的微积分课本或者维基百科，查表获得。

#### 计算图
- 计算图的前向过程，主要是为了计算函数的输出
- 计算图的后向过程，主要是为了计算函数的导数（其隐含实质是微积分运算中的链式求导法则）

#### 逻辑回归中的梯度下降(单个样本)
**这里使用计算图来计算逻辑回归的梯度下降算法；以这个例子作为开始来讲解，可以更好的理解梯度下降算法的思想**  
梯度下降：  
$$w := w - \alpha\frac{\partial{J(w,b)}}{\partial{w}}$$ 
$$b := b - \alpha\frac{\partial{J(w,b)}}{\partial{b}}$$ 

逻辑回归(其中$\sigma(x)$是sigmoid函数 ): 
$$z = w^Tx + b$$  
$$\hat{y} = a = \sigma(z)$$
$$L(a, y) = -(y\log(a)+(1-y)\log(1-a))$$

下面以单样本，两个特征为例，阐述计算图的前向过程和反向过程：  
计算图的前向过程（在单个训练样本上计算损失函数）
1. $$z = w_1x_1 + w_2x_2 + b$$
2. $$a = \sigma{(z)}$$
3. $$L(a, y)$$

计算图的反向过程（计算损失函数的导数）：
1. $$\frac{\mathbf{d}L}{\mathbf{d}a} = -\frac{y}{a} + \frac{1-y}{1-a}$$
2. $$\frac{\mathbf{d}L}{\mathbf{d}z} = \frac{\mathbf{d}L}{\mathbf{d}a}\frac{\mathbf{d}a}{\mathbf{d}z} = \left( -\frac{y}{a} + \frac{1-y}{1-a} \right) \left( a(1-a) \right) = a - y$$
3. $$\frac{\partial{L}}{\partial{w_1}} = x_1 \frac{\mathbf{d}L}{\mathbf{d}z} = x_1(a-y)$$
4. $$\frac{\partial{L}}{\partial{w_2}} = x_2 \frac{\mathbf{d}L}{\mathbf{d}z} = x_2(a-y)$$
5. $$\frac{\partial{L}}{\partial{b}} = \frac{\mathbf{d}L}{\mathbf{d}z} = a-y $$

**总结：梯度下降的迭代公式为：**  
$$w_1 := w_1 - \alpha x_1(a-y)$$ 
$$w_2 := w_2 - \alpha x_2(a-y)$$ 
$$b := b - \alpha (a-y)$$ 