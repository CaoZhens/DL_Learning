### Day5: Learning Note of Neutral Networks Basics (Video 15-18)
> 备注：今天的内容主要包括：Python基础（广播、numpy、notebook）； 深入理解逻辑回归的损失函数

#### 深入理解逻辑回归的损失函数
**结论：逻辑回归损失函数的实质是：在伯努利分布先验假设条件下的极大似然估计。**  
伯努利先验假设：  
二分类问题（不妨假设 $y^i \in [0, 1]$ ）满足伯努利分布，即：$y^i  \sim \mathbf{Bernoulli}(\phi^i) $  
其中，
$$ \phi^i = p(y^i=1|\mathbf{x}^i;\theta) = h_\theta(\mathbf{x}^i) = g(\theta^T\mathbf{x}^i) = \frac{1}{1+e^{-\theta^T\mathbf{x}^i}}$$
$$ p(y^i=0|\mathbf{x}^i;\theta) = 1-\phi^i $$

构造似然函数：  
基于伯努利分布先验假设条件，
$$p(y^i|\mathbf{x}^i;\theta) = (h_\theta(\mathbf{x}^i))^{y^i}(1 - h_\theta(\mathbf{x}^i))^{1-y^i}$$

构造似然函数
$$L(\theta) = P(\mathbf{y}|\mathbf{X};\theta) = \prod_{i=1}^m (h_\theta(\mathbf{x}^i))^{y^i}(1 - h_\theta(\mathbf{x}^i))^{1-y^i}$$

将$L(\theta)$转化为对数似然函数$l(\theta)$
$$l(\theta) = \log{L(\theta)} = \sum_{i=1}^m\left [ y^ilogh_{\theta}(\mathbf{x}^i) + (1-y^i)log(1-h_{\theta}(\mathbf{x}^i) \right ]$$

**极大化似然函数相当于极小化损失函数，即：**  
$$J(w,b) = \frac1m\sum_{i=1}^m\left [y^i\log{\hat{y}^i} + (1-y^i)\log{(1-\hat{y}^i)} \right ]$$

#### Python基础
本部分内容详见[notebook代码](https://nbviewer.jupyter.org/github/CaoZhens/DL_Learning/blob/master/dl_note_day5.ipynb)  