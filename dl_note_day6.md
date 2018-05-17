# Day6: Learning Note of Shallow neural networks (Video 1-4)
> 备注：今天的内容主要包括：神经网络的表示、输出和多样本的向量化表示
## 神经网络及其表示
以下图所示的神经网络为例：  
<img src="https://github.com/CaoZhens/DL_Learning/blob/master/pic/shallow_nn_picture_2.png" alt="" data-canonical-src="" width="400" height="200" />  
**符号表示**  
使用中括号上角标来表示神经网络的层；（注意不要与花括号上角标表示的样本混淆）  

**层**  
输入层（第0层）
隐藏层
输出层
**当我们计算网络的层数时，输入层是不算入总层数内，所以隐藏层是第一层，输出层是第二层。因此将输入层称为第零层。**

## 神经网络的计算
### 单节点的计算  
单节点神经网络如下图所示：  
<img src="https://github.com/CaoZhens/DL_Learning/blob/master/pic/shallow_nn_picture_1.png" alt="" data-canonical-src="" width="400" height="200" />  

单节点神经网络的计算包括两个步骤：  
<img src="https://github.com/CaoZhens/DL_Learning/blob/master/pic/shallow_nn_picture_3.png" alt="" data-canonical-src="" width="400" height="300" />  
1. 计算$z$
$$z = \mathbf{w}^T\mathbf{x} + b = 
{\left[ \begin{matrix}
      w_1 \\
      w_2 \\ 
      w_3 \\
    \end{matrix} \right]}^T
    {\left[ \begin{matrix}
      x_1 \\
      x_2 \\ 
      x_3 \\
    \end{matrix} \right]} + b = 
    (w_1x_1+w_2x_2+w_3x_3+b)$$
2. 通过$\sigma{(z)}$计算$a$
$$a = \sigma{(z)}$$

### 神经网络的计算 
基于单节点计算，引申至下图所示神经网络的计算：  
<img src="https://github.com/CaoZhens/DL_Learning/blob/master/pic/shallow_nn_picture_4.png" alt="" data-canonical-src="" width="400" height="200" />  
#### 单样本非向量化
写出隐藏层四个节点的计算公式 
$$z_1^{[1]} = \mathbf{w}_1^{[1]T}\mathbf{x} + b_1^{[1]}, a_1^{[1]} = \sigma{(z_1^{[1]})}$$
$$z_2^{[1]} = \mathbf{w}_2^{[1]T}\mathbf{x} + b_2^{[1]}, a_2^{[1]} = \sigma{(z_2^{[1]})}$$
$$z_3^{[1]} = \mathbf{w}_3^{[1]T}\mathbf{x} + b_3^{[1]}, a_3^{[1]} = \sigma{(z_3^{[1]})}$$
$$z_4^{[1]} = \mathbf{w}_4^{[1]T}\mathbf{x} + b_4^{[1]}, a_4^{[1]} = \sigma{(z_4^{[1]})}$$

#### 单样本向量化
**理解向量化的技巧:同一层的不同节点，纵向堆叠起来！**  
$$\mathbf{W}^{[1]} = \left[ \begin{matrix}
      {--} & {\mathbf{w}_1^{[1]}}^T  &{--}  \\  
      {--} & {\mathbf{w}_2^{[1]}}^T  &{--}  \\
      {--} & {\mathbf{w}_3^{[1]}}^T  &{--}  \\ 
      {--} & {\mathbf{w}_4^{[1]}}^T  &{--}  
    \end{matrix} \right] , 
    \mathbf{b}^{[1]} = \left[ \begin{matrix}
       {b_1^{[1]}} \\  
       {b_2^{[1]}} \\
       {b_3^{[1]}} \\ 
       {b_4^{[1]}} 
    \end{matrix} \right], 
    \mathbf{z}^{[1]} = \left[ \begin{matrix}
       {z_1^{[1]}} \\  
       {z_2^{[1]}} \\
       {z_3^{[1]}} \\ 
       {z_4^{[1]}} 
    \end{matrix} \right] $$

即：
$$\mathbf{z}^{[1]} = \mathbf{W}^{[1]}\mathbf{x} + \mathbf{b}^{[1]}, 
\mathbf{a}^{[1]} = \sigma{(\mathbf{z}^{[1]})}$$

其中：
```python
z.shape = (4, 1)
b.shape = (4, 1)
W.shape = (4, 3)
```

#### 多样本非向量化
for i = 1 to m:
$$\mathbf{z}^{[1](i)} = \mathbf{W}^{[1]}\mathbf{x}^{(i)} + \mathbf{b}^{[1]}$$
$$\mathbf{a}^{[1](i)} = \sigma{(\mathbf{z}^{[1](i)})}$$
$$\mathbf{z}^{[2](i)} = \mathbf{W}^{[2]}\mathbf{a}^{[1](i)} + \mathbf{b}^{[1]}$$
$$\mathbf{a}^{[2](i)} = \sigma{(\mathbf{z}^{[2](i)})}$$

#### 多样本向量化
将上述单样本的向量化进一步扩展至多样本： 
**将不同训练样本堆叠起来放入矩阵各列中。**  
$$\mathbf{X} = \left[
    \begin{matrix}
      \vdots & \vdots &  &\vdots  \\  
      {\mathbf{x}^{(1)}} & {\mathbf{x}^{(2)}} & \cdots & {\mathbf{x}^{(m)}}   \\
      \vdots & \vdots & & \vdots \\
    \end{matrix}
  \right]$$
$$ \mathbf{Z}^{[1]}=\left[\mathbf{z}^{[1](1)}, \mathbf{z}^{[2](2)},..., \mathbf{z}^{[1](m)}\right]$$
$$ \mathbf{A}^{[1]}=\left[\mathbf{a}^{[1](1)}, \mathbf{a}^{[2](2)},..., \mathbf{a}^{[1](m)}\right]$$
即：  
$$\mathbf{Z}^{[1]} = \mathbf{W}^{[1]}\mathbf{X} + \mathbf{b}^{[1]}$$
$$\mathbf{A}^{[1]} = \sigma{(\mathbf{Z}^{[1]})}$$
$$\mathbf{Z}^{[2]} = \mathbf{W}^{[1]}\mathbf{A}^{[1]} + \mathbf{b}^{[2]}$$
$$\mathbf{A}^{[2]} = \sigma{(\mathbf{Z}^{[2]})}$$