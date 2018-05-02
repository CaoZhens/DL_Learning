### Day1: Learning Note of Introduction of Deep Learning
1. 什么是神经网络？
以房屋价格预测为例；在CS229(机器学习课程)中，也使用了这个例子，当时是为了引出**Linear Regression**。  
> 在学习CS229时，我忽略了一点：价格永远不会是负数的，但线性回归得到的模型可能会让价格为负。

该模型可以简单地修正为：从趋近于零开始，然后变成一条直线。  
修正后的函数被称作**ReLU激活函数(Rectified Linear Unit)**。
> Rectify修正方式可以简单地理解成max(0,x)

这可能是最简单的神经网络。我们把房屋的面积作为神经网络的输入（我们称之为x），通过一个节点（一个小圆圈），最终输出了价格（我们用y表示）。其实这个小圆圈就是一个单独的神经元。  
把这些单个神经元叠加在一起，就可以组成更大的神经网络。

2. 利用神经网络作监督学习
- 应用领域：
    - 在线广告
    - 计算机视觉（图像识别）
    - 语音识别
    - 自动驾驶
- 基础算法：
    - NN  (Standard Neural Network)
    - CNN (Convolutional Neural Network): 主要用于图像识别
    - RNN (Recurrent Neural Network): 主要用于序列数据（如音频）

3. 神经网络兴起的原因
- 算力的提升，如硬件性能（CPU/GPU）
- 数据规模的大幅增长
- 算法模型规模的增长