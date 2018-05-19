# Day1: 深度学习的实用层面(Practical aspects of Deep Learning) Video1-3
> 今天的主要学习内容包括：训练／验证／测试集； 偏差／方差；机器学习过程中利用偏差和方差评估的基本方法

## 训练／验证／测试集
### 神经网络训练概要
**训练神经网络时，我们需要做出许多决策（超参数设置）**  
- 神经网络的层数
- 每层含有的隐藏单元数
- 学习速率
- 每层使用的激活函数

不可能一开始就能准确设置出这些信息和超参数。  
**应用型机器学习是一个多次迭代的过程**  
初步想法 -> 编码运行 -> 完善 -> 迭代更新

### 数据集划分
训练集 + 简单交叉验证集（验证集） + 测试集  
训练集用于模型训练  
验证集用于模型选择  
测试集用于评估，无偏评估算法的运行状况

备注：  
在传统机器学习（小数据量）时代，常见做法是70%-30%或者60%-20%-20%。  
但是在大数据量时代，数据量很可能是百万级别，那么验证集和测试集占数据总量的比例会趋向于变得更小，可能不需要20%的数据做验证集。

总结：  
在机器学习中，我们通常将样本分成训练集，验证集和测试集三部分，数据集规模相对较小，适用传统的划分比例，数据集规模较大的，验证集和测试集要小于数据总量的20%或10%。  

补充：  
现代深度学习的另一个趋势是训练集和测试集分布不匹配。针对这种情况，根据经验，需要确保验证集和测试集的数据来自同一分布。  

## 偏差／方差
高偏差（high bias) "欠拟合" ：例如：训练集误差15%，验证集误差16%
高方差（high variance) "过拟合"  例如：训练集误差1%，验证集误差11%
（高偏差&高方差：训练集误差15%，验证集误差30%）
（低偏差&低方差：训练集误差1%，验证集误差11%）

## 机器学习过程中利用偏差和方差评估的基本方法
如果偏差较高（评估训练集的性能）：  
- 选择更大的网络（更多隐藏层 or 更多隐藏单元）
- 花费更多时间训练网络
- 更先进的算法：新的网络架构

如果方差较高（评估验证集的性能）：  
- 更多数据
- 正则化
- 更先进的算法：新的网络架构