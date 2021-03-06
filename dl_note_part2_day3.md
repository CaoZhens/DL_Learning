# Day3: 深度学习的实用层面(Practical aspects of Deep Learning) Video7-9
> 今天的主要学习内容包括：dropout正则化；对dropout的直观理解；其它正则化方法

## Dropout正则化（随机失活）
### 基本原理
dropout遍历神经网络的每一层，并设置消除神经网络中节点的概率。  
遍历后，根据节点消除概率，随机消除一些节点，并删除该节点所有连线（包括入和出）；  
最后得到一个节点更少，规模更小的网络。  

### Inverted Dropout(反向随机失活)
以三层神经网络举例：  
定义向量$\mathbf{d}^{[3]}$， 并看它是否小于某个门限值（keep-prob）
```python
d3 = np.random.rand(a3.shape[0], a3.shape[1]) < keep_prob
```
如果设置keep—prob=0.8， 它表示保留某个隐藏单元的概率是0.8；（或者消除任意一个隐藏单元的概率是0.2）。  
然后：  
```python
a3 = np.multiply(a3, d3)
```
最后：  
```python
a3 /= keep_prob
```
**反向随机失活方法通过除以keep-prob，确保$a^{[3]}$的期望值不变。**  

## 直观理解Dropout
dropout类似L2正则化，可以起到收缩权重的效果：  
对于某个神经网络单元而言，它不应该过分依赖于任何一个输入（特征），因为该单元的输入可能随时被dropout清除；  

如果担心某些层比其它层更容易发生过拟合，可以把某些层的keep-prob值设置得比其它层更低。

## 其它正则化方法
- data augmentation 数据扩增
以图片为例： 水平翻转图片 or 随意裁剪图片（随意翻转+放大裁剪）  
- early stoppping
以梯度下降为例：在中间点停止迭代过程，我们得到一个w值中等大小的弗罗贝尼乌斯范数，与L2正则化相似，选择参数w范数较小的神经网络

## 归一化输入
**目的是为了加速训练神经网络**  
归一化的两个步骤：
- 0均值
- 归一化方差
