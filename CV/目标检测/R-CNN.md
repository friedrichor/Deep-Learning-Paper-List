[TOC]

# 前言
原论文地址：[Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524)


# 总体结构
![在这里插入图片描述](https://img-blog.csdnimg.cn/85f7b648852a482a85fdd983424f884d.png?,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAZnJpZWRyaWNob3I=,size_20,color_FFFFFF,t_70,g_se,x_16)
# 算法流程
1. 一张图像生成 1k~2k 个候选区域（使用Selective Search方法）
2. 对每一个候选区域，使用深度网络提取特征
3. 特征送入每一类的SVM分类器，判别是否属于该类
4. 使用回归器精细修正候选框位置
## 1. 候选区域生成
&emsp;&emsp;利用Selective Search算法通过图像分割分割的方法得到一些原始区域，然后使用一些合并策略将这些区域合并，得到一个层次化的区域结构，而这些结构就包含着可能需要的物体。
 <img src="https://img-blog.csdnimg.cn/8c85a28b9b7447a9b64362e88c650d2e.png"   width="50%">
&emsp;&emsp;如上图，使用SS(Selective Search)算法可以得到若干个候选框，这些候选框就可能包含着我们需要的目标。

## 2. 对每个候选区域，使用深度网络提取特征
&emsp;&emsp;将2000个候选区域缩放到 227×227 pixel，接着将候选区域输入事先训练好的AlexNet CNN网络，获取4096维（这个维数是AlexNet定义的）的特征，得到 2000×4096维矩阵。

![在这里插入图片描述](https://img-blog.csdnimg.cn/b96648058f514fbea3c4c32e90613089.png)

## 3. 特征送入每一类的SVM分类器，判定类别
&emsp;&emsp;将 2000×4096 维特征与 20 个 SVM 组成的权值矩阵4096×20相乘，获得2000×20维矩阵表示每个建议框是某个目标类别的得分。分别对上述2000×20维矩阵中每一列即每一类进行**非极大值抑制**剔除重叠建议框，得到该列即该类中得分最高的一些建议框。

说明：
 1. 这里的 SVM分类器是二分类器，所以对于判别每一个类别，都有一个专门的分类器。
 2. 2000×20维矩阵是一个评分矩阵，表示这2000个建议框属于每一种类别(共20类)的分数。
3. 如下图，左面表示一个建议框拥有的4096维特征（应该是行向量）， 将这个特征输入到 SVM 分类器中，就能判别这个建议框是属于哪个类别。
<img src="https://img-blog.csdnimg.cn/8d6fe0aa3ba543afbda4cc613531709d.png"   width="50%">
4. 矩阵相乘示意图
 <img src="https://img-blog.csdnimg.cn/95f6a0bb44944d88b53698e4a90f5dd1.png"   width="50%">
  假设第一个类别判断的是狗，2000×20矩阵的第一行第一列 = 2000×4096维矩阵的第一行(代表第一个建议框的特征) $\times$ 4096×20维矩阵的第一列(代表检测出狗的参数)，就得到了第一个建议框是狗的分数，以此类推，我们就能得到2000个建议框中分别是每一类别的分数。

**非极大值抑制剔除重叠建议框：**
&emsp;&emsp;$IoU (Intersection\ over\ Union)$表示 $(A\cap B)/(A\cup B)$
<center> <img src="https://img-blog.csdnimg.cn/d5f37ddf83f747be85777c7e310c3878.png"   width="50%">

<center> <img src="https://img-blog.csdnimg.cn/1d9547e2b0c5416c8e760df2f93b574b.png"   width="50%">

&emsp;&emsp;对于每一个类别，首先寻找得分最高的目标，然后计算其他目标与该目标的 IoU值，删除所有 IoU值大于给定阈值的目标，然后把这个最高得分的目标存起来，再在剩下的建议框中重复上述操作，直到遍历完所有建议框。
![在这里插入图片描述](https://img-blog.csdnimg.cn/c59efc1b9c674e31b48ee5d66d130d38.png?,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAZnJpZWRyaWNob3I=,size_20,color_FFFFFF,t_70,g_se,x_16)
&emsp;&emsp;如上图，通过SVM分类器后，A建议框判别为向日葵的概率为98%（是评分最高的一个），B建议框判别为向日葵的概率为86%，通过计算IoU，删除B，保留A。（注意这里并不代表每一次都保留A而把其他的剔除，上面讲到用完A这个得分最高的目标后会把它存起来，也就是说第二次的剔除流程中已经没有A了，再从剩余的那些建议框中选择最高得分的目标进行之前的操作）

## 4. 使用回归器精细修正候选框位置
&emsp;&emsp;对NMS(Non-Maximum Suppression，非极大值抑制)处理后剩余的建议框进一步筛选。接着分别用20个回归器对上述20个类别中剩余的建议框进行回归操作，最终得到每个类别的修正后的得分最高的bounding box。（依旧针对CNN输出的特征向量进行预测）
<center> <img src="https://img-blog.csdnimg.cn/332bc22a0e104f6cbf4057891922e9ae.png"   width="50%">

&emsp;&emsp;如上图,黄色框口P表示建议框Region Proposal，绿色窗口G表示实际框Ground Truth(这是人工标定的)，红色窗口 $\hat G$ 表示Region Proposal进行回归后的预测窗口，可以用最小二乘法解决的线性回归问题。 

# R-CNN框架
![在这里插入图片描述](https://img-blog.csdnimg.cn/e31a734e209e49eca1049b63bd84bdd6.png?,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAZnJpZWRyaWNob3I=,size_19,color_FFFFFF,t_70,g_se,x_16)
# 存在的问题
1. 测试速度慢：
测试一张图片约53s (CPU)。用Selective Search算法提取候选框用时约2秒，一张图像内候选框之间存在大量重叠，提取特征操作冗余。
2. 训练速度慢：
过程及其繁琐 。
3. 训练所需空间大：
对于SVM和bbox回归训练，需要从每个图像中的每个目标候选框提取特征，并写入磁盘。对于非常深的网络，如VGG16，从VOC07训练集上的5k图像上提取的特征需要数百GB的存储空间。


<hr>

参考来源：[1.1Faster RCNN理论合集](https://www.bilibili.com/video/BV1af4y1m7iL?p=1)