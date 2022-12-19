[TOC]

# 前言

原论文地址：[Fast R-CNN](https://arxiv.org/abs/1504.08083)

# 总体架构
![在这里插入图片描述](https://img-blog.csdnimg.cn/0e10a003e11b4103b769c4a028d2cc3a.png?,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAZnJpZWRyaWNob3I=,size_16,color_FFFFFF,t_70,g_se,x_16)

# 算法流程
1. 一张图像生成1K~2K个候选区域（使用Selective Search方法）
2. 将图像输入网络得到相应的**特征图**，将SS(Selective Search)算法生成的候选框投影到特征图上获得相应的**特征矩阵**
3. 将每个特征矩阵通过ROI pooling层缩放到**7x7大小的特征图**，接着将特征图展平通过一系列全连接层得到预测结果
（ROI：Region of Interest 感兴趣区域）


## 与R-CNN的不同
&emsp;&emsp;在R-CNN中，我们分别训练了SVM分类器（预测目标所属的类别）和BBox回归器（调整候选区域边界框）。而在Fast R-CNN中，将这两种功能结合到一个网络当中，这就不用单独训练分类器和回归器了。

&emsp;&emsp;R-CNN依次将每个候选框区域输入到卷积神经网络中得到特征（如下图），这就是需要2000次正向传播，但这2000个候选框中有大量冗余，很多区域都重叠了，计算一次就可以的事情R-CNN一直在重复地做。
<center><img src="https://img-blog.csdnimg.cn/e9c7404669fd48bbb4ad303e81a92082.png" width="50%">
<center><b><font size ='3'>R-CNN</font></b></center></font>

&emsp;&emsp;Fast R-CNN将整张图像送入网络，紧接着从特征图像上提取相应的候选区域（参考SPP Net）。这些候选区域的特征**不需要再重复计算**。 

<center><img src="https://img-blog.csdnimg.cn/0e1c7522823745dfb4409847a147ae34.png" width="50%">
<center><b><font size ='3'>Fast R-CNN</font></b></center></font>

## 训练数据的采样
通过SS算法可以得到2000个候选框，但在训练时只使用一小部分就可以了。且数据分为正样本和负样本，正样本就是候选框中确实存在我们需要检测的目标，负样本就是候选框没有我们要检测的目标（可以理解成背景）。
- 为什么需要分正负样本？
- 假如我想要判别猫和狗，若全是正样本，数据集样本不平衡，猫的样本数量远大于狗，那么训练出来的网络在预测时就更偏向于判定为猫，这样肯定是不对的。放到目标检测中，若全是正样本，即便候选框中是一个背景，网络也会强行把它认为成一个我们检测的一个类别中。

在原论文中，作者提出对于每张图片，从2000个候选框中采集64个候选区域。对于每个候选区域，它与真实框(ground-truth)的 IoU 大于0.5，那么就把他划分成正样本，把与每个真实框的 IoU 的最大的值在0.1~0.5的认定为负样本。

## RoI pooling
<center><img src="https://img-blog.csdnimg.cn/0e1c7522823745dfb4409847a147ae34.png">
有了候选区域样本之后，使用RoI pooling将每个样本缩放成统一的尺寸。

如下图，将图片划分成7×7等分，对于每一小块区域执行最大池化(max pooling)操作，这样就得到了一个7×7的特征矩阵。无论候选区域的尺寸是多大的，都将缩放成7×7矩阵，这就不限制输入图像的尺寸（与R-CNN不同，R-CNN要求输入图像尺寸为227×227）。

<center><img src="https://img-blog.csdnimg.cn/0e1c7522823745dfb4409847a147ae34.png">

## 分类器
![在这里插入图片描述](https://img-blog.csdnimg.cn/f15e88c05e5a4adb9111c527f63a0c4d.png?,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAZnJpZWRyaWNob3I=,size_20,color_FFFFFF,t_70,g_se,x_16)
输出 N+1 个类别的概率（N为检测目标的种类，1为背景）共 N+1 个结点。分类器即上图中蓝色框部分，全连接层（FC）即需要 N+1 个结点。

## 边界框回归器
![在这里插入图片描述](https://img-blog.csdnimg.cn/6573d045e6cc4fb2b0d7b937140cd18d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAZnJpZWRyaWNob3I=,size_20,color_FFFFFF,t_70,g_se,x_16)
输出对应 N+1 个类别的候选边界框回归参数$(d_x,d_y,d_w,d_h)$，共 (N+1)×4 个结点。

那么怎么用回归参数来预测的呢？

<img src="https://img-blog.csdnimg.cn/e99b192eb8f44cd4a5f69df7bd449bc4.png" width="40%">$\quad\quad\quad$ <img src="https://img-blog.csdnimg.cn/3f6c18ae610042c1ba42a0195d14628f.png" width="40%">


其中，$P_x,P_y,P_w,P_h$分别为候选框的中心$x,y$坐标以及宽高
$\hat{G_x},\hat{G_y},\hat{G_w},\hat{G_h}$分别为最终预测的边界框中心$x,y$坐标以及宽高

根据上面的公式可以看出$d_x,d_y$就是用来调整边界框中心位置的，$d_w,d_h$用来调整宽高，从而把黄色区域调整到红色区域。

## Multi-task loss
  <center><img src="https://img-blog.csdnimg.cn/1165edb0d44e481384703082a755d5c9.png"></center>    

$p$ 是分类器预测的 softmax 概率分布 $p=(p_0, ..., p_k)$，$p_0$ 即是预测为背景的概率，以此类推  
$u$ 对应目标真实类别标签  
$t^u$对应边界框回归器预测的对应类别 $u$ 的回归参数 $(t_x^u,t_y^u,t_w^u,t_h^u)$  
$v$ 对应真实目标的边界框回归参数 $(v_x,v_y,v_w,v_h)$    
<hr>

### 分类损失：
![在这里插入图片描述](https://img-blog.csdnimg.cn/1165edb0d44e481384703082a755d5c9.png)
使用交叉熵损失，原文使用如下公式：
![在这里插入图片描述](https://img-blog.csdnimg.cn/675541107f474b3c8fb60f6ebbf67219.png#pic_center)
$p$是分类器预测的 softmax 概率分布$p=(p_0, ..., p_k)$，$p_0$即是预测为背景的概率，以此类推
$u$对应目标真实类别标签
<hr>

**交叉熵损失：**
![在这里插入图片描述](https://img-blog.csdnimg.cn/fdfcf214328d44059a64edbf6774cf17.png?,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAZnJpZWRyaWNob3I=,size_16,color_FFFFFF,t_70,g_se,x_16)
假设真实标签的one-hot编码是[0,0,...,1,...,0]，预测的softmax概率为[0.1,0.3,...,0.4,...,0.1]，那么$Loss=-\log(0.4)$
<hr>

### 边界框回归损失：
![在这里插入图片描述](https://img-blog.csdnimg.cn/1165edb0d44e481384703082a755d5c9.png)
$u$对应目标真实类别标签
$t^u$对应边界框回归器预测的对应类别$u$的回归参数$(t_x^u,t_y^u,t_w^u,t_h^u)$
$v$对应真实目标的边界框回归参数$(v_x,v_y,v_w,v_h)$

其中，$[u\geq1]$是艾佛森括号，当$u\geq1$时，这个值为1，$u<1$时即为0。$u\geq1$时，说明这是检测目标中的一个类别，这就是正样本；$u<1$时（即$u=0$），就说明是负样本，那么损失函数中就没有边界框回归损失这一项。

至于$(v_x,v_y,v_w,v_h)$是如何计算的，同样使用到了下图。
<img src="https://img-blog.csdnimg.cn/e99b192eb8f44cd4a5f69df7bd449bc4.png" width="40%">$\quad\quad\quad$ <img src="https://img-blog.csdnimg.cn/3f6c18ae610042c1ba42a0195d14628f.png" width="40%">  
$v_x=(G_x-P_x)/P_w$，同理$v_y=(G_y-P_y)/P_h$
$v_w=\ln(G_w/P_w)$，同理$v_h=\ln(G_h/P_h)$
<hr>

![在这里插入图片描述](https://img-blog.csdnimg.cn/bf2296c420a24db2b78c1abac26e973a.png#pic_center)  
展开，上式 $=smooth_{L1}(t_x^u-v_x)+smooth_{L1}(t_y^u-v_y)+smooth_{L1}(t_w^u-v_w)+smooth_{L1}(t_h^u-v_h)$

![在这里插入图片描述](https://img-blog.csdnimg.cn/e1616888ef70444bb090ab572c7abb5c.png#pic_center)  
关于损失函数的学习，可以参考：[回归损失函数1：L1 loss, L2 loss以及Smooth L1 Loss的对比](https://www.cnblogs.com/wangguchangqing/p/12021638.html)

# Fast R-CNN框架
![在这里插入图片描述](https://img-blog.csdnimg.cn/381b616be2984cf598b8f76469ab8326.png?,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAZnJpZWRyaWNob3I=,size_19,color_FFFFFF,t_70,g_se,x_16#pic_center)
对比于R-CNN的四部分框架，Fast R-CNN只包含两部分。第一部分就是SS算法获取候选框，第二部分将特征提取、分类、BBox回归融合到一个CNN网络

![在这里插入图片描述](https://img-blog.csdnimg.cn/6de0125eb6d14c9f989e3546da6595e3.png?,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAZnJpZWRyaWNob3I=,size_19,color_FFFFFF,t_70,g_se,x_16#pic_center)

<hr>

参考来源：[1.1Faster RCNN理论合集](https://www.bilibili.com/video/BV1af4y1m7iL?p=2&spm_id_from=pageDriver)
