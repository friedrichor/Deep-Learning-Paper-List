[TOC]

# 0. 前言
论文标题：An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale
论文网址：[https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
源码网址：[https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)

# 1. 背景
&emsp;&emsp;Transformer结构最初提出是针对NLP领域的。<font color=#6699FF>尤其是BERT预训练语言模型，在大量通用语料上进行预训练，然后在任务相关的数据集上再训练并进行fine-tune（微调），这样BERT模型就可以应用到不同的任务中，其论文发表时提及在11个不同的NLP任务中取得SOTA表现，是NLP发展史上里程碑式的成就。</font>那么Transformer结构应用到CV领域是否同样会取得良好的结果呢？这就是本文所要探讨的。

&emsp;&emsp;在CV领域，注意力机制通常都依赖于卷积网络，要么与卷积网络结合应用，要么在保持整体结构的条件下替换卷积网络的某些组成部分。本文证明了注意力机制不必依赖于CNN，而是将Transformer直接应用于图像patches序列，这也可以很好地完成图像分类任务。

&emsp;&emsp;文章提出了Vision Transformer(以下简称 **ViT**)模型，将它应用于对大量数据进行预训练并进行多个中型或小型图像的识别基准测试（如ImageNet, CIFAR-100, VTAB等）。与先前取得SOTA表现的卷积网络相比较，Vision Transformer取得了良好成果，同时需要训练的计算资源也更少。

&emsp;&emsp;受Transformer结构中的self-attention启发，人们开始尝试将类似CNN的架构与self-attention相结合，也有一些将其完全替代卷积。后者虽然理论上很有效，但由于使用了特殊的attention模式，还不能在当时的硬件加速器上进行拓展。因此，经典的ResNet类（包括与其类似的）架构仍是SOTA表现。

# 2. 现状分析
&emsp;&emsp;Transformer结构最初是应用于NLP领域的，在很多NLP任务中都取得SOTA效果，是里程碑式的成就，使得目前NLP任务中首选的结构大多都是Transformer。<font color=#6699FF>（以往都是采用卷积的形式，如RNN、LSTM这样的时序网络，但这都存在一些明显的问题，如RNN的记忆长度比较短，于是提出了LSTM解决记忆的问题，但他们都有一个额外的问题，就是无法并行化，只能先计算T0时刻的数据，然后才能计算T0+1时刻的数据，这样的话训练效率就比较低。而Transformer就可以解决以上问题，记忆长度理论上可以无限长，同时可以并行化）</font>

&emsp;&emsp;Transformer中比较重要的就是Self-attention和Multi-head Self-attention，这也是为什么Transformer效果比较好的一个原因。**关于Self-attention和Multi-head Self-attention我会在后面举例说明一下。**

&emsp;&emsp;针对图像使用self-attention最开始的应用是要求每个pixel都关注其他的pixel。由于pixel数目二次的代价，这并不能改变实际的输入大小。为了将Transformer应用到图像中，近年来有过很多尝试：
1. Parmar等人仅在每个查询的pixel的局部邻近范围内应用self-attention，而不是全局应用；
2. Hu等人、Ramachandran等人、Zhao等人都使用了局部的Multi-Head点积self-attention blocks来完全替代卷积；
3. Child等人提出了Sparse Transformers，采用可扩展的近似全局的self-attention，以便应用于图像；
4. Weissenborn等人将self-attention应用于不同尺寸的blocks。

&emsp;&emsp;与本文所讲的ViT模型最相似的是Cordonnier等人2020年提出的一个模型：从输入图像中提取大小为2×2的patches并在顶层应用self-attention。此外，Cordonnier等人提出的模型只是使用2×2 pixels的小patch，这就使得该模型只适用于小分辨率的图像，但ViT(Vision Transformer)也可以处理中等分辨率的图像。

&emsp;&emsp;另一个相关的模型是image GPT(iGPT)，它在降低图像分辨率和颜色空间后，将Transformer应用于图像pixels。该模型以无监督的方式作为生成模型进行训练，然后对结果进行fine-tune或线性探测，以获得分类性能。<font color=#6699FF>（这里其实和NLP任务中使用Transformer的过程是很相似，同样是无监督学习，以BERT为例，其通过一个可训练的[CLS]参数，将其连接一个分类器，通过fine-tune从而使模型能够进行相应的分类任务）</font>

# 3. 任务&结论（简介）
&emsp;&emsp;受Transformer在NLP领域成功扩展的启发，作者尝试将Transformer直接应用于CV领域。但应用于NLP领域的Transformer结构中，输入的内容是一个一个单词（通过将语句分词获得），然后才能进行embedding操作。那么对于图像来说，作者将一个图像分割成一些patches，并将这些patches进行embedding作为Transformer的输入。图像patches的操作就相当于NLP中tokens（words）的处理方式。然后作者用有监督的方式训练模型从而实现图像分类。<font color=#6699FF>（这里和BERT的训练方式是不一样的，BERT使用的是大量无标注语料，因此使用的是无监督学习，而无监督学习主要分为自回归(AR, auto regression)和自编码(AE, auto encoding)，BERT使用的就是AE方法，从损坏的输入数据中利用上下文信息重构原始数据，使用mask模型）</font>

&emsp;&emsp;在中型数据集（如ImageNet）上训练时，使用Transformer的效果并没有ResNet好，因为Transformer缺乏了CNN中固有的inductive biases，比如translation equivariance（平移等变性）和 locality（位置信息），也就是说当识别的物体在图像中的不同位置时，CNN是可以学习出来的，而Transformer很难学到，因此在数据量不足的数据集上训练时的效果并不是很好。

&emsp;&emsp;然而，如果在大型数据集（14M-300M images）上训练，使用Transformer的训练效果是更好的。ViT模型在足够规模的数据集上训练时获得了很好的成果。在ImageNet-21k 或in-house JFT-300M上进行预训练时，ViT在多个图像识别任务中都击败或接近于SOTA。

# 4. 整体框架
![在这里插入图片描述](https://img-blog.csdnimg.cn/0dc5a2b0eba04a5a906ebc17f5c38ae8.png)
&emsp;&emsp;将一个图像分割成固定尺寸的patches，线性嵌入每个patch，并在添加了Position Embedding，将得到的向量序列输入到标准的Transformer Encoder中。为了进行分类，在序列中添加了一个额外的可学习的 ”classification token” 。
大体上可以分为三部分：
1. Linear Projection of Flattened Patches（Embedding层）
2. Transformer Encoder（上图右侧的结构）
3. MLP Head（最终用于分类的层结构）

# 5. 流程
1. 将输入的图片分成一个一个小的patches
![在这里插入图片描述](https://img-blog.csdnimg.cn/2d4808f44e444898a294d0c5ea673bfa.png)
2. 将每个patches输入到Linear Projection of Flattened Patches（可以理解为Embedding层），通过Embedding层就可以把patches转换成一个个向量（图中粉色多边形部分 ），也就是token。
![在这里插入图片描述](https://img-blog.csdnimg.cn/a3a192ad706c476e88250bb4ab27c004.png)
3. 在这一系列 token 前面加上一个 token（图中粉色星号部分 ，这个token是用于分类的，其实就和BERT结构很相似，对应的就是BERT中的[CLS]向量）
![在这里插入图片描述](https://img-blog.csdnimg.cn/57b4db0269b2430ca3c1348253e0ae30.png)
4. 考虑位置信息，在每个token中添加Position Embedding（对应图中紫色部分）
![在这里插入图片描述](https://img-blog.csdnimg.cn/f1428f8422a045e98de1ba36d9cf0c39.png)
5. 将最终的 token 输入到 Transformer Encoder（结构如整体框架右侧图所示，这一部分我会在后面讲述）中。
<center><img src="https://img-blog.csdnimg.cn/34f41123becf46dcb772862c6af3dad6.png" width="80%">

6. 得到输出MLP Head（正常的Transformer Encoder输入多少个token就会有多少个输出，但ViT只是用于分类，因此只需要把第一个token经过encoder后的输出提取出来就行）
<center><img src="https://img-blog.csdnimg.cn/b8dd3fe1d3374e00a97fec2472fea90a.png" width="60%">

7. 通过MLP Head得到最终的分类结果
<center><img src="https://img-blog.csdnimg.cn/99f5abedba43481397ca6995fc2d4302.png" width="30%">

# 6. 模型
## Embedding层
&emsp;&emsp;对于标准的Transformer，要求输入的是token序列，即二维矩阵[num_token, token_dim]。在代码实现中，直接通过一个卷积层来实现。以ViT-16为例，使用卷积核大小为16×16，stride为16，卷积核个数为768(即token_dim)，然后再将高度和宽度的维度进行展平。
[224,224,3]->[14,14,768]->[196,768]

&emsp;&emsp;在输入Transformer Encoder之前需要加上[class]token以及Position Embedding（两者都是可训练参数）：
&emsp;&emsp;拼接[class]token：concat([1,768], [196,768]) -> [197,768]
&emsp;&emsp;叠加Position Embedding：[197,768] -> [197,768]

## Transformer Encoder层
<center><img src="https://img-blog.csdnimg.cn/a12fc426cf2147bfaf9b383bdf42c360.png" width="25%">&emsp;&emsp;&emsp;&emsp;<img src="https://img-blog.csdnimg.cn/c8d2a3530b664476be6ab22945ff5d8d.png" width="20%">

上图左侧这是论文里给出的Transformer Encoder结构，就是将Encoder Block重复堆叠L次。上图右侧是根据论文给出的源码重新画的结构（源码中在Multi-Head Attention后又进行了Dropout，但有实验用了DropPath（这个不是本文作者做的），效果是比Dropout好的）。

**Layer Normalization:**
[Pytorch官方](https://pytorch.org/docs/master/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm)给出了如下公式：
![在这里插入图片描述](https://img-blog.csdnimg.cn/bca447ec2fcd4f0f934d3e36d1e0567c.png)
推荐一个大佬的博文：[Layer Normalization解析](https://blog.csdn.net/qq_37541097/article/details/117653177?spm=1001.2014.3001.5506)

**Multi-Head Attention:**

Transformer论文(Attention Is All You Need) 给出了如下结构：
![在这里插入图片描述](https://img-blog.csdnimg.cn/2e976cc6618d420aa8526285eb728689.png#pic_center)
对于Scaled Dot-Product Attention(self-attention)，有计算公式：
![在这里插入图片描述](https://img-blog.csdnimg.cn/f92299c809374cfe9629a1a6e6d30b21.png#pic_center)
结合Scaled Dot-Product Attention的结构与计算公式，举例如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/333e361b85664eb0909d4ec7bf984093.png#pic_center)
最终的效果就是
<center><img src="https://img-blog.csdnimg.cn/2005583c9b0144a589d32090ccbd9566.png" width="50%">

Multi-Head Attention的计算其实就是基于self-attention(Scaled Dot-Product Attention)的，论文给出的计算公式如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/cc72463f8a1a4a4994403154bff47f37.png#pic_center)
举例如下：
![在这里插入图片描述](https://img-blog.csdnimg.cn/bda9a075edc64f108f4e9be7eaf32865.png)
**MLP Block:**

<center><img src="https://img-blog.csdnimg.cn/c0d46513cfc04a05a4ccd42df13a9f35.png" width="30%">

根据源码可以知道MLP Block的结构如上图所示，就是 Dense全连接层（结点数变为输入的4倍） -> GELU -> Dropout ->  Dense全连接层（还原成原来的数量） -> Dropout 

## MLP Head
训练ImageNet21k时用的是Linear Layer + tanh + Linear Layer，而如果是其他数据集的话，只有一个Linear Layer就足够了。

# 7. 部分实验
&emsp;&emsp;论文评估了ResNet，Vision Transformer(ViT)和hybrid(CNN与Transformer结合)三种模型，对不同大小的数据集进行了预训练，并测评了很多benchmark（基准测试）任务。

## 数据集
&emsp;&emsp;为了探索模型的泛用性，使用ILSVRC-2012 ImageNet数据集1k分类和1.3M图像（下面都称为ImageNet），其超集ImageNet-21k有着21k分类和14M图像，JFT有着18k分类和303M高分辨率图像。将训练前的数据集与Kolesnikov等人的下游任务的测试集进行去重。将这些数据集上训练的模型应用到几个benchmark任务中：ImageNet上的原始验证标签和清理后的真实标签，, CIFAR-10/100，Oxford-IIIT Pets，Oxford Flowers-102。对于这些数据集，预处理采用Kolesnikov等人的方法。
&emsp;&emsp;同时也评估了19-task VTAB classifification suite。VTAB中每个任务使用了1000个训练样例，对于不同任务评估low-data transfer。这些任务被分为三组：Natural任务（例如Pets, CIFAR等），Specialized任务（医疗和卫星图像），Structured任务（需要几何理解，如局部化）。

## 模型变体
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/1a28add423b34ecdaa7a09e4b72322f9.png#pic_center)

&emsp;&emsp;我们将ViT的配置建立在BERT的基础上，“Base”和“Large”模型直接采用BERT，此外添加了更大的“Huge”模型。例如，ViT-L/16表示具有输入patch的大小为16×16的“Large”变体。注意Transformer的序列长度与patch大小的平方成反比，因此patch尺寸较小的模型训练成本更高。


## Position Embedding消融实验
&emsp;&emsp;不使用或使用不同维度的Position Embedding会对模型产生什么样的影响呢？

&emsp;&emsp;论文附件处给出了在ImageNet 5-shot linear数据集上对Position Embedding消融的实验结果：
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/b6a4c1f088b8417780c8a995f41b8181.png#pic_center)

&emsp;&emsp;如上表，当没有使用Position Embedding时，得到的结果是61.382%，而使用1维、二维或相对位置编码时，都达到了64%，与不使用Position Embedding时相差了3个百分点，而对于是使用了几维、使用什么方式的Position Embedding，这个差别是不大的。源码中默认使用的是1-D Pos. Emb.，因为一维的相对来说更简单，且参数比较少，效果也更好。

&emsp;&emsp;关于使用不同的Position Embedding最终的结果差异不大这个问题，作者推测由于Transformer Encoder操作是在patch-level级别的输入，而不是pixel-level级别的输入，因此如何编码空间信息是不那么重要的。更具体地说，patch-level级别的输入，空间维度比原始的pixel-level小得多，例如14×14(patch)而不是224×224(pixel)，对于这些不同的位置编码策略，学习在这个分辨率中表示的空间关系更容易。尽管如此，网络学习的Position Embedding相似度取决于训练的超参数。

# 8. 改进思路&相关论文
[计算机视觉中的transformer模型创新思路总结](https://zhuanlan.zhihu.com/p/440940056)

# 相关文献及下载地址
1. Vision Transformer
论文名称：An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale
论文网址：[https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)
源码网址：[https://github.com/google-research/vision_transformer](https://github.com/google-research/vision_transformer)
2. Transformer
论文名称：Attention Is All You Need
论文网址：[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)
3. Layer Normalization
论文名称：Layer Normalization
论文网址：[https://arxiv.org/abs/1607.06450](https://arxiv.org/abs/1607.06450)