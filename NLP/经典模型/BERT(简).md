[TOC]

# 模型架构
BERT的基础transformer结构（encoder部分）：
![在这里插入图片描述](https://img-blog.csdnimg.cn/4cd54983690a45538652b9440c819b86.png?,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAZnJpZWRyaWNob3I=,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)
**输入部分：**
对于transformer来说，输入部分会进行两个操作，包括Input Embedding和Positional Encoding两部分。
Input Embedding就是将输入转为词向量，可以是随机初始化，也可以是使用word2vec。
Positional Encoding就是位置编码，用正余弦三角函数来代表它。

以上是输入部分进行的操作，那么输入又是什么呢？
实际上输入由三部分组成：Input = token embedding + segment embedding + position embedding 
![在这里插入图片描述](https://img-blog.csdnimg.cn/36ae345ac8dd4fc2a230e36534c9e9c6.png?,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAZnJpZWRyaWNob3I=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
首先看Input部分，重点关注两个部分：
1. 正常词汇：my, dog, is, cute, he, likes, play, ##ing(这种符号是分词后的产物，不用特意关注，当做正常词看就行)
2. 特殊词汇：[CLS], [SEP]
两种特殊词汇的产生是由于BERT的预训练任务有一个是NSP(Next Sentence Prediction)二分类任务，是去判断**两个句子**之间的关系。[SEP]就是为了告诉机器，在这个符号之前的是一个句子，在这个符号之后的是另一个句子。[CLS]是用于二分类的特殊符号，在训练时会将[CLS]的输出向量接一个二分类器来做二分类任务。但请注意，[CLS]的输出向量并不能代表句子的语义信息（用CLS做无监督的文本相似度时效果很差）。bert pretrain模型直接拿来做sentence embedding效果甚至不如word embedding，CLS的embedding效果最差（也就是pooled output），把所有普通token embedding做pooling勉强能用（这也是
开源项目bert-as-service的默认做法），但也不会比word embedding更好。

- Token Embeddings：对所有词汇进行正常的embedding，比如随机初始化等。
- Segment Embeddings：用于区分两个句子，如上图所示，第一个句子的单词全是$E_A$，第二个句子的单词全是$E_B$。
- Position Embeddings：这一部分和基础结构中输入部分的Positional Encoding操作是**不同的**。Positional Encoding使用正余弦函数，而Position Embeddings使用的是随机初始化，让模型自己学习出来Embedding。

最开始那张图仅仅是基础结构，因为在原论文中使用的是多个encoder堆叠在一起，如BERT-BASE结构是通过12个encoder堆叠在一起。

# 预训练步骤
分为MLM(Mask Language Model)和NSP(Next Sentence Prediction)两步
## MLM(Mask Language Model)
BERT在预训练时使用的是大量的无标注语料，所以预训练任务要考虑用无监督学习来做。

无监督目标函数：
1. AR(Auto Regressive)：自回归模型，只考虑单侧的信息，典型的就是GPT
2. AE(Auto Encoding)：自编码模型，从损坏的输入数据中预测重建原始数据，可以使用上下文的信息，这也是BERT使用的方法。

例如有语句：【我爱吃饭】

AR：$P(我爱吃饭) = P(我)P(爱|我)P(吃|我爱)P(饭|我爱吃)$

AE：mask之后：【我爱mask饭】
&emsp;&emsp;$P(我爱吃饭|我爱mask饭) = P(mask = 吃|我爱饭)$
打破了原本文本，让他进行文本重建，模型要从周围文本中不断学习各种信息，努力地让他能够预测或无限接近mask这里应该填“吃”。
但mask模型也有缺点：
若mask后【我爱mask mask】
优化目标：$P(我爱吃饭|我爱mask mask) = P(吃|我爱)P(饭|我爱)$
这里“吃”和“饭”模型会认为是相互独立的，但实际上我们知道“吃”和“饭”这两个词并不是独立的，室友一定关联的。

下面将介绍mask的具体过程：
随机mask 15%的单词，但并不是这些单词都要替换成mask。这15%的单词中，选出其中80%的单词直接替换成mask，其中10%的单词原封不动，剩下10%替换成其他单词，可以看代码更好地理解一下：

```python
for index in mask_indices:
	# 80% of the time, replace with [MASK]
	if random.random() < 0.8:
		masked_token = "[MASK]"
	else:
		# 10% of the time, keep original
		if random.random() < 0.5:
			masked_token = tokens[index]
		# 10% of the time, replace with random word
		else:
			masked_token = random.choice(vocab_list)
```

## NSP 
NSP样本如下：
1. 从训练语料库中取出两个连续的段落作为正样本
2. 从不同文档中随机创建一对段落作为负样本
缺点：主题预测（两段文本是否来自同一文档）和连贯性预测（两个段落是不是顺序关系）合并成一个单项任务。由于主题预测是非常简单的，非常容易去学习，导致NSP很容易没有效果。

# 下游任务微调BERT
![在这里插入图片描述](https://img-blog.csdnimg.cn/6ad68c586c624b64b06afbdc67ee58f4.png?,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAZnJpZWRyaWNob3I=,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
(a)句子对分类：也可以说是文本匹配任务，把两个句子拼接起来，用CLS输出判断，如0—不相似，1—相似；
(b)单个句子分类：用CLS输出做分类；
(c)问答
(d)序列标注任务：把每一个Token输入，做一个softmax，看他属于哪一个。

# 如何提升BERT下游任务表现
最简单的步骤：
1. 获取谷歌中文BERT
2. 基于任务数据进行微调

再改进一些（四步骤，以做微博文本情感分析为例）：
1. 在大量通用语料上训练一个LM(Language Model，语言模型，以下简称LM)（Pretrain）；
——一般不去做，直接用中文谷歌BERT
2. 在**相同领域**上继续训练LM（Domain transfer）;
——在大量*微博文本*上训练这个BERT
3. 在**任务相关**的小数据上继续训练LM（Task transfer）；
——在*微博情感文本*上（有的文本不属于情感分析的范畴）
4. 在任务相关数据上做具体任务（Fine-tune）

一般情况下，先 Domain transfer，再进行 Task transfer，最后 Fine-tune，性能是最好的。

如何在相同领域数据中进行further pre-training
1. 动态mask：每次epoch去训练的时候mask，每次训练的mask很大概率是不一样的，而不是一直使用同一个。
2. n-gram mask：比如 ERNIE 和 SpanBert 都是类似于做了实体词的mask

参数：
Batch size：16，32——影响不太大
Learning rate(Adam)：5e-5，3e-5，2e-5，尽可能小一点，避免灾难性遗忘
Number of epoch：3，4
Weighted decay修改后的Adam，使用warmup，搭配线性衰减