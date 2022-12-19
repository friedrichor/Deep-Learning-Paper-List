[TOC]

# 0 前言
论文网址：[Learning to Select Knowledge for Response Generation in Dialog Systems](https://arxiv.org/abs/1902.04911)
源码网址：[Posterior-Knowledge-Selection](https://github.com/bzantium/Posterior-Knowledge-Selection)

本文是 Learning to Select Knowledge for Response Generation in Dialog Systems 中 Posterior Knowledge Selection（PostKS）模型的理论部分（不含实验）


# 1 背景
1. 传统的Seq2Seq模型成功实现了生成流畅的回复，但是它倾向于生成信息较少的回复（通用的回复），如“我不知道”、“那很酷”，这就导致了对话并不那么有吸引力。
2. 数据集Persona-chat和Wizard-of-Wikipedia在回复生成中引入了与对话相关的知识（例如Persona-chat中的个人简介）用于指导对话流程。Dinan等人使用ground-truth知识来指导知识选择，这是优于那些不使用这些知识的。然而，ground-truth知识实际上是很难获得的。
3. 现有的研究大多是基于 输入语句 与 知识 之间的语义相似度来选择知识，这种语义相似度被认为是**知识的先验分布**。然而，**先验分布不能有效地指导正确的知识选择**，因为不同的知识可以用于对同一个的输入语句产生影响，从而生成不同的回复。相反，在给定输入语句及其相应的回复时，从**输入语句和回复**（区别于先验分布中的仅使用输入语句）中推导出的**后验分布可以有效地指导知识选择**，因为它考虑了**回复**中实际使用的知识。先验分布和后验分布的差异给学习过程带来了困难：模型在没有回复的情况下，仅根据先验分布难以选择合适的知识，在推理过程中难以获得正确的后验分布。这种差异会阻止模型通过使用适当的知识来产生正确的回复。
<hr>

&emsp;&emsp;对于第3点，详细解释一下：假如说数据集中的输入语句是$X$，它的回复是$Y$，对于这个对话对儿有若干个知识 ${\{K_i\}}_{i=1}^N$。先验分布就是说我在只知道$X$的情况下，要挑选一个合适的知识$K_i$的准确性；而后验分布是不仅知道$X$，且知道$Y$的条件下，挑选一个合适的知识$K_i$的准确性，很显然后验分布是更准确，但后验分布只能在训练中得到，实际应用中，以聊天机器人为例，用户发送给系统一句话，因为回复是需要机器自己生成的，机器仅能依据用户的输入语句选择这个语句的知识，这时就只能得到先验分布。


<hr>

<center><img src="https://img-blog.csdnimg.cn/0fbcdd13249f4e7aa11db8ce498e87dd.png" width="60%">

<table>
<tr>
    <td>Utterance</td>
    <td>嗨！我没有最喜欢的乐队，但我最喜欢的读物是黄昏</td>
  </tr>
  <tr>
    <td rowspan="4">Profiles/Knowledge</td>
  </tr>
  <tr><td>K1：我喜欢红辣椒乐队</tr>
  <tr><td>K2：我的脚是六码女的</tr>
  <tr><td>K3：我想成为一名记者，但我却在西尔斯卖洗衣机</tr>
  <tr>
    <td>R1(no knowledge)</td>
    <td>你是干什么的？</td>
  </tr>
  <tr>
    <td>R2(use K2)</td>
    <td>我买了一双六码女鞋</td>
  </tr>
  <tr>
    <td>R3(use K3)</td>
    <td>我是个好记者</td>
  </tr>
  <tr>
    <td>R4(use K3)</td>
    <td>我也喜欢读书，并希望成为一名记者，但现在只能卖洗衣机</td>
  </tr>
  <tr>
    <td>Response</td>
    <td>我喜欢写作！想当记者但是我只能在西尔斯将就着卖洗衣机。</td>
  </tr>
</table>

&emsp;&emsp;如上表中的对话，在这个数据集中，每一个角色都会与一个作为知识的角色相关联。两个角色根据相关的知识交换信息，给定一个语句Utterance，根据是否使用了适当的知识Knowledge，可以产生不同的回复R。

&emsp;&emsp;$R1$ 没有使用任何知识，因此生成一个信息较少的答复，而其他的答复拥有更多的信息，因为他们吸收了外部的知识（Profiles/Knowledge部分）。然而，在这些知识中，$K1$ 和 $K3$ 都与 输入语句Utterance 相关。如果我们仅仅根据 Utterance （即先验信息）选择 knowledge，而没有使用 $K3$ 来产生应答 $R$（即后验信息），则很难产生正确的应答，因为可能没有选择适当的知识。如果通过选择错误的知识（如 $R2$ 使用了 $K2$）或与实际回复无关的知识（如$K1$）来训练模型，可以看出它们是完全无用的，因为它们不能提供任何有用的信息。请注意，在回复$R$ 生成中**适当地**吸收知识也很重要。例如，虽然 $R3$ 选择了正确的知识 $K3$，但是它由于不恰当地使用知识，导致生成了不太相关的回复。只有 $R4$ 对 knowledge 进行了适当的选择，并在生成回复时适当地合并了它。

&emsp;&emsp;为了解决上述差异，提出将后验分布与先验分布分离。在知识的后验分布中，Utterance 和 回复Response都被利用，而先验分布在没有预先知道Response的情况下有效。然后，我们试着最小化它们之间的距离。具体来说，在训练过程中，我们的模型被训练成最小化先验分布和后验分布之间的 KL divergence 收敛，这样我们的模型可以利用先验分布准确地逼近后验分布。然后，在推理过程中，模型仅根据先验分布（即生成回复时不存在后验）对知识进行采样，并将采样后的知识纳入到响应生成中。在此过程中，该模型可以有效地学习利用适当的知识生成适当的、信息丰富的响应。

# 2 Posterior Knowledge Selection 模型（PostKS）
假设一个语句 $X=x_1 x_2⋯x_n( x_t  是 X 中第 t 个单词)$，存在一个知识集合${\{K_i\}}_{i=1}^N$，目标是从集合中选择适当的知识，并通过选择的知识生成应答 $Y=y_1 y_2⋯y_n$。PostKS模型的架构如图所示，主要由四部分组成：
- 对话编码器(Utterance Encoder)：将$X$编码成一个对话向量$x$，并将其输入知识管理器。
- 知识编码器(Knowledge Encoder)：将每一个$K_i$作为输入，然后把它编码成知识向量$k_i$，当回复$Y$可用时将其编码成向量$y$。
- 知识管理器(Knowledge Manager)：由两个子模块组成：先验知识模块和后验知识模块。给定先前编码的 $x$ 和 ${\{K_i\}}_{i=1}^N$(如果 $y$ 可用)，知识管理器负责选择一个适当的$k_i$并使它(与一个基于注意的上下文向量$c_t$)进入解码器。（这里的 $y$ 是否可用是针对先验和后验的，先验就是不可用，后验就是可用）
- 解码器(Decoder)：根据所选知识$k_i$以及基于注意的上下文向量$c_t$生成回复。

<center><img src="https://img-blog.csdnimg.cn/de2f4c2019be400f921b74c8af1e68e1.png" width="60%">

## 2.1 对话编码器&知识编码器(Utterance Encoder&Knowledge Encoder)
&emsp;&emsp;在对话编码器中使用GRU的双向RNN(由前向RNN和反向RNN组成)，对于语句$X=x_1 x_2⋯x_n$，前向RNN从左向右读取$X$并获得每个$x_t$从左向右的隐藏状态 ($\overrightarrow{h_t}$)  ，同理反向RNN从右向左读取$X$并获得每个$x_t$从右向左的隐藏状态 ($\overleftarrow{h_t}$ )  ，这两个隐藏状态合并成一个总的隐藏状态$h_t$：

<center><img src="https://img-blog.csdnimg.cn/7622b3308e60424e912346974aa1b0dc.png" width="60%">

&emsp;&emsp;利用隐藏状态并定义 $x=[ \overrightarrow{h_t};\overleftarrow{h_t}]$，这个向量会传送到知识管理器来优化知识的选择，同时它将会作为解码器的初始隐藏状态。
&emsp;&emsp;知识编码器的结构与对话管理器相同，但它们之间不共享参数。知识编码器使用双向RNN将每一个知识$K_i$(如果回复$Y$可用)转化成向量$K_i$并传入知识管理器。 
## 2.2 知识管理器(Knowledge Manager)

<center><img src="https://img-blog.csdnimg.cn/753cf8bbafd14b5d9ddc20d004821ee7.png" width="60%">

给定编码后的语句$x$和知识集合${\{K_i\}}_{i=1}^N$，知识管理器的目标是选择一个合适的个性$k_i$  ，当应答$y$可用时，模型也将同时利用$y$得到$k_i$ 。知识管理器由先验知识模块和后验知识模块两个模块组成。

&emsp;&emsp;在先验知识模块中，定义了知识上的条件概率分布 $p(k│x)$：

<center><img src="https://img-blog.csdnimg.cn/666ab03f725044079eebba1ce52539de.png" width="40%">

使用点乘来衡量 $k_i$ 与 $x$ 的关联性，$p(k│x)$表示仅有$x$的时候，即它在不知道应答的情况下工作，因此它是知识先验分布。但不同对话相关的知识各有不同，所以在训练中仅仅使用先验分布来选择知识是十分困难的。

&emsp;&emsp;在后验知识模块中，通过考虑输入语句及其应答，定义了知识的后验分布$p(k│x,y)$:

<center><img src="https://img-blog.csdnimg.cn/6ac890734d474e20add32194c2f5081d.png" width="50%">

其中，MLP是一个全连接层。通过比较先验分布，后验分布是比较准确的，可以获得应答$Y$使用的知识内容。

&emsp;&emsp;显然，后验分布在选择知识时是优于先验分布的（因为后验分布是在已知$x$和$y$的条件下的概率，而先验分布是在仅已知$x$条件下的概率），但在推理生成应答阶段中，后验分布是未知的，因此期望先验分布能够尽可能地接近后验分布。为此，引入了Kullback-Leibler divergence loss(KLDivLoss)作为辅助损失函数，用来衡量先验分布和后验分布的接近性：

<center><img src="https://img-blog.csdnimg.cn/579cbfa4795e40f2b54516d5f46e387f.png" width="50%">

其中 θ 为模型参数。

&emsp;&emsp;在最小化KLDivLoss时，后验分布$p(k│x,y)$可以被视为标签，模型使用先验分布$p(k│x)$ 来精确地近似 $p(k│x,y)$。因此，即使在推理过程中后验分布是未知的（因为真实回复$Y$是未知的），也可以有效地利用先验分布 $p(k│x)$ 对适当的知识进行抽样，从而产生合适的回复。作者认为它是第一个神经模型：将后验分布作为指导，使准确的知识查找和高质量的回复生成成为可能。
## 2.3 解码器
&emsp;&emsp;解码器通过合并所选知识$k_i$逐字生成应答，使用分级门控融合单元(Hierarchical Gated Fusion Unit, 简称HGFU)。HGFU提供了一种将知识融合到应答生成中的方法，由话语GRU、知识GRU和融合单元三个主要部分组成。

&emsp;&emsp;话语GRU和知识GRU遵循标准GRU结构，基于上一个状态$s_{t-1}$和上下文向量$c_t$分别对最后生成的$y_{t-1}$和选择的知识$k_i$生成隐藏状态：

<center><img src="https://img-blog.csdnimg.cn/c818d4dd22054268a831403f45278ac3.png" width="50%">

然后，融合单元将它们组合在一起，生成整体的隐藏状态：

<center><img src="https://img-blog.csdnimg.cn/96233ce2620743f6af596a2e44076c8d.png" width="30%">

其中，$r=σ(W_z [tanh⁡(W_y s_t^y );tanh⁡(W_k s_t^k )])，W_z，W_y，W_k$是参数。门控 $r$ 控制 $s_t^y和 s_t^k$对最终隐藏状态$s_t$的影响比例，以便于它们能够灵活的融合。

&emsp;&emsp;获得隐藏状态$s_t$后，下一个单词$y_t$ 根据下面的概率分布生成：

<center><img src="https://img-blog.csdnimg.cn/678ac5e089e6467ebdbc54e823e4e259.png" width="30%">

## 2.4 损失函数
<center><img src="https://img-blog.csdnimg.cn/753cf8bbafd14b5d9ddc20d004821ee7.png" width="60%">

仍然是看这个图，除了刚刚在上面 *2. 知识管理器(Knowledge Manager)* 中提到的 KLDivLoss，还用了NLL Loss 和 BOW Loss。

NLL Loss：用于衡量 真实response 和 模型生成的 response 之间的差异：

<center><img src="https://img-blog.csdnimg.cn/72dfaabd89d0430bb2e76a938e1b4191.png" width="50%">

BOW Loss：通过加强知识与真实response的关联来确保采样知识 $k_i$ 的准确性。论文中给出，令 $w=MLP(k_i)\in R^{|V|}$，其中 $|V|$是 vocabulary size，定义 
$$p(y_t|k_i)=\frac{exp(w_yt)}{\sum_{v\in V}exp(w_v)} \quad$$

<center><img src="https://img-blog.csdnimg.cn/25505594ccaa4e989bc3fc0881daeb15.png" width="50%">

整个模型的损失函数就是将 KLDivLoss、NLL Loss 和 BOW Loss 三者相加：

<center><img src="https://img-blog.csdnimg.cn/a05a9790c83b4da7bfa0501be020dcc1.png" width="40%">

在代码中也就是将这三个损失相加后作为整个模型的损失，共同调整模型参数：

<center><img src="https://img-blog.csdnimg.cn/40e222b2bf924f3493a58b22ca1787f8.png" width="40%">


# 3 Q & A
## 3.1 先验知识模块相关问题
在知识管理器 Knowledge Manager 的先验知识模块中，定义了知识上的条件概率分布 $p(k│x)$：

<center><img src="https://img-blog.csdnimg.cn/666ab03f725044079eebba1ce52539de.png" width="40%">

这里可能会有疑问，给定了 知识（$\{k_i\}_{i=1}^N$） 和 输入语句（$x$） 不就能知道哪个 $k_i$ 的概率最大了吗？

首先看 Manager 类中，下面是源码（model.py 中）：
<img src="https://img-blog.csdnimg.cn/19a93e913dc7468e8ddabc7a4a7b211a.png" width="70%">


在 forward 中，有个 if 条件语句，用来判断 $y$ 是否可用的（$y$ 即是 $response$，训练时 $y$ 已知，测试时 $y$ 未知），所以 满足`y is not None`时（训练时），计算了知识的先验和后验分布，而不满足`y is not None`时（测试时），仅仅能计算知识的先验分布。

在训练时（图中165-188行），`prior` 为知识的先验分布，其计算代码如下：

```python
prior = F.log_softmax(
	torch.bmm(x.unsqueeze(1), K.transpose(-1, -2)), dim=-1
).squeeze(1)
```
这么乍一看确实是 给定 $\{k_i\}_{i=1}^N$ 和 $x$ 就能知道哪个 $k_i$ 的概率最大，这里确实是没涉及调整 Manager 中的模型参数，但是这其中的 $x$ 和 $K$ 是通过编码器（Utterance Encoder和Knowledge Encoder）得到的（$x$ 和 $K$是通过模型计算得到的，并不是一成不变的），所以说 先验知识模块 这里应该是通过不断修改 编码器 中的模型参数来进行训练、优化的。

看下源码中训练的部分（train.py）:
<img src="https://img-blog.csdnimg.cn/b53df8af20194edb80e68f5557ebeda9.png" width="70%">

$x$ 是通过encoder（源码 model.py 中的 Encoder 类）得到的，而$K$ 是通过Kencoder（源码 model.py 中的 KnowledgeEncoder 类）得到的。训练时即是通过不断调整 Encoder 和 KnowledgeEncoder 中 GRU 的参数，来不断优化先验分布的，来让知识的先验分布不断地贴近后验分布，以提高选择知识 $K_i$ 的准确率。

综上，公式确实是没有问题，不过这里的 $x$ 并不是原始的输入语句，$\{k_i\}_{i=1}^N$也不是原始的知识，他们都是编码后的，训练时公式中的 $x$ 和 $k_i$ 是变化的，所以才能不断调整先验分布。不过这里确实有些误导因素，因为调整先验分布这部分的参数却不是在 Knowledge Manager 中实现的。