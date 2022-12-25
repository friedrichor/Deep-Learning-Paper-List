# Deep-Learning-Paper-List
List of deep learning papers, including CV (Computer Vision), NLP (Natural Language Processing), Multimodal and other research directions.

备注：表格中“简介”一列中加粗的部分为该论文中提出的模型名称。

# 多模态

### 多模态综述

|paper|源码|收录|简介|笔记|讲解|
|:-:|:-:|:-:|:-:|:-:|:-:|
|[Multimodal Machine Learning: A Survey and Taxonomy](https://ieeexplore.ieee.org/document/8269806)|-|IEEE 2019|-|-|-|
|[Multimodal Learning with Transformers: A Survey](https://arxiv.org/abs/2206.06488)|-|2022|-|-|-|

### 多模态数据集

|paper|源码|收录|简介|笔记|
|:-:|:-:|:-:|:-:|:-:|
| [Image-Chat: Engaging Grounded Conversations](https://aclanthology.org/2020.acl-main.219/) |  -   |ACL 2020|Image-Chat|-|
|[PhotoChat: A Human-Human Dialogue Dataset With Photo Sharing Behavior For Joint Image-Text Modeling](https://aclanthology.org/2021.acl-long.479/)|  -   |ACL/IJCNLP 2021|**PhotoChat**; 提出两个任务: 分享图片意图预测, 图像检索|-|
|[MMDialog: A Large-scale Multi-turn Dialogue Dataset Towards Multi-modal Open-domain Conversation](https://arxiv.org/abs/2211.05719)|[code](https://github.com/victorsungo/MMDialog)|2022|**MMDialog**; 大规模多模态对话数据集，相比于 PhotoChat 大了很多; 意图预测(文本/图像/停止); 检索式、生成式|正在写|

### 多模态经典模型

|paper|源码|收录|简介|笔记|讲解|
|:-:|:-:|:-:|:-:|:-:|:-:|
|[ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision](http://proceedings.mlr.press/v139/kim21k.html)|[code](https://github.com/dandelin/vilt)|ICML 2021|**ViLT**; 视觉和语言预训练 (Vision-and-Language Pre-training, VLP) 提高了各种视觉和语言联合下游任务的性能，作者提出了很小的 VLP 模型|[笔记](https://friedrichor.github.io/Deep-Learning-Paper-List/Multimodal/多模态经典模型/ViLT：Vision-and-Language%20Transformer%20Without%20Convolution%20or%20Region%20Supervision.html)|[视频](https://www.bilibili.com/video/BV14r4y1j74y/)|
|[Learning Transferable Visual Models From Natural Language Supervision]()|[code](https://github.com/OpenAI/CLIP)|ICML 2021|**CLIP**; 对比学习; Zero-shot; Prompt; 迁移能力很强，仅仅 zero-shot 就能和有监督训练的模型打成平手|[笔记](https://friedrichor.github.io/Deep-Learning-Paper-List/Multimodal/多模态经典模型/CLIP.html)|[视频](https://www.bilibili.com/video/BV1SL4y1s7LQ/)|

### 多模态对话

|paper|源码|收录|简介|笔记|
|:-:|:-:|:-:|:-:|:-:|
|[Learning to Respond with Stickers: A Framework of Unifying Multi-Modality in Multi-Turn Dialog](https://arxiv.org/abs/2003.04679)|[code](https://github.com/gsh199449/stickerchat)|WWW 2020|open-domain; 检索式; 可回复出表情包|-|
|[Towards Expressive Communication with Internet Memes: A New Multimodal Conversation Dataset and Benchmark](https://arxiv.org/abs/2109.01839)|[code](https://github.com/lizekang/DSTC10-MOD)|-|open-domain; 检索式; 可回复出表情包|-|
|[Multimodal Dialogue Response Generation](https://aclanthology.org/2022.acl-long.204/)|[code](https://aclanthology.org/2022.acl-long.204/)|ACL 2022|**Divter**; open-domain; 生成式; 可生成图像|[笔记](https://friedrichor.github.io/Deep-Learning-Paper-List/Multimodal/多模态对话/Multimodal%20Dialogue%20Response%20Generation.html)|
|[MMDialog: A Large-scale Multi-turn Dialogue Dataset Towards Multi-modal Open-domain Conversation](https://arxiv.org/abs/2211.05719)|[code](https://github.com/victorsungo/MMDialog)|2022|**MMDialog**; 大规模多模态对话数据集，相比于 PhotoChat 大了很多; 意图预测(文本/图像/停止); 检索式、生成式|正在写|

### Text-to-Image

|paper|源码|收录|简介|笔记|讲解|
|:-:|:-:|:-:|:-:|:-:|:-:|
|[Taming Transformers for High-Resolution Image Synthesis](https://openaccess.thecvf.com/content/CVPR2021/html/Esser_Taming_Transformers_for_High-Resolution_Image_Synthesis_CVPR_2021_paper.html)|[code](https://github.com/CompVis/taming-transformers)|CVPR 2021|**VQGAN**|-|-|
|[Zero-Shot Text-to-Image Generation](http://proceedings.mlr.press/v139/ramesh21a.html)|[code](https://github.com/openai/DALL-E)|ICML 2021|**DALL-E**|-|-|
|[CogView: Mastering Text-to-Image Generation via Transformers](https://proceedings.neurips.cc/paper/2021/hash/a4d92e2cd541fca87e4620aba658316d-Abstract.html)|[code](https://github.com/THUDM/CogView)|NeurIPS 2021|**CogView**; Transformer 和 VQ-VAE 相结合的生成模型|-|-|
|[NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion](https://arxiv.org/abs/2111.12417)|[code](https://github.com/microsoft/NUWA)|ECCV 2022|**NÜWA**; 不仅可以生成图像，还可以生成较短的视频|-|-|
|[GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models](https://proceedings.mlr.press/v162/nichol22a.html)|[code](https://github.com/openai/glide-text2im)|ICML 2022|**GLIDE**; 扩散模型; 生成高质量逼真图像|-|-|
|[Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125)|-|2022|**DALL-E 2**; 扩散模型; 结合 CLIP 和 GLIDE，相比于 DALL-E 和 GLIDE，具有更好的多样性，且计算效率更高|-|[视频](https://www.bilibili.com/video/BV17r4y1u77B/)|
|[Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding](https://arxiv.org/abs/2205.11487)|-|2022|**Imagen**; 扩散模型; 与 DALL-E 2 类似，但性能更强|-|-|
|[High-Resolution Image Synthesis with Latent Diffusion Models](https://ieeexplore.ieee.org/document/9878449)|[code](https://github.com/CompVis/latent-diffusion)|CVPR 2022|**LDMs**; 扩散模型|-|-|

### 多模态情感分析

|paper|源码|收录|简介|笔记|
|:-:|:-:|:-:|:-:|:-:|
| [Few-Shot Multi-Modal Sentiment Analysis with Prompt-Based Vision-Aware Language Modeling](https://ieeexplore.ieee.org/document/9859654) |        [code](https://github.com/yynj98/PVLM)         |      ICME 2022      |           **PVLM**; Few-shot; Prompt           |    -     |
| [Unified Multi-modal Pre-training for Few-shot Sentiment Analysis with Prompt-based Learning]() |       [code](https://github.com/yynj98/UP-MPF)        | ACM Multimedia 2022 |                **UP-MPF**; Few-shot; Prompt                |    -     |
| [Multimodal Sentiment Detection Based on Multi-channel Graph Neural Networks](https://aclanthology.org/2021.acl-long.28/) |   [code](https://github.com/YangXiaocui1215/MGNNS)    |   ACL/IJCNLP 2021   |                 **MGNNS**; GNN                 |    -     |
| [Image-Text Multimodal Emotion Classification via Multi-View Attentional Network](https://ieeexplore.ieee.org/document/9246699) |    [code](https://github.com/YangXiaocui1215/MVAN)    |      IEEE 2021      |                    **MVAN**                    |    -     |
| [Few-shot Multimodal Sentiment Analysis based on Multimodal Probabilistic Fusion Prompts](https://arxiv.org/abs/2211.06607) | [code](https://github.com/YangXiaocui1215/MultiPoint) |        2022         | **MultiPoint**; Few-shot; Probabilistic Fusion |    -     |

### 视频领域

|paper|源码|收录|简介|笔记|讲解|
|:-:|:-:|:-:|:-:|:-:|:-:|
|[Two-Stream Convolutional Networks for Action Recognition in Videos](https://proceedings.neurips.cc/paper/2014/hash/00ec53c4682d36f5c4359f4ae7bd7ba1-Abstract.html)|  -   | NIPS 2014 |               视频领域中应用深度学习的开山之作               | [笔记](https://friedrichor.github.io/Deep-Learning-Paper-List/Multimodal/视频领域/Two-Stream%20Convolutional%20Networks%20for%20Action%20Recognition%20in%20Videos.html) |[视频](https://www.bilibili.com/video/BV1mq4y1x7RU/)|
|[VideoBERT: A Joint Model for Video and Language Representation Learning](https://ieeexplore.ieee.org/document/9009570)|-|ICCV 2019|**VideoBERT**; 对动作分类任务直接进行 zero-shot 推理时，就能与先前的有监督训练好的 S3D 取得差不多的效果|[笔记](https://friedrichor.github.io/Deep-Learning-Paper-List/Multimodal/视频领域/VideoBERT：A%20Joint%20Model%20for%20Video%20and%20Language%20Representation%20Learning.html)|-|


# NLP

### 经典模型

|paper|源码|收录|简介|笔记|讲解|
|:-:|:-:|:-:|:-:|:-:|:-:|
| [Attention is All you Need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html) |-|   NIPS 2017    |                    **Transformer** 的提出                    |                              -                               | [视频](https://www.bilibili.com/video/BV1pu411o7BE/) |
| [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://aclanthology.org/N19-1423/) | [code](https://github.com/google-research/bert) | NAACL-HLT 2017 | **BERT**; 在 11 个不同的 NLP 任务中取得 SOTA，NLP 中里程碑式的成果 | [笔记](https://friedrichor.github.io/Deep-Learning-Paper-List/NLP/经典模型/BERT.html) | [视频](https://www.bilibili.com/video/BV1PL411M7eQ/) |


### 对话

|paper|源码|收录|简介|笔记|讲解|
|:-:|:-:|:-:|:-:|:-:|:-:|
| [DIALOGPT : Large-Scale Generative Pre-training for Conversational Response Generation](https://aclanthology.org/2020.acl-demos.30/) |        [code](https://github.com/microsoft/DialoGPT)         |   ACL 2020   | **DialoGPT** |  -   |  -   |
| [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://jmlr.org/papers/v21/20-074.html) | [code](https://github.com/google-research/text-to-text-transfer-transformer) |  JMLR 2020   |    **T5**    |  -   |  -   |
| [Language Models are Few-Shot Learners](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) |                              -                               | NeurIPS 2020 |  **GPT-3**   |  -   |  -   |
| [Recipes for Building an Open-Domain Chatbot](https://aclanthology.org/2021.eacl-main.24/) |                              -                               |  EACL 2021   | **Blender**  |  -   |  -   |
| [Towards a Human-like Open-Domain Chatbot](https://arxiv.org/abs/2001.09977) | [code](https://github.com/google-research/google-research/tree/master/meena/) |     2020     |  **Menna**   |  -   |  -   |

### 文本分类

|paper|源码|收录|简介|笔记|讲解|
|:-:|:-:|:-:|:-:|:-:|:-:|
| [Towards Unified Prompt Tuning for Few-shot Text Classification](https://arxiv.org/abs/2205.05313) |  -   | 2022 | 提出了统一的 Prompt Tuning 模版用于 Few-shot 文本分类 |[笔记](https://friedrichor.github.io/Deep-Learning-Paper-List/NLP/文本分类/Towards%20Unified%20Prompt%20Tuning%20for%20Few-shot%20Text%20Classification.html)|  -   |


# CV

### 图像分类


|paper|源码| 收录 |简介|笔记|讲解|
|:-:|:-:|:-:|:-:|:-:|:-:|
|[An Image Is Worth 16x16 Words: Transformers For Image Recognition At Scale](https://arxiv.org/abs/2010.11929)|[code](https://github.com/google-research/vision_transformer)|ICLR 2021|**Vision Transformer**|[笔记](https://friedrichor.github.io/Deep-Learning-Paper-List/CV/经典模型/Vision%20Transformer.html)|[博客](https://blog.csdn.net/qq_37541097/article/details/118242600)；<br>[视频](https://www.bilibili.com/video/BV1Jh411Y7WQ/)；<br>[视频(代码)](https://www.bilibili.com/video/BV1AL411W7dT/)|
|[Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)|[code](https://github.com/microsoft/Swin-Transformer)|ICCV 2021|**Swin Transformer**|-|[博客](https://blog.csdn.net/qq_37541097/article/details/121119988)；<br>[视频](https://www.bilibili.com/video/BV1pL4y1v7jC/)；<br>[视频(代码)](https://www.bilibili.com/video/BV1yg411K7Yc/)|
|[A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)|[code](https://github.com/facebookresearch/ConvNeXt)|CVPR 2022|**ConvNeXt**|-|[博客](https://blog.csdn.net/qq_37541097/article/details/122556545)；<br>[视频](https://www.bilibili.com/video/BV1SS4y157fu/)；<br>[视频(代码)](https://www.bilibili.com/video/BV14S4y1L791/)|


### 目标检测

|paper|源码|收录|简介|笔记|讲解|
|:-:|:-:|:-:|:-:|:-:|:-:|
| [Rich feature hierarchies for accurate object detection and semantic segmentation](https://arxiv.org/abs/1311.2524) |                         -                          | CVPR 2014 |    **R-CNN**     | [笔记](https://friedrichor.github.io/Deep-Learning-Paper-List/CV/目标检测/R-CNN.html) | [视频](https://www.bilibili.com/video/BV1af4y1m7iL?p=1) |
|        [Fast R-CNN](https://arxiv.org/abs/1504.08083)        |  [code](https://github.com/rbgirshick/fast-rcnn)   | ICCV 2015 |  **Fast R-CNN**  | [笔记](https://friedrichor.github.io/Deep-Learning-Paper-List/CV/目标检测/Fast%20R-CNN.html) | [视频](https://www.bilibili.com/video/BV1af4y1m7iL?p=2) |
| [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://proceedings.neurips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html) | [code](https://github.com/ShaoqingRen/faster_rcnn) | NIPS 2015 | **Faster R-CNN** | [笔记](https://friedrichor.github.io/Deep-Learning-Paper-List/CV/目标检测/Faster%20R-CNN.html) | [视频](https://www.bilibili.com/video/BV1af4y1m7iL?p=3) |

# 会议、论坛总结

