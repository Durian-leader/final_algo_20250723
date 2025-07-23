需求：撰写决赛设计文档的算法设计部分，符合要求，尽可能多的拿到分数
（明天下午答辩，明天上午我打算试试vad后先用SPK/OFFICIAL_all_target_vs_others.ipynb训练得到的模型做二分类，然后再五分类，但只看四分类，不看其他说话人。，因为直接做的五分类对于负样本的效果较差）

注意要写算法的设计思路，要合理，因为这个我记得占分数。综合考虑 数据预处理、PC和板子对齐、模型架构、等各方面因素。充分参考工程内所有内容，和初赛设计文档。完成决赛设计文档算法部分的撰写

该工程是做决赛用的工程。
VAD/vad_model_quant_mfcc12_cls2_sigmoid.tflite
SPK/checkpoints/spk-OFFICIAL_ALL-NO_NOISE_hid512_9X12 (adopted)
是最终采用的模型。


我提供的信息，你参考着选取：

VAD选择CNN单层网络的合理性参考文献：
VOICE ACTIVITY DETECTION IN NOISY ENVIRONMENTS
条目类型 	期刊文章
作者 	Nicklas Hansen
摘要 	Automatic speech recognition (ASR) systems often require an always-on low-complexity Voice Activity Detection (VAD) module to identify voice before forwarding it for further processing in order to reduce power consumption. In most real-life scenarios recorded audio is noisy and deep neural networks have proven more robust to noise than the traditionally used statistical methods. This study investigates the performance of three distinct low-complexity architectures – namely Long Short-Term Memory (LSTM) Recurrent Neural Networks (RNN), Gated Recurrent Unit (GRU) RNNs and an implementation of DenseNet. Furthermore, the impact of Focal Loss (FL) over the Cross-Entropy (CE) criterion during training is explored and ﬁndings are compared to recent VAD research.
语言 	en
文库编目 	Zotero
添加日期 	2025/5/18 21:03:01
修改日期 	2025/5/18 21:03:01


选择seliro教师模型做VAD的原因：

模型的输出更容易被新的模型学会，而不是人类标注的。
这一观点最早可以追溯到 **Caruana 等人在 KDD 2006 的论文《Model Compression》**，他们提出用一个性能更强的“教师”模型去为大量无标注样本打 pseudo-label，再让“学生”网络模仿这些 **模型输出**，而不是直接模仿 **人类标注**。作者指出，用这种教师-学生方式训练出的学生模型“几乎不损失性能，却容易收敛而且更小更快” [Cornell University Computer Science](https://www.cs.cornell.edu/~caruana/compression.kdd06.pdf)。

随后 **Hinton, Vinyals & Dean 在 2015 年的经典论文《Distilling the Knowledge in a Neural Network》** 将这一思想系统化为 “knowledge distillation”。他们明确写道：

> “当软目标（soft targets）熵较高时，每个训练样本携带的信息量远大于硬目标（hard labels），梯度方差也更小，因此学生模型往往**可以用更少的数据、更高的学习率就学会教师模型的行为**。” [arXiv](https://arxiv.org/pdf/1503.02531)

这段论述直接阐明了“**模型输出（软标签）对新模型来说比人类标注更容易学习**”的核心观点，并使得知识蒸馏成为今天广泛采用的模型压缩与迁移范式。


初赛时候是识别两个特定人和其他人，具体要求可以看初赛赛题要求.md
初赛时候的设计文档：初赛设计文档.md

决赛是识别四个特定人和其他人。具体赛题要求：决赛赛题要求.md

