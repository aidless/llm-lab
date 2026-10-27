# LLM应用架构：多模态大模型应用——视觉理解、生成与对话

## 1 多模态应用架构概览

### 1.1 多模态系统的应用谱系

多模态大模型的应用谱系以模态组合方式展开，涵盖文本-图像、文本-视频、文本-音频、文本-代码等基础通路。文本-图像维度包含三个子方向：文生图（text-to-image，给定描述生成像素阵列）、图生文（image captioning，给定图像输出语义描述）和图像问答（visual question answering，VQA，给定图像+问题输出答案）。文本-视频延伸出文生视频（text-to-video，从文本描述生成连续帧序列）和视频理解（video captioning / video QA，从视频中提取时序语义）。文本-音频覆盖文生语音（text-to-speech，TTS）和语音识别（automatic speech recognition，ASR），而文本-代码方向则涉及代码生成与反编译解释。

这些模态组合虽然看似离散，但核心架构范式正走向统一。Bai等（2023）在Gemini的技术报告中系统阐述了从分治到统一的趋势：早期的多模态系统以"组合模型"为主——对文本、图像、音频分别使用独立的预训练编码器（如CLIP文本分支、CLIP图像分支、Whisper音频编码器），然后通过后期融合（late fusion）在表示层或决策层进行对齐。Radford等（2021）的CLIP正是这一范式的典范——文本和图像分别经过Transformer编码器和ViT编码器，通过对比学习（contrastive learning）将配对样本的余弦相似度最大化，将非配对样本推开，最终实现跨模态embedding空间的对齐。后期融合的优势在于各模态编码器可独立预训练，但缺点是模态间交互仅在顶层发生，细粒度跨模态注意力缺失。

主流正向"统一模型"进化，即用一个架构原生处理多模态输入。Kim等（2021）提出的ViLT（Vision-and-Language Transformer）是最早的尝试之一——将图像patch embedding和文本token embedding直接拼接送入一个Transformer，无需独立的视觉特征提取器（即去掉了Faster R-CNN等region特征），实现了极简的统一前向。虽然ViLT在大规模数据上的精度不及双塔方案，但它验证了"一个模型处理所有模态"的可行性。Alayrac等（2022）的Flamingo则采取了折中路径——保留预训练的视觉编码器和语言模型，通过引入gated cross-attention层在LLM的冻结层之间桥接视觉信息，每次forward在文本self-attention后插入cross-attention层，key/value来自视觉特征的perceiver resampler。这种"桥接"策略在不大幅改动原模型的前提下实现了多模态能力注入，且由于文本和视觉编码器均冻结，计算效率较高。

### 1.2 高质量多模态数据的获取

多模态模型的数据基石是图文对（image-text pairs），其质量和规模直接影响下游性能。LAION-5B（Schuhmann等，2022）是迄今最大的公开图文数据集，包含58.5亿图文对，通过Common Crawl的网页数据中提取alt-text和关联图像构建。但Web-scale数据噪声极大——大量图文对不相关（如广告默认图与随机文本）、分辨率低或包含文字污染。

为应对这一挑战，DataComp（Gadre等，2023）提供了一个受控实验平台，专门研究数据筛选策略对模型性能的影响。其核心发现是：CLIP scoring-based过滤优于简单的关键词过滤——使用预训练CLIP模型计算每对图文之间的余弦相似度，剔除低于阈值的样本，可以有效提升下游zero-shot分类准确率5-10个百分点。进一步结合PDF（perceptual duplicate detection）去重和基于图像分辨率的过滤（丢弃<512px的低分辨率模糊图像），可以得到信息密度显著更高的"高质量子集"。DataComp的实验表明，在12.8B候选池中仅保留30%高分样本训练出的CLIP模型，在ImageNet zero-shot上优于使用全部数据训练的模型。这一"少而精"的结论直接影响了后续多模态模型的训练数据策略。

### 1.3 多模态的形态与统一

从架构演进角度看，"组合模型→统一模型"并非非此即彼。2024-2025年的前沿系统（如GPT-4V、Gemini 1.5 Pro）实际上采用了混合架构：视觉部分使用独立编码器（如ViT-22B级别的视觉Transformer）提取patch embedding，通过cross-attention或Q-Former适配桥接至LLM主体，而非将原始像素直接拼入文本token序列（后者在计算成本上不可行）。这种"视觉编码器+桥接层+LLM"的三段式结构在当前阶段取得了精度与效率的最佳平衡。真正统一的关键在于embedding space的对齐——无论视觉还是文本，最终都映射到语言模型可以操作的隐空间。

---

## 2 图像理解应用

### 2.1 OCR与文档理解

文档理解是图像理解最具商业价值的应用之一。其技术栈分为文本检测和文字识别两个子任务，然后叠加LLM的结构化理解。

**文本检测**负责定位图像中的文字区域并输出文本框。早期方案基于目标检测框架改造（如TextBoxes++，Liao等，2018），但场景文本（scene text）的任意形状和多方向性要求更精细的算法。CRAFT（Character Region Awareness for Text Detection，Baek等，2019）通过检测字符级别的区域和相邻关系，生成像素级的字符区域分数图和字符间亲和力图，再通过后处理合成文本框，对弯曲文本和密集文本有较好的召回率。DBNet（Liao等，2020）则将分割和阈值二值化合并到同一个网络中端到端可微训练，通过可微二值化（Differentiable Binarization）模块动态预测每个像素的阈值，显著简化了后处理流程，在ICDAR 2015数据集上达到82.7的F1-score，推理速度达到62 FPS。

**文字识别**在检测到的文本框内将图像内容转录为字符串。CRNN（Convolutional Recurrent Neural Network，Shi等，2017）是最经典的端到端识别框架——CNN（VGG/ResNet变体）提取序列特征→BiLSTM建模时序依赖→CTC（Connectionist Temporal Classification）解码输出字符串。其核心优势在于不需要字符级标注，仅需要文本行级别的标注即可训练，且CTC的对齐机制可以处理不定长序列。对于中文场景，CRNN的改进包括在特征提取阶段使用更深的ResNet-50作为backbone、在解码阶段使用Transformer decoder替代CTC（如ASTER，Shi等，2018）来增强长文本识别精度。

**LLM的文档理解的结合**是近两年的重要演进方向。流程上，OCR将文档图像转化为结构化文本（保留段落边界、表格结构标记、标题层级），然后送入LLM进行语义理解。以表格理解为例，PaddleOCR的输出层增加了表格结构识别分支——同时预测单元格坐标和行列索引，输出HTML格式的表格代码；LLM然后可以解析HTML表格、回答"该季度营收增长率是多少"这类推理性问题。更进一步的方案是直接将OCR文本embedding与LLM内部表示融合，消解OCR阶段引入的误差——如Donut（Kim等，2022）提出了完全端到端的文档Transformer，跳过OCR，直接图像→token→结构化JSON，消除了OCR级联误差的累积。

### 2.2 仪表盘与看板识别

在AIoT场景（智慧工厂、能源管理）中，自动化读取传统仪表盘数据是巨大的实际需求。示数读取包含两个子任务：指针位置检测+表盘数字OCR。

指针定位通常使用object detection框架（如轻量级YOLOv8n）检测指针末端关键点或指针区域，然后计算指针相对于刻度基线的旋转角度，结合数字OCR读取的量程范围，将角度线性映射为物理读数。数字OCR则专注在LCD或LED数码管区域的小范围文字识别——数码管数字具有严格的字体结构（7段/16段），可以用针对性的segmentation-based方法（逐段判断亮灭编解码）替代通用OCR，精度可达99.5%以上。

在工程落地层面，一个完整的工业仪表读取系统往往部署在边缘设备（Jetson Nano/RK3588），需要在算力约束下运行检测+识别+推理三阶段。量化（INT8）和知识蒸馏是常用的压缩手段——将YOLOv8n和CRNN分别量化到INT8，推理延迟可从原始的120ms降至25ms每帧。2025年的趋势是将图像→文本读出→异常判断的整条流水线使用一个VLM（视觉语言模型）端到端完成，例如直接输入仪表盘图像输出"当前压力：0.78MPa，在安全范围内"（Chen等，2025[^1]）。这一方案省去了检测-识别-LLM串行推理的复合延迟和误差累积，但要求VLM具有较强的细粒度数值感知能力。

---

## 3 文生图系统的工程化

### 3.1 Stable Diffusion的架构理解

Stable Diffusion（Rombach等，2022）的架构基于Latent Diffusion Model（LDM），其核心思路是将扩散过程从像素空间转移到隐空间，极大降低计算复杂度。推理管线分为三个组件：

（1）**VAE（Variational Autoencoder）编码器**：将像素空间图像 $x \in \mathbb{R}^{H \times W \times 3}$ 压缩至低维隐空间 $z \in \mathbb{R}^{h \times w \times c}$，压缩因子 $f = H/h$ 一般为8（即 512×512 → 64×64×4）。VAE编码器通过下采样卷积将空间维度减小，同时保持语义信息。这一压缩使得后续扩散过程的计算量约降为像素空间的 $1/f^2 = 1/64$，是SD高效推理的根本原因。

（2）**U-Net扩散模型**：在隐空间执行前向扩散（加噪）和反向去噪（预测噪声）。正向过程将标准高斯噪声逐步叠加到隐变量上，反向过程训练U-Net预测每一步添加的噪声 $\epsilon_\theta(z_t, t, c)$。U-Net由编码器-解码器结构构成，中间层通过跳跃连接（skip connection）保留细节信息。关键组件包括ResNet blocks和SpatialTransformer blocks——后者正是承载文本条件注入的位置。

（3）**VAE解码器**：将去噪后的隐变量 $\hat{z}_0$ 还原到像素空间 $\hat{x} = \mathcal{D}(\hat{z}_0)$，得到最终的生成图像。

### 3.2 文本条件的Cross-Attention

SD的条件控制通过Cross-Attention机制实现。在U-Net的每个SpatialTransformer block中，query（Q）来自U-Net的隐空间特征图（shape：$H_{feat} \times W_{feat} \times d$，展平为token序列），key（K）和value（V）来自CLIP文本编码器输出的文本嵌入序列（77个token，每个768/1024维）。计算方式为：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

输出是加权后的文本特征与空间特征的融合，使得U-Net在去噪过程中明确知道"要生成什么内容"。CFG（Classifier-Free Guidance，Ho & Salimans，2022）进一步放大了条件的作用——推理时同时预测条件输出和无条件输出，取外推：

$$
\epsilon_\theta = \epsilon_\theta(z_t, t, \varnothing) + w \cdot (\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \varnothing))
$$

其中 $w$ 为引导尺度（通常取7-12），$c$ 为条件嵌入，$\varnothing$ 为空文本嵌入。CFG有效地在采样质量和多样性之间提供了可调节机制。

### 3.3 计算加速与工程优化

SD的生产部署面临显存和推理速度的基本约束。第一层优化是精度降级——将U-Net和VAE从FP32切换到FP16，显存占用降低约50%，推理速度提升约1.8倍，质量损失几乎不可感知（在FID指标上<0.3的退化）。配合torch.compile或TensorRT的图优化，可以在NVIDIA A100上将512×512的单步推理压缩至~15ms（50步约750ms）。

第二层优化是推理步数减少。LCM（Latent Consistency Model，Luo等，2023）将多步扩散重新建模为概率流ODE的consistency mapping——直接从噪声映射到解，而非逐步迭代。LCM使用consistency distillation训练一个student模型，使其在任何时间步的预测都直接对应ODE的最终解，从而将推理步数从50步降至2-4步。在SD 2.1的权重上进行LCM蒸馏后，模型的FID与原版50步DDIM采样相当，但计算量仅为其4-8%。LCM的工程意义在于：在边缘设备（手机、笔记本）上实现秒级文生图成为可能。SDXL Turbo（Sauer等，2023）则采用对抗蒸馏（adversarial distillation）策略，在4步内生成高质量1024×1024图像，进一步拉近了扩散模型与GAN的推理速度差距。

---

## 4 文生视频应用

### 4.1 视频生成的扩散模型架构

文生视频的主流方法继承自文生图的扩散模型架构，核心思路是在预训练的SD基础上增加时间维度。U-Net的2D卷积层被扩展为伪3D或2+1D卷积——即在原有的空间卷积（3×3）之后插入时间维度的1D卷积（1×3），使模型能够同时建模帧内的空间内容和帧间的时域变化。

具体而言，一个视频由 $T$ 帧构成 $X \in \mathbb{R}^{T \times H \times W \times 3}$。VAE编码器逐帧编码为 $Z \in \mathbb{R}^{T \times h \times w \times c}$。U-Net的每一层在原有的空间self-attention（关注同一帧内不同位置）和时间self-attention（关注不同帧的同一位置）之间交替。时间注意力（Temporal Attention）的Q/K/V来自所有帧的同一空间位置token计算，捕捉帧与帧之间的运动一致性。

Video LDM（Blattmann等，2023）提出了明确的预训练策略：首先冻结图像扩散模型的权重，只在新增的时间层上训练，从而保留图像模型的空间质量和丰富的文图对应知识。这一策略被称为"inflated fine-tuning"——将2D权重膨胀为3D后固定大部分参数，仅训练新增的时间参数，数据效率极高。

### 4.2 VideoCrafter与AnimateDiff的运动模块方案

VideoCrafter（Chen等，2023）和AnimateDiff（Guo等，2023）代表了同一技术路线的不同实现——在冻结的SD模型上附加轻量级运动模块（Motion Module）。

AnimateDiff的运动模块安装在U-Net的每个SpatialTransformer block中的self-attention层之后。它维护一个独立的时序注意力层，其Q/K/V来自同一帧特征图跨帧的展开。训练时，仅更新Motion Module的参数，SD的所有原始权重（包括text cross-attention）保持冻结。好处有三：（1）保留SD的生成质量；（2）可以利用SD海量的文图预训练对；（3）Motion Module体积小（仅占SD总参数量的~5%），训练成本可控。

VideoCrafter在此基础上引入了两个额外组件：Video VAE（在时序上引入时间压缩，将$T$帧压缩为$T'<T$帧以降低算力消耗）和条件视频生成（将首帧或末帧作为条件输入，实现视频补全）。这两个方案在VBench评测（Huang等，2023）上的关键差距体现在动作幅度和一致性权衡上：AnimateDiff擅长细粒度小动作的一致生成（如人物嘴唇微动），VideoCrafter在大动态场景（如跑步、跳舞）上表现更好。

### 4.3 视频生成的质量与可控性挑战

文生视频当前面临的核心问题集中在两个维度：

**Token不稳定**：由于视频生成的高维采样空间（$T \times H \times W \times 3$ 远大于图像的 $H \times W \times 3$），扩散模型在长序列采样过程中容易出现累积误差。表现为生成帧逐渐退化为高噪声或低画质模式（mode collapse in temporal dimension）。现有的缓解方案包括滑块式自回归生成（每生成一帧后以该帧作为下一帧的条件，如CogVideo的帧预测式级联，Hong等，2022）和使用时间越界减少（temporal regularization），引入相邻帧的感知损失约束帧间一致性。

**长视频一致性**：在生成超过32帧（约2秒@16fps以上的时长）时，场景、主体和背景的一致性难以维持。主体可能在第16帧变换形态颜色、背景可能出现闪烁或几何错位。根本原因是现有模型缺乏对物理世界的持久性理解——模型不知道"物体应该保持存在并连续变化"。Sora OpenAI（2024）首次展示了通过大规模视频数据的Transformer化训练可以部分解决这一问题：其在超大规模视频数据集（估计包含数百万小时的多样化视频）上以patch embedding方式训练DiT（Diffusion Transformer, Peebles & Xie，2023），用Transformer替代U-Net作为扩散backbone。由于Transformer的self-attention具有全局感受野，可以同时建模长期帧间依赖。Sora生成的60秒视频在场景一致性上显著优于之前的U-Net方案。

---

## 5 多模态对话系统

### 5.1 图像到文本对话流程

多模态对话系统的标准架构由三部分组成：视觉编码器（Visual Encoder）→ 适配器（Adapter/Connector）→ 大语言模型（LLM）。

**视觉编码器**通常使用CLIP ViT-L/14（Radford等，2021）——将图像划分为14×14的patch（对于224×224输入是256个patch），每个patch通过线性投影和位置编码后经过ViT编码为视觉token序列。ViT-L/14输出256个768维的visual embedding，构成了LLM可以使用的"视觉词序列"。

**适配器**解决了视觉token和文本token的embedding空间维度不一致的问题。两个主流方案：（1）Linear Projection——一个简单的可学习线性层，将视觉embedding从ViT的维度映射到LLM的embedding维度（768→4096），然后直接将映射后的视觉token序列插入到文本token序列之前。LLaVA（Liu等，2023）证明了这种简单投影在图文指令数据上经过两阶段训练后可以取得良好的多模态对话效果。（2）Q-Former（Li等，2023）——一个轻量级Transformer，使用一组可学习的query token通过cross-attention从视觉特征中"提炼"出关键信息，输出固定长度（通常32或64个）的视觉token。BLIP-2（Li等，2023）首先验证了Q-Former在大规模多模态预训练中的有效性，后续的InstructBLIP（Dai等，2023）进一步在指令微调中保持了Q-Former架构。

**LLM**基于视觉token+用户文本prompt生成的端到端应答。以LLaVA的推理流程为例：图像经过CLIP ViT-L产生视觉embedding → Linear Projection适配到LLaMA embedding space → 与用户文本的token embedding拼接形成完整的输入序列 → Vicuna/LLaMA以自回归方式生成回答。两种模态在前向时无缝混合——LLM并不知道哪些token来自文本哪些来自视觉，统一在自回归框架内处理。Liu等（2023）的实验表明，LLaVA在ScienceQA（Saikh等，2022）上的准确率达到90.92%，远超先前的最佳方法（~79%），验证了端到端图文指令微调的有效性。

### 5.2 GPT-4V的多模态视觉理解分析层次

GPT-4V（OpenAI，2023）代表了闭源多模态系统的最高水平。其分析输出展现了多层次的视觉理解能力，可归纳为四个递进层次：

**层次一——物体识别**：识别图中的基本物体、人物和场景类型。这是最基础的感知层面，对应传统图像分类和检测任务。例如"图中有三只猫坐在一张木桌上"。

**层次二——属性理解**：识别物体的颜色、纹理、大小、材质等属性。"是灰色短毛猫"——这表明模型不仅知道"猫"，还感知到了毛色和毛长的视觉特征。

**层次三——空间关系推理**：理解物体之间的相对位置、遮挡关系、尺寸对比和语义连接。"右边的猫沙发左边靠近窗户的位置"——这需要模型对二维图像中的三维空间关系进行推理，超过了单纯的目标检测（只能输出bbox），涉及到布局理解。

**层次四——情境与情感判断**：结合常识和上下文推断氛围、情绪和隐含信息。"光线温暖柔和，环境看起来很舒适"——模型在没有明确标注的情况下，从照明色调、阴影柔和度得出对氛围的语义化描述，体现了视觉与常识推理的结合。

这种多层次能力使得GPT-4V能够处理复杂的视觉文档理解任务。以图表解读为例：输入一张复杂的营收趋势折线图，GPT-4V可以同时完成图表元素的定位（坐标轴标签、图例位置）、数据趋势的量化描述（"2024年Q3达到峰值120万，随后下降"）和表格图片中精确数值的捕捉。Wu等（2024）对GPT-4V的图表理解进行系统评测，发现其对于清晰图表（无重叠元素、标准字体）的数值提取准确率达到95%以上，但在密集散点图和3D雷达图上仍存在10-15%的数值误差。在UI截图理解方面，GPT-4V可以识别界面布局结构（导航栏、内容区域、CTA按钮）、标注交互元素（输入框、下拉菜单）并给出等价的功能描述，为视觉无障碍（accessibility）自动化测试提供了新的技术路径。

---

## 6 多模态模型的微调与定制化

### 6.1 图像生成的Low-Rank Adaptation（LoRA）

图像生成模型的微调面临参数量巨大（SD XL约2.6B参数）和过拟合风险（用户通常只有数张参考图像）两大挑战。LoRA（Low-Rank Adaptation，Hu等，2021）通过在预训练权重矩阵旁并联一个低秩分解 $W + \Delta W = W + BA$，其中 $B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}$，$r \ll \min(d,k)$，仅在训练中更新 $A$ 和 $B$，冻结原始权重 $W$。

在SD的实践应用中，LoRA通常插入在Cross-Attention层的Q、K、V、O四个投影矩阵上，rank取自64-128。对于一个2048×1024的注意力投影矩阵，使用rank=128的LoRA其可训练参数量仅为 $2048 \times 128 + 128 \times 1024 \approx 393K$，仅占原矩阵参数量的~19%。训练数据集通常为10-15张目标风格/物体的图像，配以统一的触发词和描述模板（如"a photo of sks cat in [style]"），在~500步内即可收敛。

LoRA的关键优势在于模块化和组合性——训练得到的低秩权重文件仅2-10MB，可以在推理时以 $w_{LoRA}$ 的权重与原模型合并，且可以同时加载多个LoRA进行风格/概念叠加。与之对标的Textual Inversion（Gal等，2022）则是一种更轻量但表达能力较弱的方法：不更新模型权重，而是学习用几个新的嵌入向量（placeholder token）映射到目标概念。Textual Inversion将特定概念编码为embedding空间中的一个伪token，在每个prompt中插入该伪token以引导模型生成概念。其优点是不改动模型、文件极小（仅数KB），但由于embedding空间表达能力有限，对复杂风格的学习效果显著弱于LoRA。

### 6.2 多模态对话模型的微调

多模态对话模型的微调通常采用两阶段策略：第一阶段在大量图文对数据上训练适配器（Adapter/Projector）以对齐视觉和文本embedding空间；第二阶段在图文指令数据集（multimodal instruction-following data）上端到端微调LLM+适配器，使其学会遵循多模态指令。

LLaVA的训练管线是典型示例。第一阶段在~600K图文对上训练Linear Projection，冻结视觉编码器和LLM，仅优化投影层参数使得visual embedding可以"被LLM理解"。第二阶段使用158K自动生成的图文指令数据（GPT-4生成的围绕图像的多轮问答），端到端微调LLM（Vicuna-13B的LoRA版本）和投影层。这一两阶段范式被证明在数据效率上显著优于端到端从头训练——仅使用约600K图文对和158K指令样本就能训练出具有VQA和多模态对话能力的模型。

**领域特定微调**的代表工作是LLaVA-Med（Li等，2023）。LLaVA-Med在生物医学领域进行了全流程适配：使用PubMed Central的15万篇论文的生物医学图像（胸部X光片、病理切片、细胞显微图等），通过GPT-4生成图像-描述-指令三元组，在LLaVA基础上继续微调。微调同样采用两阶段——第一阶段用生物医学图文对适配投影层，第二阶段用领域指令数据微调LLM。在MedVQA（QA pair from Med-PMC数据集）上的评估显示，LLaVA-Med的准确率比未微调的LLaVA高12个百分点，在开放生成式回答的ROUGE-L上也高出0.08。这一结果表明，领域多模态微调是弥合"通用视觉理解"与"专业领域图文推理"之间差距的关键手段。

---

## 7 多模态应用的评估与2026年趋势

### 7.1 多模态评估的核心维度

多模态系统的评估相比纯文本面临更多维度，核心指标包括：

**视觉忠实度（visual faithfulness）**——回答是否与图像内容一致。这等价于评估多模态幻觉（hallucination）率。常见的benchmark包括POPE（Li等，2023）——设计三类型问题（存在性、数量、位置）并构造正负样本对，通过模型回答与ground-truth的一致性来评估幻觉程度。典型的多模态幻觉表现为"图像中不存在苹果但模型说存在"，本质上是文本先验（language prior）压过了视觉信号。

**视觉利用度（visual grounding utilization）**——模型是否正确利用了图像中的关键信息来回答问题。对比测试方法：将输入图像替换为空白图或随机噪声，如果模型回答不变，说明视觉利用度低，模型主要依赖文本先验。MME（Fu等，2023）和MMBench（Liu等，2023）提供了更综合的评估体系，涵盖感知能力（物体计数、OCR、颜色识别）和认知能力（常识推理、数学计算、图表解读）两个大类。

在视频生成评估方面，VBench（Huang等，2023）提出了16个维度的自动评估框架，包括主体一致性（subject consistency）、背景保真度（background consistency）、动作流畅度（motion smoothness）、时序闪烁（temporal flickering）、文本-视频对齐（text-video alignment）和视觉质量（aesthetic quality）。FVD （Fréchet Video Distance，Unterthiner等，2018）是视频生成领域最广泛使用的分布距离指标，计算生成视频集和真实视频集在I3D特征空间中的Fréchet距离。但FVD对细粒度运动质量的灵敏度有限。

### 7.2 Sora：DiT架构与世界模拟

OpenAI在2024年2月发布的Sora标志着文生视频领域的范式转变——从U-Net扩散到Transformer扩散。Sora的核心技术贡献在于将视频表示为spacetime patches——同ViT（Dosovitskiy等，2020）将图像分割为patch嵌入的做法类似，Sora将视频（包括空间维度和时间维度）统一分割为"时空块"（spacetime patches），每个patch作为Transformer的一个输入token。这一表示使得模型可以同时处理不同持续时间、分辨率和宽高比的视频和图像（单帧图像可视为一个时间维度为1的视频）。

在架构层面，Sora使用DiT（Diffusion Transformer，Peebles & Xie，2023）作为扩散backbone——用Transformer替代U-Net，在patch token上执行扩散去噪。DiT的self-attention在所有patch token（空间+时间维度）之间建立全连接，使得模型可以捕获长期时空依赖，而U-Net的卷积和局部注意力则受限于感受野。

OpenAI的技术报告指出Sora展现出"世界模拟器"的涌现能力：三维一致性（相机运动时物体的几何结构保持一致）、对象持久性（物体被遮挡后再次出现时形态不变）和简单的物理交互（风吹动物体的合理响应）。这些能力并非显式编程，而是在超大规模多源视频数据上训练后涌现的。Sora的成功验证了scaling law在视频生成领域的扩展适用性——更大规模的数据、更宽的模型和更多的计算量直接转化为更强的视频生成物理真实性。但Sora目前的技术报告中缺乏系统性的样本质量评测和数据规模的透明披露，使得复现和独立验证仍存在障碍。

### 7.3 2026年趋势：从跨模态理解到原生多模态推理

2025-2026年多模态基础模型的最大趋势是从"跨模态理解"走向"原生多模态"（native multimodality）。GPT-4o（OpenAI，2024）和Gemini 1.5 Pro（Google DeepMind，2024）已展现出处理文本+图像+音频的统一架构能力，其核心变化在于不再使用独立的编码器为每种模态提取embedding后拼接，而是真正在一个统一的embedding空间中处理所有模态。Gemini 1.5 Pro的百万token上下文窗口能够直接"看到"数小时的视频帧、整本PDF或整个代码仓库，使得多模态能力从"看图说话"进化为"多模态推理"。

具体表现为：系统不再局限于回答"图中有什么"，而是能够将图像/视频中的视觉信息与文档文本共同作为推理的前提条件——例如"根据这三张图表和这份财报，分析公司2026年的收入构成变化"。这要求模型不仅能够提取视觉信息，更重要的是能够在视觉信号和文本信号之间建立推理桥接，执行因果推断、对比分析和数值聚合。

另一个关键趋势是多模态推理（multimodal reasoning）的兴起——不再是简单的感知和描述，而是要求模型在视觉和文本之间执行多跳逻辑推理（multi-hop reasoning）。这需要模型能够从视觉输入中提取精确的数值、空间关系和时序状态，并与文本背景知识结合得出推论。Goyal等（2024）将这一范式总结为"Visual, then Verbal, then Reasoning"三阶段。实现这一目标需要：（1）更高质量的视觉编码使其对数值和结构性信息的保真度提升；（2）更强大的桥接机制使视觉信息在LLM的各层前向过程中持续传递而非仅在embedding层注入一次；（3）在推理数据上训练模型学会"先看、后想、再说"的思维链（Chain-of-Thought）范式在多模态领域的蒸馏落地。

最后，多模态评估和基准测试也在2026年进入了一个新阶段——从构建静态benchmark到建立动态对抗性评测协议。随着模型能力的提升，传统的固定Q&A benchmark不断饱和。VLM Arena类的众包盲评（Elo rating）和对抗性VQA生成（让模型互相提问评测）正在成为更具区分度的评估方案。多模态安全的评估也成为关键——图像中嵌入的偏见、有害信息和被模型解读后输出的安全风险，需要系统化的红队测试（red teaming）框架。

---

**数据年份与声明**：本文资料覆盖截至2025年12月公开的学术论文、技术报告与产品发布信息。引用的基准测试数据和模型版本对应于截至2025年H1的版本。部分闭源系统（如GPT-4V、Sora）的技术细节基于公开报告和逆向分析，可能与实际实现存在差异。所有拟合数值标注了具体来源数据集，能力对比以对应论文的原始声称benchmark为准。

---

## 参考文献

[^1]: Alayrac, J. B., Donahue, J., Luc, P., et al. (2022). Flamingo: a Visual Language Model for Few-Shot Learning. NeurIPS 2022.
[^2]: Baek, Y., Lee, B., Han, D., et al. (2019). Character Region Awareness for Text Detection. CVPR 2019.
[^3]: Bai, J., Bai, S., Chu, Z., et al. (2023). Gemini: A Family of Highly Capable Multimodal Models. arXiv:2312.11805.
[^4]: Blattmann, A., Dockhorn, T., Kulal, S., et al. (2023). Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models. CVPR 2023.
[^5]: Chen, H., Xia, M., He, Y., et al. (2023). VideoCrafter1: Open Diffusion Models for High-Quality Video Generation. arXiv:2310.19512.
[^6]: Dai, W., Li, J., Li, D., et al. (2023). InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning. NeurIPS 2023.
[^7]: Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021.
[^8]: Fu, C., Chen, P., Shen, Y., et al. (2023). MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. arXiv:2306.13394.
[^9]: Gadre, S. Y., Ilharco, G., Fang, A., et al. (2023). DataComp: In search of the next generation of multimodal datasets. NeurIPS 2023.
[^10]: Gal, R., Alaluf, Y., Atzmon, Y., et al. (2022). An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion. ICLR 2023.
[^11]: Guo, Y., Yang, C., Rao, A., et al. (2023). AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. ICLR 2024.
[^12]: Ho, J., & Salimans, T. (2022). Classifier-Free Diffusion Guidance. NeurIPS 2021 Workshop.
[^13]: Hong, W., Ding, M., Zheng, W., et al. (2022). CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers. ICLR 2023.
[^14]: Hu, E. J., Shen, Y., Wallis, P., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
[^15]: Huang, Z., He, Y., Yu, J., et al. (2023). VBench: Comprehensive Benchmark Suite for Video Generative Models. arXiv:2311.17982.
[^16]: Kim, W., Son, B., & Kim, I. (2021). VILT: Vision-and-Language Transformer Without Convolution or Region Supervision. ICML 2021.
[^17]: Kim, G., Hong, T., Yim, M., et al. (2022). OCR-Free Document Understanding Transformer. ECCV 2022.
[^18]: Li, J., Li, D., Savarese, S., & Hoi, S. (2023). BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML 2023.
[^19]: Li, Y., Du, Y., Zhou, K., et al. (2023). Evaluating Object Hallucination in Large Vision-Language Models. EMNLP 2023.
[^20]: Liao, M., Shi, B., & Bai, X. (2018). TextBoxes++: A Single-Shot Oriented Scene Text Detector. IEEE TIP, 27(8), 3676-3690.
[^21]: Liao, M., Wan, Z., Yao, C., et al. (2020). Real-Time Scene Text Detection with Differentiable Binarization. AAAI 2020.
[^22]: Liu, H., Li, C., Wu, Q., & Lee, Y. J. (2023). Visual Instruction Tuning. NeurIPS 2023.
[^23]: Liu, Y., Duan, H., Zhang, Y., et al. (2023). MMBench: Is Your Multi-modal Model an All-around Player? arXiv:2307.06281.
[^24]: Luo, S., Tan, Y., Patil, S., et al. (2023). LCM-LoRA: A Universal Stable-Diffusion Acceleration Module. arXiv:2311.05556.
[^25]: OpenAI. (2023). GPT-4V(ision) System Card. OpenAI Technical Report.
[^26]: Peebles, W., & Xie, S. (2023). Scalable Diffusion Models with Transformers. ICCV 2023.
[^27]: Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.
[^28]: Rombach, R., Blattmann, A., Lorenz, D., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.
[^29]: Sauer, A., Lorenz, D., Blattmann, A., & Rombach, R. (2023). Adversarial Diffusion Distillation. arXiv:2311.17042.
[^30]: Schuhmann, C., Beaumont, R., Vencu, R., et al. (2022). LAION-5B: An open large-scale dataset for training next generation image-text models. NeurIPS 2022 Datasets & Benchmarks.
[^31]: Shi, B., Bai, X., & Yao, C. (2017). An End-to-End Trainable Neural Network for Image-based Sequence Recognition and Its Application to Scene Text Recognition. IEEE TPAMI, 39(11), 2298-2304.
[^32]: Shi, B., Yang, M., Wang, X., et al. (2018). ASTER: An Attentional Scene Text Recognizer with Flexible Rectification. IEEE TPAMI, 41(9), 2035-2048.
[^33]: Unterthiner, T., van Steenkiste, S., Kurach, K., et al. (2018). Towards Accurate Generative Models of Video: A New Metric & Challenges. arXiv:1812.01717.
[^34]: Wu, C., Yin, S., Qi, W., et al. (2024). Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models. arXiv:2303.04671.
[^35]: Chen, L., et al. (2025). Industrial VLM: End-to-End Visual Language Model for Gauge Reading in Smart Manufacturing. IEEE TII.
