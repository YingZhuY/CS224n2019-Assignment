# CS224n系列 Notes 
[课程链接](https://web.stanford.edu/class/cs224n/) &emsp; [中文翻译](https://zhuanlan.zhihu.com/p/31977759) &emsp;[MyAssignment]()

------
概念、语言符号 --encoding--> 想法或事物

etc. Machine Translation、Semantic Analysis、Coreference、Question Answering

------

One-hot

单词表示之间的联系不能显示出来 --> 考虑低维子空间

$$(w^{hotel})^Tw^{motel}=(w^{hotel})^Tw^{cat}=0$$

------

SVD Based Method

遍历很大数据集 --统计词--> 共现矩阵X --SVD分解--> $USV^T$ --行-->词向量

------
Word-Document Matrix

相关的 word(i) 会在同一个 document(j) 中一起出现 --大量统计--> $R^{|V|*M}$ -> 和文档规模M成正比 --> Window based Co-occurrence Matrix

------
SVD 


根据奇异值（矩阵对角线上元素），按期望的捕获方差百分比截断
$$\frac {\Sigma_{i=1}^{k} \sigma_i} {\Sigma_{i=1}^{|V|} \sigma_i}$$

<div align="center">
<img src="./SVD1.png" width="80%" height="80%" />
<img src="./SVD2.png" height="80%" width="80%"/>
</div>

>加入新词矩阵维度会改变，矩阵稀疏性，计算复杂度O(n^2),词频极剧不平衡

>忽略'the''has'等`功能词`，使用`ramp window`-基于文档中词之间距离给共现计数加权值
------

Word2vec

包含两个算法：continuous bag-of-words (CBOW) 和 skip-gram，两个训练方法：negative sampling （抽取负样本）和 hierarchical softmax（树结构）

`CBOW` 对每个单词学习两个向量，v（input vector）当单词是上下文，u（output vector）当单词是中心词，我们需要学习$V$和$U$两个矩阵（通过目标函数交叉熵）

* $w_i$ 词汇表中的单词i
* $V∈R^{n*|V|}$ 输入词矩阵
* $v_i$ V的第i列，单词$w_i$的输入向量表示
* $U∈R^{|V|*n}$ 输出词矩阵
* $u_i$ U的第i行，单词$w_i$的输出向量表示
* $H(\hat{y},y)=-\Sigma_{j=1}^{|V|}y_ilog(\hat{y})$ -> $=y_jlog(\hat{y})$ 损失函数（交叉熵）y是one-hot向量，式子可简化
* minimize $J=-logP(w_c|w_{c-m},...,w_{c-1},w_{c+1},...,w_{c+m})$
  
  $=-logP(u_c|\hat{v})=-log\frac{exp(u_c^T \hat{v})}{\Sigma_{j=1}^{|V|}exp(u_j^T \hat{v})}$
  
  $=-u_c^T \hat{v}+log \Sigma_{j=1}^{|V|} exp(u_j^T \hat{v})$ 优化目标函数，$\hat{v}$为经过$V$后的向量平均
* $U_{new} \leftarrow U_{old} \leftarrow \alpha \nabla_UJ$ &emsp;$V_{new} \leftarrow V_{old} \leftarrow \alpha \nabla_VJ$ 计算梯度-SGD更新参数-词向量会发生变化

`Skip-Gram Model` 类似CBOW

* minimize $J=-logP(w_{c-m},...,w_{c-1},w_{c+1},...,w_{c+m}|w_c)$
  
  $=-log\prod_{j=0,j \not=m}^{2m}P(w_{c-m+j}|w_c)$
  
  $=-log \prod_{j=0,j \not=m}^{2m}P(u_{c-m+j}|v_c)$
  
  $=-log \prod_{j=0,j \not=m}^{2m}\frac{exp(u_{c-m+j}^{T}v_c)}{\Sigma_{k=1}^{|V|}exp(u_k^Tv_c)}$
  
  $=-\Sigma_{j=0,j\not=m}^{2m}u_{c-m+j}^Tv_c +2m log\Sigma_{k=1}^{|V|}exp(u_k^Tv_c)$ 目标函数
* note that $J=-\Sigma_{j=0,j \not=m}^{2m}logP(u_{c-m+j}|v_c) =\Sigma_{j=0,j \not=m}^{2m}H(\hat{y},y_{c-m+j})$ 交叉熵

`Negative Sampling` 对$|V|$的求和计算量是非常大的（时间复杂度），考虑在每个训练的时间步不去遍历整个词汇表，而是仅仅抽取一些负样例。$P_n(w)$抽样概率和词频的顺序相匹配。

* $P(D=1|w,c)$表示$(w,c)$是来自语料库，$P(D=0|w,c)$表示不是
* $P(D=1|w,c,\theta)=\sigma(v_c^Tv_w)=\frac{1}{1+e^{(-v_c^Tv_w)}}$
* 如果中心词和上下文确实在语料库中，就最大化概率$P(D=1|w,c)$，如果确实不在，就最大化$P(D=0|w,c)$，极大似然估计$\theta$
* $\theta=arg max_\theta \prod_{(w,c)\in D }P(D=1|w,c,\theta) \prod_{(w,c)\in \widetilde{D} }P(D=0|w,c,\theta)$
   
   $=arg max_\theta \prod_{(w,c)\in D }P(D=1|w,c,\theta) \prod_{(w,c)\in \widetilde{D} }(1-P(D=1|w,c,\theta))$ 
   
   $=arg max_\theta \Sigma_{(w,c)\in D }logP(D=1|w,c,\theta)+ \Sigma_{(w,c)\in \widetilde{D} }log(1-P(D=1|w,c,\theta))$
   
   $=arg max_\theta \Sigma_{(w,c)\in D }log\frac{1}{1+exp(-u_w^Tv_c)}+ \Sigma_{(w,c)\in \widetilde{D} }log(1-\frac{1}{1+exp(-u_w^Tv_c)})$ 
   
   $=arg max_\theta \Sigma_{(w,c)\in D }log\frac{1}{1+exp(-u_w^Tv_c)}+ \Sigma_{(w,c)\in \widetilde{D} }log(\frac{1}{1+exp(u_w^Tv_c)})$
* 则$J=-\Sigma_{(w,c)\in D }log\frac{1}{1+exp(-u_w^Tv_c)}- \Sigma_{(w,c)\in \widetilde{D} }log(\frac{1}{1+exp(u_w^Tv_c)})$
* Skip-gram的新目标函数 $-log\sigma(u_{c-m+j}^T*v_c)-\Sigma_{k=1}^Klog\sigma(-\widetilde{u}_k^T*v_c)$
* CBOW的新目标函数 $-log\sigma(u_c^T*\hat{v})-\Sigma_{k=1}^K log\sigma(-\widetilde{u}_k^T*\hat{v})$
* $\widetilde{u}_k$抽样 指数为3/4的Unigram模型，可见，$bombastic$现在被抽样的概率是之前的三倍，而$is$才高一点点
  $is:0.9^{3/4}=0.92$
  $Constitution:0.09^{3/4}=0.16$
  $bombastic:0.01^{3/4}=0.032$

------

Language Models(Unigrams,Bigrams,etc.)

$P(w_1,w_2,...,w_n)$ 语言模型给有效的句子高概率

$P(w_1,w_2,...,w_n)=\prod_{i=1}^nP(w_i)$ `Unigram model` 完全独立

$P(w_1,w_2,...,w_n)=\prod_{i=1}^nP(w_i|w_{i-1})$ `Bigram model` 成对概率 （共现窗口为1）

------
Hierarchical Softmax

Hierarchical Softmax对低频词往往表现得更好，负采样对高频词和较低维度向量表现更好。Hierarchical Softmax使用一个二叉树（哈夫曼树）来表示词表中的所有词（叶节点），复杂度$O(log(|V|))$

* $p(w|w_i)=\prod_{j=1}^{L(w)-1} \sigma([n(w,j+1)=ch(n(w,j))]*v_{n(w,j)}^T v_{w_i})$ 随机漫步概率计算
* $[x]=\begin{cases}
    1 & \text{True}\\
    -1 & \text{otherwise}
\end{cases}$ 区分路径往左往右，保证归一化
* 计算点积来比较输入向量 $v_{w_i}$对每个内部节点向量$v_{n(w,j)}^T$的相似度，最小化$-logP(w|w_i)$，更新路径上节点向量
  
  $P(w_2|w_i)=p(n(w_2,1),left)*p(n(w_2,2),left)*p(n(w_2,3),right)$

  $=\sigma(v_{n(w_2,1)}^T v_{w_i})*\sigma(v_{n(w_2,2)}^T v_{w_i})*\sigma(-v_{n(w_2,3)}^T v_{w_i})$


------

------
Global Vectors for Word Representation（GloVe）
GloVe由一个加权最小二乘模型组成，在全局词-词共现统计上训练，仅对单词共现矩阵中非零元素训练 --> 类比任务

* 如何量化评估词向量的质量？
  * `Intrinsic Evaluation` 生成的词向量在特定中间子任务（如词类比 a:b :: c:?）上的评估，简单且计算速度快
  * 类比任务计算词向量的最大余弦相似度$d=argmax_i \frac{(x_b-x_a+x_c)^T x_i}{|x_b-x_a+x_c|}$
  * 城市类比任务考虑问题：具有相同名称的不同城市（答案不唯一），在不同时间点的首都不同的国家（语料库过时）
  * 使用内在评估系统（如类比系统）来调整词嵌入中的超参数。模型的表现高度依赖使用词向量的模型；语料库更大模型的表现更好；对于极高或者极低维度的词向量，模型的表现较差（过高的维度可能捕获语料库中无助于泛化的噪声-即所谓的高方差问题）。
  * 评估词向量质量（人为评估的数据集）：人给出的相似度评分 vs 对应的余弦相似度
  * `Extrinsic Evaluation` 对一组在实际任务（如QA）中生成的词向量的评估，复杂且计算速度慢。通常，优化表现不佳的外部评估系统我们难以确定哪个特定子系统存在错误，还需要进一步的内部评估
  * 很多NLP的外部任务都可以表述为分类任务，情感分类（正面、负面or中性）、命名实体识别（给定一个上下文和一个中心词，我们想把中心词分类为许多类别之一）$\{x^{(i)},y^{(i)}\}_1^N$ --> 分类（逻辑分类/SVM/非线性分类模型-神经网络） --> 训练权重（梯度下降/L-BFGS/牛顿法）
* 用不同的词向量来捕获同一个单词在不同场景下的不同用法。固定大小窗口 --上下文词向量的加权平均--> 上下文 --k-means--> 上下文聚类 --> 每个出现的单词都重新标签为其相关联的类，对各类训练对应的词向量。
* 在训练集很小(词覆盖率小)的情况下，词向量不应该重新训练--> 词在词空间中移动 --> performance $\downarrow$
* syntactic test -- semantic test
------

`sigmoid` 二元逻辑回归单元 $a=\frac{1}{1+exp(-(w^Tx+b))}$

`Maximum Margin Objective Function` 保证对“真”标签数据计算的得分要比“假”标签数据高，误差为（$s_{False}-s_{True}$）否则为0。即$minimize J=max(s_{False}-s_{True},0)$

反向传播，误差共享/分配，偏置更新，梯度检验，L2正则化（在损失函数J上增加一个正则项，$\lambda$）($\lambda$太大会令很多权值都接近于0，则模型就不能在训练集上学到有意义的东西，经常在训练、验证和测试集上的表现都非常差；$\lambda$太小会让模型仍旧出现过拟合的现象)；`Dropout` 本质上作的是一次以指数形式训练许多较小的网络，并对其预测进行平均

激活函数 sigmoid、tanh、hard tanh、soft sign（类tanh，饱和地较慢）、relu（CV better）、leaky relu、

$hardtanh(z)=\begin{cases}
-1 & \text{z<1} \\
z & \text{-1≤ z≤ 1}\\
1 & \text{z>1}
\end{cases}$

`Data Preprocessing` Mean Subtraction、Normalization、Whitening

`Parameter Initialization` 通常初始化为分布在0附近的很小的随机数，或者按照输入（fan-in）输出（fan-out）单元数初始化：

$W\rightarrow U[-\sqrt{\frac{6}{n^{(l)} +n^{(l+1)}}},\sqrt{\frac{6}{n^{(l)} +n^{(l+1)}}}]$

`learning rate` 太大可能损失函数难以收敛，太小可能不会在合理的时间内收敛或者困在局部最优点。annealing 指数衰减 $\alpha(t)=\alpha_0e^{-kt}$ 允许学习率随着时间减少 $\alpha(t)=\frac{\alpha_0\tau}{max(t,\tau)}$ ；$\alpha_0$ -起始学习率 $\tau$ -可调参数

`Momentum Updates`动量方法

`Adaptive Optimization Methods` AdaGrad（之前很少被更新的参数就用比现在更大的学习率更新）、RMSProp、Adam

------
SST (Stanford Sentiment Treebank) 数据集

------

------

Dependency Parsing Dependency structure
* Dependency Parsing
  
  * constituency=phrase structure grammar=context-free grammars(CFGs)
  * words are given a category(part of speech=pos) 词性标注，词根据种类规则组成短语，短语递归地组成更大的短语
  * 依存关系描述为从head（被修饰的主题）指向dependent（修饰语）的箭头，每个单词都依存于唯一的一个节点
  * 连接介词短语会出现歧义现象，形容词修饰的歧义，动词短语歧义，长句子中各部分修饰的究竟是谁，特别是新闻上
  * 通过dependency paths可以发现语义关系
  * 假root根节点
  * annotated data 标注的数据
  * discriminative classifier 判别分类器
  * 准确度评估方式 $Acc=\frac{ n_{correct deps}}{ n_{all deps}}$

* Language Models
  * 语言模型 $\rightarrow$ 预测下一个单词的概率分布
  * n-gram n个连续的单词块，统计不同n-gram的频率
    * unigrams:"the","students","opened","their"
    * bigrams:"the students","students opened","opened their"
    * trigrams:"the students opened","students opened their"
    * 4-grams:"the students opened their"
  * get probabilities $\rightarrow$ large corpus of text, count!
  * 具体例子下，context很重要；补空的n-gram组合从未出现过，概率为0$\rightarrow$sparsity problem 稀疏性问题，设置一个小的 $\Delta$，稀疏词的小概率存在$\rightarrow$smoothing；增加 n-gram中的 n可以使自动生成的文本（知道什么使用用逗号隔开）语义更连贯，但数据稀疏性更严重，模型更大
  * 4-gram，分母概率$P(x_i,x_{i+1},x_{i+2},x_{i+3})$接近0时，考虑往下降的模型，3-gram即$P(x_{i+1},x_{i+2},x_{i+3})$
  $$P(w|students~opened~their)=\frac{count(students~opened~their~w)}{count(students~opened~their)}$$
  * words/one-hot vectors $\rightarrow$ word embeddings $\rightarrow$ hidden states $\rightarrow$ output distribution
  * 使用RNN（Recurrent Neural Networks）的优点：能处理任意长度的输入；步骤t可以使用来自许多步骤的信息；对于较长的输入，模型大小不会增加，模型的大小是固定的，会重复应用相同的权重；<br />缺点：计算很慢，（相当长的序列）无法并行计算；对处理之前出现的信息表现并不好
  
  <div align="center">
    <img src="./RNNsample.png" width="60%" height="60%" />
  </div>

  * You can train a RNN-LM on any kind of text, then generate text in that style. (Obama speeches，一个持续的背景，Harry Potter，recipes，无法记住整体发生了什么，paint color name油漆的颜色名，字符级语言模型)
  * 评估标准，perplexity，lower perplexity is better
  
    $\begin{aligned}
        perplexity&=\prod_{t=1}^T(\frac{1}{P_{LM}(x^{t+1}|x^{t},...,x^{1}) })^{1/T} \\
        &=\prod_{t=1}^T(\frac{1}{\hat{y_{x_{t+1}}^{(t)}}})^{1/T} \\
        &=exp(\frac{1}{T}\sum_{t=1}^T-log \hat{y_{x_{t+1}}^{(t)}} ) \\
        &=exp(J(\theta))
    \end{aligned}$
    
  * a benchmark task，帮助衡量我们理解语言的进度，grammar,syntax,logic and reasoning,real-world knowledge；subcomponent of many NLP tasks，产生文本估计文本概率，手机打字，使用更少的动作进行通信，google，语音识别，speech recognition，handwriting recognition,authorship identification，作者的写作风格，machine translation,summarization,dialogue
  * tagging task part-of-speech tagging 词性标注 and named entity recognition 命名实体识别 
  * sentence classification 句子分类，句子标签，句子向量instead所有词向量
  * a general purpose encoder module 通用编码器模块 $\rightarrow$question answering SQuAD challenge
  
  <div align="center">
    <img src="./RNN-QA.png" width="80%" height="80%" />
  </div>

  * vanilla RNN (simple) GRU LSTM multi-layer RNNs (complex)

* OOV 未知单词 out of vocabulary 文本中的OOV率，在实验中加入伪字 `<UNK>` 对这些潜在未知单词进行建模 

------

Vanishing gradient problem $\rightarrow$ LSTM GRU

* Syntactic recency 语法就近性 $\rightarrow$ dependency parse
Sequential recency 顺序就近性
* cause bad updates $\rightarrow$ 从某个点重新训练
* solutions: Gradient clipping $\rightarrow$ threshold 陡崖 overfitting?
$$if( ||\hat{g}|| \ge threshold ) then \\ \hat{g} \leftarrow \frac{threshold}{||\hat{g}||}\hat{g}$$
截断模型效果图（虚线，对比实线）

<div align="center">
  <img src="./gradient-clipping.png" width="50%" height="50%" />
</div>

skip-connections "ResNet"Residual connections，直接跳过部分网络 add more direct connections,Highway connections(gate) Dense connections "DenseNet" 
* Bidirectional RNNs 
* Multi-layer RNNs 第一层RNN的输出作为第二层RNN的输入，2 to 4 layers is best for the encoder RNN, 4 layers is best for the decoder RNN, skip-connections/dense-connections are needed to train deeper RNNs(e.g. 8 layers) 不会太深，因为它不能并行计算。Transformer-based networks(e.g. BERT) can be up to 24 layers/12 layers $\rightarrow$ have a lot of skipping-like connections.
  
------

Machine Translation (MT) $\rightarrow$ sequence-to-sequence $\rightarrow$ attention
* statistical Machine Translation $argmax_yP(x|y)P(y)$ 基于概率模型
  $P(y)$英语 平行语料库（多种语言）对齐 $\rightarrow$ 英语句子和法语句子相互对应 直接对应物，不改变顺序（不能捕获排序差异）
  * 一个翻译模型，告诉我们一个源语言中最有可能被翻译为的句子/短语
  * 一个语言模型，告诉我们给定句子/短语的整体可能性
* alignment can be one to many, a fertile word $\rightarrow$ it needs to have several corresponding English words to translate it. or many to many. 并行数据
* 找到最佳序列 $\rightarrow$ decoding, heuristic search algorithm, discard hypotheses that are too low-probability
  $\rightarrow$ too many separately-designed subcomponents,lots of feature engineering to capture, extra resources, tables of equivalent phrases
* 2014 Neural Machine Translation MT research, sequence to sequence(seq2seq) $\rightarrow$ two RNNs:Encoder RNN produces an encoding of the source sentence,编码固定大小的“上下文向量”; Decoder RNN is a **Language Model** that generates target sentence,conditioned on encoding 使用来自解码器生成的上下文向量来初始化第一层的隐藏层，然后逐词的生成输出，结尾的\<EOS\>标记.在第一个时间步将一层接一层地运行这个三层LSTM，将最后一层的输出经过softmax生成第一个输出单词，然后将这个词传递到下一个时间步的第一层，重复上述流程生成输出单词。

<div align="center">
  <img src="./MTencoder-decoder.png" width="80%" height="80%" />
</div>

* seq2seq not just for MT, Summarization(long text to short text), Dialogue (previous utterances to next utterance), Parsing (input text to output parse as sequence), Code generation (natural language to Python code)
* greedy decoding $\rightarrow$ taking **argmax** on each step of the decoder作为下一步的输入，整个句子的最大状态，不能go back；exhaustive search decding $\rightarrow$ search through the space of all possible French translations $\rightarrow$ Beam search decoding (K) $P \in (0,1) logP< 0$ 选越接近 0 的，进入下一步分支时，概率相乘，即需要和之前的log值相加。产生 END 令牌，先放到一边；或者一旦达到时间步T，我们就结束Beam Search；或者一旦我们收集了至少 N个已完成的假设，我们就结束Beam Search。the top one $\rightarrow$ 仅看最后的分数时，会偏向取 the shorter one
$$\frac{1}{t} \sum_{i=1}^t log P_{LM}(y_i|y_1,...,y_{i-1},x)$$
* NMT advantage: better performance(More fluent, better use of context, better use of phrase similarities), 没有很多需要单独优化的子组件，less human engineering effort, no feature , same method for all language pairs; disadvantages: 更难解释，hard to debug, difficult to control（比如我们想将a word固定翻译为特定的某个词，加入规则或加入一个相当简单的样式，安全问题）
* MT的评估 $\rightarrow$ BLUE Bilingual Evaluation Understudy, 比较同一句话的一个或几个人类书面翻译，计算基于n-gram精度的相似性得分，a penalty for short translations, 缺点：a good translation can get a poor BLUE score because it has low n-gram overlap with the human translation.
* 2014, First seq2seq paper published
  2016, Google Translate switches from SMT to NMT
* 存在的问题: Out-of-vocabulary words；训练集和测试集的域不匹配Domain mismatch(用维基的语料库和人对话)；Maintaining context over longer text(新闻文章或书籍)；Low-resource language pairs都是基于大的平行语料库，部分语种数量少，like somali $\rightarrow$ 平行文本的最佳资源之一是 the Bible,宗教风格，无意义的输入；一些特定的错误，common sense的翻译，保留了训练集中的偏见
* attetion $\leftarrow$ 信息瓶颈 Information bottleneck

<div align="center">
  <img src="./MTattention.png" width="80%" height="80%" />
</div>

  * encoder hidden state $h_1,...,h_N \in R^h$
  * 在每个时间步t $\rightarrow$ decoder hidden state $s_t \in R^h$
  * 计算每步的attention scores $e^t$
  $$e^t=[s_t^Th_1,...,s_t^Th_N] \in R^N$$
  * softmax $\rightarrow$ get the attention distribution $\alpha^t$ (this is a probability distribution and sums to 1)
  $$\alpha^t=softmax(e^t) \in R^N$$
  * 使用$\alpha^t$ $\rightarrow$ get the attention output $a_t$
  $$a_t=\sum_{i=1}^N \alpha_i^t h_i \in R^h$$
  我们称之为$a$的注意力输出向量与encoder隐藏状态大小是相同的
  * 最后将注意力输出$a_t$与decoder隐藏状态连接起来继续。
  $$[a_t;s_t] \in R^{2h}$$
  * attention 增加了可解释性，能有效翻译长句，attention distribution表，可以看到encoder每一步focusing on什么，自动对齐（注意力可以认为是“对齐”）
  
  <div align="center">
    <img src="./attentionvec-compute.png" width="70%" height="70%" />
    <img src="./attention-table.png" width="70%" height="70%" />
  </div>

------

* 语言模型计算特定序列${w_1,...,w_m}$中多个单词的出现概率
$$P(w_1,...,w_m)=\prod_{i=1}^{i=m}P(w_i|w_1,...,w_{i-1})\approx \prod_{i=1}^{i=m}P(w_i|w_{i-n},...,W_{i-1})$$

  机器翻译需要进行单词排序和单词选择
$$p(w_2|w_1)=\frac{count(w_1,w_2)}{count(w_1)}$$
$$p(w_3|w_1,w_2)=\frac{count(w_1,w_2,w_3)}{count(w_1,w_2)}$$
  由公式知，考虑分子，如果$w_1,w_2,w_3$在语料中从未出现过，那么$w_3$的概率就是0$\rightarrow$在每个单词计数后面加上一个很小的$\delta$，进行平滑操作；考虑分母，如果$w_1,w_2$在语料中从未出现过，那么$w_3$的概率将会无法计算$\rightarrow$这里可以只考虑$w_2$，"backoff" 增加 n 会让稀疏问题更加严重，所以一般$n\le5$。

* RNN的损失函数$\rightarrow$交叉熵
$$J^{(t)}(\theta)= \sum_{j=1}^{|V|}y_{t,j}* log(\hat{y_{t,j}})$$
语料库大小为T时，即
$$J=-\frac{1}{T} \sum_{t=1}^{T}J^{(t)}(\theta)=-\frac{1}{T} \sum_{t=1}^{T}\sum_{j=1}^{|V|}y_{t,j}* log(\hat{y_{t,j}})$$

* 在训练 RNN 的编码和解码时使用不同的权值（$W^{(hh)}$矩阵不同），使这两个单元解耦，让两个RNN模块中的每一个进行更精准的预测。
* 给定一个德语词序列ABC，它的英语翻译是XY，在训练时不使用ABC$\rightarrow$XY，而是使用CBA$\rightarrow$XY，这么处理的原因是A更有可能翻译成X。因此对前面讨论的梯度弥散问题，反转输入句子的顺序有助于降低输出短语的错误率。
* LSTM Input：Does $x^{(t)}$ matter? New memory:Compute new memory Forget:Should $c^{(t-1)}$ be forgotten? Output/Exposure: How much $c^{(t)}$ should be exposed?

<div align="center">
  <img src="./WhatLSTM.png" width="80%" height="80%" />
</div>

* Google的多语言模型，构建一个可以翻译任意两种语言的单独系统，实现“零数据翻译”，有日语-英语数据和韩语-英语数据，Google发现多语言NMT系统对这些数据进行训练后实际上可以产生合理的日语-韩语翻译。维持一个**独立于所涉及的实际语言**的输入/输出句子的内部表示。
* 评估机器学习翻译的质量已经成为一个研究领域，提出了许多评估的方法TER，METEOR，MaxSim，SEPIA和RTE-MT，或者比如在查询检索任务中度量了翻译的质量（提取搜索查询的正确网页$\rightarrow$忽略了句法和语法）
* k=4，BLEU分数仅计算大小 $\le$ 4的所有n-grams，以及惩罚
$$p_n=\frac{\# matched\_n-grams}{\# n-grams\_in\_candidate\_translation}$$
$$BLEU=\beta \prod_{i=1}^k p_n^{w_n}$$
$$\beta=e^{min(0,1-\frac{len_{ref}}{len_{MT}})}$$
其中 $len_{ref}$是参考翻译的句子长度，$len_{MT}$是机器翻译的句子长度
$$BLEU=\beta \prod_{i=1}^k p_n^{w_n}$$
* NCE(Noise Contrastive Estimation)想法是通过随机地从负样本中抽取K个单词来近似“softmax”；Hierarchical Softmax能够更有效率地计算（通过计算树上的一个路径）出来“Softmax”（不能在GPU上并行计算），这两个方法只用在训练阶段。
* 将词汇量限制在一个很小的数量上，用一个标签\<UNK\>替换限制后的词汇表外的单词。K=15k,30k,50k（k个最常用）和 K'=10,20（k'个可能目标词）

<div align="center">
  <img src="./candidate-list.png" width="60%" height="60%" />
</div>

* Byte Pair Encoding 基本思想：从字符的词汇开始，并且在数据集中继续扩展词汇与最常用的n-gram对（l,o,w,e,r,n,s,t,i,d,es,est），直到选择所有n-gram对或词汇大小达到某个阈值。
* 基于字符的模型来实现开放词汇表示
* 关于罕见单词的字符的深层LSTM模型
* Azure Piazza conda env create --file local_env.yml

------

* Baselines, Benchmarks, Evaluation, Error analysis, Paper writing








