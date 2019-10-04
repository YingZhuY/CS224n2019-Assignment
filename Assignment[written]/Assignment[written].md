# Assignment2 [written]
相关参数，详见 [notes](https://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf)

* $n$ 向量维度，$|V|$ 词汇表大小
* $w_i$ 词汇表中的单词i
* $V∈R^{|V|*n}$ 输入词矩阵
* $v_i∈R^{1*n}$ $V$的第 $i$行，单词 $w_i$的输入向量表示
* $U∈R^{|V|*n}$ 输出词矩阵
* $u_i∈R^{1*n}$ $U$的第 $i$行，单词 $w_i$的输出向量表示
* $y∈R^{1*|V|}$ one-hot 行向量，the true probabilities
* $\hat y∈R^{1*|V|}$ 行向量，the predicted probabilities

p.s. 注意，此处的输入矩阵 $V$（即作业程序中的 centerWordVectors）与 notes中的矩阵维度描述相反。另外，做这个作业我的小心得是，搞清楚**相关参数维度**以及整个**运行过程**，对分析偏导和理解程序，很重要。可以 debug查看$V、U$矩阵相关参数如下

![](./U-V矩阵参数.png)

## (a)
交叉熵损失 $\begin{aligned}H(\hat{y},y)=-\sum_{w\in Vocab}y_w log(\hat{y_w})\end{aligned}$ 

其中 $y$ 是one-hot向量($j=o$处为1，其余为0)，$H(\hat{y},y)$ 可简化，即有 

$\begin{aligned} H(\hat{y},y)=-\sum_{w\in Vocab}y_w log(\hat{y_w})=-\sum_{j=1}^{|V|}y_j log(\hat{y_j})=-y_o log(\hat{y_o}) =-1*log(\hat{y_o}) = -log(\hat{y_o}) \end{aligned}$


## (b)
$\begin{aligned} J_{naive-softmax}(v_c,o,U)=-log P(O=o|C=c)=-log(\frac{exp(u_o^Tv_c)}{\sum_{w \in Vocab} exp(u_w^Tv_c)}) \\
=-u_o^Tv_c+log(\sum_{w \in Vocab} exp(u_w^Tv_c)) \end{aligned}$

逐级求导有

$\begin{aligned}\frac{\partial J_{naive-softmax}(v_c,o,U)}{\partial{v_c}}& =-u_o+ \frac{1}{\sum_{w \in Vocab} exp(u_w^Tv_c)}*\sum_{w\in Vocab}exp(u_w^Tv_c)u_w \\
& =-u_o+ \sum_{w\in Vocab}\frac{exp(u_w^Tv_c)}{\sum_{w \in Vocab} exp(u_w^Tv_c)}u_w \\
& =-u_o+\sum_{w\in Vocab}P(O=w|C=c)u_w \\
& =-u_o+\sum_{w\in Vocab}\hat{y_w}u_w 
\end{aligned}$

受（a）中启发，考虑将 $\sum$ 项中的 $\hat{y_w}$拆分为 $y_w+(\hat{y_w}-y_w)$ ，有$\begin{aligned}\sum_{w\in Vocab}y_wu_w=u_o \end{aligned}$，上式为

$\begin{aligned}\frac{\partial J_{naive-softmax}(v_c,o,U)}{\partial{v_c}}& =-u_o+\sum_{w\in Vocab}\hat{y_w}u_w \\
& =-u_o+\sum_{w\in Vocab}[ y_w+(\hat{y_w}-y_w) ]u_w \\
& =-u_o+\sum_{w\in Vocab}y_wu_w + \sum_{w\in Vocab}(\hat{y_w}-y_w)u_w \\
& =-u_o+u_o+\sum_{w\in Vocab}(\hat{y_w}-y_w)u_w \\
& =\sum_{w\in Vocab}(\hat{y_w}-y_w)u_w \end{aligned}$

展开化简有

$ \begin{aligned} \frac{\partial J_{naive-softmax}(v_c,o,U)}{\partial{v_c}} & =\sum_{j=1}^{|V|}(\hat{y_w}-y_w)u_w =(\hat{y_1}-y_1)u_1+(\hat{y_2}-y_2)u_2+...+(\hat{y_{|V|}}-y_{|V|})u_{|V|} \\
&= \left ( \begin{matrix} \hat{y_1}-y_1 & \hat{y_2}-y_2 & \cdots & \hat{y_{|V|}}-y_{|V|}\end{matrix} \right ) 
\left ( \begin{matrix} u_1 \\ u_2 \\ \vdots \\ u_{|V|} \end{matrix} \right ) =(\hat y-y)U
\end{aligned}$ 


## (c)
由（b）中 $$J_{naive-softmax}(v_c,o,U)
=-u_o^Tv_c+log(\sum_{w \in Vocab} exp(u_w^Tv_c))$$

当 $w\not=o$ 时 $y_w=0$

$\begin{aligned}\frac{\partial J_{naive-softmax}(v_c,o,U)}{\partial{u_w}}& =0+ \frac{1}{\sum_{w \in Vocab} exp(u_w^Tv_c)}*\sum_{w\in Vocab}exp(u_w^Tv_c)v_c \\
& =\sum_{w\in Vocab}\frac{exp(u_w^Tv_c)}{\sum_{w \in Vocab} exp(u_w^Tv_c)}v_c \\
& =\sum_{w\in Vocab}P(O=w|C=c)v_c \\
& =\sum_{w\in Vocab}\hat{y_w} v_c \\
& =\hat{y_1}v_c+\hat{y_2}v_c+\hat{y_{|V|}}v_c \\
& =\hat yv_c  \Rightarrow (\hat{y_w}-y_w)v_c
\end{aligned}$

当 $w=o$ 时 $y_w=1$

$\begin{aligned}\frac{\partial J_{naive-softmax}(v_c,o,U)}{\partial{u_w}}& =-v_c+ \sum_{w\in Vocab}P(O=w|C=c)v_c \\
& =-v_c+\sum_{w\in Vocab}\hat{y_w} v_c \\
& =(\hat{y_o} -1)v_c  \Rightarrow (\hat{y_w}-y_w)v_c
\end{aligned}$

另外，考虑到维度问题， $y,\hat y∈R^{1*|V|}, v_c ∈R^{1*n},\frac{\partial J_{naive-softmax}(v_c,o,U)}{\partial{u_w}}∈R^{|V|*n}$，综上有$\frac{\partial J_{naive-softmax}(v_c,o,U)}{\partial{u_w}}=(\hat y-y)^Tv_c$

## (d)
$\begin{aligned} \frac{\partial\sigma(x)}{\partial x} =\frac{e^x(e^x+1)-e^xe^x}{(e^x+1)^2}=\frac{e^x}{(e^x+1)^2}=\frac{1}{e^x+1}*\frac{e^x}{e^x+1}=(1-\sigma(x))*\sigma(x)\end{aligned}$

## (e)
$\begin{aligned} \frac{\partial J_{neg-sample}(v_c,o,U)}{\partial v_c} & =-\frac{(1-\sigma(u_o^Tv_c))*\sigma (u_o^Tv_c)*u_o}{\sigma(u_o^Tv_c)}  -\sum_{k=1}^K \frac{(1-\sigma(-u_k^Tv_c))*\sigma (-u_k^Tv_c)*(-u_k)}{\sigma(-u_k^Tv_c)} \\
& =-(1-\sigma(u_o^Tv_c))*u_o-\sum_{k=1}^K (1-\sigma(-u_k^Tv_c))*(-u_k) \\
& =-(1-\sigma(u_o^Tv_c))*u_o+\sum_{k=1}^K (1-\sigma(-u_k^Tv_c))*u_k \end{aligned}$

$\begin{aligned} \frac{\partial J_{neg-sample}(v_c,o,U)}{\partial u_k}& =0 -\sum_{k=1}^K \frac{1}{\sigma(-u_k^Tv_c)}*(1-\sigma(-u_k^Tv_c))*\sigma (-u_k^Tv_c)*(-v_c) \\
& =-\sum_{k=1}^K (1-\sigma(-u_k^Tv_c))*(-v_c) \\
& =\sum_{k=1}^K (1-\sigma(-u_k^Tv_c))*v_c \end{aligned}$

$\begin{aligned} \frac{\partial J_{neg-sample}(v_c,o,U)}{\partial u_o}& =-\frac{(1-\sigma(u_o^Tv_c))*\sigma (u_o^Tv_c)*v_c}{\sigma(u_o^Tv_c)} \\
& =-(1-\sigma(u_o^Tv_c))*v_c
\end{aligned}$

## (f)
$(i) \begin{aligned}\frac{\partial J_{skip-gram}(v_c,w_{t-m},...w_{t+m},U)}{\partial U}= \sum_{-m\leq j\leq m,j\not =0}\frac{\partial J(v_c,w_{t+j},U)}{\partial U}\end{aligned}$

$(ii) \begin{aligned}\frac{\partial J_{skip-gram}(v_c,w_{t-m},...w_{t+m},U)}{\partial v_c}=\sum_{-m\leq j\leq m,j\not =0}\frac{\partial J(v_c,w_{t+j},U)}{\partial v_c} \end{aligned}$

$(iii) \begin{aligned} \frac{\partial J_{skip-gram}(v_c,w_{t-m},...w_{t+m},U)}{\partial v_w}=0 (w\not ={c}) \end{aligned}$

> p.s. 这块我花了不少时间，学到这儿再回去看 Chris Manning 第一个video最后的推导，思路会清晰很多，部分式子理解可参考[此网站](https://shomy.top/2017/07/28/word2vec-all/)


# Assignment3 [written]

## 1.
### (a)
i. 当横向与纵向的梯度变化差别很大时，SGD 会朝最小值振荡前进，梯度下降的速度很慢且我们不能使用大的学习率。Adam计算梯度的指数加权平均，然后用这个梯度来更新权重。将 $m$看作速度，$\beta_1$看作摩擦，后边的倒数项看成加速度，指数加权平均大的方向得到更大的动量，可以更快的朝着最小值方向移动。详见[吴恩达视频](https://www.bilibili.com/video/av49445369/?p=63)
![](./momentum.png)

ii. 学习率

### (b)
i. $E_{P_{drop}}[h_{drop}]_i=E_{P_{drop}}[\gamma d h]_i =\gamma E_{P_{drop}}[d h]_i =\gamma (1-P_{drop})h_i=h_i$

则有 $\gamma=\frac{1}{1-P_{drop}}$

ii. 训练时随机失活可以减少数据过拟合，而测试评估时使用 dropout会随机失活部分cell，从而可能产生不同的结果，我们不希望得到随机的预测结果，所以在评估时不用dropout

## 2.

### (a)
Stack|Buffer|New dependency|Transition
-----|------|--------------|----------
[ROOT]|[I,parsed,this,sentence,correctly]| |Initial Configuration
[ROOT,I]|[parsed,this,sentence,correctly]| |SHIFT
[ROOT,I,parsed]|[this,sentence,correctly]| |SHIFT
[ROOT,parsed]|[this,sentence,correctly]|parsed $\rightarrow$I|LEFT-ARC
[ROOT,parsed,this]|[sentence,correctly]| |SHIFT
[ROOT,parsed,this,sentence]|[correctly]| |SHIFT
[ROOT,parsed,sentence]|[correctly]|sentence $\rightarrow$this|LEFT-ARC
[ROOT,parsed]|[correctly]|parsed$\rightarrow$sentence|RIGHT-ARC
[ROOT,parsed,correctly]|[]| |SHIFT
[ROOT,parsed]|[]|parsed$\rightarrow$correctly|RIGHT-ARC
[ROOT]|[]|$ROOT\rightarrow$parsed|RIGHT-ARC


### (b)
2n
将所有的 n个词移进$\rightarrow$ n steps + 每个词被且仅被指向一次$\rightarrow$ n steps

### (c)
![](./3.2.c.png)

### (d)
![](./3.2.d.png)

### (e)
![](./3.2.e.png)

>p.s. 因为有随机 dropout，所以每次运行得到的数据不一定相同

![](./3.2.e2.png)

![](./3.2.e3.png)

### (f)
i. Verb Phrase Attachment Error; wedding$\rightarrow$fearing; heading$\rightarrow$fearing
ii. Coordination Attachment Error; makes$\rightarrow$rescue;rush$\rightarrow$rescue
iii. Prepositional Phrase Attachment Error; named$\rightarrow$Midland; guy$\rightarrow$Midland
iv. Modifier Attachment Error; elements$\rightarrow$most; crucial$\rightarrow$most





