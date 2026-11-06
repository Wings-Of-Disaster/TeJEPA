# TeJEPA：教师约束、预测目标与谱几何

本分支发布方法定义、数学推导和 Lean 证明文件。实验与训练实现另行处理。

核心问题是：**怎样把教师知识约束到最终保留的学生 backbone 上，并在保持教师匹配目标的同时调整误差几何？**

## 方法定义

**本说明以已有完整方法为准：原 LeJEPA pred＋CLS＋RKD＋改进 Patch＋改进 Gram＋TeReg。** 谱 Gram 与可靠性 Patch 保留在主目标中，TeReg 替代 SIGReg。教师引导的两项精简方案仅是未采用的理论候选，见第4、5节；它没有取代此处的方法定义。本分支仅发布理论和证明，不修改训练实现。

$$
\boxed{
L=\lambda_{\mathrm{pred}}L_{\mathrm{pred}}
+\beta\left(L_{\mathrm{CLS}}+\lambda_rL_{\mathrm{RKD}}
+\lambda_pL_{\mathrm{Patch}}+\lambda_gL_{\mathrm{Gram}}\right)
+\lambda_TL_{\mathrm{TeReg}}.
}
$$

| 部分 | 明确定义 | 理论职责 |
|---|---|---|
| pred | 学生各视图经原 projector 后，匹配两个 global 视图的投影均值 | 原 LeJEPA 多视图一致性；不使用教师目标 |
| CLS | 第一 global 视图的学生 CLS 经可学习对齐头后，以 MSE 匹配教师 | 逐图全局特征监督 |
| RKD | 上述对齐 CLS 的批内余弦相似度，经 softmax 后与教师分布做 KL 匹配 | 图像间关系监督 |
| 改进 Patch | 两个 global 视图中按裁剪/翻转建立重叠对应，并按教师一致性分配正权重 | 局部特征监督与有效区域覆盖 |
| 改进 Gram | 对齐并归一化 patch 特征的 Gram，经移位平方根谱变换及范数归一化后匹配 | 空间关系与谱误差度量 |
| TeReg | 第一 global 视图的同裁剪 backbone CLS，在固定坐标中做教师锚定联合匹配 | 将教师表示约束直接落实到 backbone |

已有完整配方的系数为 $\lambda_{\mathrm{pred}}=0.95$、$\beta=\lambda_r=1$、$\lambda_p\approx0.471686$、$\lambda_g\approx0.162640$、$\lambda_T\approx0.242291$。这些数值描述已有配方，不是上述理论给出的唯一最优权重。

**TeReg 不直接约束原 projector。** 原 projector 只接收 pred 梯度；CLS 对齐头接收 CLS/RKD 梯度；patch 对齐头接收 Patch/Gram 梯度；backbone 接收全部损失的梯度。教师、TeReg 的正交坐标和标定尺度冻结。

已确定的设计原则：

- 教师冻结，不接收梯度。
- TeReg 直接作用于学生 backbone，使用冻结、全维、可逆的坐标变换。
- TeReg 采用教师锚定的联合分布目标；没有额外的高斯 SIGReg 项。
- 原 pred 的学生均值目标与后文候选的教师预测目标严格区分；前者允许常量 projector 零损失，不能借用后者的方差下界来宣称已经解决此问题。

**损失结构审计：** 这六项具有不同作用位置，但不能据此断言它们在零点上全部独立。第5.4节给出精确包含关系、有限误差下的非等价性和局部覆盖反例；这些结论不改变上述完整配方。

## 1. TeReg：从边际相同到同图对应

### 1.1 固定坐标

设学生和教师的全局特征均属于 $\mathbb R^d$。用成对训练样本标定正交变换 $R$、均值和正标量尺度，随后冻结：

$$
S(x)=\frac{(h_S(x)-\mu_S)R}{r_S},\qquad
T(x)=\frac{h_T(x)-\mu_T}{r_T},
\quad R^\top R=I,\quad r_S,r_T>0.
$$

它们分别是学生和教师在比较坐标中的表示。正交变换加固定非零缩放保留学生特征的中心化秩和协方差谱形状。该结论要求全维正交映射，不适用于任意可学习 MLP 或降维映射。

### 1.2 联合特征函数目标

仅匹配 $S$ 与 $T$ 的边际分布，无法识别教师样本的重新排列。因此把同图教师特征作为锚点：

$$
\Delta(a,b)=\mathbb E_x\left[
e^{i b^\top T(x)}\left(e^{i a^\top S(x)}-e^{i a^\top T(x)}\right)
\right],
$$

$$
\mathcal L_{\mathrm{TeReg}}=\mathbb E_{a,b}|\Delta(a,b)|^2.
$$

$a,b$ 独立取自具有正带宽的高斯频率分布。高斯分布用于检测频率，不要求学生或教师特征服从高斯。

**总体目标的识别性：**

$$
\mathcal L_{\mathrm{TeReg}}=0
\quad\Longleftrightarrow\quad
S(x)=T(x)\ \text{几乎处处成立}.
$$

证明：被积函数非负；频率密度处处为正，且 $\Delta$ 连续，所以零积分推出所有频率上的差均为零。取 $b=-a$，得到

$$
\mathbb E e^{i a^\top(S-T)}=1\quad\text{对所有 }a.
$$

由特征函数唯一性，$S-T$ 的分布为零点质量。反向直接代入即可。

这个证明针对完整总体频率积分。一次有限随机 sketch 的零损失不具有相同识别保证。

### 1.3 核形式与有限配对例子

定义正带宽高斯核 $k_s,k_t$。有限批次的精确核形式为：

$$
\widehat{\mathcal L}=
\frac1{N^2}\sum_{i,j}k_t(T_i,T_j)
\left[k_s(S_i,S_j)+k_s(T_i,T_j)-k_s(S_i,T_j)-k_s(T_i,S_j)\right].
$$

它是联合经验分布 $(S,T)$ 与 $(T,T)$ 的乘积核 MMD 平方，即一个有偏 V-statistic。随机 Fourier 估计器对频率取期望等于这个批次核目标。

考虑两张教师表示不同的图像，学生恰好交换两者的对应关系。令两个核的非对角值分别为 $u,v\in[0,1)$，则：

$$
\widehat{\mathcal L}_{\mathrm{joint}}=(1-u)(1-v)>0.
$$

移除教师锚点，相当于令 $v=1$，上述值变为零。这个有限例子把“边际匹配无法识别交换、联合目标能识别”的差别直接写成可检查的代数命题。

### 1.4 约束落实到 backbone

在总体零点上：

$$
h_T(x)=\frac{r_T}{r_S}(h_S(x)-\mu_S)R+\mu_T.
$$

教师与学生的成对距离仅差固定比例，协方差谱形状相同；教师的线性读出可通过此可逆仿射关系转换为学生的线性读出。这是理想匹配点的表示性质。

## 2. 谱 Gram：保持关系目标，调整谱敏感度

这一节推导主方法中改进 Gram 的理论性质。

令 $P$ 为进入关系损失的 patch 特征矩阵，$G=PP^\top\succeq0$。如果 $P$ 经过可学习对齐头，本节结论首先作用于对齐后的 patch 空间；将性质传回原始 backbone 需要额外的可逆性条件。

定义精确谱变换与稳定化归一化：

$$
B_\delta(G)=\sqrt{G+\delta I}-\sqrt\delta I,\qquad
\Phi(G)=\frac{B_\delta(G)}{\|B_\delta(G)\|_F+\epsilon},
\quad\delta,\epsilon>0.
$$

### 2.1 保持零点

设 $B=B_\delta(G)$，直接平方可恢复：

$$
G=B^2+2\sqrt\delta B.
$$

因此 $B_\delta$ 是单射。稳定化归一化 $B\mapsto B/(\|B\|_F+\epsilon)$ 也是单射：其输出范数为 $r/(r+\epsilon)$，对 $r\ge0$ 严格递增；先恢复范数，再恢复矩阵。

于是精确实数运算中：

$$
\|\Phi(G_S)-\Phi(G_T)\|_F^2=0
\quad\Longleftrightarrow\quad G_S=G_T.
$$

$\epsilon>0$ 对这里的归一化单射性证明重要。形式上的单射性也不代表浮点环境能稳定恢复任意细小的尺度差异。

### 2.2 谱重加权与有界敏感度

对应标量变换 $f_\delta(\lambda)=\sqrt{\lambda+\delta}-\sqrt\delta$：

$$
f_\delta'(\lambda)=\frac1{2\sqrt{\lambda+\delta}},
$$

$$
f_\delta(x)-f_\delta(y)
=\frac{x-y}{\sqrt{x+\delta}+\sqrt{y+\delta}}.
$$

这明确减弱大特征值方向的绝对敏感度。若两矩阵可同时正交对角化，未归一化的变换后距离精确等于各特征值误差平方乘以对应分母平方的倒数。

一般情况下，在 $G$ 的特征基中，精确平方根变换的 Fréchet 导数为：

$$
\bigl[U^\top DB_\delta(G)[H]U\bigr]_{ij}
=\frac{[U^\top HU]_{ij}}
{\sqrt{\lambda_i+\delta}+\sqrt{\lambda_j+\delta}}.
$$

分母不包含特征值之差；重根不引入奇异的特征值间隙分母，并且

$$
\|DB_\delta(G)[H]\|_F\le\frac1{2\sqrt\delta}\|H\|_F.
$$

这是精确平方根映射相对于 Gram 扰动的界，不是整个神经网络参数梯度的界。

还可不依赖可交换性推导有限差分界。设 $A=\sqrt{G+\delta I}$、$C=\sqrt{H+\delta I}$、$X=A-C$，则

$$
AX+XC=G-H.
$$

取 Frobenius 内积并使用 $A,C\succeq\sqrt\delta I$ 与 Cauchy–Schwarz 不等式，得到

$$
\|B_\delta(G)-B_\delta(H)\|_F
\le\frac1{2\sqrt\delta}\|G-H\|_F.
$$

若另外 $G,H\preceq MI$，由同一恒等式还得到反向界：

$$
\frac{\|G-H\|_F}{2\sqrt{M+\delta}}
\le\|B_\delta(G)-B_\delta(H)\|_F.
$$

因此，这个设计的理论职责是保持原关系匹配目标，同时重塑谱方向的误差权重。它不会在逐 patch 精确匹配的零点之外凭空增加新的教师信息。

上述矩阵结论针对精确平方根。有限次数数值迭代的误差、浮点舍入与反向传播误差需另行分析。

## 3. Patch 权重：覆盖约束与监督分配

这一节推导主方法中的局部损失。先按裁剪位置和翻转建立真实重叠区域的空间对应，再讨论权重。

设有效对应位置共有 $n>0$ 个，误差 $e_j\ge0$，教师确定的权重 $w_j\in[\rho,1]$，其中 $\rho>0$：

$$
L_0=\frac1n\sum_j e_j,\qquad
L_w=\frac{\sum_jw_je_j}{\sum_jw_j}.
$$

分子与分母分别夹在相应上下界之间，故

$$
\rho L_0\le L_w\le\rho^{-1}L_0,
\qquad L_w=0\iff e_j=0\ \text{对所有有效位置成立}.
$$

权重下界保留有效对应区域上的零误差目标。没有有效对应时需要单独定义回退目标，不能套用 $n>0$ 的证明。

指数一致性权重可以从另一个明确问题出发理解。对教师不一致代价 $d_j\ge0$，解熵正则化选择问题：

$$
\min_{q\in\Delta_n}\sum_jq_jd_j+\tau\sum_jq_j\log q_j,
\qquad q_j^\star\propto e^{-d_j/\tau},\quad\tau>0.
$$

对带底线的权重 $w_j=\rho+(1-\rho)e^{-d_j/\tau}$，归一化分布可写为均匀分布与该 Gibbs 分布的混合：

$$
\frac{w_j}{\sum_kw_k}=\alpha\frac1n+(1-\alpha)q_j^\star,
\quad\alpha=\frac{n\rho}{n\rho+(1-\rho)\sum_ke^{-d_k/\tau}}.
$$

这个分解提供“一致性选择＋覆盖底线”的解释。它不等同于证明教师一致性就是标签正确率，带底线的混合分布也不是上面无约束熵问题本身的最优解。

## 4. 当前 pred 的定义，以及尚未采用的教师预测候选

当前方法保留两个 global、六个 local 视图。设学生 projector 输出为 $z_{i,v}=p(h_S(v_i))\in\mathbb R^{d_z}$，参考均值为 $c_i=(z_{i,g_1}+z_{i,g_2})/2$，则实际 pred 定义为

$$
L_{\mathrm{pred}}=\frac1{N|\mathcal V|d_z}\sum_{i,v}\|z_{i,v}-c_i\|^2.
$$

任意 backbone 都允许选择 $p(\cdot)=c$ 令此项为零。由于 TeReg 不经过 $p$，单靠 backbone 上的 TeReg 并不能排除这个 projector 解。

**以下是独立的、尚未采用的候选推导，不能当作当前 pred 的性质或定义。** 若把原目标替换为同图参考视图上的停止梯度教师向量 $Y=T(U)$，并令新预测头为 $q$，平方风险为

$$
L_{\mathrm{teacher\text{-}pred}}=\mathbb E\|q(h(V))-Y\|^2.
$$

对常量输出有精确分解：

$$
\mathbb E\|c-Y\|^2
=\|c-\mathbb EY\|^2+\operatorname{tr}\operatorname{Cov}(Y).
$$

因此非退化教师目标排除了常量预测的零风险捷径。这个命题没有声称所有常量预测都不是驻点。

该候选中的 $Y=T(U)$ 来自同图参考裁剪，它未必能由输入视图 $V$ 完全预测。对应目标是条件预测，而非假定所有裁剪必须具有完全相同的教师表示。

设 $\mathbb E\|Y\|^2<\infty$，定义

$$
m(V)=\mathbb E[Y\mid V],\qquad
R_\star=\mathbb E\|Y-m(V)\|^2.
$$

任意仅依赖 $V$ 的预测器 $f$ 均满足精确风险分解：

$$
\mathbb E\|f(V)-Y\|^2
=R_\star+\mathbb E\|f(V)-m(V)\|^2.
$$

因此，跨视图目标的最优值一般是 $R_\star$，无需为零。

进一步，对所有常量预测器取最优值，得到：

$$
\min_c\mathbb E\|c-Y\|^2-R_\star
=\mathbb E\|m(V)-\mathbb EY\|^2.
$$

**常量预测严格次优的准确条件是 $m(V)$ 非常量。** 仅有教师目标 $Y$ 的方差非零，尚不足以保证输入视图含有可预测信息。这个区别决定了视图分布所需的理论条件。

## 5. 兼容性：区分共同目标与优化路径

对非负目标 $L_1,L_2,L_3$ 与正系数，有

$$
L_1+\lambda L_2+\mu L_3=0
\iff L_1=L_2=L_3=0.
$$

但要证明共同零点存在，还需另外给出学生可同时实现这些表示的条件。对于跨视图预测，正确的共同最优性条件可能涉及不可约残差，而非所有项为零。

### 5.1 未采用的两项候选：TeReg 与教师引导预测的共同最优性

这一节只分析第4节中的两项候选 $L_{\mathrm{candidate}}=\lambda_TL_{\mathrm{TeReg}}+\lambda_PL_{\mathrm{teacher\text{-}pred}}$，不构成六项主目标的整体最优性证明。明确要求：

1. **表示可实现性：** 学生函数类中存在 $S_\star$，满足 $S_\star(V)=T(V)$ 几乎处处。
2. **教师均值充分性与预测头表达能力：** 允许的预测头类中存在 $q_\star$，满足 $q_\star(T(V))=m(V)$ 几乎处处。

第二条要求教师输入视图表示保留预测参考视图条件均值所需的信息；它是实质假设，不由“教师冻结”自动推出。

在这两个条件下：

$$
L_{\mathrm{TeReg}}(S_\star,T)=0,
\qquad L_{\mathrm{teacher\text{-}pred}}(S_\star,q_\star)=R_\star.
$$

对任何学生和预测头，$L_{\mathrm{candidate}}\ge\lambda_PR_\star$，且上述解取得该下界。因此：

$$
\boxed{\min L_{\mathrm{candidate}}=\lambda_PR_\star,\qquad
\operatorname*{argmin}L_{\mathrm{candidate}}
=\{(S,q):L_{\mathrm{TeReg}}=0,\ L_{\mathrm{teacher\text{-}pred}}=R_\star\}.}
$$

这个命题在所列可实现性和充分性条件下证明两个目标不存在必然的全局最优性冲突。它不假设不同裁剪的教师表示完全一致。若充分性或可实现性不成立，共同最优性结论不能直接沿用。

### 5.2 Gram/Patch 是否数学上必需

TeReg 约束全局 CLS；它没有要求 patch Gram 相同。在表示层面，完全相同的 CLS 可以分别伴随 $P_T=I_2$ 与 $P_S=\left[\begin{smallmatrix}1&0\\1&0\end{smallmatrix}\right]$，二者的 Gram 分别为单位矩阵和全一矩阵。这说明全局匹配不蕴含局部关系匹配。

所以，显式 patch 几何若是方法目标，Gram/Patch 提供独立约束。当前完整方法保留这两项：TeReg 负责 backbone 的全局 CLS 匹配，Patch/Gram 则监督局部特征和空间关系。两项精简候选没有包含这组显式空间约束，不能替代完整方法的定义。

### 5.3 梯度兼容性

共同最优解存在也不推出各项在每一步的梯度同向。一个可严格证明的一阶充分条件是：对于非零主梯度 $g$ 和已包含权重的辅助梯度 $r$，如果 $\|r\|\le\eta\|g\|$、$\eta<1$，则

$$
\langle g,g+r\rangle
\ge(1-\eta)\|g\|^2>0.
$$

这是关于指定主损失的一阶方向命题；额外损失、有限步长和带状态的优化器更新需要各自的条件。

### 5.4 完整配方的结构审计：冗余与互补必须分别定义

这里区分两种不同命题：

- **零点独立：** 其余损失全为零时，这一项还能非零。它意味着新增了独立的精确约束。
- **有限误差非等价：** 一项损失不能由另一项的标量数值恢复，或两者在非零误差时采用不同的误差度量。它不要求零点独立。

#### CLS 与 RKD：零点包含，但有限误差不等价

记进入 CLS-MSE 和 RKD 的同一组对齐特征为 $A_i$，教师特征为 $T_i$。则

$$
L_{\mathrm{CLS}}=0\Longrightarrow A_i=T_i\ \forall i
\Longrightarrow L_{\mathrm{RKD}}=0.
$$

后一个推论对当前相同的归一化、温度和 softmax 规则直接成立。因此不存在“CLS 为零、RKD 非零”的反例，**六项在零点上完全独立的总命题不成立**。Lean 已检查该蕴含及对应的反例不存在性。

这不意味着 RKD 是 CLS 损失数值的重复计算。令

$$
T=\begin{bmatrix}1&0\\0&1\end{bmatrix},\qquad
A=2T,\qquad
B=\begin{bmatrix}1&0\\1&0\end{bmatrix}.
$$

两者的未平均 CLS 平方误差都为 $2$。但对任意温度 $\tau>0$：

$$
L_{\mathrm{RKD}}(A,T)=0,\qquad L_{\mathrm{RKD}}(B,T)>0.
$$

$A$ 的归一化相似关系与教师相同；$B$ 的相似关系全部相同，产生均匀行分布，而教师的行分布非均匀。Lean 直接检查了余弦、softmax 和严格 KL 正性。反例中所有特征范数至少为 $1$，因此在实现的归一化阈值小于 $1$ 时，精确余弦与实际归一化定义一致。该反例说明不存在标量函数 $f$ 满足所有表示上的 $L_{\mathrm{RKD}}=f(L_{\mathrm{CLS}})$。实现中的均值归约只乘正系数，不改变这些等式、严格不等式及非等价性。

RKD 的局部数学角色可以更精确地表述。令教师一行 logits 的 softmax 为 $\pi$，学生 logits 的扰动为 $u$，则

$$
\mathrm{KL}\bigl(\pi\,\|\,\mathrm{softmax}(z_T+u)\bigr)
=\log\sum_j\pi_j e^{u_j}-\sum_j\pi_ju_j
=\tfrac12u^\top F_\pi u+O(\|u\|^3),
$$

$$
F_\pi=\operatorname{diag}(\pi)-\pi\pi^\top\succeq0,
\qquad
u^\top F_\pi u=\sum_j\pi_j\left(u_j-\sum_k\pi_ku_k\right)^2.
$$

因此 RKD 在教师附近给特征误差施加额外的关系方向权重。加权方差恒等式和该二次型非负性已通过 Lean；KL 的 Taylor 展开及其与神经网络 Jacobian 的组合是解析论证。它说明误差度量改变，不证明优化条件数必然改善或训练必然加速。

#### 实际 Patch 与 Gram：部分覆盖下双向不可替代

Patch 只在有效重叠/插值支撑上观察局部误差，Gram 使用全部 patch 的关系。对一个只观察第一个位置的有限覆盖模型，取单位标量 patch：

$$
P_T=(1,1),\qquad P_S=(1,-1).
$$

覆盖位置误差为零，但两个 Gram 的平方 Frobenius 距离为 $8$。反过来，取 $P_S=(-1,-1)$，Gram 完全相同，覆盖位置的特征平方误差为 $4$。单位 patch 的平方误差与余弦误差只差正系数。

这两个有限反例均已通过 Lean。前一个对应真实插值实现时，需要被改动 token 不进入任何有效插值支撑；不能把任意未被中心点选中的 token 都视为不受采样影响。第2节精确谱变换的单射性将不同原始 Gram 转为不同的变换后 Gram；数值迭代仍有单独的近似分析边界。

**完整覆盖是不同情形：** 如果每个对齐 patch 都已精确匹配，Gram 必然匹配。这个条件性包含关系也已通过 Lean，不能为了声称独立而省略它。

#### 固定 backbone 约束与可学习 CLS 读出：参数不同，不能互换

以固定坐标中的两个教师值 $T=(-1,1)$ 为例。学生取 $S=2T$，可学习标量读出取 $a=1/2$，则对齐 CLS 误差为零，但固定坐标误差 $\sum_i(S_i-T_i)^2=2$。反过来，$S=T$ 而读出取 $a=-1$，固定坐标误差为零，CLS 误差为 $8$。

Lean 已检查这些有限读出反例。将固定坐标不匹配换成总体 TeReg 非零，使用第1节的总体识别性解析结论；不把有限频率 sketch 当成总体识别定理。反例说明可学习读出能够吸收尺度变化，而固定坐标目标约束的是读出之前的表示。

#### pred：有独立的多视图误差，但有可消去的 projector 捷径

教师损失不直接依赖原 projector 参数，所以改变 projector 可以改变 pred 而保持教师损失不变。不过，若允许常量 projector，任意 backbone 均可令 pred 为零。对其余损失记为 $F(\theta)$、非负系数为 $\lambda$，则

$$
\min_p\{F(\theta)+\lambda L_{\mathrm{pred}}(\theta,p)\}=F(\theta).
$$

Lean 检查了常量零解、固定其余项时 pred 可改变的例子，以及逐 backbone 的最小值上下界。因此，不能把原 pred 表述成一个不可绕过的 backbone 防坍塌约束；它可能影响联合训练路径，但这里没有将这种影响冒充成独立最优解约束。

以上审计支持的准确表述是：**完整配方包含全局读出、固定坐标表示、部分覆盖局部特征、全局空间关系以及多视图误差；其中存在明确的零点包含关系，同时存在有限误差度量和观察覆盖的互补性。** 不使用“所有损失在所有意义上均无冗余”的表述。

## 6. 哪些命题经过 Lean 检查

以下证明使用 Lean 4.19.0，mathlib 固定在 `c44e0c8ee63ca166450922a373c7409c5d26b00b`。发布的每个定理都打印并检查其公理依赖；只允许 Lean/mathlib 的标准逻辑基础，不含 `sorry`、`admit` 或自定义公理。

| 文件 | 已检查范围 |
|---|---|
| [JointAnchoring.lean](formal/JointAnchoring.lean) | 两样本交换的联合核因式分解、正性、无锚点时的零值；满足平方方程的根经过同一平移后的单射性 |
| [SpectralDistillation.lean](formal/SpectralDistillation.lean) | 标量谱逆变换、单射性、差商；带正 epsilon 的范数归一化单射性；正权重保零点；非负目标组合的零点代数 |
| [DistillationLoss.lean](formal/DistillationLoss.lean) | 有限联合 CF 估计器非负性与教师零点；变换后平方差与加权 Patch 的基本性质；教师权重界；带假设的一阶梯度预算命题 |
| [ProjectorAnchor.lean](formal/ProjectorAnchor.lean) | 有限教师回归的常量分解、方差下界及正总损失 |
| [TeacherPrediction.lean](formal/TeacherPrediction.lean) | 有限分组条件均值的残差正交性、风险分解、预测最优性；显式教师充分性下的风险保持；可预测信息非退化时常量严格次优；带风险下界的联合最优性代数 |
| [LossRelations.lean](formal/LossRelations.lean) | CLS零误差蕴含RKD零误差；任意正温度下相同CLS误差却有不同RKD的余弦/softmax/KL反例；关系二次型非负性；部分覆盖Patch/Gram双向反例及完整匹配时的包含关系；可学习读出与固定坐标的反例；原pred的projector消去性质 |

**解析推导与 Lean 证书的边界：** 总体特征函数识别性、核等价、矩阵平方根的完整函数演算及 Fréchet 界、Gibbs 最优性、一般概率空间中的条件期望结论在本文中给出数学论证或适用条件，没有标为已完成 Lean 形式化。条件预测的有限分组、逐坐标版本已经检查；教师充分性和可实现性仍是显式前提。`transformedGram` 的基本平方差定理也不单独构成矩阵平方根算法正确性的证明。

## 7. 复现证明检查

安装对应 Lean 工具链后，在本分支根目录运行：

```bash
lake update
lake exe cache get
python3 scripts/check_proofs.py
```

检查器会核对 mathlib 版本、逐文件编译、检查每个定理的公理依赖，并生成 [formal/verification.json](formal/verification.json)。证书只记录软件版本、相对文件名、内容哈希和证明结果。
