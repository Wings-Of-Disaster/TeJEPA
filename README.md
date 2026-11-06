# TeJEPA：教师约束、预测目标与谱几何

本分支发布方法定义、数学推导和 Lean 证明文件。实验与训练实现另行处理。

核心问题是：**怎样把教师知识约束到最终保留的学生 backbone 上，并在保持教师匹配目标的同时调整误差几何？**

## 方法定义

**当前方法规格采用两个目标：TeReg＋教师引导的跨视图预测。** 谱 Gram 和可靠性 Patch 保留为独立的备选理论，不加入主目标；没有额外 CLS-MSE、RKD、Patch、Gram 或 SIGReg 项。这是理论方法规格，本分支没有宣称训练实现已经切换到此配方。

从同一图像取输入视图 $V$ 和参考视图 $U$。$U$ 为全局视图，$V$ 可来自定义好的全局/局部视图分布。教师在两个视图上分别产生停止梯度的特征。固定坐标后的特征记为 $S(V),T(V),T(U)$，预测头为 $q$：

$$
\boxed{
L=\lambda_TL_{\mathrm{TeReg}}\bigl(S(V),T(V)\bigr)
+\lambda_PL_{\mathrm{pred}},\qquad
L_{\mathrm{pred}}=\mathbb E_{U,V}\|q(S(V))-\operatorname{sg}T(U)\|^2,
\quad\lambda_T,\lambda_P>0.
}
$$

| 部分 | 明确定义 | 理论职责 |
|---|---|---|
| TeReg | 输入视图上的同图、同裁剪学生/教师联合特征函数匹配 | 将教师表示约束落实到 backbone |
| 教师引导 pred | 从输入视图预测同图参考视图的冻结教师特征 | 学习跨视图可预测信息，排除学生自生成常量零目标的捷径 |
| 预测头 $q$ | 学生预测端的可学习映射 | 承担跨视图条件预测；下游使用 backbone 时可丢弃 |
| Gram/Patch | 独立的关系/局部理论推导 | 仅作为后续需要显式空间约束时的候选 |

这里的 pred 在数学上是跨视图回归蒸馏，已经改变了原学生均值一致性目标；它不应被描述成“原 pred 原样保留”。全局 TeReg 也不等于显式 patch 几何监督，本方法规格不作这种替代声明。

已确定的设计原则：

- 教师冻结，不接收梯度。
- TeReg 直接作用于学生 backbone，使用冻结、全维、可逆的坐标变换。
- TeReg 采用教师锚定的联合分布目标；没有额外的高斯 SIGReg 项。
- 预测目标必须明确说明目标由谁产生、使用哪个视图，以及是否允许常量预测取得零损失。

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

这一节是关系损失的理论推导，是否加入最终配方由“方法定义”决定。

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

这一节同样属于局部损失的理论推导。先按裁剪位置和翻转建立真实重叠区域的空间对应，再讨论权重。

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

## 4. 预测目标必须处理常量解

学生 projector 自身产生多视图均值目标时，设 $z_{i,v}=p(h(v_i))$，有

$$
L_{\mathrm{self}}=\mathbb E_{i,v}\|z_{i,v}-\overline z_i\|^2.
$$

任意 backbone 都允许选择 $p(\cdot)=c$ 令此项为零。由于 TeReg 不经过 $p$，单靠 backbone 上的 TeReg 并不能排除这个 projector 解。

若预测目标改为停止梯度的教师向量 $Y$，平方风险为

$$
L_{\mathrm{teacher\text{-}pred}}=\mathbb E\|q(h(V))-Y\|^2.
$$

对常量输出有精确分解：

$$
\mathbb E\|c-Y\|^2
=\|c-\mathbb EY\|^2+\operatorname{tr}\operatorname{Cov}(Y).
$$

因此非退化教师目标排除了常量预测的零风险捷径。这个命题没有声称所有常量预测都不是驻点。

本方法中 $Y=T(U)$ 来自同图参考裁剪，它未必能由输入视图 $V$ 完全预测。正确目标是条件预测，而非假定所有裁剪必须具有完全相同的教师表示。

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

### 5.1 TeReg 与教师引导预测的共同最优性

对主方法，使用共同最优性而非共同零点来论证。明确要求：

1. **表示可实现性：** 学生函数类中存在 $S_\star$，满足 $S_\star(V)=T(V)$ 几乎处处。
2. **教师均值充分性与预测头表达能力：** 允许的预测头类中存在 $q_\star$，满足 $q_\star(T(V))=m(V)$ 几乎处处。

第二条要求教师输入视图表示保留预测参考视图条件均值所需的信息；它是实质假设，不由“教师冻结”自动推出。

在这两个条件下：

$$
L_{\mathrm{TeReg}}(S_\star,T)=0,
\qquad L_{\mathrm{pred}}(S_\star,q_\star)=R_\star.
$$

对任何学生和预测头，$L\ge\lambda_PR_\star$，且上述解取得该下界。因此：

$$
\boxed{\min L=\lambda_PR_\star,\qquad
\operatorname*{argmin}L
=\{(S,q):L_{\mathrm{TeReg}}=0,\ L_{\mathrm{pred}}=R_\star\}.}
$$

这个命题在所列可实现性和充分性条件下证明两个目标不存在必然的全局最优性冲突。它不假设不同裁剪的教师表示完全一致。若充分性或可实现性不成立，共同最优性结论不能直接沿用。

### 5.2 Gram/Patch 是否数学上必需

TeReg 约束全局 CLS；它没有要求 patch Gram 相同。在表示层面，完全相同的 CLS 可以分别伴随 $P_T=I_2$ 与 $P_S=\left[\begin{smallmatrix}1&0\\1&0\end{smallmatrix}\right]$，二者的 Gram 分别为单位矩阵和全一矩阵。这说明全局匹配不蕴含局部关系匹配。

所以，显式 patch 几何若是方法目标，Gram/Patch 提供独立约束；若目标是教师全局表示迁移和跨视图条件预测，它们不是上述识别性、共同最优性与常量次优性命题的必要前提。主方法据此只保留两个目标，同时保留第2、3节作为局部几何扩展的完整理论依据。

### 5.3 梯度兼容性

共同最优解存在也不推出各项在每一步的梯度同向。一个可严格证明的一阶充分条件是：对于非零主梯度 $g$ 和已包含权重的辅助梯度 $r$，如果 $\|r\|\le\eta\|g\|$、$\eta<1$，则

$$
\langle g,g+r\rangle
\ge(1-\eta)\|g\|^2>0.
$$

这是关于指定主损失的一阶方向命题；额外损失、有限步长和带状态的优化器更新需要各自的条件。

## 6. 哪些命题经过 Lean 检查

以下证明使用 Lean 4.19.0，mathlib 固定在 `c44e0c8ee63ca166450922a373c7409c5d26b00b`。发布的每个定理都打印并检查其公理依赖；只允许 Lean/mathlib 的标准逻辑基础，不含 `sorry`、`admit` 或自定义公理。

| 文件 | 已检查范围 |
|---|---|
| [JointAnchoring.lean](formal/JointAnchoring.lean) | 两样本交换的联合核因式分解、正性、无锚点时的零值；满足平方方程的根经过同一平移后的单射性 |
| [SpectralDistillation.lean](formal/SpectralDistillation.lean) | 标量谱逆变换、单射性、差商；带正 epsilon 的范数归一化单射性；正权重保零点；非负目标组合的零点代数 |
| [DistillationLoss.lean](formal/DistillationLoss.lean) | 有限联合 CF 估计器非负性与教师零点；变换后平方差与加权 Patch 的基本性质；教师权重界；带假设的一阶梯度预算命题 |
| [ProjectorAnchor.lean](formal/ProjectorAnchor.lean) | 有限教师回归的常量分解、方差下界及正总损失 |
| [TeacherPrediction.lean](formal/TeacherPrediction.lean) | 有限分组条件均值的残差正交性、风险分解、预测最优性；显式教师充分性下的风险保持；可预测信息非退化时常量严格次优；带风险下界的联合最优性代数 |

**解析推导与 Lean 证书的边界：** 总体特征函数识别性、核等价、矩阵平方根的完整函数演算及 Fréchet 界、Gibbs 最优性、一般概率空间中的条件期望结论在本文中给出数学论证或适用条件，没有标为已完成 Lean 形式化。条件预测的有限分组、逐坐标版本已经检查；教师充分性和可实现性仍是显式前提。`transformedGram` 的基本平方差定理也不单独构成矩阵平方根算法正确性的证明。

## 7. 复现证明检查

安装对应 Lean 工具链后，在本分支根目录运行：

```bash
lake update
lake exe cache get
python3 scripts/check_proofs.py
```

检查器会核对 mathlib 版本、逐文件编译、检查每个定理的公理依赖，并生成 [formal/verification.json](formal/verification.json)。证书只记录软件版本、相对文件名、内容哈希和证明结果。
