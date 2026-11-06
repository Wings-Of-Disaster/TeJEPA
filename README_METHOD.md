# 当前方法与 TeX 的对应说明

当前方法保留 LeJEPA 的多视图预测，使用冻结教师进行语义与空间蒸馏，并以教师约束替代 SIGReg。教师约束同时覆盖最终保留的 backbone 表示和训练用的 projector 输出。

**输入与视图分配**

每张图片生成两个 global 裁剪和六个 local 裁剪。学生处理全部八个视图，教师处理相同的两个 global 视图。

| 损失 | 使用的视图 | 作用 |
|---|---|---|
| pred | 全部八个学生视图 | 各视图的 projector 输出接近两张 global 输出的均值 |
| CLS | 两个 global 分别计算，再平均 | 学生特征经可学习对齐头后，匹配同裁剪的教师全局特征 |
| RKD | 两个 global 分别计算，再平均 | 匹配每个视图内、每设备 batch 中的图像相似关系 |
| SPM（改进 Patch） | 两个 global 的有效重叠区域 | 匹配局部特征，按教师跨视图一致性分配正权重 |
| SGM（改进 Gram） | 两个 global 各自的完整 patch 网格 | 经移位平方根谱变换和 Frobenius 归一化后，匹配 patch 关系 |
| backbone TeReg | 汇总两个 global 的同裁剪学生—教师配对 | 在冻结坐标中进行教师锚定的联合统计匹配 |
| projector 教师约束 | 学生与教师各自先对两个 global 求均值 | 对学生投影中心施加教师配对 MSE 和联合 CF 约束 |

**完整目标**

$$
\mathcal L
=c_{\mathrm{pred}}\mathcal L_{\mathrm{pred}}
+\mathcal L_{\mathrm{SEM}}
+\lambda_t\mathcal L_{\mathrm{TeReg}}
+\lambda_a\mathcal L_{\mathrm{proj}},
$$

$$
\mathcal L_{\mathrm{SEM}}
=\mathcal L_{\mathrm{CLS}}+\mathcal L_{\mathrm{RKD}}
+\lambda_p\mathcal L_{\mathrm{SPM}}
+\lambda_g\mathcal L_{\mathrm{SGM}},
\qquad
\mathcal L_{\mathrm{proj}}
=\mathcal L_{\mathrm{proj\text{-}MSE}}
+\mathcal L_{\mathrm{proj\text{-}CF}}.
$$

各系数在方法公式中用超参数符号表示，具体数值在实验设置中说明。不额外加入 SIGReg。

**两处教师约束的区别**

backbone TeReg 直接监督 backbone 的全局特征。固定的均值、尺度和正交映射提供比较坐标；联合统计匹配比较“学生—教师配对”和“教师—教师配对”。两个 global 的配对一起汇总，每步使用共享的随机 Fourier 频率，下一步重新采样。

projector 教师约束监督原预测分支的全局中心。记两个学生 global 的投影均值为 $c_i$，两个教师 global 特征的均值经过冻结的中心化、缩放和等距映射后为 $y_i$：

- 配对 MSE：逐张图片计算 $\|c_i-y_i\|^2$，再平均。
- 联合 CF：使用随机正弦、余弦统计，匹配 $(c_i,y_i)$ 与 $(y_i,y_i)$ 的联合分布统计。

原 pred 保持不变，六个 local 仍通过 pred 与这个全局中心联系。当教师目标具有非零方差时，常量 projector 无法让教师配对 MSE 降为零。

教师始终冻结。projector 教师约束与 pred 均向 projector 和 backbone 传播梯度；CLS、RKD、SPM、SGM 和 backbone TeReg 使用关闭学生随机层、保留梯度的监督前向。原预测分支及其 projector 教师约束保持原生训练模式。推理时只保留学生 backbone。

**相对所提供 TeX 的修改**

| 位置 | 原 TeX | 当前定义 |
|---|---|---|
| 总目标 | pred＋SEM＋TeReg | 显式写出预测系数，并加入 projector 教师约束 |
| CLS、RKD | 只用第一个 global | 两个 global 分别计算后平均 |
| backbone TeReg | 使用两个 global | 保留此定义 |
| projector 描述 | 仅说明它不在 TeReg 的直接约束内 | 说明 backbone TeReg 的作用位置，并补充 projector 自身的教师锚定约束 |
| 学生前向模式 | 所有教师监督分支均关闭随机层 | 区分确定性监督前向与原预测分支上的 projector 教师约束 |

原 TeX 中的初始化与适配框架、固定坐标、TeReg 联合匹配、SPM 和 SGM 定义保留。projector 教师约束单独补充，不改变原八视图 pred 的定义。


**数学推导与 Lean 证明**

详细推导见 [理论说明 README](README.md)。证明源码、固定工具链和验证证书一并提供；当前证书记录 **64 个通过检查的形式化命题**。

| 文件 | 对应的数学结论 |
|---|---|
| [DistillationLoss.lean](formal/DistillationLoss.lean) | 联合 CF 非负性和教师匹配零点；SPM 正权重界；基本损失性质及带条件的一阶梯度界 |
| [JointAnchoring.lean](formal/JointAnchoring.lean) | 两样本交换时，联合匹配能区分边际匹配不能区分的配对变化；带平方方程条件的谱根单射性 |
| [ProjectorAnchor.lean](formal/ProjectorAnchor.lean) | 配对 MSE 的常量输出分解与方差下界，解释教师锚定如何排除常量 projector 的零损失捷径 |
| [SpectralDistillation.lean](formal/SpectralDistillation.lean) | 标量谱变换的逆关系、单射性和差商；带正 epsilon 的范数归一化单射性；正权重保零点 |
| [LossRelations.lean](formal/LossRelations.lean) | CLS/RKD 的包含关系与有限误差非等价性；部分覆盖 Patch/Gram 的互补反例；固定坐标与可学习读出的区别 |
| [TeacherPrediction.lean](formal/TeacherPrediction.lean) | 补充的条件预测风险分解，以及显式充分性假设下的共同最优性；这是另一预测目标的理论分析，与当前 pred 定义分开 |

需要区分：零点包含关系与有限误差互补性是不同命题。例如 CLS 完全匹配会使 RKD 同样为零，但相同的非零 CLS 误差可以对应不同的 RKD。`LossRelations.lean` 中 projector 可消去的结论只针对不含 projector 教师锚定项的历史表达式；当前带锚定的方法对应 `ProjectorAnchor.lean` 的正下界。

Lean 形式化范围以各文件中的定理前提为准。总体特征函数识别性、完整矩阵平方根函数演算等解析推导与有限代数证明分别说明；证明不等同于下游准确率或优化收敛保证。

验证证书见 [formal/verification.json](formal/verification.json)。工具链与依赖版本固定后，可在仓库根目录运行：

```bash
lake update
lake exe cache get
python3 scripts/check_proofs.py
```
