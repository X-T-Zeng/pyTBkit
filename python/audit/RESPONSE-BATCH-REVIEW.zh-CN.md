# 谱数据复用与响应处方审阅

基线：官方 BUAA-TCMP/TBkit @ `48e5217593598c8e1d634cdee3976e1dc6984516`。
本轮在独立分支开发，未修改官方原始源码，未向官方 MATLAB 仓库提交补丁。

## 从物理定义核查有限 eta

对厄米 Hamiltonian 的孤立能带，在正交框架和 A=i<u|du> 约定下：

`Omega_ab,n = -2 sum_(m!=n) Im(v_a,nm v_b,mn)/(E_n-E_m)^2`。

这是实数、关于 a/b 反对称；在完整有限维轨道框架内，各带曲率之和为0。
官方 NCTE_k 把上述实数分母替换为 `(Delta+i*eta)^2`，但仍在分式外仅对
速度乘积取虚部。对两带中某一实数乘积虚部 p，得到

`Omega_n=-2p/(Delta+i*eta)^2`，`Omega_m=-conj(Omega_n)`。

因此全带和一般为 `2i Im(Omega_n)`，不再为0，也不再是实数带曲率。
这不是数值舍入：独立带对求和与直接运行原始 NCTE_k 都重现该结果，
选定倾斜两带例子的最终 alpha 也有非零虚部。

**结论的边界**：此有限 eta 路径不能直接称作标准 Berry 曲率的展宽，也不能
未经推导就视为已验证的实 DC 系数。它是否对应某种复响应/寿命处方，需要
完整的因果、耗散、频率及物理单位推导；这里不据此断言所有复 Kubo 响应错误。
同样不通过擅自取 real、改成绝对值分母或引入新散射模型来“修复”原式。
Python 继续提供明确的干净极限。eta=0 时与原式匹配 k_B*T 后验证一致。

官方 `abs(Delta)+eta>eps` 还会在 eta>eps 时包含对角项及简并带对。
对角项理论上应因速度厄米性消失，但这种规则不是简并子空间几何的替代。
当前 Python 默认拒绝未定义的简并单带量，omit 仅为显式比较模式。

## 架构修改及物理约束

过去每次 response_kernel 调用都重做本征问题、导数旋转与几何收缩。
现在明确分成三层，数据由调用方持有，不使用跨模型全局缓存：

1. `prepare_spectral`：一次批量求谱，得到 E、正交本征矢和带基底 dH/dk。
2. `prepare_geometry`：生成实数干净极限几何，保留 frame、gap_tol、
   degeneracy 策略及每个 k 点的简并标记。
3. `response_from_geometry`：仅对 mu/T/张量索引加权，不再访问模型或求谱。

SpectralBatch/GeometryBatch 对输入做有限性、形状和相应厄米性/正交性检查，
复制为只读数组；源模型在准备后修改，不会改变已有快照。
快照不是“最新模型”的自动缓存，不跨参数变化或不同框架盲目复用。

`frame='model'` 使用传入模型本身的导数定义；对于周期 HR 是系数框架。
`frame='point-orbital'` 显式加入轨道中心的点轨道嵌入，且不重复嵌入已有包装模型。
这不是完整 Wannier 偶极/位置连接，也不是非正交基的响应理论；后者明确拒绝。
能量和 mu 在 Fermi 权重阶段须为 eV、T 为 K；梯度未除 hbar，SI 系数未自动加入。

Numba 数组内核拆为 band_velocities 和 geometry_from_velocities，可在用户
自己的 nopython 管线调用；原 spectral_geometry 签名作为组合接口保留。
原 geometric_tensors/response_kernel 接口继续可用。

## 验证方式

- 解析 massive Dirac 的 Berry 曲率与量子度量，不依赖另一个同源实现。
- 随机本征态相位变换后几何不变；简并时只比较有意义的策略或子空间量。
- 同一快照反复计算温度/化学势/张量通道，只调用一次源模型的批量求谱。
- 点轨道快照与显式 OrbitalEmbedding 的 NumPy 参考一致。
- Numba 与 NumPy 收缩、分块与单点形状、只读数据和独立用户 @njit 管线。
- 原始官方有限 eta 的反例，以及 eta=0 的原式对照。

## 性能和内存边界

复用基准比较同样9次响应的单次 API 与准备 API，**两条路径的计时均包含
所需准备工作**。固定随机种子、线程数，区分首次 JIT/缓存载入与预热中位数。
只报告该工作流的改善，不据此宣称底层内核普遍加速或比 MATLAB 快。

快照存储完整本征矢/速度矩阵，是 O(nk*dim*norb^2)；nbytes 不包括 Python 对象、
临时量和 JIT 内存。大的 k 网格应在 integrate_bz 的每个小批次内部准备并复用，
不要将整个网格当永久缓存。稀疏模型请求完整响应仍会生成密集本征问题。
