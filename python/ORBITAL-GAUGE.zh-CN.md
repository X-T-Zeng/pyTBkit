# 点轨道嵌入、模型代数与 Wilson 边界

HR 的 `orbitals` 是分数坐标，晶格向量为矩阵 A 的行，Cartesian 中心 `tau=orbitals@A`。
不自动把中心包回晶胞，也不把相差整数胞的坐标当作同一个 Bloch 基底。
默认模型导数是其系数框架导数；点轨道假设需显式使用：

```python
import numpy as np
from tbkit import HR, models, embed_orbitals, wilson_loop, prepare_geometry

base = models.ssh()
model = HR(base.vectors, base.hopping, orbitals=[[0], [.4]])
embedded = embed_orbitals(model)
scaled = 2 * embedded
assert embed_orbitals(scaled) is scaled  # 不重复加入位置项
k = (2*np.pi*np.arange(151)/151)[:, None]
phase = wilson_loop(scaled, k, boundary_sewing=scaled.boundary_sewing([2*np.pi]))
# phase = -0.6*pi，等价于 1.4*pi；正数倍缩放不改变占据子空间。
geometry = prepare_geometry(scaled, k, frame='point-orbital')
```

约定 `D_jj=exp(i*k.tau_j)`，`H_emb=D† H D`，因此
`dH_emb=D†[dH+i(H*tau-tau*H)]D`，导数尚未除 hbar。
`S_G=diag(exp(-i*G.tau_j))` 满足 `H_emb(k+G)=S_G H_emb(k) S_G†`，
假设底层系数模型在该倒格矢 G 下周期。`wilson_loop` 的 k 不含重复端点；
最后一个链结目标是 `S_G@u(k_first)`。用户负责选择回路绕行对应的 G；
反向回路使用 -G。传入幺正矩阵并不能证明它就是该模型的物理 sewing。

`orbital_frame='point-orbital'` 与 `requires_boundary_sewing=True` 是内置模型代数
传播的框架标记。`embed_orbitals` 在此框架中幂等；直接重复构造
`OrbitalEmbedding(embedded)` 明确报错。`prepare_spectral/prepare_geometry` 识别标记，
包括经过缩放、相加、相减与嵌套代数的对象，不再依靠具体 Python 类判断。

相加要求相同框架、同一晶格和相同分数中心。中心比较为绝对 `1e-12`、rtol=0，
单位是晶格分数，而非 Cartesian 长度；这是一项明确的表示相容容限。
不同框架或已知/缺失中心混用会报错；需要用户先做明确的基底与元数据变换。
此 PR 不提供自动基底变换或完整一般 Bloch sewing 协议。PR #13 的晶格匹配收紧
独立审阅；本分支未将其未合并代码混入。

`boundary_sewing` 要求有限实 Cartesian 倒格矢；其整数分量检查使用 atol=1e-10、
rtol=0，不会因倒格矢序号增大而允许非整数偏离。极大绕行的相位精度仍受输入浮点数
限制，不承诺任意大整数或不可分辨中心的精度。

仅接受周期框架的 Chern/Wilson-flow/nested/Z2 等入口不会用几个 H(k) 采样点来
覆盖显式 sewing 要求，即使原子极限的 H 恰好为常矩阵。通用带 sewing 的二维
拓扑网格、投影 mirror sector 的连接传播仍待后续实现；不要把局部谱相同当作
所有闭合几何量相同。

数值层保留 NumPy 参考；`backend='numba'` 使用纯数组
`tbkit.gauge_kernels.embed_hamiltonian/embed_gradient`，可从用户 `@njit` 调用。
原生内核要求上述已校验形状的有限数组，不包含对象、隐藏缓存或隐式厄米化；
Fortran 仍求底层 Fourier 数据，随后由 NumPy 加入位置项。

本功能假设正交点轨道，不恢复完整 Wannier 偶极矩阵或非正交位置连接。
原点与轨道位置会影响极化约定；比较计算时应保留相同原点、电子填充和离子背景。
研究例子、独立符号导数和验证依据见 [审计报告](audit/ORBITAL-GAUGE-REVIEW.zh-CN.md)。
