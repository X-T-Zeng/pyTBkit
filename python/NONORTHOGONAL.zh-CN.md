# 非正交轨道谱与可靠性报告

本接口研究普通正定轨道 Gram 矩阵 `Hc=E Sc`，本征矢为列且 `C†SC=I`。
不定度量 BdG、完整非正交 Berry 连接/群速度不在当前接口中。

```python
import numpy as np
from tbkit import HK, NonOrthogonal

sx = np.array([[0., 1.], [1., 0.]])
sz = np.diag([1., -1.])
h = HK({(0,): .4*sz, (1,): sx})
s = HK({(0,): np.diag([1., 2.])})
model = NonOrthogonal(h, s)

report = model.spectrum([[.2], [.6]])
print(report.energy)               # (nk, norb)
print(report.overlap_condition)    # 每个 k 的 Gram 条件估计
print(report.residual)             # 每个 k、每条带的范数残差
print(report.metric_error)         # 每个 k 的 ||C† S C-I||F
print(model.band_gradient([.2]))   # (..., dimension, band)
```

`spectrum` 始终保留 batch 轴，数据只读、归调用者持有，无全局模型缓存。
`single` 标记输入是否为单点。现有 `bands` 单点/批量返回形状、列本征矢和可写数组保留。
H/S 在每次调用时读取，模型修改不会回写已经取得的报告。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `condition_limit` | `1e8` | Gram 特征值比预算；超限在广义对角化前报错 |
| `check_tol` | `1e-8` | 逐带相对范数残差、Frobenius Gram 框架误差的上界 |
| `band_gradient(..., gap_tol=None)` | 无绝对截止 | 默认使用局部整个能谱带宽的 `1e-9` |
| 显式 `gap_tol>0` | 输入能量单位 | 使用绝对阈值，关闭默认相对项 |
| 显式 `gap_rtol>=0` | 无量纲 | 与绝对项取最大；精确简并总是拒绝 |

所有检查不自动厄米化、正交化或删基。默认条件预算是新增的兼容性边界：过去可计算
但不可靠的输入可能现在报错。条件数取决于坐标基底，放宽预算或重整基底须明确决定，
不能从残差小推断病态模型的物理前向误差小。输入本身舍入掉的信息也无法由求解器恢复。

可用 `backend='numpy'` 运行模型与独立 NumPy 诊断参考；`backend='numba'` 使用模型的
Numba 后端及可组合的数组诊断内核。广义对角化仍是 SciPy/LAPACK。
低层入口 `tbkit.generalized_kernels.generalized_diagnostics` 仅接受已经验证且分别缩放
的有限 H/S 数组和一致单位的 E/C，可直接嵌套在用户 `@njit` 中；不传入 Python 模型对象。

```bash
PYTHONPATH=python python python/examples/nonorthogonal_reliability.py --output work/nonorthogonal
NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONPATH=python python python/benchmarks/generalized_diagnostics.py
```

研究例子输出 PNG/SVG、NPZ 和 JSON，分别展示能量单位与基底病态程度；原始数据不受
画图下限影响。固定官方 MATLAB 的 Umat 换基错误采用原始表达式级 Octave 验证。
证明、适用边界与全部验证记录见 [物理审计](audit/NONORTHOGONAL-REVIEW.zh-CN.md)。
PR #10 的共享 Frobenius 厄米检查与本轮互补，尚未合并；本分支只对非正交调用显式
要求零绝对容差，不复制或覆盖其全局稠密/稀疏实现。
