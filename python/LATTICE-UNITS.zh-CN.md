# 晶格、近邻与长度单位

pyTBkit 的数值晶格使用**行向量**，不隐含将 Å 转为米。所有长度数据采用共同单位，
Cartesian k 采用对应逆长度单位。若晶格乘 alpha，Cartesian k 要除 alpha；
分数 k、分数原子位置和整数晶胞像不变，cutoff 与显式距离 tol 同样乘 alpha。
Htrig 的 frequencies 是 Cartesian 长度向量，也要同步变换。

```python
import numpy as np
from tbkit import Structure
from tbkit.slater_koster import from_structure
from tbkit.lattice import reciprocal_lattice

a = 1e-10  # 这里自行选择米；能量积分 t 仍使用调用者的能量单位
structure = Structure(a*np.eye(3), [[0, 0, 0]], ['X'])
bonds = structure.neighbors(1.1*a)
assert len(bonds.i) == 6
model = from_structure(structure, [['s']], 1.1*a, {'ss_sigma': 1.})
B = reciprocal_lattice(structure.lattice)
energy = model.bands(np.array([[.1, .2, .3]]) @ B)
```

| 接口 | 当前约定 |
| --- | --- |
| `Structure.neighbors(cutoff)` | 默认 `tol=None` 为 cutoff 的 `1e-10`，返回 `tol<d<=cutoff+tol` 的有向键 |
| `neighbors(..., tol=value)` | 显式 tol 是调用者的绝对长度单位，保留此语义 |
| `neighbors(..., max_images=n)` | 分配前限制复制的原子像，默认2,000,000；超限报错 |
| `structure.volume` | 正体积，单位为长度的三次方；不能有限表示时明确报错 |
| `structure.log_volume` | 对同一数值体积取自然对数，适合极端量级的记录 |
| `integrate_bz(..., normalization='average')` | 每胞 BZ 平均；不需要构造体积因子 |
| `normalization='states'` | `integral d^d k/(2*pi)^d`，常数1为 `1/volume` |
| `normalization='integral'` | Cartesian `d^d k` 测度，常数1为 `(2*pi)^d/volume` |

近邻搜索内部改变计算尺度，输出的位移与距离仍是原长度单位。原子不自动包回晶胞，
整数像标签保持原有含义；默认仍有分辨阈值，需更小阈值时显式设置 tol。
距离依赖的 SK 参数回调收到原单位的距离，其参数单位也须一致转换。

构造器改用相对数值秩检查，没有绝对行列式阈值。复数、非有限、奇异或数值秩亏的
晶格拒绝；不自动正交化或改变胞方向。共享 Bloch 基底的晶格使用 `1e-12` 相对
Frobenius 比较，不能直接合并近似但不同的原胞，也不猜测不同单位或等价换基。
极端病态胞、不可表示的倒格矢/绝对测度仍需要换单位、基底或精度研究。

科研例子：`PYTHONPATH=python python python/examples/lattice_units.py`。
[物理反例、官方源对照与验证边界](audit/LATTICE-REVIEW.zh-CN.md)。
