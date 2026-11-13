# TBkit Python 0.2

这是原仓库内的原生 Python 实现，不调用 MATLAB。按 `TBkit → HR / HK / Htrig` 分层，提供从建模、Wannier 插值、能带、拓扑到简单输运的可执行研究路径。默认 Numba 后端；NumPy 后端用于交叉验证和适合 BLAS 的负载。**这是经过测试的核心版本，尚未完整移植原仓库所有类和方法。** 详细范围见 `PORTING.md`，当前完整测试及实测性能见 `VALIDATION.zh-CN.md`。

## 安装与运行

从 TBkit 仓库根目录：

```bash
source codex-env/activate.sh
python -m pip install -e './python[test,plot,crystal]'
export NUMBA_NUM_THREADS=2
python python/tools/build_fortran.py  # 可选 Fortran/OpenMP
python -m pytest python/tests -q
python python/examples/research.py
python python/benchmarks/benchmark.py
```

新机器可直接执行 `bash python/install.sh`，完成隔离环境、Python 包、Fortran 后端和完整验证。将此命令作为仓库根目录的 Cloud Install script 即可。只准备基础环境时运行 `bash codex-env/install.sh`。前一阶段的 `codex-env/requirements.lock` 固定了已验证的数值依赖。使用 `pip install --no-deps -e ./python` 可保持这些依赖版本不变；可编辑安装的构建工具仍需本地可用或能够下载。无 Octave 时 MATLAB 数值对照测试会明确标为 skipped，其他测试照常执行。示例输出到根目录 `work/python-example/`。

```python
import numpy as np
from tbkit import models, kmesh, chern_number, berry_curvature

model = models.qi_wu_zhang(mass=-1)
k = kmesh([61, 61])
energies = model.bands(k, backend="numba", chunk_size=1024)
omega = berry_curvature(model, k)
print(chern_number(model, (31, 31)))  # +1，采用本文约定
```

自定义 HR：

```python
from tbkit import HR
# H(k) = H0 + T exp(+ik) + T† exp(-ik)
h0 = np.array([[0, .6], [.6, 0]])
t = np.array([[0, 0], [1., 0]])
model = HR([[0], [1], [-1]], [h0, t, t.T])
e, u = model.bands([[0.], [np.pi]], eigenvectors=True)
finite_sparse = model.cut([100])
supercell = model.supercell([3])
# 模型编辑返回新对象；不会污染其他模型的编译数组
modified = model.with_hopping([0], np.diag([.1, -.1]))
```

Wannier90：

```python
from tbkit import HR, write_wannier90, load_hdf5
# lattice 为实空间基矢（按行），单位自行统一（通常 Angstrom）
model = HR.from_wannier90("wannier90_hr.dat", lattice=np.eye(3))
energies = model.bands([[0., 0., 0.], [.1, 0., 0.]])
model.save("model.h5")
restored = load_hdf5("model.h5")
write_wannier90(restored, "normalized_hr.dat")
```

`hr.dat` 不包含晶格与轨道中心，不能从文件中恢复它们。未提供 lattice 时使用单位矩阵；degeneracy 在导入时除入跃迁，导出时统一写为 1。`tbkit.io.apply_wsvec(model, path)` 可将 `wsvec.dat` 的轨道对位移修正展开到新的实空间跃迁；输入 degeneracy 已在读入 HR 时吸收。缺少非零跃迁的修正记录会报错。

HK 与符号建模：

```python
import sympy as s
from tbkit import HK
kx, ky, m = s.symbols("kx ky m", real=True)
h = s.Matrix([[m, kx-s.I*ky], [kx+s.I*ky, -m]])
kp = HK.from_sympy(h, (kx, ky), parameters={m: .2})
print(kp.bands([[.1, .2]]))
lattice_model = kp.kp2tb(spacing=1.)
# HR.to_hk(order, center) 生成 q=k-center 的 Taylor 模型
```

`kp2tb` 是总阶数不超过 2 的中心差分晶格正则化，不是全 BZ 精确等价变换；对线性项用 sin，对平方项用 2(1-cos)，混合项使用中心差分乘积。HR↔Htrig 是可公度频率下的精确变换。

## 接入自己的 Numba 程序

Python 模型对象不直接传入 `@njit`，将连续数组传给公开 `tbkit.kernels` 即可；这些函数可以在 nopython 模式中调用。以下路径已纳入测试：

```python
from numba import njit, set_num_threads
from tbkit import kernels

set_num_threads(2)
```

完整调用示例：

```python
@njit
def occupied_energy(k, vectors_cart, hopping):
    h = kernels.fourier(k, vectors_cart, hopping)
    total = 0.0
    for q in range(len(k)):
        total += np.linalg.eigvalsh(h[q])[0]
    return total / len(k)

value = occupied_energy(k, model.cartesian_vectors, model.hopping)
```

可调用的内核包括 `fourier`、`fourier_gradient`、`polynomial`、`polynomial_gradient`、`trig`、`trig_gradient`。它们使用 complex128/float64，`nogil=True`、`cache=True`，按 k 点 `prange` 并行，每个线程独占输出，不使用改变求和顺序的并行归约，也不启用 fastmath。首次调用需要编译；不同维数的模型可能复用同一数组类型签名，数组 dtype/连续性/只读属性变化可能触发新签名。

Numba 的线程由 `NUMBA_NUM_THREADS` 或 `numba.set_num_threads()` 控制，**不由 OMP_NUM_THREADS 保证控制**。避免 Numba 线程和 BLAS 多线程同时占满机器；当前激活脚本把 BLAS 设为单线程。单 k 点任务或小批量可能被并行调度开销主导，可改用 `backend="numpy"`，或把 k 点合并批量执行。符号求值、文件 I/O、模型构造不放进 JIT 热路径。

## 数值与布局约定

- 索引从 0 开始。跃迁布局 `(nR, norb, norb)`，能谱 `(nk, norb)`，本征矢为列，导数 `(nk, dim, norb, norb)`；单 k 输入去掉最前面的批量轴。
- Fourier 采用 `exp(+i k·R)`。默认 Cartesian k，单位为弧度/长度。`HR.hamiltonian/gradient(..., coordinates="fractional")` 使用 `exp(+2πi k_frac·R_int)`，导数也是对相应输入坐标求导。通用 bands/topology 接口使用 Cartesian k。
- 轨道中心作为元数据保存；采用周期 orbital gauge，不自动乘入轨道中心相位。Wilson loop 的闭合连接也使用该 gauge。
- Hermitian 检查不通过就报错；`hermitize()` 是显式 `(H_R+H_-R†)/2` 投影，会改变非厄米输入，需由用户决定。
- 能量单位随模型（通常 eV）；`fermi` 采用 eV/K。导数是 dH/dk，不自动除以 ħ，Berry 曲率单位为长度平方，传输为无量纲。没有自旋简并的隐含乘 2。
- 单带 Kubo 曲率遇到简并报错；占据子空间 Chern/Wilson 用重叠矩阵并检查能隙与奇异链接。网格上未命中不等于排除所有能隙闭合，研究结果应检查网格收敛。
- Chern 仅适用于二维周期 BZ，符号为连接 `A=i<u|du>`；QWZ m=-1 的占据带为 +1。不要直接将非周期 k·p 模型的有限区域当作完整 BZ。
- `surface_green` 仅支持最近邻主层结构。`principal_layer(model, axis=..., transverse_fractional=...)` 可将长程跃迁自动分组为最近邻主层，再用 `chain_transmission` 验证理想导线。返回前检查 Riccati 方程残差，小主层在需要时做 Newton 修正；大主层若残差异常会报错，建议检查展宽与主层构造。`transmission` 接收已经嵌入 device 维数的 retarded 自能。

## 性能与内存

`bands` 分块生成 H(k)，将暂存量限制在 chunk_size；本征矢仍按调用要求返回。仅要部分本征值可用 `sparse_bands`（SciPy shift-invert），有限尺寸 `cut` 直接组装 CSR。`HR` 使用密集跃迁块；大轨道模型改用 `SparseHR`，其 CSR 跃迁、Numba COO 插值、超胞/slab/有限切片和 HDF5 保存均无需将跃迁密集化。显式请求完整 H(k) 或全部能谱仍会生成密集矩阵；部分谱应使用 sparse_bands。

基准记录第一次调用（可能含磁盘缓存载入）、预热后中位数、固定随机种子、线程数及误差。实际快照见 `VALIDATION.zh-CN.md`、`benchmarks/results.json` 和 `benchmarks/sparse-results.json`。Numba 在部分小轨道/稀疏负载上更快，密集负载可能由 BLAS/NumPy 胜出；共享云主机也有计时抖动。**没有 MATLAB 全库性能对比结论。**

继承原仓库 LGPL-2.1 许可；来源与对应算法见 `NOTICE` 和 `PORTING.md`。


## 0.2 新增研究流程

- `Structure`：POSCAR、周期邻居；`from_structure`：实 s/p/d Slater–Koster 建模。
- `SparseHR`：真正的 CSR 存储，稀疏超胞/slab/切片与部分本征谱。
- `NonOrthogonal`：Hc=ESc 及带梯度；不套用错误的正交基 Berry 公式。
- `finite_group/invariant_kp`：有限群与符号约束基底；可选 spglib 给出空间群、原子映射，`sewing_matrix/symmetrize_hr` 包含分数平移 Bloch 相位与 s/p/d 自旋表示。
- `geometric_tensors/response_kernel/integrate_bz`：Numba 曲率/度量/BCP、NCTE/SOAHC 和流式积分。
- `nested_polarization/mirror_chern/inversion_indicators/z2_wilson`：BBH、镜面拓扑、反演对称强/弱 TI、无反演二维 Z2；各自适用条件见函数文档。
- `landau_hamiltonian/plane_wave_hamiltonian`、`SpinModel`、`Hckt/Subckt`：Landau/连续模型、自旋和 RLC 数值核心。
- `read_eigenval/read_kpoints/read_procar/read_win/read_wout`：真实仓库 DFT 与 Wannier 文件的独立读入；PROCAR 保留电荷和三个自旋投影分量。
- `Qnum/BasisFunc/spherical_harmonic/coupled_basis`：精确量子数、角向基及 Clebsch–Gordan 耦合；径向波函数和一般晶体不可约表示投影不在此接口中。

所有模型热路径默认 Numba。HR 另有 `backend="fortran"`，先运行
`python python/tools/build_fortran.py` 编译；构建不在 import 时自动触发。
Fortran 阵列转换有开销，不保证优于 Numba，应按实际负载测量。

```bash
# 一次运行所有测试、研究例子、真实外部数据例子和性能基准
python python/tools/validate.py --output work/python-validation --benchmarks
```

长线状态与明确未完事项保存在 `LONG_TASK.zh-CN.md`。代码与 CI 已发布；
周期开发任务从 2026-11-14 起每天北京时间 08:00 附近接续，通过分支和 PR
供维护者审阅。它不是全天连续运行，也不代表全库迁移已完成。


MATLAB reference: BUAA-TCMP/TBkit @ `48e5217593598c8e1d634cdee3976e1dc6984516`; see `upstream.json`.

## 复用谱与几何：温度、化学势和多个响应通道

```python
from tbkit import prepare_spectral, prepare_geometry, response_from_geometry

spectral = prepare_spectral(model, k, backend="numba", frame="model")
geometry = prepare_geometry(spectral, gap_tol=1e-8, degeneracy="raise")
for temperature in [50., 200., 500.]:
    berry = response_from_geometry(
        geometry, [-.2, 0., .2], kind="berry", indices=(0, 1),
        temperature=temperature,
    )
```

同一批 k 点只准备一次谱和几何，后续加权不再次对角化。快照独立于源模型，
由调用方持有，只读且不自动更新。参数或基底改变后应重新准备。
所有快照数组保留批次维，单点准备的响应输出会去掉该维，兼容原单点 API。

`frame="point-orbital"` 显式选择点轨道嵌入；`frame="model"` 保留传入模型的框架。
对 HR 而言，后者是周期系数框架。两者不能混用，也不包含 hr.dat 中缺失的完整
Wannier 偶极矩阵。快照接口当前只适用于正交基；Fermi 权重使用 eV/K。

Numba 用户可直接将谱数组传入 `kernels.band_velocities` 和
`kernels.geometry_from_velocities`；两者都支持 nopython 调用。
大网格应在 `integrate_bz` 的每个小批次内准备/复用，快照不是磁盘或分布式缓存。

可执行例子：`python python/examples/response_sweep.py`。
有限 eta 审计、理论边界及复用基准见 `audit/RESPONSE-BATCH-REVIEW.zh-CN.md`。
