# pyTBkit

以 **[BUAA-TCMP/TBkit](https://github.com/BUAA-TCMP/TBkit)** 为 MATLAB 蓝本，
采用 Python 架构实现的科学计算工具包。核心数组内核可接入 **Numba nopython**，
并提供 NumPy、稀疏计算与可选 Fortran/OpenMP 后端。

目标是成为多功能、好用、计算高效且物理结果可信的研究工具，而非逐行翻译 MATLAB。
精确符号建模、解析推导与科学展示是核心功能，和 Numba 数值计算共同发展。
**目前是持续开发的研究核心版本，不宣称已完成原库全部 API，也不笼统声称快于 MATLAB。**

## 安装

Python 3.12 为当前验证环境。安装 Python 包与可选科学工具：

```bash
git clone https://github.com/X-T-Zeng/pyTBkit.git
cd pyTBkit
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,plot,crystal]'
```

在支持的 Linux x86_64 云环境完整安装隔离工具链、Octave、科学依赖和可选 Fortran 后端：

```bash
bash python/install.sh
```

此脚本使用用户级工具链，不安装商业 MATLAB。固定的 Python 依赖版本位于
`python/requirements-tested.txt`；数值一致性不代表跨 CPU 位级一致。

## 快速例子

```python
from tbkit import models, kmesh, chern_number

model = models.qi_wu_zhang(mass=-1)
k = kmesh([61, 61])
energy = model.bands(k, backend="numba", chunk_size=1024)
print(chern_number(model, (31, 31)))  # +1，A=i<u|du> 约定
```

主要功能包括 HR/HK/Htrig、稀疏模型、Wannier 插值、结构/SK、空间群与
nonsymmorphic sewing、对称性约束 k·p、谱/导数/拓扑、响应积分、主层输运、
连续模型、角动量、自旋及 RLC 数值核心。适用条件和未完功能见迁移清单。

## 验证

```bash
export NUMBA_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python python/tools/prepare_upstream.py  # 获取固定官方版本和独立补丁副本
python python/tools/build_fortran.py     # 需 Fortran/Meson/Ninja；可选后端
python -m pytest python/tests -q
python python/tools/validate.py --output work/python-validation --benchmarks
```

完整验证需 Octave、Fortran 和 crystal/plot/test 依赖。没有 Octave/Fortran 时对应
单测明确跳过；完整验证脚本的高级例子要求这些后端，不能据部分测试声称全通过。

参考库固定为官方提交 `48e5217593598c8e1d634cdee3976e1dc6984516`；原始源码与
建议补丁分开保存，避免“修改参考后再证明一致”。不随 latest 自动漂移。

## 文档与维护

- [使用和物理约定](python/README.zh-CN.md)
- [符号建模、公式展示与迁移清单](python/SYMBOLIC.zh-CN.md)
- [迁移范围与边界](python/PORTING.md)
- [物理与代码审计](python/audit/PHYSICS-AUDIT.zh-CN.md)
- [长线目标与接续清单](python/LONG_TASK.zh-CN.md)
- [验证记录](python/VALIDATION.zh-CN.md)
- [协作方式](CONTRIBUTING.md)

Python 是后续开发主线。所有物理修复先有独立推导或反例、回归测试与适用条件，
再比较性能；多方改动通过分支和 PR 审阅。CI 支持 push/PR、手动与每周回归。

采用与原库一致的 LGPL-2.1 许可证；来源与历史上下文见 `python/NOTICE`。
