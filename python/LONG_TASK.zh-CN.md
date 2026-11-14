# TBkit Python 长线迁移工作记录

## 接续轮：最少生成元与有限群秩

`feature/group-minimal-generators`依赖尚未合并的群论PR #5，基线为`6720b6f`；
应先审阅父PR，再审阅此增量。新增精确BFS最小生成元搜索、`rank/is_generating`、
Numba纯数组闭包及预算异常。C6不可约简二元组与全局单生成元的反例已测试，
并用组合穷举、GF(2)维数和SymPy置换群独立验证。完整本分支177项测试、
所有例子及四项基准通过，见audit/generator-validation.json。

阶64的C2^6预制表上1,024次闭包与集合参考完全一致，预热中位数约17.7倍；
该结果不代表整体搜索或相对MATLAB的加速。首次JIT/缓存加载和线程见
audit/group-closure-benchmark.json。MATLAB本轮只追加generator/rank源码对照，
不新增候选补丁，不宣称完整运行其对象组合枚举。

下一步：同构/自同构、一般不可约表示生成、CG/重数空间基底，优先给出可独立验证的
小群及量子模型例子。仍需保留反幺正/投影表示、Bloch小群等未覆盖边界。

目标：按 Python 架构复现原 TBkit 的研究功能，核心热路径可由 Numba
编译；模块完成后必须有可运行例子、解析/独立数值验证，适合时直接与
原 MATLAB 数值段作对照。不以文件数、空接口或别名数量充当完成度。

## 当前交付

0.1 实现 HR/HK/Htrig、Wannier、基本拓扑/输运与 Numba 后端；
0.2 扩展结构/SK、稀疏存储、广义谱、对称性/约束建模、响应积分、
高级拓扑、Landau/平面波、自旋与电路数值核心。具体边界见 PORTING.md。

本记录是可恢复的工作清单，不代表全天不间断运行或承诺一周内完成。

代码已发布至 https://github.com/X-T-Zeng/pyTBkit 。2026-11-13 已启用
“pyTBkit 物理审计与开发”周期任务：从 2026-11-14 起每天北京时间 08:00
附近接续一轮；实际启动时间可浮动。任务从最新 main 和未合并 PR 接续，
在独立分支中开发并提交 PR，不自动合并，不强制推送 main。
若环境、权限或资源阻断运行，任务应报告阻断，不得虚报已完成。

GitHub Actions 的 push/PR/手动/每周回归是另一条验证流程，不等同于
自动开发任务。当前本地141项测试已通过，远端 CI 状态应以实际运行结果为准。

## 接续开发顺序与验收

1. **晶体对称性接线（数值核心已验证，完整接口待补）**：已完成真实金刚石
   nonsymmorphic Bloch sewing、s/p/d 自旋表示、HR 对称化及非零 k 测试。
   继续补高角动量、磁空间群及原仓库所有表示接口。
   群论基础已另建 `AbstractFiniteGroup/FiniteGroup/MatrixRepresentation`，覆盖
   群公理、子群/陪集/商群、普通特征标及投影，见 GROUPS.zh-CN.md。
   本分支完整161项测试及全部例子通过，独立快照见audit/group-validation.json；
   不覆盖其他未合并PR的验证记录。本轮未比较性能。
   后续优先补一般不可约表示自动生成、投影后的重数空间分离/CG系数、
   双群因子系统与反幺正核心表示；不将现有Hamiltonian相位商当作完整表示。
2. **外部数据闭环（主要格式已验证，变体待补）**：POSCAR/KPOINTS/EIGENVAL/
   非共线 PROCAR、Wannier win/wout 已用真实仓库文件验证。继续扩展格式变体、
   vasprun.xml、DFT 拟合全流程与更大真实材料的 golden fixtures。
3. **全量模型操作（部分完成）**：已完成稀疏 supercell/slab/切片、主要模型
   HDF5 保存和线性代数。继续补所有原符号参数/模型编辑接口与真实大模型基准。
4. **拓扑扩展（部分完成）**：已实现无反演二维 Wilson Z2，并测试自旋混合
   QSH/平庸模型。继续补一般三维组装、更高维 nested flow 和所有对称指标。
5. **输运与非线性响应（部分完成）**：已实现长程跃迁自动主层化及理想链
   透射验证。继续补多端口、所有响应公式/单位和真实体系的收敛与守恒检查。
6. **附属模型族（部分完成）**：高角动量符号基、Vampire、SPICE/HSPICE
   兼容、原 SpinModel/Hckt/Braid 的完整接口与示例。商业软件本体不自动安装。
7. **性能后端（部分完成）**：基于真实研究负载选择 NumPy/Numba/Sparse，
   已验证 HR 的可选 Fortran/OpenMP 后端；继续扩展调度与线程策略；所有加速先验证误差再报告计时。
8. **发布验收（待完成）**：原仓库研究示例逐项迁移、外部文件 fixture、
   文档与 API 对应表、兼容边界、CI 全通过后才标记完整版本。

## 恢复和验证命令

```bash
cd /workspace/pyTBkit
source codex-env/activate.sh
bash python/install.sh
python python/tools/validate.py --output work/python-validation --benchmarks
```

结果保存在 `work/python-validation/validation.json`、JUnit、各步骤日志、
例子数据和图像中；不依赖聊天记录记住测试结果。

若验证失败，先修复或明确归因后更新验证记录。存在 pending 功能时不得
将此清单或全库迁移状态改为 completed。


MATLAB reference: BUAA-TCMP/TBkit @ `48e5217593598c8e1d634cdee3976e1dc6984516`; see `upstream.json`.


## 最近一轮：谱与响应复用

已验证显式只读 SpectralBatch/GeometryBatch、Numba 可组合数组内核、mu/T/张量扫描
复用和点轨道框架选择；保持单次API兼容。有限eta的官方NCTE路径已复现其复数和
全带曲率和偏离标准几何的行为，未擅自引入散射模型或取real替代物理推导。
详见 audit/RESPONSE-BATCH-REVIEW.zh-CN.md；下一步应继续核查响应处方与单位，
以及完整Wannier位置连接/非正交响应，而非直接复制全部公式。
