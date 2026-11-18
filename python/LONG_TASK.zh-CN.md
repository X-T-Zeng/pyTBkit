# TBkit Python 长线迁移工作记录

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

## 2026-11-18：非正交谱可靠性

本轮从 main 独立分支接续。新增普通正定 Gram 谱的只读诊断、条件预算、
无量纲残差与 metric 门；band_gradient 默认按本点带宽分辨简并，显式绝对阈值保留。
研究例子分别检验单位缩放与近相关基底，Numba 内核和完整求解流程分别计时。
官方 EIGENCAR_gen 的静态 Umat 原始块在 Octave 复现了不同时变换 S 的谱改变；
未新增未经完整类验证的 MATLAB 补丁。审计见 audit/NONORTHOGONAL-REVIEW.zh-CN.md。

PR #10 的共享厄米检查仍待合并；本分支不覆盖它。下一轮继续正定 Gram 的输入误差
传播、方向简并微扰、非正交连接与速度/响应，并核查 H/S 晶格元数据容差。

本轮独立分支178项测试、12步通过；与PR #2–#10临时组合375项、27步通过。
诊断内核约3倍于NumPy参考，完整广义谱流程速度接近；两者分开记录，保留慢的首版基准。
完整引用提交和源码hash、官方原始Umat反例、适用边界均已存档。
