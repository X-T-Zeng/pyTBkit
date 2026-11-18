# TBkit Python 长线迁移工作记录

## 接续轮：最少生成元与有限群秩

`feature/group-minimal-generators`依赖尚未合并的群论PR #5，基线为`6720b6f`；
应先审阅父PR，再审阅此增量。新增精确BFS最小生成元搜索、`rank/is_generating`、
Numba纯数组闭包及预算异常。C6不可约简二元组与全局单生成元的反例已测试，
并用组合穷举、GF(2)维数和SymPy置换群独立验证。完整本分支177项测试、
所有例子及四项基准通过，见audit/generator-validation.json。
增量已提交PR #6；与PR #2/#3/#4及父PR #5临时自动组合后，220项测试、
全部例子及基准通过，见audit/generator-integration-validation.json。远端main未合并。

阶64的C2^6预制表上1,024次闭包与集合参考完全一致，预热中位数约17.7倍；
该结果不代表整体搜索或相对MATLAB的加速。首次JIT/缓存加载和线程见
audit/group-closure-benchmark.json。MATLAB本轮只追加generator/rank源码对照，
不新增候选补丁，不宣称完整运行其对象组合枚举。

下一步：同构/自同构、一般不可约表示生成、CG/重数空间基底，优先给出可独立验证的
小群及量子模型例子。仍需保留反幺正/投影表示、Bloch小群等未覆盖边界。

## 长线目标

目标：按 Python 架构复现原 TBkit 的研究功能，核心热路径可由 Numba
编译；模块完成后必须有可运行例子、解析/独立数值验证，适合时直接与
原 MATLAB 数值段作对照。不以文件数、空接口或别名数量充当完成度。

符号构建、解析推导和展示呈现是明确的核心要求。不能因为优化数值计算就省略
原库用于教学、公式展示和研究表达的功能。SymPy 是必需依赖；接续开发先读
[SYMBOLIC.zh-CN.md](SYMBOLIC.zh-CN.md) 的源码盘点与未完成清单，持续补齐
HR/Htrig 符号参数、Gamma 基、轨道/群表示及展示工作流，并保留精确性到显式数值化。

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
自动开发任务。当前本地161项测试已通过，远端 CI 状态应以实际运行结果为准。

## 接续轮：自动普通复不可约表示

`feature/automatic-finite-irreps`依赖群论PR #5（6720b6f），独立于最少生成元PR #6。
从复正则表示的中心分块和右作用重数空间提取全部普通irrep，返回明确的数值表示，
保留手工精确SymPy路径；不把浮点结果猜成精确根式。C3复共轭、Q8四元数型、
S4/A4/A5等独立字符判据及四面体量子模型已验证。
完整本分支178项测试、所有例子和四项基准通过，见audit/automatic-irrep-validation.json。
准备好的分块内核相对向量化NumPy约2.9倍；只适用于该负载，完整生成器未声称等比例加速。
测量线程、误差、JIT/缓存加载、完整生成耗时见audit/regular-block-benchmark.json。
已提交PR #7。补充Q8×C3真正二维复型回归；与PR #2/#3/#4/#5/#6临时组合，
保留双方新增接口/内核/例子并手动解决5处文本冲突后，237项测试、19个验证步骤通过。
精确提交及冲突处理见audit/automatic-irrep-integration-validation.json；远端main未合并。

下一步仍优先推进精确代数数域irrep、CG与重数空间基底；反幺正核心表示、投影因子系统、
Bloch小群不可直接套普通字符公式。MATLAB本轮为源码入口审阅，未新增补丁。

## 接续开发顺序与验收

**符号与呈现主线**：已建立独立 SymbolicHamiltonian、精确允许项输出、解析导数/
总阶数展开、Pauli 张量积分解、算符和基态 LaTeX/Jupyter 展示，以及符号到 Numba
的显式编译路径。尚未完成全部原符号模型操作、径向函数、Gamma/群表示展示。
每轮功能需同时检查参数可编辑、公式可读、物理正确与数值化一致。

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

## 2026-11-14：输运能量单位与因果性审计

独立分支 `fix/scale-invariant-transport`，保留尚未合并的符号PR #2与绘图PR #3。
已改为无量纲表面格林函数/Caroli求解、相对厄米/因果性检查，并将额外设备吸收
改成显式参数，避免隐含1e-12的能量尺度。加入解析单链/单能级、有限多轨道链、
基底旋转和能量原点平移的独立检验，以及可运行的缩放/吸收图例。
官方原始GW_iter的绝对阈值失效已单独执行复现，本轮未修改该MATLAB原式。
原式缺陷复现不能解释为原库正确；完整依据见 audit/TRANSPORT-REVIEW.zh-CN.md。
本分支完整验证161项通过、0失败/错误/跳过，所有例子成功；独立记录保存在
audit/transport-validation.json，不覆盖其他未合并分支的验证快照。本轮不比较性能。

下一步优先检查大主层小eta下的残差修正和系统eta收敛。当前大于16轨道的主层若
残差超限仍会明确报错；未完成所有输运算法、多端口和散射/寿命模型迁移。

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


## 2026-11-16：角动量与旋转基础审计

`fix/angular-rotation-invariants`直接基于main 13a5832，独立于未合并PR #2–#8。
修复近似半整数被误接纳、轴归一化量级依赖、隐藏小振幅剪枝、轨道正交性相对容差，
并用SU(2)半角4pi约化处理巨大有限角，保留2pi自旋负号。新增Numba纯数组生成器，
以及受校验的SymPy标量angular_cg准备入口，不构造不必要的完整高j矩阵。
原始Python与MATLAB先执行存档，再修复Python；原始高j阶乘溢出和Racah消减已独立复现，
没有新增高j MATLAB候选补丁，不把PR #8的小j修复外推为高j稳定性。
完整本分支178项测试、12个步骤（全部例子与四项基准）成功，见audit/angular-validation.json。
生成器构造相对向量化NumPy本次约2.9–8.9倍；只代表准备阶段，不是expm/CG整体加速。
JIT/缓存加载、预热、误差、线程及公共校验API计时见audit/angular-generator-benchmark.json。
已提交PR #9。与未合并PR #2–#8临时组合，保留全部符号呈现和数值功能后，
300项测试、23个验证步骤通过。分两步在六个共享文件解决文本冲突，详见
audit/angular-integration-validation.json。组合仅用于本地验证，不自动合并main。

下一步：稳健高j批量CG/3j算法与明确误差控制、Wigner–Eckart/选择定则；继续保留响应、
Wannier连接、输运等全库接续清单，不把本轮角动量范围当成全量迁移完成。

MATLAB reference: BUAA-TCMP/TBkit @ `48e5217593598c8e1d634cdee3976e1dc6984516`; see `upstream.json`.


## 2026-11-15：有限群CG与官方角动量CG审计

分支`feature/finite-group-clebsch-gordan`依赖PR #7（3096541），不改main与官方pin。
新增常量精确SymPy/NumPy/Numba的普通有限群CG及对齐的重数基底，符号H可保留自由参数。
D3解析系数、交换奇偶、E的三重副本、复基底、C3电荷及Q8反对称不变态已验证；
两个等价E可混合但每个能量仍二重简并，提供可运行推导与科研图。
官方角动量入口另复现参数数目及未定义输出两个错误，候选补丁只改调用形式/输出名，
358组精确系数、两个入口共716个数值已对照；不是完整Y_l__m或高j稳定性验证。
完整本分支204项测试、全部例子和五项基准通过，见audit/coupling-validation.json。
Numba内核减少重复写入，8/16/32维最后记录约2.26/2.03/1.46倍；全分解有时NumPy更快且计时有波动，
NumPy优化路径仍保留。初始32维曾较慢，快照也保留，见audit/matrix-unit-benchmark.json。
PR #8与PR #2/#3/#4/#5/#6/#7临时组合后，263项测试与21个验证步骤（含6项基准）通过。
五处文本冲突处理及精确提交见audit/coupling-integration-validation.json。
已兼容Octave 8/10的未定义输出错误措辞，并让验证失败尾日志进入CI控制台；科学容差不变。
临时merge验证后撤销，不自动合并main。

接续优先：规范固定/6j或Wigner–Eckart，反幺正核心表示及物理选择定则；并继续独立审阅
响应处方、Wannier连接与输运等全库清单。不要把普通有限群CG视为全部角动量/空间群迁移。

## 最近一轮：谱与响应复用

已验证显式只读 SpectralBatch/GeometryBatch、Numba 可组合数组内核、mu/T/张量扫描
复用和点轨道框架选择；保持单次API兼容。有限eta的官方NCTE路径已复现其复数和
全带曲率和偏离标准几何的行为，未擅自引入散射模型或取real替代物理推导。
详见 audit/RESPONSE-BATCH-REVIEW.zh-CN.md；下一步应继续核查响应处方与单位，
以及完整Wannier位置连接/非正交响应，而非直接复制全部公式。

## 绘图质量要求

美观是默认要求：统一字体/数学排版、配色、线宽、留白、分图编号与图例，
同时保证物理单位、正负号、能量参考和归一化清楚。每类图应实际生成并查看，
导出论文用PDF/SVG与高分辨率PNG；不隐式平移、平滑、连接断带或裁剪物理特征。
公共样式与图例见 [PLOTTING.zh-CN.md](PLOTTING.zh-CN.md)。
符号示例、轨道/自旋/拓扑/输运等后续可视化也按同一规范验收；本轮统一
主线bandplot、研究例子与DFT例子，不代表全部原MATLAB绘图功能已迁移。

## 2026-11-17：谱检查的尺度与基底审计

直接基于最新 main，在独立分支修正默认固定绝对厄米容差接受小尺度错误模型的问题。
稠密/稀疏谱、非正交 H/S 与导数共享逐矩阵 Frobenius 判据；Numba 纯数组诊断内核
可组合调用，包含次正规和极大有限分量的数值保护。独立两带/Gram 反例和可运行绘图例子
见 audit/HERMITICITY-REVIEW.zh-CN.md，原始 MATLAB 重叠表达式单独验证，无新增候选补丁。

下一轮优先：审计正定但病态的 Gram 矩阵、非正交带梯度/简并阈值的量纲与条件数，
随后再扩展一般非正交几何。群论、符号展示和角动量相关 PR 仍须审阅后合入，不能把
临时组合测试当作已发布。继续核查其余模块的隐藏绝对阈值，不重复宣称所有模块完成。

本轮独立分支179项测试、12步验证通过；与PR #2–#9的临时组合338项、25步通过。
报告含完整提交与源码hash。基准保留云端墙钟波动和较慢结果，不用选择性最快值
宣称固定加速；稳定资源上的大矩阵后端调度仍待实测。

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


## 2026-11-18：维护者授权统一集成

维护者本次要求自动处理所有现有PR，故将PR #2–#11统一集成，在同一个确切版本
复跑375项测试、27个步骤并记录107个Python源码文件hash。所有原始PR提交历史保留，
通过集成PR的远端CI后使用GitHub合并接口更新main；不强制推送main。
源PR清单、原CI、依赖和测试见audit/integration-prs-20261118.json及integration-validation-20261118.json。

本次授权解决当前积压PR；周期开发继续按每轮明确范围、物理验证和可审阅PR交付，
后续自动化的合并策略以维护者的新指示为准。历史审计中的“待合并”是当时状态，
当前集成状态以本节、GitHub PR记录和最新源清单为准。
