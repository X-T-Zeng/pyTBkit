# 当前验证快照：pyTBkit 0.2.1

## 2026-11-21：点轨道规范与模型代数分支

从 main a1f1102 独立修复并新增26项回归，完整 **401 passed，0 failures/errors/skipped**，
29个步骤全部通过（含所有研究例子和10项基准）。Fortran与Octave实际执行；
只有一条Matplotlib/Pyparsing依赖弃用警告。报告及10个被测文件hash见
`audit/gauge-validation.json`，原始日志在 `work/gauge-validation/`。
当前分支未包含仍待审阅的长度单位PR #13（其独立424项测试/CI success）；
401与424属于不同分支，不能相加当作已合并main的完成度。
例子、Numba/NumPy计时和适用边界见 `audit/ORBITAL-GAUGE-REVIEW.zh-CN.md`。
独立代码提交eeac457的GitHub CI已success。与PR #13的本地临时组合450项/31步
也全部通过，117个Python文件hash见 `audit/gauge-lattice-integration-validation.json`；
此结果不代表main已合并两轮改动。

## 2026-11-18：PR #2–#11 集成快照

历史集成版本：**375 passed，0 failures，0 errors，0 skipped**；27个步骤全部通过，
包含全部研究例子与9项基准，仅1条Matplotlib/Pyparsing依赖弃用警告。Fortran/OpenMP
后端和Octave参考继续执行。数值源码与此前通过375项测试的组合逐文件一致（107个Python文件）。
当前完整报告见 `audit/integration-validation-20261118.json` / `validation-snapshot.json`，
源PR与提交清单见 `audit/integration-prs-20261118.json`；完整日志位于 `work/merge-all-validation/`。
以下分支记录是历史测量，不代表当前测试数量，也不代表全库880个MATLAB文件已完成迁移。

## 历史符号分支记录

符号功能分支完整验证结果：**161 passed，0 failures，0 errors，0 skipped**。
Python 3.12.15；NumPy/Numba/稀疏及已编译 Fortran/OpenMP 后端；Octave 实际执行。
测试、6组基础研究例子、绘图、真实官方 DFT/Wannier 文件例子、高级研究例子均成功。
原始结果：validation-snapshot.json；完整日志：work/symbolic-validation/。

MATLAB 基线严格固定为 BUAA-TCMP/TBkit @
48e5217593598c8e1d634cdee3976e1dc6984516（2026-07-12），880 个源文件。
官方原始 checkout 和候选补丁副本分开保留；参考下载、重复初始化、
拒绝覆盖修改过的参考库都有回归验证。补丁未推送到官方 MATLAB 仓库。

本轮新增重点：C3/C4/C6、C3v+T 三阶5参数表面空间与BHZ二阶7参数空间的
完整张成空间检验；反幺正/反对称性和群投影；近整数胞矢截断；晶格元数据；
非厄米导数与简并 HF；Wilson 端点规范、点轨道位置与边界 sewing；Fermi 双尾；
官方 FFT 缓存、applyRU/applyU 数值分支与第4输出位置速度符号。

部分测试明确重现原始官方缺陷，再验证候选补丁，不能把绿色测试解释为
官方库无缺陷或全库迁移已完成。完整证据与假设在 audit/PHYSICS-AUDIT.zh-CN.md。

性能快照在 benchmarks/；这些历史测量不是本次跨机器安装测试的新基准，
不能据此宣称全面快于 MATLAB。完整验证器可以用 --benchmarks 重新测量。


## 接续验证：共享谱与响应扫描

SpectralBatch/GeometryBatch 准备和复用、随机本征态相位不变性、点轨道框架、
解析 Dirac 度量/曲率、简并策略与用户 @njit 管线测试通过。官方 NCTE 有限 eta
路径由原式执行与独立带对求和刻画；干净极限匹配，复响应处方未自动移植。

完整例子与性能记录：examples/response_sweep.py、benchmarks/response_reuse.py、
benchmarks/response-reuse-results.json。8轨道/512k/9次响应，Numba 2线程、BLAS 1线程，
预热单次API工作流363.29ms、准备复用77.23ms，约4.70倍；两者均包括所需准备工作，
输出最大差异0。首次准备258.46ms可能含JIT/缓存载入；不是MATLAB对比结论。
完整日志在 work/spectral-batch-validation/，本次测试141项、全部例子及三组基准通过。


## 精确符号层与展示验证

本轮完整验证161项测试，0失败/错误/跳过；原有例子和新增符号例子全部通过。
新增20项覆盖精确参数/不可变矩阵、同时替换、总阶数展开与非零中心、TR/PH/C3/C4、
Pauli张量重构、公式导出、符号变量命名冲突、Numba nopython与原几何管线接线。
解析两带能谱最大差异2.22e-16，二阶局部模型动量减半的误差比7.997，符合三阶余项。

源码候选索引在固定官方880个源文件中定位252个相关文件，校验通过；这不是迁移比例。
CLI示例生成的公式/收敛图已查看；LaTeX源文件已导出，没有声称PDF或Jupyter前端已运行。
本轮未重新测量性能基准；前述性能数字仍为各自历史工作负载。
