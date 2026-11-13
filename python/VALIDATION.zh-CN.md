# 当前验证快照：pyTBkit 0.2.1

独立仓库 `/workspace/pyTBkit` 完整验证结果：**133 passed，0 failures，0 errors，0 skipped**。
Python 3.12.15；NumPy/Numba/稀疏及已编译 Fortran/OpenMP 后端；Octave 实际执行。
测试、6组基础研究例子、绘图、真实官方 DFT/Wannier 文件例子、高级研究例子均成功。
原始结果：validation-snapshot.json；完整日志：work/python-validation/。

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
