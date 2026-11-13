# 实际验证结果

验证日期：2026-11-13（Asia/Shanghai）。仓库：X-T-Zeng/TBkit；宿主 Debian 13，Linux x86_64。

已在本次云环境完成用户级安装及测试，所有测试命令退出码为 0。

| 项目 | 结果 |
| --- | --- |
| Python 独立 venv | `/workspace/pyTBkit/.venv`，Python 3.12.15 |
| 科学计算库 | NumPy 2.2.6、SciPy 1.15.3、Numba 0.61.2、SymPy 1.14.0、Matplotlib 3.10.3、h5py 3.13.0、pytest 8.3.5 |
| 原生工具链 | GCC/G++/GNU Fortran 16.2.0；CMake 4.2.3 |
| GNU Octave | 10.3.0 |
| Fortran | LAPACK 本征值、OpenMP 实际 2 线程及求和通过 |
| C/C++ | CMake/Ninja 构建成功，FFTW 与 C++ 两项 CTest 全部通过 |
| Python–Fortran | f2py + Meson 构建 OpenMP 扩展成功，Python 调用及两带模型能谱交叉验证通过 |
| Python 数值与 I/O | pytest：3 passed；覆盖 SciPy/SymPy 积分、Numba、HDF5、FFT、PNG |
| Octave `.m` | 复 Hermitian 本征值、数值积分、FFT 全部通过 |
| Python 依赖一致性 | `python -m pip check`：No broken requirements found |
| 工具链锁文件 | 有效 `@EXPLICIT` 格式及 MD5；对空前缀离线 dry-run 成功，缓存命中，0 B 下载；未进行第二份完整安装 |
| 脚本检查 | bash 语法检查、git diff --check 通过 |

为提高云主机间的可移植性，已将 `_x86_64-microarch-level` 约束为 `1=5_x86_64`，替换初始求解得到的高级 CPU 构建，并在替换后重新执行全部测试。锁文件保留替换后的基础 x86_64 构建。仍需 Linux x86_64、兼容 glibc 和软件包源；不是跨操作系统锁文件。

测试有 14 条 Matplotlib 3.10.3 使用旧 pyparsing API 产生的弃用警告，未影响结果，未将其隐藏。无界面环境运行 Octave 时关闭 GUI 属正常行为。

系统没有 sudo 和 cloud-agent-packages，系统包安装路径不可用。实际采用仓库内 micromamba 工具链与单独 venv；只读主目录缓存和 Conda 注册问题已在脚本中通过重定向缓存、初始化本地前缀解决。没有修改 HOME 或系统 Python，也未关闭 TLS 校验。当前网络策略允许实际所用下载源。

未安装商业 MATLAB。未声称整个 TBkit 在 Octave 下兼容，也未验证完整 Wannier/输运工作流。未修改 GitHub 远端或发布云环境快照；当前接口没有保存云环境 Install script 设置的工具，请将 `bash codex-env/install.sh` 填入云环境设置。

完整本地安装及测试输出分别在忽略版本控制的 `work/toolchain-install.log`、`work/install.log`、`work/test.log`；工具链可移植性调整及锁文件预演记录在 `work/portability-install.log`、`work/replay.log`。
