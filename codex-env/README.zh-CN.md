# Codex Cloud 科学计算环境（Linux x86_64）

在仓库根目录运行：

```bash
bash codex-env/install.sh
source codex-env/activate.sh
bash codex-env/test.sh
```

将 Codex Cloud 的 Install script 设置为 `bash codex-env/install.sh`（工作目录为 TBkit 仓库根目录）。任务中的 shell 应先 `source codex-env/activate.sh`；安装脚本里的激活不会传播到新的 shell。仓库内脚本和锁文件可纳入版本控制；本次没有发布云环境快照或推送 GitHub。

## 隔离与依赖

没有 sudo 的 Linux 云环境使用用户级 micromamba，工具链装在 `.toolchain/`，Python 使用独立 `.venv/`，缓存和构建位于 `work/`；均不写系统 Python 或 shell 启动文件。安装器固定 micromamba 2.10.0，工具链使用 conda-forge。通过初始化本地空前缀并使用 install，避免向 `~/.conda/environments.txt` 注册。首次求解后保存显式软件包 URL 锁文件 `toolchain-linux-64.lock`，后续新建工具链从该文件恢复。Python 使用 `requirements.lock` 固定所有解析后的版本；输入依赖见 `requirements.in`。锁定软件包仍依赖相应下载源持续可用。

包括 Python 3.12、NumPy、SciPy、Numba、SymPy、Matplotlib、h5py、pytest、pandas、mpmath、threadpoolctl、Meson、Ninja；GNU C/C++/Fortran、CMake、Make、pkg-config、OpenBLAS、BLAS、LAPACK、FFTW、OpenMP 及 GNU Octave。GNU 编译器使用 conda 平台前缀（`$CC`、`$CXX`、`$FC`），不要求系统存在名为 gfortran 的命令。

激活默认 OpenMP、BLAS 单线程，固定 Python 哈希种子，Matplotlib 使用无界面的 Agg。测试显式用两个 OpenMP 线程；生产计算可以设置 `OMP_NUM_THREADS`，避免 BLAS 与 OpenMP 嵌套超额并行。未启用 `-ffast-math` 或针对宿主 CPU 的 `-march=native`。容差验证不承诺跨 CPU/编译器的位级一致；随机研究计算应另外显式保存 RNG 种子、输入和运行参数。

实际执行结果见 [TEST-REPORT.zh-CN.md](TEST-REPORT.zh-CN.md)。工具链固定基础 x86_64 microarch 构建，避免求解器按当前 CPU 选择 AVX2/AVX-512 专用库。

## 验证范围

- Fortran 独立编译，LAPACK 对角化，OpenMP 两线程求和。
- C/C++ 用 CMake/Ninja 编译，CTest 验证 FFTW 与 C++ 程序。
- NumPy f2py 使用 Meson 编译 Fortran OpenMP 扩展并由 Python 调用；两带模型能谱与 NumPy、SciPy 本征值交叉验证。
- SciPy 与 SymPy 积分、Numba JIT、HDF5 读写、FFT 及无界面 PNG 输出。
- Octave 执行 `tests/smoke.m`，验证复 Hermitian 两带模型、积分和 FFT。

TBkit 为 MATLAB 框架。本测试只证明这些兼容语法的最小 `.m` 脚本可运行，未验证整个 TBkit；MATLAB classdef、Symbolic Toolbox、图形及较新 MATLAB API 可能需要移植。没有下载、安装或运行商业 MATLAB。当前测试也不代表已验证量子输运或 Wannier 插值的完整研究算法。

## 权限与网络

当前系统没有 sudo 和 scoped package installer，因此采用上述实际用户级安装路径。需要 HTTPS GitHub 及其 release 重定向域、conda.anaconda.org、PyPI 与 files.pythonhosted.org。保留平台代理和 CA 配置，不关闭证书校验。遇到明确网络策略拒绝时，应在云环境设置中放行所需源，或在联网且相同 Linux 平台的机器准备软件包缓存/离线镜像及对应锁文件；无网络且无缓存时无法完成首次安装。

有系统权限的 Debian/Ubuntu 环境可以改用管理员提供的系统依赖，再独立创建 Python venv：

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-dev build-essential gfortran cmake ninja-build pkg-config libopenblas-dev libblas-dev liblapack-dev libfftw3-dev octave git curl
python3 -m venv .venv
.venv/bin/python -m pip install -r codex-env/requirements.lock
```

此为替代路径，不与当前工具链混用；系统 Python 版本需与锁文件支持范围一致，使用系统编译器时也需调整激活脚本。云平台若提供 `cloud-agent-packages`，优先使用该受限安装器而非直接 apt。
