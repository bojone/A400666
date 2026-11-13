# A400666：拼接数整除问题的 GPU 搜索

**中文 | [English](README.en.md)**

**整个项目完全由 Kimi K3.1 Pro [thinkg: max] + Kimi-Code 自主完成。**

本项目记录 [OEIS A400666](https://oeis.org/A400666) 中 `a(14)`、`a(15)`、`a(16)`、`a(17)` 的搜索、优化和独立验证。定义

$$c(n)=\mathrm{concat}(1,2,\ldots,n),\qquad a(n)=\min\\{k > n:c(n)\mid c(k)\\}.$$

例如 $`c(14)=1234567891011121314`$。程序从不实际构造巨大的 $`c(k)`$，而是在模 $`c(n)`$ 或其大素因子下递推。

## 结果

| 项 | 结果 | 搜索结论 |
|---|---:|---|
| $`a(14)`$ | `675924569493048966` | 17 位及以下无解；完整 18 位区间仅此一解 |
| $`a(15)`$ | `70685893246361615705` | 19 位及以下无解；20 位有三个解，取最小者 |
| $`a(16)`$ | `10152437877876943330588` | 22 位及以下无解；23 位有十个解，取最小者 |
| $`a(17)`$ | `553943104744829340555260` | 23 位及以下无解；完整 24 位区间仅此一解 |

$`a(15)`$ 的另外两个 20 位解为 `73394079098572402415`、`74341537919751115530`。$`a(16)`$ 的全部 10 个 23 位解见 [a16/results/d23.summary](a16/results/d23.summary)。$`a(17)`$ 的轨道坐标与完整覆盖证据见 [a17/results/](a17/results/)。四个答案都在完整复合模数 $`c(n)`$ 下通过余数为零的检查，而不只是在大素因子下检查。

前 13 个公布值也已复核整除性和最小性，结果见 [a14/oeis_results/verification.txt](a14/oeis_results/verification.txt)。

## 设备与时间

- GPU：8 张 NVIDIA GPU，架构由本机自动检测。
- CUDA：生产环境使用 CUDA 12.9，编译器路径为 `/usr/local/cuda-12.9/bin/nvcc`。
- CPU：共 288 线程；$`a(15)`$ 和 $`a(17)`$ 的 AVX-512 搜索均使用 272 个工作线程。
- $`a(14)`$：全量 GPU 搜索约 80 分钟。
- $`a(15)`$：约 69 小时，包含算法迭代、实验、搜索、CPU/GPU 交接和最终审计，并非一次不变的批处理耗时。
- $`a(16)`$：主流程用时 277.448 秒，包含续跑、设备初始化和最终审计；此前的 2–6 位试跑另耗时 69.346 秒。
- $`a(17)`$：总探索跨度约 79.53 小时，包含算法开发、优化、断点续跑、低位 CPU 补漏和最终审计，并非一次不间断的批处理。

公开版本不列出 GPU/CPU 的具体型号；历史构建字段已去标识化，架构相关的反汇编文件不纳入发布。

这些是本机记录，不是对其他硬件的性能承诺。其他 NVIDIA 架构可尝试覆盖 `GPU_ARCH`，但原始吞吐数据不能照搬；CPU 生产程序要求编译目标支持 AVX-512。

## 目录

```text
A400666/
├── README.md               # 默认中文总览
├── README.en.md            # English overview
├── LICENSE
├── SHA256SUMS
├── a14/
│   ├── README.md             # 默认中文版：推导、执行过程、结果与复现
│   ├── README.en.md          # English version
│   ├── Makefile
│   ├── src/                  # CUDA 搜索、调度、独立验证
│   ├── results/              # 36 个分片的日志、断点和候选
│   └── oeis_results/         # 前 13 项的穷举复核证据
├── a15/
│   ├── README.md             # 默认中文版：各数位区间的降维与完整搜索过程
│   ├── README.en.md          # English version
│   ├── EXPERIMENTS.md        # 实测加速及未采用方案
│   ├── Makefile
│   ├── requirements.txt      # 审计与公式验证的 Python 依赖
│   ├── src/                  # 生产 CUDA/C++/Python 源码
│   ├── experiments/          # 对照与 rejected 实验源码
│   ├── results15/            # 有效日志、断点、迁移清单和最终审计
│   └── cpu_results/          # 272 线程搜索及 CPU/GPU 交接证据
├── a16/
│   ├── README.md             # 默认中文版：双素因子搜索与验证
│   ├── README.en.md          # English version
│   ├── Makefile
│   ├── src/                  # CUDA 搜索、调度管理与独立验证
│   ├── results/              # 日志、断点、候选和最终审计
│   └── campaign16.log        # 主流程记录
└── a17/
    ├── README.md             # 默认中文版：小因子过滤、周期降维与八卡搜索
    ├── README.en.md          # English version
    ├── Makefile
    ├── requirements.txt      # 重新生成过滤器时的 Python 依赖
    ├── src/                  # CUDA/CPU 搜索、调度、快照与独立验证
    ├── filters/              # 小因子剩余类及生成审计
    ├── experiments/          # 对照和未采用实验的源码
    ├── results/              # 完整断点、日志、候选、优化记录和最终审计
    └── campaign17_main.log   # 24 位主流程记录
```

为保持生产脚本原样，`src/results`、`src/oeis_results`、`src/results15` 和 `src/cpu_results` 是指向对应证据目录的相对软链接；`a17` 还使用 `src/filters` 指向过滤器数据。`a15/experiments/` 中的 `search15_prune.cu`、`search15_period_fast.cu` 也是指向生产源码的相对软链接，避免维护两份不一致的共享实现。从源码运行程序时，先进入各自的 `src/`；从项目根目录执行时，推荐使用下面的 `make -C` 命令。

## 只审计已有结果（不需要 GPU）

需要 Python 3。$`a(15)`$ 的部分公式验证和最终审计在导入时使用 SymPy；本次环境版本为 1.14.0。a16 和 a17 的审计只使用 Python 标准库，a17 仅在重新生成小因子过滤器时需要 SymPy。

```bash
python3 -m pip install -r a15/requirements.txt
make -C a14 audit
make -C a14 verify-first13
make -C a15 audit
make -C a16 audit
make -C a17 audit
```

a15 审计应输出：

```text
a(15)=70685893246361615705
block_sums_remainder=0
matrix_remainder=0
...
FINAL_INDEPENDENT_AUDIT_OK
```

a16 审计还应输出 `a(16)=10152437877876943330588` 和 `FINAL_A16_AUDIT_OK`；a17 审计应输出 `a(17)=553943104744829340555260`、`FINAL_A17_AUDIT_OK` 和 `INDEPENDENT_MATRIX17_VERIFIED`。

各项审计会重新计算保存候选的整块拼接余数，核对分片配置、终点、完成日志以及 CPU/GPU 边界，并检查没有更小的已保存解。它们**不是**在几分钟内从零重扫整个搜索空间；最小性依赖所附完整搜索证据及搜索算法本身。独立余数验证器采用块和法和矩阵法，不复用 CUDA 搜索中的周期降维公式。

若要重新检查文件传输是否损坏：

```bash
sha256sum -c SHA256SUMS
```

清单覆盖常规文件；软链接由版本控制系统按链接本身保存。

## 构建与轻量自检

```bash
# 若 nvcc 不在 PATH：
make -C a14 NVCC=/usr/local/cuda-12.9/bin/nvcc
make -C a15 NVCC=/usr/local/cuda-12.9/bin/nvcc
make -C a16 NVCC=/usr/local/cuda-12.9/bin/nvcc
make -C a17 NVCC=/usr/local/cuda-12.9/bin/nvcc

# 默认使用 GPU 0；不会启动全量搜索
make -C a14 selftest GPU_ID=0
make -C a15 selftest GPU_ID=0
make -C a16 selftest GPU_ID=0
make -C a17 selftest GPU_ID=0
```

也可以只执行 `make -C a15 audit`、`make -C a15 selftest-cpu` 或 `make -C a15 selftest-gpu`。`make all` 只编译，不启动搜索。a15 的 CPU 自检通过共享封装入口执行同一组 AVX-512 数值检查，并在结束后清除故意生成的 $`k=15`$ 测试候选；a17 的自检同时覆盖 CUDA 数值路径和 CPU 过滤器。

## 从零重跑

请先阅读 [a14/README.md](a14/README.md)、[a15/README.md](a15/README.md)、[a16/README.md](a16/README.md) 和 [a17/README.md](a17/README.md)。完整任务涉及数天计算、8 张 GPU、272 个 CPU 工作线程和多个并发阶段，不能将“编译成功”或“自检通过”当作全量复现。

本目录带有已完成断点，脚本会从断点继续；要独立重跑，应复制项目并使用新的空结果目录，不要覆盖所附证据。$`a(15)`$ 的 `campaign15.py` 是**接续协调器**：运行它之前仍需按说明启动低位 CPU 搜索和首轮 20 位 GPU 搜索，它不是从零启动所有任务的一键脚本。$`a(17)`$ 的 `make -C a17 run` 则依次覆盖 2–20 位 CPU 过滤、21–24 位 GPU 搜索和最终审计。

## 源码范围与许可

核心 CUDA/C++ 程序针对每个 $`n`$ 的固定模数和素因子编写了常量，并非任意 $`n`$ 的通用求解器。Python 的块和法和矩阵法余数验证则可用于给定的 $`k`$ 和模数。历史实验源码单独放在 `a15/experiments/` 和 `a17/experiments/`，不应误当成最终生产路径。

不附带本机编译的 CUDA/CPU 可执行文件。文档记录的生产二进制 SHA-256 仅用于对应当时日志；不同编译器、驱动或架构产生的哈希不同并不表示结果不同。

采用 [MIT License](LICENSE)。历史快照不包含在此精简发布中；当前审计所需的状态、清单、候选与完整日志均保留。
