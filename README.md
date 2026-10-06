# 论文一代码骨架:跨域轴承故障诊断中的目标域谱指纹标定

第一篇论文的实验代码,当前版本完成 **S2 跨工况设置(CWRU load0 → load3,10 类细分)**。

> **⚠️ 论文定位已于 2026-10-05 重定位**(依据详见 `接手文档.md` 第 4 节)。
> 原主线"物理仿真孪生补样本"被三组独立实验一致证否:孪生合成的九个故障类谱形几乎全同
> (实测各类在 2800-3300 Hz 都约 53%),而真实类间差异是**每颗故障轴承自身的录制指纹**,
> 物理模型无从预测 —— 仅用孪生训练迁移到真实域时 macro_f1 只有 **0.12**(随机水平 0.10)。
> 现主方法 `fp_dann`(用目标支撑集估计逐类谱包络再着色生成)在上表中的标准协议下达
> **0.9996 ± 0.0005**(对照 `dann` 0.9705 ± 0.0569)。
>
> **但更重要的发现是:该协议本身有"同录制指纹泄漏"** —— 支撑窗与测试窗来自同一次录制。
> 换成跨录制评估后,所有方法掉 **0.13~0.34** macro_f1,且 `fp_dann` 与"只用支撑集"打平。
> 完整数据见 `tools/probe_leakage.py` 与 `接手文档.md` 第 4.7 节。**论文按此重定位。**

## 0. 环境安装(RTX 5060 必读)

RTX 5060 是 Blackwell 架构(算力 **sm_120**),**旧版 PyTorch(CUDA 12.1/12.6 编译)无法运行**,
会报 `CUDA error: no kernel image is available for execution on the device`。
必须装 **CUDA 12.8(cu128)及以上**编译的轮子。

**Windows 用户直接用 `install.bat`,不用手动敲下面的命令。**

```bash
# 1) 先卸载可能存在的旧版
pip uninstall -y torch

# 2) 安装 cu128 版本
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

- **不要写死版本号**。torch 的可用版本取决于你的 Python 版本:本机实测 Python **3.14** 在 cu128 源上
  只有 `2.9.0 / 2.9.1 / 2.10.0 / 2.11.0`(均含 `+cu128` 的 cp314 Windows 轮子),写 `torch==2.7.1` 会直接报
  `Could not find a version that satisfies the requirement`。不写版本号让 pip 自动选,最稳。
- **本项目不需要 torchvision / torchaudio**(代码只用 `torch` + `numpy` + `scipy`),不要顺手装上,能省一半下载量。
- **国内下载慢的替代源**(实测可用,均有 cp314 Windows 轮子):
  - 阿里云 PyTorch 镜像 `https://mirrors.aliyun.com/pytorch-wheels/cu128/`
  - 清华 PyPI 镜像 `https://pypi.tuna.tsinghua.edu.cn/simple`
- **驱动**:RTX 50 系需要较新的 NVIDIA 驱动(建议装最新版;若 `nvidia-smi` 识别不到显卡,先升级驱动)。
- **验证**:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_arch_list())"
```

期望输出:`cuda_avail` 为 `True`,且 `arch_list` 中包含 `sm_120`。

安装其余依赖:

```bash
pip install -r requirements.txt
```

(requirements.txt 不含 torch,只含 numpy/scipy + matplotlib/tqdm;torch 按上文 cu128 方式装。)

## 0.5 Windows 用户(一键脚本)

仓库里带两个 Windows 启动脚本,在文件资源管理器里双击或命令行运行均可:

- **`install.bat`** — 一键环境安装:检查 Python → 卸载旧 torch → 安装 cu128 版 PyTorch(RTX 50 系必需)→ 安装其余依赖 → 验证 CUDA 与 `sm_120`。
- **`run_s2.bat`** — 实验启动器,用法与 `run_s2.sh` 一致:

  ```bat
  run_s2.bat            ← 默认跑现主方法 fp_dann
  run_s2.bat smoke      ← 链路冒烟(6 个方法各 2 epoch,先跑这个)
  run_s2.bat all        ← 六方法对比(论文主表)
  run_s2.bat dann       ← 单跑某个方法
  ```

  覆盖参数:`set K=5 & set CLASSES=10 & set EPOCHS=60 & set DATA=data\cwru & run_s2.bat all`。

> `run_s2.sh` 面向 Linux / macOS / WSL;Windows 请用 `run_s2.bat`。
> 注意 `requirements.txt` 不含 torch(torch 的 `+cu128` 版本号无法从 PyPI 直接解析,必须用 `install.bat` 或按第 0 节的 `--index-url` 方式安装)。

**关于 .bat 的编码(重要)**:两个 `.bat` 保存为 **GBK 编码 + CRLF 换行**,这是 cmd.exe 在中文 Windows 下最稳的组合。
如果你要改动它们,请用 VS Code / Notepad++ 并**保持 GBK(ANSI)编码**;若用记事本另存为 UTF-8,中文会变乱码。
(常见症状:双击后报 `'cal' 不是内部或外部命令` 之类——那是换行被存成了 LF,cmd 逐行读取时偏移漂移、吞掉行首字符,改回 CRLF 即可。)

## 1. 数据准备

当前只用到 CWRU。下载官方 12k 驱动端数据:

- 官方入口:[CWRU Bearing Data Center](https://csegroups.case.edu/bearingdatacenter/pages/welcome-case-western-reserve-university-bearing-data-center-website)
- 或从 Kaggle / GitHub 镜像获取(文件名必须保持官方命名,如 `Normal_0.mat`、`IR007_0.mat`)。

解压后目录组织(把 `data/cwru` 直接指向包含 .mat 的文件夹即可,脚本会递归扫描):

```
data/
└── cwru/
    ├── Normal_0.mat ... Normal_3.mat
    ├── B007_0.mat ... OR028_3.mat
    └── (48k 等其它目录可忽略,脚本只认文件名)
```

XJTU-SY(论文 S3 用,当前骨架未接入):官方 [GitHub](https://github.com/WangBiaoXJTU/xjtu-sy-bearing-datasets),国内可用[百度网盘](https://pan.baidu.com/s/1OaY82azTXHBwjiCjA_jRcw)。
Paderborn(论文 S1 用):官方[数据中心](https://mb.uni-paderborn.de/kat/forschung/datacenter/bearing-datacenter/)。

## 2. 快速开始

```bash
# 先跑最简下界(纯源域训练,不迁移)
bash run_s2.sh source_only

# 跑现主方法(目标域指纹标定 + 生成式增广 + DANN)
bash run_s2.sh fp_dann

# 物理消融(指纹 + 目标转速冲击串,实测更差)
bash run_s2.sh fp_phys_dann

# 原主方法(物理孪生合成域,已被证否,作对照)
bash run_s2.sh dt_dann

# 六方法依次跑(论文主表)
bash run_s2.sh all

# 快速冒烟:六个方法各 2 epoch,验证代码链路(几分钟)
bash run_s2.sh smoke
```

训练日志与最终结果写入 `results/`(每设置一条 CSV:test_acc, macro_f1, 每类召回率)。

**方法一览**(详见 `train.py` 顶部注释与 `接手文档.md` 第 4 节):

| 方法 | 一句话 | 10 类 S0→S3 的 macro_f1(5 seed) |
|---|---|---|
| `source_only` | 仅源域训练,测目标域 —— 下界 | 0.7776 ± 0.0732 |
| `dann` | 源域 + 目标无标签池对抗适应(A2 消融) | 0.9705 ± 0.0569 |
| `rand_dann` | A3 消融:孪生域换成循环平移+噪声 | 0.9709 ± 0.0392 |
| `dt_dann` | 原主方法:物理孪生合成域 | 0.8154 ± 0.1379 |
| `fp_phys_dann` | 目标指纹着色 + 目标转速冲击串 | 0.9807 ± 0.0294 |
| **`fp_dann`** | **现主方法:目标指纹着色 + 白噪声** | **0.9996 ± 0.0005** |

## 3. 冒烟测试核对清单(拿到机器后第一步)

装好环境、下好 CWRU 后,先跑一次冒烟确认链路全通:

```bash
EPOCHS=2 bash run_s2.sh smoke
```

观察点:
- 六个方法都打印 `[final] ... acc=... macro_f1=...`,并在 `results/smoke/` 下生成 6 条 CSV;
- 2 epoch 不追求指标,只看链路是否全通;
- 无报错、无挂起(单进程总时长约几分钟)。

若冒烟通过,再跑正式实验(`bash run_s2.sh all`,默认 60 epoch)。

**注意**:默认 `--classes 10`。4 类设置的基线已饱和(`source_only` 12 个方向全部 100%,
天花板效应下无法验证任何方法差异),仅作历史对照保留:`CLASSES=4 bash run_s2.sh all`。

## 4. 文件说明

| 文件 | 作用 |
|---|---|
| `data_utils.py` | CWRU 读取、切窗、归一化、few-shot 目标域划分(支撑集/无标签池/测试集)、缓存(缓存键绑定数据目录与类别体系,避免不同数据源互相污染) |
| `twin_calib.py` | **现主方法核心**:从目标支撑集估计逐类谱包络(`estimate_fingerprints`),用它着色激励源生成目标域风格样本(`synth_target_like`) |
| `dt_synth.py` | 物理仿真孪生信号合成器(McFadden-Smith 冲击响应模型,按 CWRU 6205 轴承几何参数)。作为主方法已被证否,保留作对照 |
| `models.py` | ResNet1D 骨干、分类头、域判别器、梯度反转层 |
| `train.py` | 统一训练入口:`--method source_only \| dann \| dt_dann \| rand_dann \| fp_dann \| fp_phys_dann`,`--classes 4\|10` |
| `run_s2.sh` / `run_s2.bat` | S2 跨工况(load0→load3)示例命令(支持 `smoke` 冒烟模式) |
| `tools/` | 诊断与探针脚本:`probe_fp.py`(合成域迁移+域判别器)、`probe_fewshot.py`(少样本增广对照)、`probe_simreal.py`、`probe_envelope.py`、`compare_classes.py` 等 |
| `接手文档.md` | 项目状态、环境修复记录、全部实验证据与结论(第 4 节是核心) |

## 5. 训练入口参数

```bash
python train.py --method fp_dann --classes 10 --src_load 0 --tgt_load 3 --k 5 --seed 0 \
    --epochs 60 --window 1024 --synth_per_class 400
```

- `--method` :`source_only` / `dann` / `dt_dann` / `rand_dann` / `fp_dann` / `fp_phys_dann`
- `--classes` :`10`(默认,正常 + B/IR/OR × 007/014/021)或 `4`(粗分类)
- `--k` :目标域每类带标签样本数(1/3/5/10)
- `--synth_per_class` :增广域每类样本数
- `--pseudo` :开启目标域伪标签自训练(**当前主方法未使用**;`fp_*` 与 `dt_*`/`rand_*` 均可叠加)
- `--data_root` :CWRU .mat 所在目录,默认 `data/cwru`
- `--update manual` :用纯张量手写梯度下降替代优化器,仅供**无 GPU 的受限环境**(沙箱/无驱动虚机)做逻辑冒烟;正常 GPU 环境请用默认的 `adam`。两种更新对训练逻辑等价,`manual` 只是去掉动量与调度,指标差异可忽略。

> 说明:开发环境的沙箱(gVisor,无 GPU)与 torch 原生多线程不兼容,无法在沙箱内跑完整多轮训练(会在某个 forward/backward 处死锁,位置随环境抖动)。代码逻辑已用"模块级单测 + 单周期 forward/backward"验证通过;`--update manual` 即为沙箱冒烟保留的路径。**在你本机的 RTX 5060 上请直接用默认配置跑,不受此限制。**

## 6. 下一步(论文节奏)

按重定位后的框架推进(详见 `接手文档.md` 第 7 节 M3):

1. **跑 12 个跨负载方向 × 3 seed 的方法对比主表**(`fp_dann` / `fp_phys_dann` / `dann` / `dt_dann` / `source_only`);
2. **k 扫描**(1/3/5/10/20),回答"指纹估计需要几个标签"以及方法在极少数标签下的优势幅度 ——
   注意 `support_only` 在 k=1 时已达 0.893,这是必须在论文里正面讨论的对照;
3. **指纹泄漏的方法学控制**:CWRU 窗口切分使支撑窗与测试窗来自同一次录制,共享同一指纹,
   会让任何带目标标签的方法虚高。用"load3 支撑训练 → load0/1/2 测试"量化无泄漏条件下的真实泛化;
4. 起草论文 `IEEE TIM`(LaTeX,`xelatex` + `ctex`)。
