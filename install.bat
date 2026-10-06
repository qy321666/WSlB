@echo off
setlocal
chcp 936 >nul
cd /d "%~dp0."

echo ============================================================
echo   论文一 环境安装脚本  ^(Windows / RTX 50 系 / CUDA 12.8^)
echo ============================================================
echo.

rem ---------- 1. 挑一个 64 位的 Python ----------
rem PyTorch 只发布 64 位轮子。若选到 32 位解释器, 所有源都会报
rem "Could not find a version that satisfies the requirement torch (from versions: none)"。
rem 这里按版本从高到低逐个探测, 只接受 64 位。
set PY=
for %%v in (3.14 3.13 3.12 3.11 3.10 3) do call :try_py %%v
if defined PY goto :py_ready
python -c "import sys;sys.exit(0 if sys.maxsize>2**32 else 1)" >nul 2>&1
if not errorlevel 1 set PY=python
if defined PY goto :py_ready

echo [错误] 没有找到 64 位 Python
echo        PyTorch 只有 64 位版本, 32 位 Python 装不了。
echo        请安装 64 位 Python 3.10+ 并勾选 "Add Python to PATH":
echo        https://www.python.org/downloads/
echo        若你机器上有多个 Python, 可先运行 diag.bat 看清单。
pause
exit /b 1

:py_ready
echo [1/5] Python 命令: %PY%
%PY% -c "import sys,sysconfig;print('      版本',sys.version.split()[0],' 64位',sys.maxsize>2**32,' 平台',sysconfig.get_platform())"
echo.

rem ---------- 2. 升级 pip, 清理旧 torch ----------
echo [2/5] 升级 pip 并清理可能存在的旧版 torch ...
%PY% -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>&1
if errorlevel 1 %PY% -m pip install --upgrade pip >nul 2>&1
%PY% -m pip uninstall -y torch >nul 2>&1
echo.

rem ---------- 3. 安装 PyTorch cu128 ----------
rem 不锁版本: 可用版本取决于 Python 版本, 锁版本会报
rem "Could not find a version that satisfies the requirement"
echo [3/5] 安装 PyTorch cu128 ...  约 2~4 GB, 请耐心等待
%PY% -m pip install torch --index-url https://download.pytorch.org/whl/cu128
if not errorlevel 1 goto :torch_ok
echo.
echo [提示] 官方源失败, 换阿里云 PyTorch 镜像重试 ...
%PY% -m pip install torch -i https://mirrors.aliyun.com/pytorch-wheels/cu128/
if not errorlevel 1 goto :torch_ok
echo.
echo [提示] 仍失败, 换清华 PyPI 镜像重试 ...
%PY% -m pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
if not errorlevel 1 goto :torch_ok
echo.
echo [错误] PyTorch 安装失败
echo        若三个源都报 "(from versions: none)", 多半是 Python 位数不对:
echo        请运行 diag.bat, 确认上面 [1/5] 打印的 "64位 True"。
pause
exit /b 1

:torch_ok
echo.

rem ---------- 4. 其余依赖 ----------
echo [4/5] 安装其余依赖 numpy/scipy/matplotlib/tqdm ...
%PY% -m pip install -r requirements.txt
if not errorlevel 1 goto :deps_ok
echo [提示] 默认源失败, 换清华镜像重试 ...
%PY% -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if not errorlevel 1 goto :deps_ok
echo [错误] 依赖安装失败
pause
exit /b 1

:deps_ok
echo.

rem ---------- 5. 验证 GPU ----------
echo [5/5] 验证 PyTorch / CUDA / sm_120 ...
%PY% -c "import torch; print('  torch      :', torch.__version__); print('  cuda_avail :', torch.cuda.is_available()); print('  arch_list  :', torch.cuda.get_arch_list())"
if errorlevel 1 (
    echo.
    echo [警告] 验证未通过, 请检查:
    echo   - torch 是否装成功, 可单独运行 pip show torch 查看
    echo   - 若 cuda_avail 为 False: 显卡驱动过旧, nvidia-smi 能否识别到显卡
    echo   - 若 arch_list 不含 sm_120: 装成了旧 CUDA 版本的 torch, 需重装 cu128 版
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   安装完成
echo ============================================================
echo   下一步:
echo     1^) 下载 CWRU 数据到 data\cwru\  ^(命名见 README^)
echo     2^) 先跑链路冒烟:  run_s2.bat smoke
echo     3^) 再跑正式实验:  run_s2.bat all
echo.
pause
exit /b 0

rem ---------- 子过程: 探测某个 py 版本是不是 64 位 ----------
:try_py
if defined PY exit /b 0
py -%1 -c "import sys;sys.exit(0 if sys.maxsize>2**32 else 1)" >nul 2>&1
if not errorlevel 1 set PY=py -%1
exit /b 0
