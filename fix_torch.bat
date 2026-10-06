@echo off
setlocal
chcp 936 >nul
cd /d "%~dp0."

echo ============================================================
echo   PyTorch 导入失败 诊断与修复  ^(WinError 1114 / c10.dll^)
echo ============================================================
echo.

rem 选一个 64 位解释器
set PY=
for %%v in (3.14 3.13 3.12 3.11 3.10 3) do call :try_py %%v
if defined PY goto :py_ready
python -c "import sys;sys.exit(0 if sys.maxsize>2**32 else 1)" >nul 2>&1
if not errorlevel 1 set PY=python
if not defined PY set PY=python

:py_ready
echo [1] 解释器
%PY% -c "import sys;print('  exe  :',sys.executable);print('  版本 :',sys.version.split()[0]);print('  64位 :',sys.maxsize>2**32)"
echo.

echo [2] torch 安装情况
%PY% -m pip show torch
echo.

echo [3] MSVC 运行库 (torch 依赖; 缺失会导致 c10.dll 初始化失败)
if exist "%SystemRoot%\System32\vcruntime140.dll" echo   存在   vcruntime140.dll
if not exist "%SystemRoot%\System32\vcruntime140.dll" echo   缺失   vcruntime140.dll   ^<== 需装 VC++ 运行库
if exist "%SystemRoot%\System32\vcruntime140_1.dll" echo   存在   vcruntime140_1.dll
if not exist "%SystemRoot%\System32\vcruntime140_1.dll" echo   缺失   vcruntime140_1.dll   ^<== 需装 VC++ 运行库
if exist "%SystemRoot%\System32\msvcp140.dll" echo   存在   msvcp140.dll
if not exist "%SystemRoot%\System32\msvcp140.dll" echo   缺失   msvcp140.dll   ^<== 需装 VC++ 运行库
echo.

echo [4] PATH 中可能冲突的库与其它 Python
echo   -- libiomp5md.dll (OpenMP, 装了 Anaconda 常冲突) --
where libiomp5md.dll 2>nul
echo   -- python.exe 都在哪 --
where python 2>nul
echo.

echo [5] 尝试导入 torch
%PY% -c "import torch;print('  导入成功: torch',torch.__version__,' cuda',torch.cuda.is_available())" 2>&1
echo.

echo ============================================================
echo   可选修复动作
echo ============================================================
echo   1 = 重新安装 torch  ^(强制覆盖 + 不用缓存, 修复下载损坏的 DLL^)
echo       注意: 需重新下载约 2~4 GB
echo   2 = 打开 VC++ 运行库下载页  ^(装完重启电脑再试^)
echo   0 = 只看诊断, 什么都不做
echo.
choice /c 120 /n /m "请输入 1 / 2 / 0: "
if errorlevel 3 goto :end
if errorlevel 2 goto :openredist
if errorlevel 1 goto :reinstall

:openredist
echo.
echo 正在打开 VC++ 2015-2022 x64 运行库下载页 ...
echo 下载后双击安装, 装完重启电脑, 再跑 run_s2.bat。
start "" "https://aka.ms/vs/17/release/vc_redist.x64.exe"
goto :end

:reinstall
echo.
echo 重新安装 torch ...
%PY% -m pip install --force-reinstall --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
if not errorlevel 1 goto :reinstall_check
echo [提示] 官方源失败, 换阿里云镜像 ...
%PY% -m pip install --force-reinstall --no-cache-dir torch -i https://mirrors.aliyun.com/pytorch-wheels/cu128/
if not errorlevel 1 goto :reinstall_check
echo [提示] 仍失败, 换清华镜像 ...
%PY% -m pip install --force-reinstall --no-cache-dir torch -i https://pypi.tuna.tsinghua.edu.cn/simple
if not errorlevel 1 goto :reinstall_check
echo [错误] 重装失败
goto :end

:reinstall_check
echo.
echo 重装后验证:
%PY% -c "import torch;print('  导入成功: torch',torch.__version__,' cuda',torch.cuda.is_available())" 2>&1
goto :end

:end
echo.
echo 若仍失败, 把上面全部输出复制回来。
echo.
pause
exit /b 0

rem ---------- 子过程: 探测 64 位解释器 ----------
:try_py
if defined PY exit /b 0
py -%1 -c "import sys;sys.exit(0 if sys.maxsize>2**32 else 1)" >nul 2>&1
if not errorlevel 1 set PY=py -%1
exit /b 0
