@echo off
setlocal
chcp 936 >nul
cd /d "%~dp0."

echo ============================================================
echo   环境诊断 - 请把下面全部输出复制粘贴回来
echo ============================================================
echo.

echo [A] 系统架构
echo   PROCESSOR_ARCHITECTURE  = %PROCESSOR_ARCHITECTURE%
echo   PROCESSOR_ARCHITEW6432  = %PROCESSOR_ARCHITEW6432%
echo.

echo [B] py 启动器能看到的全部 Python
py -0p
echo.

echo [C] py -3 解析到哪一个
py -3 -c "import sys,sysconfig;print('  exe     :',sys.executable);print('  版本    :',sys.version.split()[0]);print('  64位    :',sys.maxsize>2**32);print('  平台tag :',sysconfig.get_platform())"
echo.

echo [D] pip 版本与配置(配置里的 index-url / proxy 会覆盖脚本设置)
py -3 -m pip --version
py -3 -m pip config list
echo.

echo [E] 网络连通性(200 为正常)
curl -s -o nul -w "  pytorch.org : %%{http_code}\n" --max-time 20 https://download.pytorch.org/whl/cu128/
curl -s -o nul -w "  aliyun      : %%{http_code}\n" --max-time 20 https://mirrors.aliyun.com/pytorch-wheels/cu128/
curl -s -o nul -w "  tsinghua    : %%{http_code}\n" --max-time 20 https://pypi.tuna.tsinghua.edu.cn/simple/
echo.

echo [F] pip 认为可用的 wheel 标签(前 10 个, 看有没有 win_amd64)
py -3 -c "import sys;from pip._vendor.packaging.tags import sys_tags;ts=[str(t) for t in sys_tags()];print('  共',len(ts),'个标签, 前 10 个:');[print('   ',t) for t in ts[:10]]"
echo.

echo [G] 各源上 torch 的可见版本
echo   -- pytorch cu128 --
py -3 -m pip index versions torch --index-url https://download.pytorch.org/whl/cu128 2>&1
echo   -- 清华 --
py -3 -m pip index versions torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple 2>&1
echo.

echo ============================================================
echo   诊断结束, 请把以上全部内容复制回来
echo ============================================================
pause
