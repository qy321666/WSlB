@echo off
setlocal
chcp 936 >nul
cd /d "%~dp0."
set PYTHONIOENCODING=gbk:replace

rem ============================================================
rem   S2 跨工况实验启动器: CWRU load0 -^> load3
rem
rem   用法:
rem     run_s2.bat                 默认跑现主方法 fp_dann
rem     run_s2.bat <方法>          source_only / dann / dt_dann / rand_dann / fp_dann / fp_phys_dann
rem     run_s2.bat all             六个方法依次跑(论文主表)
rem     run_s2.bat smoke           快速冒烟: 各 2 epoch
rem
rem   覆盖变量:
rem     set K=5 ^& set EPOCHS=60 ^& set CLASSES=10 ^& set SRC=0 ^& set TGT=3 ^& run_s2.bat all
rem
rem   方法速查(详见 train.py 顶部注释):
rem     fp_dann      **现主方法**: 目标域谱包络指纹标定 + 白噪声着色生成式增广
rem     fp_phys_dann 其物理消融: 指纹 + 目标转速冲击串(实测更差)
rem     dt_dann      原主方法: 物理孪生合成域(已被证否, 保留作对照)
rem     rand_dann    A3 消融: 无物理增广
rem     dann         A2 消融: 仅对抗域适应
rem     source_only  下界
rem ============================================================

set METHOD=%~1
if "%METHOD%"=="" set METHOD=fp_dann
if "%K%"=="" set K=5
if "%SEED%"=="" set SEED=0
if "%EPOCHS%"=="" set EPOCHS=60
if "%CLASSES%"=="" set CLASSES=10
if "%SRC%"=="" set SRC=0
if "%TGT%"=="" set TGT=3
if "%DATA%"=="" set DATA=data\cwru
if "%OUT%"=="" set OUT=results

set PY=python
py -3 --version >nul 2>&1
if not errorlevel 1 set PY=py -3

if "%METHOD%"=="smoke" goto :smoke
if "%METHOD%"=="all" goto :all
if "%METHOD%"=="source_only" goto :run
if "%METHOD%"=="dann" goto :run
if "%METHOD%"=="dt_dann" goto :run
if "%METHOD%"=="rand_dann" goto :run
if "%METHOD%"=="fp_dann" goto :run
if "%METHOD%"=="fp_phys_dann" goto :run

echo 未知方法: %METHOD%
echo 用法: run_s2.bat [source_only ^| dann ^| dt_dann ^| rand_dann ^| fp_dann ^| fp_phys_dann ^| all ^| smoke]
exit /b 1

:run
echo.
echo ============================================================
echo   运行 %METHOD%    S%SRC%-^>S%TGT%   classes=%CLASSES%  k=%K%  seed=%SEED%  epochs=%EPOCHS%
echo ============================================================
%PY% train.py --method %METHOD% --classes %CLASSES% --src_load %SRC% --tgt_load %TGT% --k %K% --seed %SEED% --epochs %EPOCHS% --data_root "%DATA%" --out_dir "%OUT%"
if errorlevel 1 (
    echo.
    echo [错误] %METHOD% 运行失败, 请检查上方日志
    pause
    exit /b 1
)
goto :done

:all
for %%m in (source_only dann dt_dann rand_dann fp_dann fp_phys_dann) do (
    echo.
    echo ============================================================
    echo   运行 %%m    S%SRC%-^>S%TGT%   classes=%CLASSES%  k=%K%  seed=%SEED%  epochs=%EPOCHS%
    echo ============================================================
    %PY% train.py --method %%m --classes %CLASSES% --src_load %SRC% --tgt_load %TGT% --k %K% --seed %SEED% --epochs %EPOCHS% --data_root "%DATA%" --out_dir "%OUT%"
    if errorlevel 1 (
        echo.
        echo [错误] %%m 运行失败, 已终止
        pause
        exit /b 1
    )
)
goto :done

:smoke
rem 快速链路验证: 全部方法各 2 epoch
for %%m in (source_only dann dt_dann rand_dann fp_dann fp_phys_dann) do (
    echo.
    echo ============================================================
    echo   [smoke] %%m    S%SRC%-^>S%TGT%   k=%K%  seed=%SEED%  epochs=2
    echo ============================================================
    %PY% train.py --method %%m --classes %CLASSES% --src_load %SRC% --tgt_load %TGT% --k %K% --seed %SEED% --epochs 2 --data_root "%DATA%" --out_dir "%OUT%\smoke"
    if errorlevel 1 (
        echo.
        echo [错误] %%m 运行失败, 已终止
        pause
        exit /b 1
    )
)
goto :done

:done
echo.
echo ============================================================
echo   全部完成。结果在 %OUT%\
echo ============================================================
pause
