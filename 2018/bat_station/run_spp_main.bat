@echo off
cd /d %~dp0
echo "Python_sl_mainを実行しますか？"&echo.
choice  /t 5 /d n
if %errorlevel% == 1 goto :yes
echo 実行せずに終了します。&pause >nul
exit

:yes
echo 何回まで実行しますか?
set /p num=""
echo %num%回まで実行します。
python spp_copy.py
for /l %%n in (0,1,%num%) do (
  python spp_bat.py %%n %num%
)
python spp_del.py
echo 終わりました。
pause >nul
exit