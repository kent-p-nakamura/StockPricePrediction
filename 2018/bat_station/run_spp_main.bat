@echo off
cd /d %~dp0
echo "Python_sl_main�����s���܂����H"&echo.
choice  /t 5 /d n
if %errorlevel% == 1 goto :yes
echo ���s�����ɏI�����܂��B&pause >nul
exit

:yes
echo ����܂Ŏ��s���܂���?
set /p num=""
echo %num%��܂Ŏ��s���܂��B
python spp_copy.py
for /l %%n in (0,1,%num%) do (
  python spp_bat.py %%n %num%
)
python spp_del.py
echo �I���܂����B
pause >nul
exit