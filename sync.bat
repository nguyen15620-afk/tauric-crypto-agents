@echo off
chcp 65001 >nul
echo 🚀 Đang tự động kiểm thử và cập nhật lên GitHub...
set MSG=%~1
if "%MSG%"=="" set MSG=Update project changes %DATE% %TIME%
python sync_github.py "%MSG%"
pause
