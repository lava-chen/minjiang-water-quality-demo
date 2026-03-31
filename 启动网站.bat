@echo off
chcp 65001 >nul
title 岷江水系水质监测系统
cd /d D:\26dachuang

echo.
echo   =============================================
echo        岷江水系水质监测系统
echo        基于 LSTM+Attention 的溶解氧预测平台
echo   =============================================
echo.
echo   正在启动...
echo   启动后请打开浏览器访问:
echo.
echo       http://localhost:8501
echo.
echo   关闭此窗口即可停止服务
echo   =============================================
echo.

streamlit run src/app/app.py --server.port 8501 --server.headless true

pause
