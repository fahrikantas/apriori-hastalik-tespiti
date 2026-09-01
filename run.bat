@echo off
chcp 65001 >nul
title Hastalik Tahmin Uygulamasi
cd /d "%~dp0"

echo.
echo ============================================
echo   Hastal Tahmin Uygulamasi - Baslatiliyor
echo ============================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo [HATA] Python bulunamadi.
    echo Lütfen https://www.python.org/downloads adresinden Python 3.10+ kurun.
    echo Kurulum sirasinda "Add Python to PATH" kutusunu isaretleyin.
    pause
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    echo Ilk calistirmada sanal ortam olusturuluyor...
    python -m venv .venv
    if errorlevel 1 (
        echo [HATA] Sanal ortam olusturulamadi.
        pause
        exit /b 1
    )
    set "PY=.venv\Scripts\python.exe"
)

echo Paketler kontrol ediliyor / kuruluyor...
"%PY%" -m pip install --disable-pip-version-check -q -r requirements.txt

echo.
echo Uygulama tarayicinizda acilacak...
echo Kapatmak icin bu pencerede Ctrl+C kullanin.
echo.
"%PY%" -m streamlit run app.py

pause