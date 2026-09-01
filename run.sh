#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "============================================"
echo "  Hastalik Tahmin Uygulamasi - Baslatiliyor"
echo "============================================"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[HATA] python3 bulunamadi. Python 3.10+ kurun."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "Ilk calistirmada sanal ortam olusturuluyor..."
  python3 -m venv .venv
fi

. .venv/bin/activate
pip install --disable-pip-version-check -q -r requirements.txt

echo
echo "Uygulama tarayicinizda acilacak. Kapatmak icin Ctrl+C."
echo
streamlit run app.py