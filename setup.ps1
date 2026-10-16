# Windows PowerShell:  powershell -ExecutionPolicy Bypass -File setup.ps1
# Нужен Python 3.11 (https://www.python.org/downloads/ — галочка "Add to PATH") или: py -3.11
$py = "py -3.11"
try { Invoke-Expression "$py --version" | Out-Null } catch { $py = "python" }
Invoke-Expression "$py -m venv .venv"
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel
pip install -r requirements.txt -r requirements-extra.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python scripts/check_env.py
Write-Host "Готово. Активация: .\.venv\Scripts\Activate.ps1"
