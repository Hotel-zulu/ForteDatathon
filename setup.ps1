# Windows:  powershell -ExecutionPolicy Bypass -File setup.ps1
$py = "py -3.11"; try { Invoke-Expression "$py --version" | Out-Null } catch { $py = "python" }
Invoke-Expression "$py -m venv .venv"; .\.venv\Scripts\Activate.ps1
python -m pip install -U pip wheel; pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python scripts/check_env.py; python -m src.labels
