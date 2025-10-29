$ErrorActionPreference = 'Stop'

# Root is the script directory
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

Write-Host 'Creating venv (.venv) if needed...'
if (-Not (Test-Path .venv)) { python -m venv .venv }
& .\.venv\Scripts\Activate.ps1

Write-Host 'Installing backend requirements...'
pip install -r .\server\requirements.txt

Write-Host 'Starting backend (http://localhost:8000)...'
Start-Process powershell -ArgumentList "-NoExit","-Command","uvicorn NeuroID.server.app:app --reload --port 8000" | Out-Null

Start-Sleep -Seconds 2
Write-Host 'Opening frontend...'
Start-Process $([System.IO.Path]::Combine($ROOT,'frontend','index.html'))

Write-Host 'Done. If the browser did not open, open NeuroID/frontend/index.html manually.'

