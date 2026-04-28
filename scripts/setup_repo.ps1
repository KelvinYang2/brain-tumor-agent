# One-shot bootstrap helper for Windows PowerShell.
# Run from inside an empty git repo.
$ErrorActionPreference = "Stop"

Write-Host "==> Creating venv with uv (python 3.11)" -ForegroundColor Cyan
uv venv --python 3.11

Write-Host "==> Activating venv and installing project" -ForegroundColor Cyan
& .\.venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"

Write-Host "==> Verifying imports" -ForegroundColor Cyan
python -c "import src; print('src version:', src.__version__)"

Write-Host "==> Running smoke tests" -ForegroundColor Cyan
pytest tests/ -q

Write-Host "==> Setting up .env (if missing)" -ForegroundColor Cyan
if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host "    Created .env from .env.example — edit it to point at your data." -ForegroundColor Yellow
} else {
    Write-Host "    .env already exists — leaving it alone." -ForegroundColor Green
}

Write-Host ""
Write-Host "DONE. Next: edit .env, then proceed to Step 1.2 (tool wrapping)." -ForegroundColor Green
