param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$EvaluationArguments
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) { throw 'Install uv with: python -m pip install uv' }
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot '.uv-cache-cpu'
& $uvCommand.Source run --python 3.11 --with 'torch>=2.4,<3' --with numpy --with scipy --with nibabel --with matplotlib --with nbformat --with kaggle --with python-dotenv --with psutil --index https://download.pytorch.org/whl/cpu python local_evaluate_baseline.py @EvaluationArguments
exit $LASTEXITCODE
