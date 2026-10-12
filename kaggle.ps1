param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$KaggleArguments
)
# Run this from a normal PowerShell terminal. It does not change sandbox settings.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$uvCommand = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCommand) {
    throw 'uv is not installed. Install it with: python -m pip install uv'
}
& $uvCommand.Source run --python 3.11 --with kaggle --with python-dotenv --with nbformat python sync_kaggle.py @KaggleArguments
exit $LASTEXITCODE
