param(
    [switch]$Global,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'Stop'
$destination = Join-Path $PSScriptRoot '.agents/skills'
if ($Global) { $destination = Join-Path ([Environment]::GetFolderPath('UserProfile')) '.codex/skills' }

# These are actual skill directories, rather than links to collection indexes.
$sources = @(
    @{ Repo = 'hardikpandya/stop-slop'; Paths = @('.'); Names = @('stop-slop') },
    @{ Repo = 'luwill/research-skills'; Paths = @('lit-search', 'medical-imaging-review', 'research-paper'); Names = @('lit-search', 'medical-imaging-review', 'research-paper') },
    @{ Repo = 'Orchestra-Research/AI-Research-SKILLs'; Paths = @('21-research-ideation/brainstorming-research-ideas', '20-ml-paper-writing/ml-paper-writing'); Names = @('brainstorming-research-ideas', 'ml-paper-writing') },
    @{ Repo = 'K-Dense-AI/scientific-agent-skills'; Paths = @('skills/citation-management', 'skills/peer-review', 'skills/statistical-analysis', 'skills/scientific-visualization', 'skills/pytorch-lightning', 'skills/pydicom'); Names = @('citation-management', 'peer-review', 'statistical-analysis', 'scientific-visualization', 'pytorch-lightning', 'pydicom') },
    @{ Repo = 'ndpvt-web/latex-document-skill'; Paths = @('latex-document'); Names = @('latex-document') }
)

if ($CheckOnly) {
    foreach ($source in $sources) {
        for ($i = 0; $i -lt $source.Paths.Count; $i++) {
            [pscustomobject]@{ Skill = $source.Names[$i]; Repository = $source.Repo; Path = $source.Paths[$i]; Destination = Join-Path $destination $source.Names[$i] }
        }
    }
    return
}

Get-Command git -ErrorAction Stop | Out-Null
New-Item -ItemType Directory -Path $destination -Force | Out-Null
$staging = Join-Path $PSScriptRoot ('.skill-downloads/' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $staging -Force | Out-Null
$records = @()
foreach ($source in $sources) {
    $pending = @()
    for ($i = 0; $i -lt $source.Paths.Count; $i++) {
        $target = Join-Path $destination $source.Names[$i]
        if (Test-Path -LiteralPath $target) {
            if (-not (Test-Path -LiteralPath (Join-Path $target 'SKILL.md'))) { throw "Existing incomplete skill: $target. Inspect it before retrying." }
            Write-Host "Already present, keeping: $($source.Names[$i])"
        } else { $pending += $i }
    }
    if ($pending.Count -eq 0) { continue }
    $checkout = Join-Path $staging ($source.Repo.Replace('/', '--'))
    & git clone --depth 1 --filter=blob:none --sparse "https://github.com/$($source.Repo).git" $checkout
    if ($LASTEXITCODE -ne 0) { throw "Download failed for $($source.Repo). Staging retained at $staging" }
    if ($source.Paths -contains '.') {
        & git -C $checkout sparse-checkout disable
    } else {
        & git -C $checkout sparse-checkout set @($source.Paths)
    }
    if ($LASTEXITCODE -ne 0) { throw "Checkout failed for $($source.Repo)" }
    $commit = & git -C $checkout rev-parse HEAD
    if ($LASTEXITCODE -ne 0) { throw 'Could not record source revision.' }
    foreach ($i in $pending) {
        $skillSource = Join-Path $checkout $source.Paths[$i]
        $target = Join-Path $destination $source.Names[$i]
        if (-not (Test-Path -LiteralPath (Join-Path $skillSource 'SKILL.md'))) { throw "No SKILL.md at $skillSource" }
        $linked = Get-ChildItem -LiteralPath $skillSource -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }
        if ($linked) { throw "Review linked files before installing $($source.Names[$i])." }
        New-Item -ItemType Directory -Path $target | Out-Null
        Get-ChildItem -LiteralPath $skillSource -Force | Where-Object Name -ne '.git' | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $target -Recurse }
        $records += [pscustomobject]@{ Skill = $source.Names[$i]; Repository = $source.Repo; Path = $source.Paths[$i]; Commit = "$commit"; Destination = $target }
        Write-Host "Installed: $($source.Names[$i])"
    }
}
if ($records.Count -gt 0) {
    $records | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $staging 'installed.json') -Encoding UTF8
}
Write-Host "Done. Skills are in $destination and will be available on your next turn."
Write-Host "Source checkouts and installation records are retained in $staging. Dependencies have not been installed."
