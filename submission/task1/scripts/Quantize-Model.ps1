[CmdletBinding()]
param(
    [string]$InputModel = ".\models\llama-3.2-3b-instruct-f16.gguf",
    [string]$OutputModel = ".\models\llama-3.2-3b-instruct-q4_0.gguf",
    [string]$LogPath = ".\logs\quantization-q4_0.log"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Quote-NativeArgument {
    param([string]$Value)

    if ($null -eq $Value -or $Value.Length -eq 0) {
        return '""'
    }

    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    return '"' + $Value.Replace('"', '\"') + '"'
}

function Invoke-LoggedNativeProcess {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$Arguments,
        [Parameter(Mandatory)][string]$LogPath
    )

    $ArgumentString = (($Arguments | ForEach-Object {
        Quote-NativeArgument ([string]$_)
    }) -join " ")

    $StartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $StartInfo.FileName = $FilePath
    $StartInfo.Arguments = $ArgumentString
    $StartInfo.UseShellExecute = $false
    $StartInfo.CreateNoWindow = $true
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true

    $Process = New-Object System.Diagnostics.Process
    $Process.StartInfo = $StartInfo

    if (-not $Process.Start()) {
        throw "Could not start process: $FilePath"
    }

    $StdoutTask = $Process.StandardOutput.ReadToEndAsync()
    $StderrTask = $Process.StandardError.ReadToEndAsync()

    $Process.WaitForExit()

    $Output = @(
        $StdoutTask.Result.TrimEnd()
        $StderrTask.Result.TrimEnd()
    ) | Where-Object { $_ }

    [System.IO.File]::WriteAllText(
        $LogPath,
        ($Output -join [Environment]::NewLine),
        (New-Object System.Text.UTF8Encoding($false))
    )

    if ($Process.ExitCode -ne 0) {
        throw "Command failed with exit code $($Process.ExitCode). See $LogPath"
    }
}

$Quantizer = ".\build\bin\Release\llama-quantize.exe"

foreach ($RequiredPath in @($Quantizer, $InputModel)) {
    if (-not (Test-Path $RequiredPath -PathType Leaf)) {
        throw "Required file not found: $RequiredPath"
    }
}

New-Item -ItemType Directory -Path (Split-Path $OutputModel) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path $LogPath) -Force | Out-Null

Invoke-LoggedNativeProcess `
    -FilePath (Resolve-Path $Quantizer).Path `
    -Arguments @(
        (Resolve-Path $InputModel).Path,
        $OutputModel,
        "Q4_0"
    ) `
    -LogPath $LogPath

Write-Host "Quantization completed: $OutputModel"
Write-Host "Log: $LogPath"
