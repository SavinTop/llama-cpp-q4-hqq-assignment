[CmdletBinding()]
param(
    [string]$SourceModel = ".\models\Llama-3.2-3B-Instruct",
    [string]$OutputModel = ".\models\llama-3.2-3b-instruct-f16.gguf",
    [string]$LogPath = ".\logs\conversion.log"
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

if (-not (Test-Path $SourceModel)) {
    throw "Source model directory not found: $SourceModel"
}

New-Item -ItemType Directory -Path (Split-Path $OutputModel) -Force | Out-Null
New-Item -ItemType Directory -Path (Split-Path $LogPath) -Force | Out-Null

Invoke-LoggedNativeProcess `
    -FilePath "python.exe" `
    -Arguments @(
        ".\convert_hf_to_gguf.py",
        $SourceModel,
        "--outtype", "f16",
        "--outfile", $OutputModel
    ) `
    -LogPath $LogPath

Write-Host "Conversion completed: $OutputModel"
Write-Host "Log: $LogPath"
