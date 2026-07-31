[CmdletBinding()]
param(
    [string]$Model = ".\models\llama-3.2-3b-instruct-q4_0.gguf",
    [string]$OutputDir = ".\logs\task1-final",

    [int]$Threads = 10,
    [int]$ContextSize = 4096,
    [int]$InferenceTokens = 128,

    [int]$SingleTokenRepetitions = 10,
    [int]$FirstTokenRepetitions = 10,
    [int]$PerformanceRepetitions = 5,
    [int]$MemoryRepetitions = 5,

    [int]$TimeoutSeconds = 600
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Cli = ".\build\bin\Release\llama-cli.exe"
$Bench = ".\build\bin\Release\llama-bench.exe"

$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)

# ============================================================================
# Helpers
# ============================================================================

function Write-Utf8File {
    param(
        [Parameter(Mandatory)]
        [string]$Path,

        [AllowEmptyString()]
        [string]$Content
    )

    [System.IO.File]::WriteAllText(
        $Path,
        $Content,
        $script:Utf8NoBom
    )
}

function ConvertTo-NativeArgument {
    param(
        [AllowEmptyString()]
        [string]$Value
    )

    if ($null -eq $Value -or $Value.Length -eq 0) {
        return '""'
    }

    # Для наших аргументов достаточно заключить значения с пробелами
    # в кавычки. Пути и prompts передаются без вложенных кавычек.
    if ($Value -match '[\s"]') {
        $Escaped = $Value.Replace('"', '\"')
        return '"' + $Escaped + '"'
    }

    return $Value
}

function ConvertTo-NativeArgumentString {
    param(
        [string[]]$Arguments
    )

    $EscapedArguments = @(
        foreach ($Argument in $Arguments) {
            ConvertTo-NativeArgument -Value ([string]$Argument)
        }
    )

    return ($EscapedArguments -join " ")
}

function Get-CombinedOutput {
    param(
        [Parameter(Mandatory)]
        $Result
    )

    $Parts = @()

    if ($Result.Stdout -and $Result.Stdout.Trim()) {
        $Parts += $Result.Stdout.Trim()
    }

    if ($Result.Stderr -and $Result.Stderr.Trim()) {
        $Parts += $Result.Stderr.Trim()
    }

    return ($Parts -join [Environment]::NewLine)
}

function Invoke-NativeProcess {
    param(
        [Parameter(Mandatory)]
        [string]$FilePath,

        [string[]]$Arguments = @(),

        [string]$StdoutPath,
        [string]$StderrPath,

        [switch]$MeasureMemory,

        [int]$TimeoutSeconds = 0,
        [int]$PollIntervalMilliseconds = 20
    )

    $ArgumentString = ConvertTo-NativeArgumentString `
        -Arguments $Arguments

    $StartInfo = New-Object System.Diagnostics.ProcessStartInfo
    $StartInfo.FileName = $FilePath
    $StartInfo.Arguments = $ArgumentString
    $StartInfo.WorkingDirectory = (Get-Location).Path

    $StartInfo.UseShellExecute = $false
    $StartInfo.CreateNoWindow = $true
    $StartInfo.RedirectStandardOutput = $true
    $StartInfo.RedirectStandardError = $true

    $Process = New-Object System.Diagnostics.Process
    $Process.StartInfo = $StartInfo

    $Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

    if (-not $Process.Start()) {
        throw "Could not start process: $FilePath"
    }

    # Читаем оба потока асинхронно, чтобы избежать блокировки,
    # если программа много пишет в stdout или stderr.
    $StdoutTask = $Process.StandardOutput.ReadToEndAsync()
    $StderrTask = $Process.StandardError.ReadToEndAsync()

    [long]$ObservedPeakWorkingSet = 0
    [long]$ObservedPeakPrivateMemory = 0

    $TimedOut = $false

    while (-not $Process.HasExited) {
        if ($MeasureMemory) {
            try {
                $Process.Refresh()

                if ($Process.WorkingSet64 -gt $ObservedPeakWorkingSet) {
                    $ObservedPeakWorkingSet = $Process.WorkingSet64
                }

                if (
                    $Process.PrivateMemorySize64 -gt
                    $ObservedPeakPrivateMemory
                ) {
                    $ObservedPeakPrivateMemory =
                        $Process.PrivateMemorySize64
                }
            }
            catch {
                # Процесс мог завершиться между HasExited и Refresh().
            }
        }

        if (
            $TimeoutSeconds -gt 0 -and
            $Stopwatch.Elapsed.TotalSeconds -ge $TimeoutSeconds
        ) {
            $TimedOut = $true

            try {
                $Process.Kill()
            }
            catch {
            }

            break
        }

        Start-Sleep -Milliseconds $PollIntervalMilliseconds
    }

    $Process.WaitForExit()
    $Stopwatch.Stop()

    $Stdout = $StdoutTask.Result
    $Stderr = $StderrTask.Result

    [long]$WindowsPeakWorkingSet = 0

    if ($MeasureMemory) {
        try {
            $Process.Refresh()
            $WindowsPeakWorkingSet = $Process.PeakWorkingSet64
        }
        catch {
        }
    }

    $PeakWorkingSet = [Math]::Max(
        $ObservedPeakWorkingSet,
        $WindowsPeakWorkingSet
    )

    $ExitCode = if ($TimedOut) {
        $null
    }
    else {
        $Process.ExitCode
    }

    if ($StdoutPath) {
        Write-Utf8File `
            -Path $StdoutPath `
            -Content $Stdout
    }

    if ($StderrPath) {
        Write-Utf8File `
            -Path $StderrPath `
            -Content $Stderr
    }

    return [PSCustomObject]@{
        FilePath               = $FilePath
        Arguments              = $ArgumentString
        ExitCode               = $ExitCode
        TimedOut               = $TimedOut
        ElapsedSeconds         = $Stopwatch.Elapsed.TotalSeconds
        Stdout                 = $Stdout
        Stderr                 = $Stderr
        PeakWorkingSetBytes    = $PeakWorkingSet
        PeakPrivateMemoryBytes = $ObservedPeakPrivateMemory
    }
}

function Assert-NativeSuccess {
    param(
        [Parameter(Mandatory)]
        $Result,

        [Parameter(Mandatory)]
        [string]$Name
    )

    if ($Result.TimedOut) {
        throw "$Name timed out."
    }

    if ($Result.ExitCode -ne 0) {
        $Output = Get-CombinedOutput -Result $Result

        throw (
            "{0} failed with exit code {1}.`n{2}" -f
            $Name,
            $Result.ExitCode,
            $Output
        )
    }
}

# ============================================================================
# Validate input files
# ============================================================================

foreach ($RequiredFile in @($Model, $Cli, $Bench)) {
    if (-not (Test-Path $RequiredFile -PathType Leaf)) {
        throw "Required file not found: $RequiredFile"
    }
}

$ModelPath = (Resolve-Path $Model).Path
$CliPath = (Resolve-Path $Cli).Path
$BenchPath = (Resolve-Path $Bench).Path

if (Test-Path $OutputDir) {
    Remove-Item `
        -Path $OutputDir `
        -Recurse `
        -Force
}

New-Item `
    -ItemType Directory `
    -Path $OutputDir `
    -Force |
    Out-Null

$OutputPath = (Resolve-Path $OutputDir).Path

Write-Host ""
Write-Host "Output directory:"
Write-Host $OutputPath

# ============================================================================
# Read CLI help and determine the single-turn flag
# ============================================================================

Write-Host ""
Write-Host "Reading llama-cli parameters..."

$CliHelpResult = Invoke-NativeProcess `
    -FilePath $CliPath `
    -Arguments @("--help") `
    -TimeoutSeconds 60

Assert-NativeSuccess `
    -Result $CliHelpResult `
    -Name "llama-cli --help"

$CliHelpText = Get-CombinedOutput -Result $CliHelpResult

Write-Utf8File `
    -Path (Join-Path $OutputPath "llama-cli-help.txt") `
    -Content $CliHelpText

if ($CliHelpText -match "--single-turn") {
    $SingleTurnArgument = "--single-turn"
}
elseif ($CliHelpText -match "(^|\s)-st([,\s]|$)") {
    $SingleTurnArgument = "-st"
}
else {
    throw (
        "This llama-cli build does not expose a single-turn flag. " +
        "Check llama-cli-help.txt."
    )
}

Write-Host "Single-turn argument: $SingleTurnArgument"

$BenchHelpResult = Invoke-NativeProcess `
    -FilePath $BenchPath `
    -Arguments @("--help") `
    -TimeoutSeconds 60

# Некоторые версии CLI могут завершать --help ненулевым кодом,
# поэтому здесь сохраняем вывод, но не требуем ExitCode 0.
$BenchHelpText = Get-CombinedOutput -Result $BenchHelpResult

Write-Utf8File `
    -Path (Join-Path $OutputPath "llama-bench-help.txt") `
    -Content $BenchHelpText

# ============================================================================
# Environment information
# ============================================================================

Write-Host ""
Write-Host "Collecting environment information..."

$GitResult = Invoke-NativeProcess `
    -FilePath "git.exe" `
    -Arguments @("rev-parse", "HEAD") `
    -TimeoutSeconds 60

Assert-NativeSuccess `
    -Result $GitResult `
    -Name "git rev-parse HEAD"

$CliVersionResult = Invoke-NativeProcess `
    -FilePath $CliPath `
    -Arguments @("--version") `
    -TimeoutSeconds 60

Assert-NativeSuccess `
    -Result $CliVersionResult `
    -Name "llama-cli --version"

$Cpu = Get-CimInstance Win32_Processor
$OperatingSystem = Get-CimInstance Win32_OperatingSystem
$ComputerSystem = Get-CimInstance Win32_ComputerSystem

$ModelFile = Get-Item $ModelPath
$ModelHash = Get-FileHash `
    -Path $ModelPath `
    -Algorithm SHA256

$EnvironmentLines = @(
    "Task 1 environment"
    "=================="
    ""
    "Timestamp:"
    "$(Get-Date -Format o)"
    ""
    "Source model:"
    "meta-llama/Llama-3.2-3B-Instruct"
    ""
    "llama.cpp commit:"
    $GitResult.Stdout.Trim()
    ""
    "llama-cli version:"
    (Get-CombinedOutput -Result $CliVersionResult)
    ""
    "llama-bench executable:"
    $BenchPath
    ""
    "Model path:"
    $ModelPath
    ""
    "Model size:"
    "$($ModelFile.Length) bytes"
    "$([Math]::Round($ModelFile.Length / 1GB, 3)) GiB"
    ""
    "Model SHA256:"
    $ModelHash.Hash
    ""
    "Backend:"
    "CPU"
    ""
    "CPU threads:"
    "$Threads"
    ""
    "Inference context size:"
    "$ContextSize"
    ""
    "CPU:"
    (
        $Cpu |
        Select-Object `
            Name,
            NumberOfCores,
            NumberOfLogicalProcessors,
            MaxClockSpeed |
        Format-List |
        Out-String
    ).Trim()
    ""
    "Operating system:"
    (
        $OperatingSystem |
        Select-Object `
            Caption,
            Version,
            OSArchitecture |
        Format-List |
        Out-String
    ).Trim()
    ""
    "Installed RAM:"
    "$([Math]::Round(
        $ComputerSystem.TotalPhysicalMemory / 1GB,
        2
    )) GiB"
)

$CMakeCachePath = ".\build\CMakeCache.txt"

if (Test-Path $CMakeCachePath -PathType Leaf) {
    $CMakeSettings = Select-String `
        -Path $CMakeCachePath `
        -Pattern @(
            "^GGML_NATIVE:"
            "^GGML_CUDA:"
            "^GGML_VULKAN:"
            "^CMAKE_GENERATOR:"
            "^CMAKE_CXX_COMPILER:"
        )

    $EnvironmentLines += ""
    $EnvironmentLines += "Relevant CMake settings:"

    if ($CMakeSettings) {
        foreach ($Setting in $CMakeSettings) {
            $EnvironmentLines += $Setting.Line
        }
    }
    else {
        $EnvironmentLines += "No matching settings found."
    }
}

Write-Utf8File `
    -Path (Join-Path $OutputPath "environment.txt") `
    -Content ($EnvironmentLines -join [Environment]::NewLine)

Write-Host "Environment information collected."

# ============================================================================
# Inference tests
# ============================================================================

function Invoke-InferenceTest {
    param(
        [Parameter(Mandatory)]
        [string]$Name,

        [Parameter(Mandatory)]
        [string]$Prompt
    )

    Write-Host ""
    Write-Host "Running inference: $Name"

    $PromptPath = Join-Path `
        $OutputPath `
        "$Name.prompt.txt"

    $StdoutPath = Join-Path `
        $OutputPath `
        "$Name.stdout.log"

    $StderrPath = Join-Path `
        $OutputPath `
        "$Name.stderr.log"

    Write-Utf8File `
        -Path $PromptPath `
        -Content $Prompt

    # Prompt передаётся через файл:
    # так нет проблем с пробелами и кавычками в PowerShell.
    $Arguments = @(
        "-m", $ModelPath,
        "-f", $PromptPath,
        "-n", "$InferenceTokens",
        "-t", "$Threads",
        "-tb", "$Threads",
        "-c", "$ContextSize",
        "--seed", "42",
        $SingleTurnArgument
    )

    $Result = Invoke-NativeProcess `
        -FilePath $CliPath `
        -Arguments $Arguments `
        -StdoutPath $StdoutPath `
        -StderrPath $StderrPath `
        -TimeoutSeconds $TimeoutSeconds

    Assert-NativeSuccess `
        -Result $Result `
        -Name "Inference test '$Name'"

    Write-Host (
        "Completed in {0:N2} seconds." -f
        $Result.ElapsedSeconds
    )
}

Invoke-InferenceTest `
    -Name "inference-bitcoin" `
    -Prompt "What is bitcoin?"

Invoke-InferenceTest `
    -Name "inference-python" `
    -Prompt "Write a Python function to reverse a list."

# ============================================================================
# Benchmark helper
# ============================================================================

function Invoke-BenchmarkTest {
    param(
        [Parameter(Mandatory)]
        [string]$Name,

        [Parameter(Mandatory)]
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "Running benchmark: $Name"

    $JsonPath = Join-Path `
        $OutputPath `
        "$Name.json"

    $StderrPath = Join-Path `
        $OutputPath `
        "$Name.stderr.log"

    $Result = Invoke-NativeProcess `
        -FilePath $BenchPath `
        -Arguments $Arguments `
        -StdoutPath $JsonPath `
        -StderrPath $StderrPath `
        -TimeoutSeconds $TimeoutSeconds

    Assert-NativeSuccess `
        -Result $Result `
        -Name "Benchmark '$Name'"

    Write-Host (
        "Completed in {0:N2} seconds." -f
        $Result.ElapsedSeconds
    )
}

# ============================================================================
# Benchmark 1: pp1
#
# Совместимая с предыдущим отчётом метрика:
# обработка одного synthetic prompt token.
# ============================================================================

Invoke-BenchmarkTest `
    -Name "bench-pp1" `
    -Arguments @(
        "-m", $ModelPath,
        "-p", "1",
        "-n", "0",
        "-t", "$Threads",
        "-r", "$SingleTokenRepetitions",
        "-o", "json"
    )

# ============================================================================
# Benchmark 2: pg32,1
#
# Обработка synthetic prompt из 32 токенов и вычисление
# одного выходного токена.
# ============================================================================

Invoke-BenchmarkTest `
    -Name "bench-pg32-1" `
    -Arguments @(
        "-m", $ModelPath,
        "-p", "0",
        "-n", "0",
        "-pg", "32,1",
        "-t", "$Threads",
        "-r", "$FirstTokenRepetitions",
        "-o", "json"
    )

# ============================================================================
# Benchmark 3: pp512 + tg128
#
# pp512 — скорость prompt processing.
# tg128 — средняя скорость последовательной генерации.
# ============================================================================

Invoke-BenchmarkTest `
    -Name "bench-performance" `
    -Arguments @(
        "-m", $ModelPath,
        "-p", "512",
        "-n", "128",
        "-t", "$Threads",
        "-r", "$PerformanceRepetitions",
        "-o", "json"
    )

# ============================================================================
# RAM measurement
#
# Каждый прогон запускает новый процесс llama-bench.
# Внутри llama-bench используется -r 1, поскольку внешний цикл
# уже делает независимые повторы.
# ============================================================================

Write-Host ""
Write-Host "Running RAM measurements..."

$MemoryResults = @()

for ($Run = 1; $Run -le $MemoryRepetitions; $Run++) {
    Write-Host ""
    Write-Host "Memory run $Run / $MemoryRepetitions..."

    $StdoutPath = Join-Path `
        $OutputPath `
        "memory-run-$Run.json"

    $StderrPath = Join-Path `
        $OutputPath `
        "memory-run-$Run.stderr.log"

    $Arguments = @(
        "-m", $ModelPath,
        "-p", "512",
        "-n", "128",
        "-t", "$Threads",
        "-r", "1",
        "-o", "json"
    )

    $Result = Invoke-NativeProcess `
        -FilePath $BenchPath `
        -Arguments $Arguments `
        -StdoutPath $StdoutPath `
        -StderrPath $StderrPath `
        -MeasureMemory `
        -TimeoutSeconds $TimeoutSeconds `
        -PollIntervalMilliseconds 20

    Assert-NativeSuccess `
        -Result $Result `
        -Name "Memory run $Run"

    $MemoryRow = [PSCustomObject]@{
        Run                  = $Run
        ExitCode             = $Result.ExitCode
        TimedOut             = $Result.TimedOut
        ElapsedSeconds       = [Math]::Round(
            $Result.ElapsedSeconds,
            2
        )
        PeakWorkingSetMiB    = [Math]::Round(
            $Result.PeakWorkingSetBytes / 1MB,
            2
        )
        PeakPrivateMemoryMiB = [Math]::Round(
            $Result.PeakPrivateMemoryBytes / 1MB,
            2
        )
        PromptTokens         = 512
        GeneratedTokens      = 128
        Threads              = $Threads
    }

    $MemoryResults += $MemoryRow

    Write-Host (
        "Completed: {0:N2} s, peak working set: {1:N2} MiB" -f
        $MemoryRow.ElapsedSeconds,
        $MemoryRow.PeakWorkingSetMiB
    )
}

$MemoryCsvPath = Join-Path `
    $OutputPath `
    "memory-results.csv"

$MemoryResults |
    Export-Csv `
        -Path $MemoryCsvPath `
        -NoTypeInformation `
        -Encoding UTF8

$AveragePeakWorkingSet = (
    $MemoryResults |
    Measure-Object `
        -Property PeakWorkingSetMiB `
        -Average
).Average

$MinimumPeakWorkingSet = (
    $MemoryResults |
    Measure-Object `
        -Property PeakWorkingSetMiB `
        -Minimum
).Minimum

$MaximumPeakWorkingSet = (
    $MemoryResults |
    Measure-Object `
        -Property PeakWorkingSetMiB `
        -Maximum
).Maximum

$AveragePeakPrivateMemory = (
    $MemoryResults |
    Measure-Object `
        -Property PeakPrivateMemoryMiB `
        -Average
).Average

$MemorySummary = @(
    "RAM measurement summary"
    "======================="
    ""
    "Independent process runs: $MemoryRepetitions"
    "Workload: pp512 + tg128"
    "CPU threads: $Threads"
    ""
    (
        "Average peak working set: {0:N2} MiB ({1:N3} GiB)" -f
        $AveragePeakWorkingSet,
        ($AveragePeakWorkingSet / 1024)
    )
    (
        "Minimum peak working set: {0:N2} MiB" -f
        $MinimumPeakWorkingSet
    )
    (
        "Maximum peak working set: {0:N2} MiB" -f
        $MaximumPeakWorkingSet
    )
    (
        "Average peak private memory: {0:N2} MiB" -f
        $AveragePeakPrivateMemory
    )
)

Write-Utf8File `
    -Path (Join-Path $OutputPath "memory-summary.txt") `
    -Content ($MemorySummary -join [Environment]::NewLine)

Write-Host ""
$MemoryResults | Format-Table -AutoSize

Write-Host ""
Write-Host (
    "Average peak working set: {0:N2} MiB" -f
    $AveragePeakWorkingSet
)

Write-Host (
    "Maximum peak working set: {0:N2} MiB" -f
    $MaximumPeakWorkingSet
)

# ============================================================================
# Copy conversion and quantization logs
# ============================================================================

Write-Host ""
Write-Host "Copying conversion and quantization logs..."

$ExistingLogs = @(
    ".\logs\conversion.log",
    ".\logs\quantization-q4_0.log"
)

foreach ($LogPath in $ExistingLogs) {
    if (Test-Path $LogPath -PathType Leaf) {
        Copy-Item `
            -Path $LogPath `
            -Destination $OutputPath `
            -Force

        Write-Host "Copied: $LogPath"
    }
    else {
        Write-Warning "Log not found: $LogPath"
    }
}

# Кладём в результаты точную версию самого скрипта.
if ($PSCommandPath -and (Test-Path $PSCommandPath -PathType Leaf)) {
    Copy-Item `
        -Path $PSCommandPath `
        -Destination (
            Join-Path $OutputPath "Run-Task1FinalTests.ps1"
        ) `
        -Force
}

# ============================================================================
# Create archive
# ============================================================================

$ArchivePath = Join-Path `
    (Get-Location).Path `
    "task1-final-results.zip"

if (Test-Path $ArchivePath -PathType Leaf) {
    Remove-Item `
        -Path $ArchivePath `
        -Force
}

Compress-Archive `
    -Path (Join-Path $OutputPath "*") `
    -DestinationPath $ArchivePath `
    -Force

Write-Host ""
Write-Host "============================================================"
Write-Host "Task 1 tests completed successfully."
Write-Host "============================================================"
Write-Host ""
Write-Host "Results directory:"
Write-Host $OutputPath
Write-Host ""
Write-Host "Archive:"
Write-Host $ArchivePath
