# Give the space of deleted files inside WSL/Docker back to Windows.
# WSL and Docker virtual disks (.vhdx) grow but never shrink on their own.
#
# Run in PowerShell **as Administrator**, with nothing important running in WSL.
# Copy it to Windows first (the script shuts WSL down, so don't run it from \\wsl.localhost):
#   copy \\wsl.localhost\Ubuntu-24.04\home\aymen\ncm-lab\scripts\compact_wsl_disks.ps1 $HOME\Desktop\
#   powershell -ExecutionPolicy Bypass -File $HOME\Desktop\compact_wsl_disks.ps1
# Works on Windows Home (uses diskpart, not Hyper-V's Optimize-VHD).

$ErrorActionPreference = "Stop"
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this from a PowerShell opened with 'Run as administrator'."
}

# every WSL distro disk + Docker Desktop's data disk
$disks = @()
Get-ChildItem HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss | ForEach-Object {
    $p = Get-ItemProperty $_.PSPath
    if ($p.BasePath) {
        # BasePath can look like \\?\C:\Users\...; Join-Path cannot parse that prefix
        $base = $p.BasePath -replace '^\\\\\?\\', ''
        $f = [System.IO.Path]::Combine($base, "ext4.vhdx")
        if (Test-Path -LiteralPath $f) { $disks += $f }
    }
}
$docker = Join-Path $env:LOCALAPPDATA "Docker\wsl\disk\docker_data.vhdx"
if (Test-Path $docker) { $disks += $docker }

# Linux must first tell the disk which blocks are free, or compacting releases nothing
Get-ChildItem HKCU:\Software\Microsoft\Windows\CurrentVersion\Lxss | ForEach-Object {
    $name = (Get-ItemProperty $_.PSPath).DistributionName
    if ($name -and $name -notlike "docker-desktop*") {
        Write-Host "fstrim in $name..."
        wsl -d $name -u root -- fstrim -av
    }
}

Write-Host "Stopping Docker Desktop and WSL..."
Get-Process "Docker Desktop" -ErrorAction SilentlyContinue | Stop-Process -Force
wsl --shutdown
Start-Sleep -Seconds 5

foreach ($d in $disks) {
    $before = (Get-Item $d).Length / 1GB
    $script = "select vdisk file=`"$d`"`nattach vdisk readonly`ncompact vdisk`ndetach vdisk`n"
    $tmp = New-TemporaryFile
    Set-Content -Path $tmp -Value $script -Encoding ascii
    $msg = diskpart /s $tmp
    $msg | Select-String -Pattern "percent|pourcentage|error|erreur|succe" | ForEach-Object { "    $_" }
    Remove-Item $tmp
    $after = (Get-Item $d).Length / 1GB
    "{0,-90} {1,6:N1} GB -> {2,6:N1} GB" -f $d, $before, $after
}
Write-Host "Done. Start Docker Desktop again when you need it."
