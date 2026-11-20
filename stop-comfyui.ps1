# Stops only the ComfyUI server: whatever process listens on port 8188, plus the venv launcher that started it.
$conn = Get-NetTCPConnection -LocalPort 8188 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $conn) { Write-Output 'ComfyUI is not running.'; return }
$proc = Get-CimInstance Win32_Process -Filter "ProcessId = $($conn.OwningProcess)"
$parent = Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.ParentProcessId)" -ErrorAction SilentlyContinue
Stop-Process -Id $proc.ProcessId -Force
if ($parent -and $env:COMFYUI_DIR -and $parent.ExecutablePath -like "$($env:COMFYUI_DIR)\.venv*") { Stop-Process -Id $parent.ProcessId -Force -ErrorAction SilentlyContinue }
Write-Output "Stopped ComfyUI (pid $($proc.ProcessId))"
