# Starts ComfyUI in the background on http://127.0.0.1:8188 (local only) and logs to comfyui.log.
# Set COMFYUI_DIR to the ComfyUI folder (the one with main.py and its .venv).
$root = $env:COMFYUI_DIR
if (-not $root -or -not (Test-Path (Join-Path $root 'main.py'))) { Write-Error 'Set COMFYUI_DIR to your ComfyUI folder.'; return }
$log  = Join-Path $root 'comfyui.log'
$py   = Join-Path $root '.venv\Scripts\python.exe'
Start-Process -FilePath $py -ArgumentList 'main.py', '--listen', '127.0.0.1', '--port', '8188' `
  -WorkingDirectory $root -RedirectStandardOutput $log -RedirectStandardError "$log.err" -WindowStyle Hidden
Write-Output "ComfyUI starting: http://127.0.0.1:8188  (log: $log)"
