$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$PythonExe = "python"

$OutLog = Join-Path $ProjectDir "stdout.log"
$ErrLog = Join-Path $ProjectDir "stderr.log"

# reset log mỗi lần chạy
Clear-Content $OutLog -ErrorAction SilentlyContinue
Clear-Content $ErrLog -ErrorAction SilentlyContinue

Start-Process `
    -FilePath $PythonExe `
    -ArgumentList "-u -m main" `
    -WorkingDirectory $ProjectDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog
