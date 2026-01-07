$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$PythonExe = "python"

$OutLog = Join-Path $ProjectDir "stdout.log"
$ErrLog = Join-Path $ProjectDir "stderr.log"

Start-Process `
    -FilePath $PythonExe `
    -ArgumentList "-m main" `
    -WorkingDirectory $ProjectDir `
    -WindowStyle Hidden `
    -RedirectStandardOutput $OutLog `
    -RedirectStandardError $ErrLog
