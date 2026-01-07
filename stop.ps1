# Tìm các process python có command line chứa "-m main"
$targets = Get-CimInstance Win32_Process |
    Where-Object {
        $_.Name -match "python" -and $_.CommandLine -match "-m\s+main"
    }

if (-not $targets) {
    Write-Host "Không tìm thấy tiến trình python -m main đang chạy."
    exit 0
}

foreach ($p in $targets) {
    Write-Host "Đang dừng PID $($p.ProcessId): $($p.CommandLine)"
    Stop-Process -Id $p.ProcessId -Force
}

Write-Host "Đã dừng xong."
