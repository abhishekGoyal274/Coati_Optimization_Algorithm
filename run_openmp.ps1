$exe = ".\_main_openmp.exe"
$logDir = ".\_main_openmp_logs"

# Parameter ranges
$functions  = 1..10
$coatisList = @(200, 400, 600, 800, 1000)
$itersList = @(500, 750, 1000)

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

foreach ($F in $functions) {
    foreach ($C in $coatisList) {
        foreach ($I in $itersList) {

            Write-Host "Running SEQ: F=$F C=$C I=$I"

            & $exe $F $C $I
        }
    }
}

Write-Host "OpenMP runs completed."

