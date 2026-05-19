$exe = ".\_main.exe"
$logDir = ".\_main_logs"

# Parameter ranges
$functions  = 1..10
$coatisList = @(200, 400, 600, 800, 1000)
$itersList = @(500, 750, 1000)

New-Item -ItemType Directory -Force -Path $logDir | Out-Null

foreach ($F in $functions) {
    foreach ($C in $coatisList) {
        foreach ($I in $itersList) {

            $logFile = "$logDir\F${F}_C${C}_I${I}.log"

            Write-Host "Running SEQ: F=$F C=$C I=$I"

            & $exe $F $C $I
        }
    }
}

Write-Host "Sequential runs completed."
