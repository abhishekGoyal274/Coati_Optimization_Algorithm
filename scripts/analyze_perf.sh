#!/bin/bash
# Quick performance analysis of log files

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "================================================================="
echo " SEQUENTIAL TIMINGS"
echo "================================================================="
printf "%-30s | %8s | %s\n" "FILE" "TIME(ms)" "BEST_SCORE"
echo "-------------------------------|----------|------------------------"
for f in logs/sequential/*.log; do
    fname=$(basename "$f")
    time_ms=$(grep "TIME_MS" "$f" | awk '{print $2}')
    best=$(grep "BEST_SCORE" "$f" | awk '{print $2}')
    printf "%-30s | %8s | %s\n" "$fname" "$time_ms" "$best"
done

echo ""
echo "================================================================="
echo " OPENMP TIMINGS"
echo "================================================================="
printf "%-30s | %8s | %s\n" "FILE" "TIME(ms)" "BEST_SCORE"
echo "-------------------------------|----------|------------------------"
for f in logs/openmp/*.log; do
    fname=$(basename "$f")
    time_ms=$(grep "TIME_MS" "$f" | awk '{print $2}')
    best=$(grep "BEST_SCORE" "$f" | awk '{print $2}')
    printf "%-30s | %8s | %s\n" "$fname" "$time_ms" "$best"
done

echo ""
echo "================================================================="
echo " SEQ vs OMP COMPARISON (matching configs)"
echo "================================================================="
printf "%-25s | %8s | %8s | %s\n" "CONFIG" "SEQ(ms)" "OMP(ms)" "SPEEDUP"
echo "--------------------------|----------|----------|--------"
for sf in logs/sequential/*.log; do
    fname=$(basename "$sf")
    base_no_ext="${fname%.log}"
    
    # Loop over all matching OpenMP logs for different thread counts
    for of in logs/openmp/${base_no_ext}_T*.log; do
        if [ -f "$of" ]; then
            ofname=$(basename "$of")
            threads=$(echo "$ofname" | sed -E 's/.*_T([0-9]+)\.log/\1/')
            
            seq_time=$(grep "TIME_MS" "$sf" | awk '{print $2}')
            omp_time=$(grep "TIME_MS" "$of" | awk '{print $2}')
            if [ -n "$seq_time" ] && [ -n "$omp_time" ] && [ "$omp_time" -gt 0 ] 2>/dev/null; then
                speedup=$(awk "BEGIN {printf \"%.2fx\", $seq_time / $omp_time}")
                printf "%-25s | %8s | %8s | %s\n" "${base_no_ext}_T${threads}" "$seq_time" "$omp_time" "$speedup"
            fi
        fi
    done
done
