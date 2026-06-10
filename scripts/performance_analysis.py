#!/usr/bin/env python3
"""
Performance Analysis for Coati Optimization Algorithm
=====================================================
Parses sequential, OpenMP, and CUDA execution logs, computes metrics,
generates simple, self-explanatory graphs, and writes a self-contained
results folder.

Design philosophy: ONE graph = ONE idea. No complex multi-panel figures.
"""

import os
import re
import sys
import json
import csv
import shutil
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── Configuration ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEQ_LOG_DIR  = PROJECT_ROOT / "logs" / "sequential"
OMP_LOG_DIR  = PROJECT_ROOT / "logs" / "openmp"
CUDA_LOG_DIR = PROJECT_ROOT / "logs" / "cuda"
RESULTS_DIR  = PROJECT_ROOT / "results"

TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_DIR = RESULTS_DIR / TIMESTAMP
LATEST_DIR = RESULTS_DIR / "latest"

# Benchmark function names for readability
FUNC_NAMES = {
    1: "Sphere (F1)",
    2: "Schwefel 2.22 (F2)",
    3: "Schwefel 1.2 (F3)",
    4: "Schwefel 2.21 (F4)",
    5: "Rosenbrock (F5)",
    6: "Step (F6)",
    7: "Quartic Noise (F7)",
    8: "Schwefel 2.26 (F8)",
    9: "Rastrigin (F9)",
    10: "Ackley (F10)",
}

# Short names for axis labels
FUNC_SHORT = {
    1: "F1", 2: "F2", 3: "F3", 4: "F4", 5: "F5",
    6: "F6", 7: "F7", 8: "F8", 9: "F9", 10: "F10",
}

# ── Style setup ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 13,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'legend.fontsize': 11,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Clean color palette
C_SEQ  = '#e74c3c'    # Red — Sequential
C_OMP  = '#3498db'    # Blue — OpenMP
C_CUDA = '#2ecc71'    # Green — CUDA
C_IDEAL = '#95a5a6'   # Gray — Ideal line
THREAD_COLORS = {1: '#bdc3c7', 2: '#3498db', 4: '#2ecc71', 8: '#f39c12', 16: '#9b59b6'}
THREAD_MARKERS = {1: 'o', 2: 's', 4: 'D', 8: '^', 16: 'v'}


# ── Log Parsing ────────────────────────────────────────────────────────────────
def parse_log_file(filepath):
    """Parse a single log file and return a dict of metrics."""
    data = {
        'file': filepath.name,
        'func': None, 'coatis': None, 'iters': None,
        'impl': None, 'threads': None, 'time_ms': None,
        'best_score': None, 'convergence': [],
    }
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('RUN '):
                m = re.search(r'F=(\d+)', line)
                if m: data['func'] = int(m.group(1))
                m = re.search(r'COATIS=(\d+)', line)
                if m: data['coatis'] = int(m.group(1))
                m = re.search(r'ITER=(\d+)', line)
                if m: data['iters'] = int(m.group(1))
                m = re.search(r'IMPL=(\w+)', line)
                if m: data['impl'] = m.group(1)
            elif line.startswith('ITER '):
                parts = line.split()
                if len(parts) == 3:
                    try:
                        data['convergence'].append((int(parts[1]), float(parts[2])))
                    except ValueError:
                        pass
            elif line.startswith('TIME_MS'):
                parts = line.split()
                if len(parts) == 2:
                    try:
                        data['time_ms'] = int(parts[1])
                    except ValueError:
                        data['time_ms'] = float(parts[1])
            elif line.startswith('BEST_SCORE'):
                parts = line.split()
                if len(parts) == 2:
                    try:
                        data['best_score'] = float(parts[1])
                    except ValueError:
                        pass
    if data['impl'] == 'OpenMP':
        m = re.search(r'_T(\d+)', filepath.name)
        if m:
            data['threads'] = int(m.group(1))
    return data


def parse_all_logs():
    """Parse all log files and return a DataFrame."""
    records = []
    warnings = []

    if SEQ_LOG_DIR.exists():
        for logfile in sorted(SEQ_LOG_DIR.glob('*.log')):
            try:
                rec = parse_log_file(logfile)
                rec['threads'] = 1
                records.append(rec)
            except Exception as e:
                warnings.append(f"[WARN] Failed to parse {logfile.name}: {e}")
    else:
        warnings.append(f"[WARN] Sequential log directory not found: {SEQ_LOG_DIR}")

    if OMP_LOG_DIR.exists():
        for logfile in sorted(OMP_LOG_DIR.glob('*.log')):
            try:
                records.append(parse_log_file(logfile))
            except Exception as e:
                warnings.append(f"[WARN] Failed to parse {logfile.name}: {e}")
    else:
        warnings.append(f"[WARN] OpenMP log directory not found: {OMP_LOG_DIR}")

    if CUDA_LOG_DIR.exists():
        for logfile in sorted(CUDA_LOG_DIR.glob('*.log')):
            try:
                rec = parse_log_file(logfile)
                rec['threads'] = None
                records.append(rec)
            except Exception as e:
                warnings.append(f"[WARN] Failed to parse {logfile.name}: {e}")
    else:
        warnings.append(f"[WARN] CUDA log directory not found: {CUDA_LOG_DIR}")

    rows = []
    for rec in records:
        rows.append({
            'file': rec['file'], 'func': rec['func'],
            'coatis': rec['coatis'], 'iters': rec['iters'],
            'impl': rec['impl'], 'threads': rec['threads'],
            'time_ms': rec['time_ms'], 'best_score': rec['best_score'],
        })
    df = pd.DataFrame(rows)
    return df, records, warnings


# ── Metric Computation ─────────────────────────────────────────────────────────
def compute_metrics(df):
    """Compute speedup, efficiency, and other derived metrics."""
    seq = df[df['impl'] == 'SEQ'].copy()
    seq_baseline = seq.groupby(['func', 'coatis', 'iters'])['time_ms'].mean().reset_index()
    seq_baseline.rename(columns={'time_ms': 'seq_time_ms'}, inplace=True)

    omp = df[df['impl'] == 'OpenMP'].copy()
    merged = omp.merge(seq_baseline, on=['func', 'coatis', 'iters'], how='left')
    merged['speedup'] = merged['seq_time_ms'] / merged['time_ms']
    merged['efficiency'] = merged['speedup'] / merged['threads']
    merged['workload'] = merged['coatis'] * merged['iters']
    seq['workload'] = seq['coatis'] * seq['iters']

    cuda = df[df['impl'] == 'CUDA'].copy()
    if len(cuda) > 0:
        cuda_merged = cuda.merge(seq_baseline, on=['func', 'coatis', 'iters'], how='left')
        cuda_merged['speedup'] = cuda_merged['seq_time_ms'] / cuda_merged['time_ms']
        cuda_merged['workload'] = cuda_merged['coatis'] * cuda_merged['iters']
    else:
        cuda_merged = pd.DataFrame()

    return merged, seq, seq_baseline, cuda_merged


# ── Graph Helpers ──────────────────────────────────────────────────────────────

def save_fig(fig, name):
    """Save a figure to the output directory."""
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [SAVED] {path.name}")
    return path


def simple_fig(width=10, height=6):
    """Create a clean single-axis figure."""
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


# ══════════════════════════════════════════════════════════════════════════════
#  SIMPLE GRAPHS — One graph, one idea
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────── SPEEDUP GRAPHS ───────────────────────

def plot_01_avg_speedup_vs_threads(merged):
    """Simple line: average speedup vs thread count."""
    fig, ax = simple_fig()
    avg = merged.groupby('threads')['speedup'].mean()
    threads = sorted(avg.index)

    ax.plot(threads, [avg[t] for t in threads], 'o-', color=C_OMP,
            linewidth=2.5, markersize=10, label='Actual Speedup')
    ax.plot(threads, threads, '--', color=C_IDEAL,
            linewidth=2, label='Ideal (Linear) Speedup')
    ax.axhline(y=1.0, color='black', linewidth=0.8, alpha=0.3)

    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Average Speedup')
    ax.set_title('Average OpenMP Speedup vs Thread Count')
    ax.set_xticks(threads)
    ax.legend(fontsize=12)
    fig.tight_layout()
    return save_fig(fig, '01_avg_speedup_vs_threads')


def plot_02_speedup_per_function_best(merged):
    """Horizontal bar: best avg speedup per function (at optimal thread count)."""
    fig, ax = simple_fig(10, 7)
    best = merged.groupby('func')['speedup'].mean().sort_values(ascending=True)
    labels = [FUNC_NAMES.get(f, f"F{f}") for f in best.index]
    colors = [C_OMP if v >= 1 else C_SEQ for v in best.values]

    ax.barh(labels, best.values, color=colors, edgecolor='white', linewidth=0.5, height=0.6)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Baseline (1×)')

    for i, v in enumerate(best.values):
        ax.text(v + 0.02, i, f'{v:.2f}×', va='center', fontsize=11)

    ax.set_xlabel('Average Speedup')
    ax.set_title('Average OpenMP Speedup per Function')
    ax.legend(fontsize=11)
    fig.tight_layout()
    return save_fig(fig, '02_speedup_per_function')


def plot_03_speedup_at_each_thread_count(merged):
    """Grouped bar: avg speedup per function at each thread count."""
    fig, ax = simple_fig(14, 7)
    funcs = list(range(1, 11))
    threads_list = sorted(merged['threads'].unique())
    x = np.arange(len(funcs))
    width = 0.8 / len(threads_list)

    for i, t in enumerate(threads_list):
        sub = merged[merged['threads'] == t]
        avg = sub.groupby('func')['speedup'].mean()
        vals = [avg.get(f, 0) for f in funcs]
        ax.bar(x + i * width - 0.4 + width/2, vals, width,
               label=f'T={int(t)}', color=THREAD_COLORS.get(t, '#333'),
               alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels([FUNC_SHORT[f] for f in funcs], fontsize=12)
    ax.set_xlabel('Benchmark Function')
    ax.set_ylabel('Average Speedup')
    ax.set_title('OpenMP Speedup by Function and Thread Count')
    ax.legend(title='Threads', fontsize=10)
    fig.tight_layout()
    return save_fig(fig, '03_speedup_by_function_and_threads')


# ─────────────────────── EFFICIENCY GRAPHS ───────────────────────

def plot_04_efficiency_vs_threads(merged):
    """Simple bar: average parallel efficiency at each thread count."""
    fig, ax = simple_fig()
    avg = merged.groupby('threads')['efficiency'].mean()
    threads = sorted(avg.index)
    colors = [THREAD_COLORS.get(int(t), '#333') for t in threads]

    bars = ax.bar([str(int(t)) for t in threads], [avg[t] for t in threads],
                  color=colors, edgecolor='white', linewidth=0.5, width=0.6)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Ideal (100%)')

    for bar, val in zip(bars, [avg[t] for t in threads]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Parallel Efficiency')
    ax.set_title('Average Parallel Efficiency by Thread Count')
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11)
    fig.tight_layout()
    return save_fig(fig, '04_efficiency_vs_threads')


def plot_05_efficiency_per_function(merged):
    """Bar chart: efficiency at T=8 for each function."""
    fig, ax = simple_fig(12, 6)
    t8 = merged[merged['threads'] == 8]
    if t8.empty:
        t8 = merged[merged['threads'] == merged['threads'].max()]

    avg = t8.groupby('func')['efficiency'].mean().sort_values(ascending=True)
    labels = [FUNC_NAMES.get(f, f"F{f}") for f in avg.index]
    colors = ['#2ecc71' if v >= 0.3 else '#f39c12' if v >= 0.15 else '#e74c3c' for v in avg.values]

    ax.barh(labels, avg.values, color=colors, edgecolor='white', linewidth=0.5, height=0.6)

    for i, v in enumerate(avg.values):
        ax.text(v + 0.005, i, f'{v:.1%}', va='center', fontsize=11)

    ax.set_xlabel('Parallel Efficiency')
    ax.set_title('Parallel Efficiency per Function (T=8)')
    fig.tight_layout()
    return save_fig(fig, '05_efficiency_per_function')


# ─────────────────────── EXECUTION TIME GRAPHS ───────────────────────

def plot_06_avg_time_seq_vs_omp(merged, seq):
    """Simple grouped bar: Seq vs best-OMP avg time per function."""
    fig, ax = simple_fig(12, 6)
    funcs = list(range(1, 11))

    seq_avg = seq.groupby('func')['time_ms'].mean()
    # Use T=8 as the best practical OpenMP config
    omp8 = merged[merged['threads'] == 8]
    if omp8.empty:
        omp8 = merged[merged['threads'] == 4]
    omp_avg = omp8.groupby('func')['time_ms'].mean()

    x = np.arange(len(funcs))
    width = 0.35

    ax.bar(x - width/2, [seq_avg.get(f, 0) for f in funcs], width,
           label='Sequential', color=C_SEQ, edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, [omp_avg.get(f, 0) for f in funcs], width,
           label='OpenMP (T=8)', color=C_OMP, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([FUNC_SHORT[f] for f in funcs], fontsize=12)
    ax.set_xlabel('Benchmark Function')
    ax.set_ylabel('Average Time (ms)')
    ax.set_title('Execution Time: Sequential vs OpenMP (T=8)')
    ax.legend(fontsize=12)
    fig.tight_layout()
    return save_fig(fig, '06_time_seq_vs_omp')


def plot_07_time_vs_population(seq):
    """Line chart: how execution time grows with population size."""
    fig, ax = simple_fig()
    for iters in sorted(seq['iters'].unique()):
        sub = seq[seq['iters'] == iters].groupby('coatis')['time_ms'].mean()
        ax.plot(sub.index, sub.values, 'o-', markersize=8, linewidth=2,
                label=f'{int(iters)} iterations')

    ax.set_xlabel('Population Size (Number of Coatis)')
    ax.set_ylabel('Average Execution Time (ms)')
    ax.set_title('Sequential Execution Time vs Population Size')
    ax.legend(fontsize=11)
    fig.tight_layout()
    return save_fig(fig, '07_time_vs_population')


def plot_08_time_vs_threads_single_func(merged, func_id=8):
    """Line: execution time vs threads for one function (large workload)."""
    fig, ax = simple_fig()
    func_name = FUNC_NAMES.get(func_id, f"F{func_id}")

    # Pick largest workload
    sub = merged[(merged['func'] == func_id) &
                 (merged['coatis'] == merged['coatis'].max()) &
                 (merged['iters'] == merged['iters'].max())]

    if sub.empty:
        sub = merged[merged['func'] == func_id]

    avg = sub.groupby('threads')['time_ms'].mean()
    threads = sorted(avg.index)

    ax.plot(threads, [avg[t] for t in threads], 'o-', color=C_OMP,
            markersize=10, linewidth=2.5)

    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Average Execution Time (ms)')
    ax.set_title(f'Execution Time vs Threads — {func_name}')
    ax.set_xticks(threads)
    fig.tight_layout()
    return save_fig(fig, '08_time_vs_threads_F8')


# ─────────────────────── SCALABILITY GRAPHS ───────────────────────

def plot_09_scalability_heatmap(merged):
    """Clean heatmap: speedup across functions × threads."""
    fig, ax = simple_fig(10, 7)
    pivot = merged.groupby(['func', 'threads'])['speedup'].mean().reset_index()
    heatmap_data = pivot.pivot(index='func', columns='threads', values='speedup')
    heatmap_data.index = [FUNC_SHORT.get(f, f"F{f}") for f in heatmap_data.index]
    heatmap_data.columns = [f'T={int(c)}' for c in heatmap_data.columns]

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', center=1.0,
                linewidths=1.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Speedup'}, vmin=0)

    ax.set_xlabel('Thread Count')
    ax.set_ylabel('Benchmark Function')
    ax.set_title('Speedup Heatmap — Functions × Threads')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return save_fig(fig, '09_scalability_heatmap')


def plot_10_workload_vs_speedup(merged):
    """Line: speedup vs workload size, one line per thread count."""
    fig, ax = simple_fig()
    for t in sorted(merged['threads'].unique()):
        sub = merged[merged['threads'] == t]
        avg = sub.groupby('workload')['speedup'].mean().sort_index()
        ax.plot(avg.index, avg.values, marker=THREAD_MARKERS.get(int(t), 'o'),
                markersize=7, linewidth=2, alpha=0.8,
                color=THREAD_COLORS.get(int(t), '#333'), label=f'T={int(t)}')

    ax.axhline(y=1.0, color='black', linewidth=0.8, alpha=0.3)
    ax.set_xlabel('Workload Size (Coatis × Iterations)')
    ax.set_ylabel('Average Speedup')
    ax.set_title('Speedup vs Workload Size')
    ax.legend(title='Threads', fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K'))
    fig.tight_layout()
    return save_fig(fig, '10_workload_vs_speedup')


# ─────────────────────── OVERHEAD GRAPHS ───────────────────────

def plot_11_omp_overhead_t1(merged):
    """Bar: OpenMP overhead at T=1 — how much slower than sequential."""
    fig, ax = simple_fig()
    t1 = merged[merged['threads'] == 1]
    avg = t1.groupby('func')['speedup'].mean().sort_values(ascending=True)
    labels = [FUNC_SHORT.get(f, f"F{f}") for f in avg.index]

    colors = ['#e74c3c' if v < 1 else '#2ecc71' for v in avg.values]
    ax.barh(labels, avg.values, color=colors, edgecolor='white', linewidth=0.5, height=0.6)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    for i, v in enumerate(avg.values):
        overhead = (1 - v) * 100
        ax.text(v + 0.01, i, f'{overhead:.0f}% overhead', va='center', fontsize=10)

    ax.set_xlabel('Speedup (< 1 = slower than Sequential)')
    ax.set_title('OpenMP Overhead at T=1 vs Sequential')
    fig.tight_layout()
    return save_fig(fig, '11_omp_overhead_at_T1')


def plot_12_slowdown_rate_by_threads(merged):
    """Bar: percentage of runs with slowdown at each thread count."""
    fig, ax = simple_fig()
    threads = sorted(merged['threads'].unique())
    rates = []
    for t in threads:
        sub = merged[merged['threads'] == t]
        rate = 100 * len(sub[sub['speedup'] < 1]) / len(sub)
        rates.append(rate)

    colors = [THREAD_COLORS.get(int(t), '#333') for t in threads]
    bars = ax.bar([f'T={int(t)}' for t in threads], rates,
                  color=colors, edgecolor='white', linewidth=0.5, width=0.6)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{rate:.0f}%', ha='center', fontsize=12, fontweight='bold')

    ax.set_xlabel('Thread Count')
    ax.set_ylabel('Slowdown Rate (%)')
    ax.set_title('Percentage of Runs Slower than Sequential')
    ax.set_ylim(0, 110)
    fig.tight_layout()
    return save_fig(fig, '12_slowdown_rate_by_threads')


def plot_13_speedup_distribution(merged):
    """Box plot: speedup distribution at each thread count."""
    fig, ax = simple_fig()
    thread_order = sorted(merged['threads'].unique())
    colors = [THREAD_COLORS.get(int(t), '#333') for t in thread_order]

    data_to_plot = [merged[merged['threads'] == t]['speedup'].values for t in thread_order]
    bp = ax.boxplot(data_to_plot, labels=[f'T={int(t)}' for t in thread_order],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.4, label='Baseline (1×)')
    ax.set_xlabel('Thread Count')
    ax.set_ylabel('Speedup')
    ax.set_title('Speedup Distribution by Thread Count')
    ax.legend(fontsize=11)
    fig.tight_layout()
    return save_fig(fig, '13_speedup_distribution')


# ─────────────────────── AMDAHL'S LAW GRAPHS ───────────────────────

def plot_14_amdahl_actual_vs_predicted(merged):
    """Line: actual avg speedup vs Amdahl's Law curves."""
    fig, ax = simple_fig()
    avg_speedup = merged.groupby('threads')['speedup'].mean()
    threads = sorted(avg_speedup.index)

    ax.plot(threads, [avg_speedup[t] for t in threads], 'ko-', markersize=10,
            linewidth=2.5, label='Actual (Measured)', zorder=5)

    p_range = np.linspace(1, max(threads), 100)
    for f_s, ls in [(0.05, '-'), (0.1, '--'), (0.2, '-.'), (0.5, ':')]:
        amdahl = 1.0 / (f_s + (1 - f_s) / p_range)
        ax.plot(p_range, amdahl, ls, alpha=0.6, linewidth=1.5,
                label=f'{f_s:.0%} serial')

    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Speedup')
    ax.set_title("Actual Speedup vs Amdahl's Law Predictions")
    ax.set_xticks(threads)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return save_fig(fig, '14_amdahl_vs_actual')


def plot_15_serial_fraction_per_function(merged):
    """Bar: estimated serial fraction per function."""
    fig, ax = simple_fig(10, 7)
    max_t = 8  # Use T=8 (more reliable than T=16 which has overhead issues)

    serial_fractions = {}
    for func_id in range(1, 11):
        func_data = merged[(merged['func'] == func_id) & (merged['threads'] == max_t)]
        if func_data.empty:
            continue
        avg_s = func_data['speedup'].mean()
        if avg_s > 0 and max_t > 1:
            f_est = max(0, min(1, (1/avg_s - 1/max_t) / (1 - 1/max_t)))
            serial_fractions[func_id] = f_est

    if not serial_fractions:
        plt.close(fig)
        return None

    funcs = list(serial_fractions.keys())
    fracs = [serial_fractions[f] for f in funcs]
    labels = [FUNC_NAMES.get(f, f"F{f}") for f in funcs]
    sorted_idx = np.argsort(fracs)

    colors = ['#e74c3c' if f > 0.6 else '#f39c12' if f > 0.4 else '#2ecc71' for f in np.array(fracs)[sorted_idx]]
    ax.barh([labels[i] for i in sorted_idx], [fracs[i] for i in sorted_idx],
            color=colors, edgecolor='white', linewidth=0.5, height=0.6)

    for i, idx in enumerate(sorted_idx):
        ax.text(fracs[idx] + 0.01, i, f'{fracs[idx]:.1%}', va='center', fontsize=11)

    ax.set_xlabel('Estimated Serial Fraction')
    ax.set_title(f'Serial Fraction per Function (Estimated at T={max_t})')
    ax.set_xlim(0, 1.05)
    fig.tight_layout()
    return save_fig(fig, '15_serial_fraction_per_function')


# ─────────────────────── CUDA GRAPHS ───────────────────────

def plot_16_cuda_speedup_per_function(cuda_merged):
    """Bar: CUDA speedup per function."""
    if cuda_merged is None or len(cuda_merged) == 0:
        return None

    fig, ax = simple_fig(10, 7)
    avg = cuda_merged.groupby('func')['speedup'].mean().sort_values(ascending=True)
    labels = [FUNC_NAMES.get(f, f"F{f}") for f in avg.index]
    colors = [C_CUDA if v >= 1 else C_SEQ for v in avg.values]

    ax.barh(labels, avg.values, color=colors, edgecolor='white', linewidth=0.5, height=0.6)
    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)

    for i, v in enumerate(avg.values):
        ax.text(v + 0.03, i, f'{v:.2f}×', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Speedup over Sequential')
    ax.set_title('CUDA GPU Speedup per Function')
    fig.tight_layout()
    return save_fig(fig, '16_cuda_speedup_per_function')


def plot_17_cuda_vs_seq_time(cuda_merged, seq):
    """Grouped bar: Sequential vs CUDA time per function."""
    if cuda_merged is None or len(cuda_merged) == 0:
        return None

    fig, ax = simple_fig(12, 6)
    funcs = sorted(cuda_merged['func'].unique())

    seq_avg = seq.groupby('func')['time_ms'].mean()
    cuda_avg = cuda_merged.groupby('func')['time_ms'].mean()

    x = np.arange(len(funcs))
    width = 0.35

    ax.bar(x - width/2, [seq_avg.get(f, 0) for f in funcs], width,
           label='Sequential (CPU)', color=C_SEQ, edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, [cuda_avg.get(f, 0) for f in funcs], width,
           label='CUDA (GPU)', color=C_CUDA, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([FUNC_SHORT[f] for f in funcs], fontsize=12)
    ax.set_xlabel('Benchmark Function')
    ax.set_ylabel('Average Execution Time (ms)')
    ax.set_title('Execution Time: Sequential vs CUDA')
    ax.legend(fontsize=12)
    fig.tight_layout()
    return save_fig(fig, '17_cuda_vs_seq_time')


def plot_18_cuda_speedup_vs_population(cuda_merged):
    """Line: CUDA speedup vs population size."""
    if cuda_merged is None or len(cuda_merged) == 0:
        return None

    fig, ax = simple_fig()
    avg = cuda_merged.groupby('coatis')['speedup'].mean().sort_index()

    ax.plot(avg.index, avg.values, 'o-', color=C_CUDA, markersize=10, linewidth=2.5)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.4)

    for x, y in zip(avg.index, avg.values):
        ax.text(x, y + 0.05, f'{y:.2f}×', ha='center', fontsize=11)

    ax.set_xlabel('Population Size (Number of Coatis)')
    ax.set_ylabel('Average CUDA Speedup over Sequential')
    ax.set_title('CUDA Speedup vs Population Size')
    fig.tight_layout()
    return save_fig(fig, '18_cuda_speedup_vs_population')


# ─────────────────────── ALL THREE IMPLEMENTATIONS ───────────────────────

def plot_19_all_three_time(merged, seq, cuda_merged):
    """Grouped bar: Seq vs OMP(T=8) vs CUDA time per function."""
    has_cuda = cuda_merged is not None and len(cuda_merged) > 0
    fig, ax = simple_fig(14, 7)
    funcs = list(range(1, 11))

    seq_avg = seq.groupby('func')['time_ms'].mean()
    omp_avg = merged[merged['threads'] == 8].groupby('func')['time_ms'].mean()

    x = np.arange(len(funcs))
    n_bars = 3 if has_cuda else 2
    width = 0.8 / n_bars

    ax.bar(x - 0.4 + width/2, [seq_avg.get(f, 0) for f in funcs], width,
           label='Sequential', color=C_SEQ, edgecolor='white', linewidth=0.5)
    ax.bar(x - 0.4 + 1.5*width, [omp_avg.get(f, 0) for f in funcs], width,
           label='OpenMP (T=8)', color=C_OMP, edgecolor='white', linewidth=0.5)

    if has_cuda:
        cuda_avg = cuda_merged.groupby('func')['time_ms'].mean()
        ax.bar(x - 0.4 + 2.5*width, [cuda_avg.get(f, 0) for f in funcs], width,
               label='CUDA (GPU)', color=C_CUDA, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([FUNC_SHORT[f] for f in funcs], fontsize=12)
    ax.set_xlabel('Benchmark Function')
    ax.set_ylabel('Average Execution Time (ms)')
    ax.set_title('Execution Time: All Implementations')
    ax.legend(fontsize=12)
    fig.tight_layout()
    return save_fig(fig, '19_all_implementations_time')


def plot_20_all_three_speedup(merged, cuda_merged):
    """Grouped bar: OpenMP(T=8) and CUDA speedup per function."""
    has_cuda = cuda_merged is not None and len(cuda_merged) > 0
    fig, ax = simple_fig(12, 6)
    funcs = list(range(1, 11))

    omp_avg = merged[merged['threads'] == 8].groupby('func')['speedup'].mean()

    x = np.arange(len(funcs))
    width = 0.35 if has_cuda else 0.5

    if has_cuda:
        cuda_avg = cuda_merged.groupby('func')['speedup'].mean()
        ax.bar(x - width/2, [omp_avg.get(f, 0) for f in funcs], width,
               label='OpenMP (T=8)', color=C_OMP, edgecolor='white', linewidth=0.5)
        ax.bar(x + width/2, [cuda_avg.get(f, 0) for f in funcs], width,
               label='CUDA (GPU)', color=C_CUDA, edgecolor='white', linewidth=0.5)
    else:
        ax.bar(x, [omp_avg.get(f, 0) for f in funcs], width,
               label='OpenMP (T=8)', color=C_OMP, edgecolor='white', linewidth=0.5)

    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.4, label='Baseline (1×)')
    ax.set_xticks(x)
    ax.set_xticklabels([FUNC_SHORT[f] for f in funcs], fontsize=12)
    ax.set_xlabel('Benchmark Function')
    ax.set_ylabel('Speedup over Sequential')
    ax.set_title('Speedup: OpenMP vs CUDA per Function')
    ax.legend(fontsize=11)
    fig.tight_layout()
    return save_fig(fig, '20_omp_vs_cuda_speedup')


# ─────────────────────── CONVERGENCE GRAPHS ───────────────────────

def plot_21_convergence_single(records, func_id=1):
    """Convergence curve for a single function — Seq vs OMP vs CUDA."""
    fig, ax = simple_fig()
    func_name = FUNC_NAMES.get(func_id, f"F{func_id}")

    plotted = set()
    for rec in records:
        if rec['func'] != func_id or rec['coatis'] != 1000 or rec['iters'] != 1000:
            continue
        conv = rec['convergence']
        if not conv:
            continue
        x_vals = [c[0] for c in conv]
        y_vals = [max(c[1], 1e-320) for c in conv]

        if rec['impl'] == 'SEQ' and 'SEQ' not in plotted:
            ax.plot(x_vals, y_vals, 'o-', color=C_SEQ, markersize=6,
                    linewidth=2, label='Sequential')
            plotted.add('SEQ')
        elif rec['impl'] == 'OpenMP' and rec.get('threads') == 8 and 'OMP' not in plotted:
            ax.plot(x_vals, y_vals, 's-', color=C_OMP, markersize=6,
                    linewidth=2, label='OpenMP (T=8)')
            plotted.add('OMP')
        elif rec['impl'] == 'CUDA' and 'CUDA' not in plotted:
            ax.plot(x_vals, y_vals, 'D-', color=C_CUDA, markersize=6,
                    linewidth=2, label='CUDA')
            plotted.add('CUDA')

    ax.set_yscale('symlog', linthresh=1e-300)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Fitness Score')
    ax.set_title(f'Convergence — {func_name} (N=1000, Iter=1000)')
    ax.legend(fontsize=11)
    fig.tight_layout()
    return save_fig(fig, f'21_convergence_{func_id}')


def plot_22_convergence_F5(records):
    """Convergence for F5 (Rosenbrock)."""
    return plot_21_convergence_single(records, func_id=5)


def plot_23_convergence_F8(records):
    """Convergence for F8 (Schwefel 2.26 — best CUDA function)."""
    return plot_21_convergence_single(records, func_id=8)


def plot_24_convergence_F10(records):
    """Convergence for F10 (Ackley)."""
    return plot_21_convergence_single(records, func_id=10)


# ─────────────────────── EFFICIENCY HEATMAP ───────────────────────

def plot_25_efficiency_heatmap(merged):
    """Heatmap: parallel efficiency across functions × threads."""
    fig, ax = simple_fig(10, 7)
    pivot = merged.groupby(['func', 'threads'])['efficiency'].mean().reset_index()
    heatmap_data = pivot.pivot(index='func', columns='threads', values='efficiency')
    heatmap_data.index = [FUNC_SHORT.get(f, f"F{f}") for f in heatmap_data.index]
    heatmap_data.columns = [f'T={int(c)}' for c in heatmap_data.columns]

    sns.heatmap(heatmap_data, annot=True, fmt='.1%', cmap='RdYlGn',
                linewidths=1.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Efficiency'}, vmin=0, vmax=1.0)

    ax.set_xlabel('Thread Count')
    ax.set_ylabel('Benchmark Function')
    ax.set_title('Parallel Efficiency — Functions × Threads')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    return save_fig(fig, '25_efficiency_heatmap')


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT GENERATION (unchanged logic, updated graph count)
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(df, merged, seq, seq_baseline, warnings, cuda_merged=None):
    """Generate a comprehensive analysis report."""
    has_cuda = cuda_merged is not None and len(cuda_merged) > 0
    lines = []
    lines.append("# Coati Optimization Algorithm — Performance Analysis Report")
    lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    cuda_info = f", `logs/cuda/` ({len(cuda_merged)} entries)" if has_cuda else ""
    lines.append(f"**Data Source**: `logs/sequential/` ({len(seq)} entries), `logs/openmp/` ({len(merged)} entries){cuda_info}")
    lines.append("")

    lines.append("## 1. Data Summary")
    lines.append("")
    lines.append(f"- **Benchmark Functions**: F1–F10 ({len(df['func'].unique())} unique)")
    lines.append(f"- **Population Sizes**: {sorted(int(x) for x in df['coatis'].unique())}")
    lines.append(f"- **Iteration Counts**: {sorted(int(x) for x in df['iters'].unique())}")
    lines.append(f"- **Thread Counts (OpenMP)**: {sorted(int(x) for x in merged['threads'].unique())}")
    lines.append(f"- **Total Sequential Runs**: {len(seq)}")
    lines.append(f"- **Total OpenMP Runs**: {len(merged)}")
    if has_cuda:
        lines.append(f"- **Total CUDA Runs**: {len(cuda_merged)}")
    lines.append(f"- **Total Configurations**: {len(df.groupby(['func', 'coatis', 'iters']))}")
    lines.append("")

    lines.append("## 2. Speedup Analysis")
    lines.append("")
    lines.append("### Overall Speedup Statistics")
    lines.append("")

    thread_counts = sorted(merged['threads'].unique())
    lines.append("| Thread Count | Avg Speedup | Min Speedup | Max Speedup | Std Dev |")
    lines.append("|:---:|:---:|:---:|:---:|:---:|")
    for t in thread_counts:
        sub = merged[merged['threads'] == t]
        lines.append(f"| {t} | {sub['speedup'].mean():.3f} | {sub['speedup'].min():.3f} | "
                     f"{sub['speedup'].max():.3f} | {sub['speedup'].std():.3f} |")
    lines.append("")

    lines.append("### Per-Function Speedup (T=16)")
    lines.append("")
    lines.append("| Function | Avg Speedup | Assessment |")
    lines.append("|:---|:---:|:---|")
    t16 = merged[merged['threads'] == max(thread_counts)]
    for func_id in range(1, 11):
        func_data = t16[t16['func'] == func_id]
        if func_data.empty:
            continue
        avg_s = func_data['speedup'].mean()
        if avg_s >= 10: assessment = "🟢 Excellent"
        elif avg_s >= 5: assessment = "🟡 Good"
        elif avg_s >= 2: assessment = "🟠 Moderate"
        elif avg_s >= 1: assessment = "🔵 Marginal"
        else: assessment = "🔴 Slowdown"
        lines.append(f"| {FUNC_NAMES.get(func_id, f'F{func_id}')} | {avg_s:.3f} | {assessment} |")
    lines.append("")

    lines.append("## 3. Scalability Analysis")
    lines.append("")
    avg_by_threads = merged.groupby('threads')['speedup'].mean()
    for i, t in enumerate(thread_counts):
        if i == 0: continue
        prev_t = thread_counts[i - 1]
        marginal_gain = avg_by_threads.get(t, 0) - avg_by_threads.get(prev_t, 0)
        expected_gain = avg_by_threads.get(prev_t, 0) * (t / prev_t - 1)
        ratio = marginal_gain / expected_gain if expected_gain > 0 else 0
        lines.append(f"- **T={prev_t} → T={t}**: Marginal gain = {marginal_gain:.3f} "
                     f"(linear ratio: {ratio:.1%})")
    lines.append("")

    small_wl = merged[merged['workload'] <= 200000]
    large_wl = merged[merged['workload'] > 500000]
    if len(small_wl) > 0 and len(large_wl) > 0:
        lines.append(f"- **Small workloads** (≤200K): Avg speedup = {small_wl['speedup'].mean():.3f}")
        lines.append(f"- **Large workloads** (>500K): Avg speedup = {large_wl['speedup'].mean():.3f}")
    lines.append("")

    lines.append("## 4. Parallel Efficiency")
    lines.append("")
    lines.append("| Thread Count | Avg Efficiency | Interpretation |")
    lines.append("|:---:|:---:|:---|")
    for t in thread_counts:
        sub = merged[merged['threads'] == t]
        avg_e = sub['efficiency'].mean()
        if avg_e >= 0.5: interp = "Good"
        elif avg_e >= 0.3: interp = "Moderate"
        else: interp = "Poor — diminishing returns"
        lines.append(f"| {t} | {avg_e:.3f} | {interp} |")
    lines.append("")

    lines.append("## 5. Overhead Analysis")
    lines.append("")
    slowdowns = merged[merged['speedup'] < 1.0]
    lines.append(f"- **Total slowdown cases**: {len(slowdowns)} / {len(merged)} ({100*len(slowdowns)/len(merged):.1f}%)")
    t1_omp = merged[merged['threads'] == 1]
    if len(t1_omp) > 0:
        avg_t1 = t1_omp['speedup'].mean()
        lines.append(f"- **OpenMP overhead at T=1**: {(1 - avg_t1)*100:.1f}% slower than sequential")
    lines.append("")

    lines.append("## 6. Key Findings")
    lines.append("")
    overall_speedup_16 = t16['speedup'].mean() if len(t16) > 0 else 0
    lines.append(f"1. **Best thread count**: T=8 (avg speedup: {avg_by_threads.get(8, 0):.2f}×)")
    lines.append(f"2. **T=16 degrades performance**: avg {overall_speedup_16:.2f}×")
    lines.append(f"3. **Slowdown rate**: {100*len(slowdowns)/len(merged):.1f}%")
    if has_cuda:
        lines.append(f"4. **CUDA avg speedup**: {cuda_merged['speedup'].mean():.2f}×")
        lines.append(f"5. **CUDA max speedup**: {cuda_merged['speedup'].max():.2f}×")
    lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by `performance_analysis.py` at {datetime.now().isoformat()}*")
    return "\n".join(lines)


def generate_csv_summary(df, merged, seq, cuda_merged=None):
    """Generate a structured CSV summary of all metrics."""
    rows = []
    for _, row in merged.iterrows():
        rows.append({
            'function': int(row['func']),
            'function_name': FUNC_NAMES.get(int(row['func']), f"F{int(row['func'])}"),
            'population_size': int(row['coatis']),
            'iterations': int(row['iters']),
            'threads': int(row['threads']),
            'impl': 'OpenMP',
            'time_ms': row['time_ms'],
            'seq_baseline_ms': row['seq_time_ms'],
            'speedup': round(row['speedup'], 4),
            'efficiency': round(row['efficiency'], 4),
            'workload': int(row['workload']),
            'best_score': row.get('best_score', None),
        })
    for _, row in seq.iterrows():
        rows.append({
            'function': int(row['func']),
            'function_name': FUNC_NAMES.get(int(row['func']), f"F{int(row['func'])}"),
            'population_size': int(row['coatis']),
            'iterations': int(row['iters']),
            'threads': 1,
            'impl': 'Sequential',
            'time_ms': row['time_ms'],
            'seq_baseline_ms': row['time_ms'],
            'speedup': 1.0,
            'efficiency': 1.0,
            'workload': int(row['coatis'] * row['iters']),
            'best_score': row.get('best_score', None),
        })
    if cuda_merged is not None and len(cuda_merged) > 0:
        for _, row in cuda_merged.iterrows():
            rows.append({
                'function': int(row['func']),
                'function_name': FUNC_NAMES.get(int(row['func']), f"F{int(row['func'])}"),
                'population_size': int(row['coatis']),
                'iterations': int(row['iters']),
                'threads': 'GPU',
                'impl': 'CUDA',
                'time_ms': row['time_ms'],
                'seq_baseline_ms': row.get('seq_time_ms', None),
                'speedup': round(row['speedup'], 4) if pd.notna(row.get('speedup')) else None,
                'efficiency': None,
                'workload': int(row['workload']),
                'best_score': row.get('best_score', None),
            })
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  Coati Optimization Algorithm — Performance Analysis")
    print("=" * 70)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {OUTPUT_DIR}")

    # ── Step 1: Parse logs
    print("\n[1/4] Parsing log files...")
    df, records, warnings = parse_all_logs()
    print(f"  Parsed {len(df)} total log entries")
    print(f"  Sequential: {len(df[df['impl'] == 'SEQ'])} entries")
    print(f"  OpenMP: {len(df[df['impl'] == 'OpenMP'])} entries")
    print(f"  CUDA: {len(df[df['impl'] == 'CUDA'])} entries")
    if warnings:
        for w in warnings:
            print(f"  {w}")
    if df.empty:
        print("[ERROR] No data found. Exiting.")
        return

    # ── Step 2: Compute metrics
    print("\n[2/4] Computing metrics...")
    merged, seq, seq_baseline, cuda_merged = compute_metrics(df)
    print(f"  OpenMP merged dataset: {len(merged)} rows")
    if len(merged) > 0:
        print(f"  Speedup range: {merged['speedup'].min():.3f} – {merged['speedup'].max():.3f}")
    if len(cuda_merged) > 0:
        print(f"  CUDA dataset: {len(cuda_merged)} rows")
        print(f"  CUDA speedup range: {cuda_merged['speedup'].min():.3f} – {cuda_merged['speedup'].max():.3f}")

    # ── Step 3: Generate graphs
    print("\n[3/4] Generating graphs...")
    graphs = []

    # List of (function, args) pairs
    graph_calls = [
        # Speedup graphs
        (plot_01_avg_speedup_vs_threads, (merged,)),
        (plot_02_speedup_per_function_best, (merged,)),
        (plot_03_speedup_at_each_thread_count, (merged,)),
        # Efficiency graphs
        (plot_04_efficiency_vs_threads, (merged,)),
        (plot_05_efficiency_per_function, (merged,)),
        # Execution time graphs
        (plot_06_avg_time_seq_vs_omp, (merged, seq)),
        (plot_07_time_vs_population, (seq,)),
        (plot_08_time_vs_threads_single_func, (merged,)),
        # Scalability
        (plot_09_scalability_heatmap, (merged,)),
        (plot_10_workload_vs_speedup, (merged,)),
        # Overhead
        (plot_11_omp_overhead_t1, (merged,)),
        (plot_12_slowdown_rate_by_threads, (merged,)),
        (plot_13_speedup_distribution, (merged,)),
        # Amdahl's Law
        (plot_14_amdahl_actual_vs_predicted, (merged,)),
        (plot_15_serial_fraction_per_function, (merged,)),
        # CUDA
        (plot_16_cuda_speedup_per_function, (cuda_merged,)),
        (plot_17_cuda_vs_seq_time, (cuda_merged, seq)),
        (plot_18_cuda_speedup_vs_population, (cuda_merged,)),
        # All implementations
        (plot_19_all_three_time, (merged, seq, cuda_merged)),
        (plot_20_all_three_speedup, (merged, cuda_merged)),
        # Convergence
        (plot_21_convergence_single, (records, 1)),
        (plot_22_convergence_F5, (records,)),
        (plot_23_convergence_F8, (records,)),
        (plot_24_convergence_F10, (records,)),
        # Efficiency heatmap
        (plot_25_efficiency_heatmap, (merged,)),
    ]

    for func, args in graph_calls:
        try:
            result = func(*args)
            if result:
                graphs.append(result)
        except Exception as e:
            print(f"  [WARN] Failed {func.__name__}: {e}")

    print(f"  Generated {len(graphs)} graphs")

    # ── Step 4: Save results
    print("\n[4/4] Saving results...")

    report = generate_report(df, merged, seq, seq_baseline, warnings, cuda_merged)
    report_path = OUTPUT_DIR / "analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  [SAVED] {report_path.name}")

    csv_df = generate_csv_summary(df, merged, seq, cuda_merged)
    csv_path = OUTPUT_DIR / "metrics_summary.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path.name}")

    # JSON summary
    json_summary = {
        'timestamp': TIMESTAMP,
        'total_logs_parsed': len(df),
        'sequential_entries': len(seq),
        'openmp_entries': len(merged),
        'cuda_entries': len(cuda_merged) if len(cuda_merged) > 0 else 0,
        'functions': sorted(df['func'].unique().tolist()),
        'population_sizes': sorted(df['coatis'].unique().tolist()),
        'iteration_counts': sorted(df['iters'].unique().tolist()),
        'thread_counts': sorted(merged['threads'].unique().tolist()) if len(merged) > 0 else [],
        'overall_stats': {
            'avg_speedup_T16': float(merged[merged['threads'] == 16]['speedup'].mean()) if len(merged[merged['threads'] == 16]) > 0 else None,
            'max_speedup_omp': float(merged['speedup'].max()) if len(merged) > 0 else None,
            'min_speedup_omp': float(merged['speedup'].min()) if len(merged) > 0 else None,
            'avg_efficiency_T16': float(merged[merged['threads'] == 16]['efficiency'].mean()) if len(merged[merged['threads'] == 16]) > 0 else None,
            'slowdown_count': int(len(merged[merged['speedup'] < 1])) if len(merged) > 0 else 0,
            'slowdown_percentage': float(100 * len(merged[merged['speedup'] < 1]) / len(merged)) if len(merged) > 0 else 0,
            'cuda_avg_speedup': float(cuda_merged['speedup'].mean()) if len(cuda_merged) > 0 else None,
            'cuda_max_speedup': float(cuda_merged['speedup'].max()) if len(cuda_merged) > 0 else None,
        },
        'per_function_speedup_T16': {},
        'per_function_cuda_speedup': {},
        'per_thread_avg_speedup': {},
        'graphs_generated': len(graphs),
        'warnings': warnings,
    }

    for func_id in range(1, 11):
        func_data = merged[(merged['func'] == func_id) & (merged['threads'] == 16)]
        if not func_data.empty:
            json_summary['per_function_speedup_T16'][FUNC_NAMES.get(func_id, f"F{func_id}")] = {
                'avg_speedup': round(float(func_data['speedup'].mean()), 4),
                'max_speedup': round(float(func_data['speedup'].max()), 4),
                'avg_efficiency': round(float(func_data['efficiency'].mean()), 4),
            }

    if len(cuda_merged) > 0:
        for func_id in range(1, 11):
            func_data = cuda_merged[cuda_merged['func'] == func_id]
            if not func_data.empty:
                json_summary['per_function_cuda_speedup'][FUNC_NAMES.get(func_id, f"F{func_id}")] = {
                    'avg_speedup': round(float(func_data['speedup'].mean()), 4),
                    'max_speedup': round(float(func_data['speedup'].max()), 4),
                    'avg_time_ms': round(float(func_data['time_ms'].mean()), 2),
                }

    if len(merged) > 0:
        for t in sorted(merged['threads'].unique()):
            sub = merged[merged['threads'] == t]
            json_summary['per_thread_avg_speedup'][str(int(t))] = {
                'avg_speedup': round(float(sub['speedup'].mean()), 4),
                'avg_efficiency': round(float(sub['efficiency'].mean()), 4),
            }

    if len(cuda_merged) > 0:
        json_summary['per_thread_avg_speedup']['CUDA'] = {
            'avg_speedup': round(float(cuda_merged['speedup'].mean()), 4),
            'avg_efficiency': None,
        }

    json_path = OUTPUT_DIR / "metrics_summary.json"
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f"  [SAVED] {json_path.name}")

    # Create/update 'latest' copy
    try:
        import subprocess
        subprocess.run(['rmdir', '/S', '/Q', str(LATEST_DIR)], shell=True, stderr=subprocess.DEVNULL)
        if LATEST_DIR.exists() or LATEST_DIR.is_symlink():
            LATEST_DIR.unlink(missing_ok=True)
    except:
        pass

    try:
        shutil.copytree(OUTPUT_DIR, LATEST_DIR)
        print(f"  [COPY] results/latest <- {TIMESTAMP}")
    except Exception as e:
        print(f"  [WARN] Failed to copy latest: {e}")

    print()
    print("=" * 70)
    print(f"  Analysis complete! Results saved to: results/{TIMESTAMP}/")
    print(f"  Files:")
    print(f"    - analysis_report.md")
    print(f"    - metrics_summary.csv  ({len(csv_df)} rows)")
    print(f"    - metrics_summary.json")
    print(f"    - {len(graphs)} PNG graphs")
    print("=" * 70)


if __name__ == '__main__':
    main()
