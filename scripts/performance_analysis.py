#!/usr/bin/env python3
"""
Performance Analysis for Coati Optimization Algorithm
=====================================================
Parses sequential and OpenMP execution logs, computes metrics,
generates publication-quality graphs, and writes a self-contained
results folder.
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
from matplotlib.gridspec import GridSpec
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

# ── Style setup ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9,
    'figure.facecolor': '#f8f9fa',
    'axes.facecolor': '#ffffff',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
})

# Professional color palette
COLORS = {
    'seq': '#e74c3c',        # Red
    'omp_1': '#95a5a6',      # Gray
    'omp_2': '#3498db',      # Blue
    'omp_4': '#2ecc71',      # Green
    'omp_8': '#f39c12',      # Orange
    'omp_16': '#9b59b6',     # Purple
    'cuda': '#00d2ff',       # Cyan (GPU)
    'ideal': '#1a1a2e',      # Dark navy
}
THREAD_COLORS = {1: '#95a5a6', 2: '#3498db', 4: '#2ecc71', 8: '#f39c12', 16: '#9b59b6'}
THREAD_MARKERS = {1: 'o', 2: 's', 4: 'D', 8: '^', 16: 'v'}


# ── Log Parsing ────────────────────────────────────────────────────────────────
def parse_log_file(filepath):
    """Parse a single log file and return a dict of metrics."""
    data = {
        'file': filepath.name,
        'func': None,
        'coatis': None,
        'iters': None,
        'impl': None,
        'threads': None,
        'time_ms': None,
        'best_score': None,
        'convergence': [],  # list of (iter_num, score)
    }

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Header line
            if line.startswith('RUN '):
                m = re.search(r'F=(\d+)', line)
                if m: data['func'] = int(m.group(1))
                m = re.search(r'COATIS=(\d+)', line)
                if m: data['coatis'] = int(m.group(1))
                m = re.search(r'ITER=(\d+)', line)
                if m: data['iters'] = int(m.group(1))
                m = re.search(r'IMPL=(\w+)', line)
                if m: data['impl'] = m.group(1)

            # Iteration progress
            elif line.startswith('ITER '):
                parts = line.split()
                if len(parts) == 3:
                    try:
                        data['convergence'].append((int(parts[1]), float(parts[2])))
                    except ValueError:
                        pass

            # Timing
            elif line.startswith('TIME_MS'):
                parts = line.split()
                if len(parts) == 2:
                    try:
                        data['time_ms'] = int(parts[1])
                    except ValueError:
                        data['time_ms'] = float(parts[1])

            # Best score
            elif line.startswith('BEST_SCORE'):
                parts = line.split()
                if len(parts) == 2:
                    try:
                        data['best_score'] = float(parts[1])
                    except ValueError:
                        pass

    # Extract thread count from filename for OpenMP logs
    if data['impl'] == 'OpenMP':
        m = re.search(r'_T(\d+)\.log$', filepath.name)
        if m:
            data['threads'] = int(m.group(1))

    return data


def parse_all_logs():
    """Parse all log files and return a DataFrame."""
    records = []
    warnings = []

    # Parse sequential logs
    if SEQ_LOG_DIR.exists():
        for logfile in sorted(SEQ_LOG_DIR.glob('*.log')):
            try:
                rec = parse_log_file(logfile)
                rec['threads'] = 1  # Sequential is always 1 thread
                records.append(rec)
            except Exception as e:
                warnings.append(f"[WARN] Failed to parse {logfile.name}: {e}")
    else:
        warnings.append(f"[WARN] Sequential log directory not found: {SEQ_LOG_DIR}")

    # Parse OpenMP logs
    if OMP_LOG_DIR.exists():
        for logfile in sorted(OMP_LOG_DIR.glob('*.log')):
            try:
                rec = parse_log_file(logfile)
                records.append(rec)
            except Exception as e:
                warnings.append(f"[WARN] Failed to parse {logfile.name}: {e}")
    else:
        warnings.append(f"[WARN] OpenMP log directory not found: {OMP_LOG_DIR}")

    # Parse CUDA logs
    if CUDA_LOG_DIR.exists():
        for logfile in sorted(CUDA_LOG_DIR.glob('*.log')):
            try:
                rec = parse_log_file(logfile)
                rec['threads'] = None  # CUDA does not use threads
                records.append(rec)
            except Exception as e:
                warnings.append(f"[WARN] Failed to parse {logfile.name}: {e}")
    else:
        warnings.append(f"[WARN] CUDA log directory not found: {CUDA_LOG_DIR}")

    # Build DataFrame (drop convergence for the main frame)
    rows = []
    for rec in records:
        rows.append({
            'file': rec['file'],
            'func': rec['func'],
            'coatis': rec['coatis'],
            'iters': rec['iters'],
            'impl': rec['impl'],
            'threads': rec['threads'],
            'time_ms': rec['time_ms'],
            'best_score': rec['best_score'],
        })

    df = pd.DataFrame(rows)
    return df, records, warnings


# ── Metric Computation ─────────────────────────────────────────────────────────
def compute_metrics(df):
    """Compute speedup, efficiency, and other derived metrics."""
    # Get sequential baseline: for each (func, coatis, iters), use the SEQ time
    seq = df[df['impl'] == 'SEQ'].copy()
    seq_baseline = seq.groupby(['func', 'coatis', 'iters'])['time_ms'].mean().reset_index()
    seq_baseline.rename(columns={'time_ms': 'seq_time_ms'}, inplace=True)

    # Merge with OpenMP data
    omp = df[df['impl'] == 'OpenMP'].copy()
    merged = omp.merge(seq_baseline, on=['func', 'coatis', 'iters'], how='left')

    # Compute speedup and efficiency
    merged['speedup'] = merged['seq_time_ms'] / merged['time_ms']
    merged['efficiency'] = merged['speedup'] / merged['threads']

    # Workload size = coatis * iters
    merged['workload'] = merged['coatis'] * merged['iters']
    seq['workload'] = seq['coatis'] * seq['iters']

    # Process CUDA data
    cuda = df[df['impl'] == 'CUDA'].copy()
    if len(cuda) > 0:
        cuda_merged = cuda.merge(seq_baseline, on=['func', 'coatis', 'iters'], how='left')
        cuda_merged['speedup'] = cuda_merged['seq_time_ms'] / cuda_merged['time_ms']
        cuda_merged['workload'] = cuda_merged['coatis'] * cuda_merged['iters']
    else:
        cuda_merged = pd.DataFrame()

    return merged, seq, seq_baseline, cuda_merged


# ── Graph Generation ───────────────────────────────────────────────────────────

def save_fig(fig, name):
    """Save a figure to the output directory."""
    path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [SAVED] {path.name}")
    return path


def plot_01_exec_time_comparison(merged, seq, cuda_merged=None):
    """Bar chart: Sequential vs OpenMP vs CUDA execution time across workloads."""
    has_cuda = cuda_merged is not None and len(cuda_merged) > 0
    title = 'Execution Time Comparison: Sequential vs OpenMP' + (' vs CUDA' if has_cuda else '') + ' (per Function)'
    fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharey=False)
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)

    for idx, func_id in enumerate(range(1, 11)):
        ax = axes[idx // 5][idx % 5]
        func_name = FUNC_NAMES.get(func_id, f"F{func_id}")

        # Sequential data for this function
        seq_func = seq[seq['func'] == func_id].copy()
        seq_avg = seq_func.groupby(['coatis', 'iters'])['time_ms'].mean().reset_index()

        # OpenMP data with max threads (T=16)
        omp_func = merged[(merged['func'] == func_id) & (merged['threads'] == 16)]
        omp_avg = omp_func.groupby(['coatis', 'iters'])['time_ms'].mean().reset_index()

        # CUDA data for this function
        cuda_avg = pd.DataFrame()
        if has_cuda:
            cuda_func = cuda_merged[cuda_merged['func'] == func_id]
            if len(cuda_func) > 0:
                cuda_avg = cuda_func.groupby(['coatis', 'iters'])['time_ms'].mean().reset_index()

        # Create workload labels
        if len(seq_avg) == 0:
            ax.set_title(func_name, fontsize=10)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        seq_avg['label'] = seq_avg.apply(lambda r: f"C{int(r['coatis'])}\nI{int(r['iters'])}", axis=1)
        seq_avg = seq_avg.sort_values(['coatis', 'iters'])
        omp_avg['label'] = omp_avg.apply(lambda r: f"C{int(r['coatis'])}\nI{int(r['iters'])}", axis=1)
        omp_avg = omp_avg.sort_values(['coatis', 'iters'])

        x = np.arange(len(seq_avg))
        n_bars = 2 + (1 if len(cuda_avg) > 0 else 0)
        width = 0.8 / n_bars

        bars1 = ax.bar(x - 0.4 + width/2, seq_avg['time_ms'], width, label='Sequential',
                        color=COLORS['seq'], alpha=0.85, edgecolor='white', linewidth=0.5)
        if len(omp_avg) > 0:
            bars2 = ax.bar(x - 0.4 + 1.5*width, omp_avg['time_ms'], width, label='OpenMP (T=16)',
                            color=COLORS['omp_16'], alpha=0.85, edgecolor='white', linewidth=0.5)
        if len(cuda_avg) > 0:
            cuda_avg = cuda_avg.sort_values(['coatis', 'iters'])
            bars3 = ax.bar(x - 0.4 + 2.5*width, cuda_avg['time_ms'], width, label='CUDA',
                            color=COLORS['cuda'], alpha=0.85, edgecolor='white', linewidth=0.5)

        ax.set_title(func_name, fontsize=10, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(seq_avg['label'], fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('Time (ms)' if idx % 5 == 0 else '', fontsize=9)
        if idx == 0:
            ax.legend(fontsize=7, loc='upper left')

    fig.tight_layout()
    return save_fig(fig, '01_exec_time_comparison')


def plot_02_speedup_by_threads(merged):
    """Speedup curves across thread counts for each function."""
    fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharey=False)
    fig.suptitle('Speedup vs Thread Count (per Function)',
                 fontsize=16, fontweight='bold', y=1.02)

    thread_counts = sorted(merged['threads'].unique())

    for idx, func_id in enumerate(range(1, 11)):
        ax = axes[idx // 5][idx % 5]
        func_name = FUNC_NAMES.get(func_id, f"F{func_id}")

        func_data = merged[merged['func'] == func_id]
        if func_data.empty:
            ax.set_title(func_name, fontsize=10)
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue

        # Average speedup by threads and coatis
        for coatis in sorted(func_data['coatis'].unique()):
            sub = func_data[func_data['coatis'] == coatis]
            avg_speedup = sub.groupby('threads')['speedup'].mean()
            ax.plot(avg_speedup.index, avg_speedup.values,
                    marker='o', markersize=4, linewidth=1.5,
                    label=f'C={coatis}', alpha=0.7)

        # Ideal speedup line
        ax.plot(thread_counts, thread_counts, '--', color=COLORS['ideal'],
                linewidth=1.5, alpha=0.5, label='Ideal')

        ax.set_title(func_name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Threads' if idx >= 5 else '', fontsize=9)
        ax.set_ylabel('Speedup' if idx % 5 == 0 else '', fontsize=9)
        ax.set_xticks(thread_counts)
        if idx == 0:
            ax.legend(fontsize=6, loc='upper left', ncol=2)

    fig.tight_layout()
    return save_fig(fig, '02_speedup_by_threads')


def plot_03_efficiency_curves(merged):
    """Parallel efficiency (speedup / threads) vs thread count."""
    fig, axes = plt.subplots(2, 5, figsize=(24, 10), sharey=True)
    fig.suptitle('Parallel Efficiency vs Thread Count (per Function)',
                 fontsize=16, fontweight='bold', y=1.02)

    thread_counts = sorted(merged['threads'].unique())

    for idx, func_id in enumerate(range(1, 11)):
        ax = axes[idx // 5][idx % 5]
        func_name = FUNC_NAMES.get(func_id, f"F{func_id}")

        func_data = merged[merged['func'] == func_id]
        if func_data.empty:
            ax.set_title(func_name, fontsize=10)
            continue

        # Average efficiency by threads
        avg_eff = func_data.groupby('threads')['efficiency'].mean()
        ax.plot(avg_eff.index, avg_eff.values,
                marker='s', markersize=6, linewidth=2,
                color=COLORS['omp_4'], zorder=3)

        # Fill area
        ax.fill_between(avg_eff.index, avg_eff.values, alpha=0.15, color=COLORS['omp_4'])

        # Ideal efficiency line
        ax.axhline(y=1.0, linestyle='--', color=COLORS['ideal'],
                   alpha=0.5, linewidth=1.5, label='Ideal (1.0)')

        ax.set_title(func_name, fontsize=10, fontweight='bold')
        ax.set_xlabel('Threads' if idx >= 5 else '', fontsize=9)
        ax.set_ylabel('Efficiency' if idx % 5 == 0 else '', fontsize=9)
        ax.set_xticks(thread_counts)
        ax.set_ylim(0, max(1.5, avg_eff.max() * 1.2) if len(avg_eff) > 0 else 1.5)

    fig.tight_layout()
    return save_fig(fig, '03_efficiency_curves')


def plot_04_scalability_heatmap(merged):
    """Heatmap of speedup across functions and thread counts."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('Speedup Heatmap: Functions × Thread Counts',
                 fontsize=16, fontweight='bold')

    # Pivot: rows = functions, cols = threads
    pivot = merged.groupby(['func', 'threads'])['speedup'].mean().reset_index()
    heatmap_data = pivot.pivot(index='func', columns='threads', values='speedup')
    heatmap_data.index = [FUNC_NAMES.get(f, f"F{f}") for f in heatmap_data.index]

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                linewidths=1, linecolor='white', ax=ax,
                cbar_kws={'label': 'Speedup'})
    ax.set_xlabel('Thread Count', fontsize=12)
    ax.set_ylabel('Benchmark Function', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    return save_fig(fig, '04_scalability_heatmap')


def plot_05_workload_vs_speedup(merged):
    """Speedup vs workload size (coatis × iters), colored by thread count."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('Speedup vs Workload Size (Coatis × Iterations)',
                 fontsize=16, fontweight='bold')

    for threads in sorted(merged['threads'].unique()):
        sub = merged[merged['threads'] == threads]
        avg = sub.groupby('workload')['speedup'].mean().sort_index()
        ax.plot(avg.index, avg.values,
                marker=THREAD_MARKERS.get(threads, 'o'),
                markersize=7, linewidth=2, alpha=0.8,
                color=THREAD_COLORS.get(threads, '#333'),
                label=f'T={threads}')

    ax.set_xlabel('Workload (Coatis × Iterations)', fontsize=12)
    ax.set_ylabel('Average Speedup', fontsize=12)
    ax.legend(title='Threads', fontsize=10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1e3:.0f}K' if x >= 1000 else f'{x:.0f}'))

    fig.tight_layout()
    return save_fig(fig, '05_workload_vs_speedup')


def plot_06_time_by_population(merged, seq, cuda_merged=None):
    """Execution time vs population size for different thread counts."""
    has_cuda = cuda_merged is not None and len(cuda_merged) > 0
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
    fig.suptitle('Execution Time vs Population Size (by Iteration Count)',
                 fontsize=16, fontweight='bold', y=1.02)

    iter_values = sorted(seq['iters'].unique())

    for i, iters in enumerate(iter_values):
        ax = axes[i]

        # Sequential baseline (averaged across functions)
        seq_sub = seq[seq['iters'] == iters].groupby('coatis')['time_ms'].mean()
        ax.plot(seq_sub.index, seq_sub.values, marker='X', markersize=8,
                linewidth=2.5, color=COLORS['seq'], label='Sequential', zorder=5)

        # OpenMP by thread count
        for threads in sorted(merged['threads'].unique()):
            omp_sub = merged[(merged['iters'] == iters) & (merged['threads'] == threads)]
            avg = omp_sub.groupby('coatis')['time_ms'].mean()
            ax.plot(avg.index, avg.values,
                    marker=THREAD_MARKERS.get(threads, 'o'),
                    markersize=5, linewidth=1.5, alpha=0.7,
                    color=THREAD_COLORS.get(threads, '#333'),
                    label=f'OpenMP T={threads}')

        # CUDA
        if has_cuda:
            cuda_sub = cuda_merged[cuda_merged['iters'] == iters].groupby('coatis')['time_ms'].mean()
            if len(cuda_sub) > 0:
                ax.plot(cuda_sub.index, cuda_sub.values, marker='*', markersize=10,
                        linewidth=2.5, color=COLORS['cuda'], label='CUDA (GPU)', zorder=5)

        ax.set_title(f'Iterations = {iters}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Population Size (Coatis)', fontsize=11)
        if i == 0:
            ax.set_ylabel('Avg Execution Time (ms)', fontsize=11)
        ax.legend(fontsize=8)

    fig.tight_layout()
    return save_fig(fig, '06_time_by_population')


def plot_07_speedup_distribution(merged):
    """Violin + box plot of speedup distribution per thread count."""
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.suptitle('Speedup Distribution by Thread Count',
                 fontsize=16, fontweight='bold')

    thread_order = sorted(merged['threads'].unique())
    palette = [THREAD_COLORS.get(t, '#333') for t in thread_order]

    # Use seaborn violin plot
    sns.violinplot(data=merged, x='threads', y='speedup',
                   order=thread_order, palette=palette,
                   inner='box', ax=ax, alpha=0.7, cut=0)

    # Add ideal line
    for i, t in enumerate(thread_order):
        ax.plot(i, t, 'r*', markersize=15, zorder=5,
                label='Ideal' if i == 0 else None)

    ax.set_xlabel('Thread Count', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.legend(fontsize=10)

    fig.tight_layout()
    return save_fig(fig, '07_speedup_distribution')


def plot_08_efficiency_heatmap_by_workload(merged):
    """Heatmap of efficiency across workload sizes and thread counts."""
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Parallel Efficiency: Workload Size × Thread Count',
                 fontsize=16, fontweight='bold')

    merged_copy = merged.copy()
    merged_copy['workload_label'] = merged_copy['workload'].apply(
        lambda x: f'{x/1e3:.0f}K' if x >= 1000 else str(int(x)))

    pivot = merged_copy.groupby(['workload_label', 'threads'])['efficiency'].mean().reset_index()
    # Sort workload labels numerically
    pivot['workload_num'] = pivot['workload_label'].str.replace('K', '000').astype(float)
    pivot = pivot.sort_values('workload_num')

    heatmap_data = pivot.pivot(index='workload_label', columns='threads', values='efficiency')
    # Reorder rows
    workload_order = pivot.drop_duplicates('workload_label').sort_values('workload_num')['workload_label'].tolist()
    heatmap_data = heatmap_data.reindex(workload_order)

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                linewidths=1, linecolor='white', ax=ax,
                cbar_kws={'label': 'Efficiency'}, vmin=0, vmax=2.0)
    ax.set_xlabel('Thread Count', fontsize=12)
    ax.set_ylabel('Workload Size', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    fig.tight_layout()
    return save_fig(fig, '08_efficiency_heatmap_by_workload')


def plot_09_overall_summary(merged, seq, seq_baseline):
    """Summary dashboard: overall speedup, best/worst performers."""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    fig.suptitle('Overall Performance Summary Dashboard',
                 fontsize=18, fontweight='bold', y=0.98)

    # 1. Average speedup by function (T=16)
    ax1 = fig.add_subplot(gs[0, 0])
    t16 = merged[merged['threads'] == 16]
    avg_by_func = t16.groupby('func')['speedup'].mean().sort_values(ascending=True)
    colors_bar = ['#e74c3c' if v < 1 else '#2ecc71' for v in avg_by_func.values]
    bars = ax1.barh([FUNC_NAMES.get(f, f"F{f}") for f in avg_by_func.index],
                    avg_by_func.values, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax1.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline (1x)')
    ax1.set_xlabel('Average Speedup (T=16)')
    ax1.set_title('Avg Speedup by Function (16 Threads)', fontweight='bold')

    # 2. Speedup trend across thread counts (averaged)
    ax2 = fig.add_subplot(gs[0, 1])
    avg_speedup_all = merged.groupby('threads')['speedup'].agg(['mean', 'std']).reset_index()
    ax2.errorbar(avg_speedup_all['threads'], avg_speedup_all['mean'],
                 yerr=avg_speedup_all['std'], marker='o', markersize=8,
                 linewidth=2.5, color=COLORS['omp_4'], capsize=4, capthick=1.5,
                 label='Actual')
    threads_range = avg_speedup_all['threads']
    ax2.plot(threads_range, threads_range, '--', color=COLORS['ideal'],
             linewidth=2, alpha=0.5, label='Ideal')
    ax2.set_xlabel('Thread Count')
    ax2.set_ylabel('Average Speedup')
    ax2.set_title('Global Speedup Trend', fontweight='bold')
    ax2.legend()

    # 3. Average efficiency trend
    ax3 = fig.add_subplot(gs[0, 2])
    avg_eff_all = merged.groupby('threads')['efficiency'].agg(['mean', 'std']).reset_index()
    ax3.bar(avg_eff_all['threads'].astype(str), avg_eff_all['mean'],
            yerr=avg_eff_all['std'], color=[THREAD_COLORS.get(t, '#333') for t in avg_eff_all['threads']],
            edgecolor='white', linewidth=0.5, capsize=4, alpha=0.85)
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Ideal (1.0)')
    ax3.set_xlabel('Thread Count')
    ax3.set_ylabel('Efficiency')
    ax3.set_title('Avg Parallel Efficiency', fontweight='bold')
    ax3.legend()

    # 4. Execution time reduction (best case per function)
    ax4 = fig.add_subplot(gs[1, 0])
    best_omp = merged.groupby('func')['time_ms'].min().reset_index()
    best_omp.rename(columns={'time_ms': 'omp_min_ms'}, inplace=True)
    best_seq = seq.groupby('func')['time_ms'].max().reset_index()
    best_seq.rename(columns={'time_ms': 'seq_max_ms'}, inplace=True)
    compare = best_omp.merge(best_seq, on='func')
    compare['reduction_pct'] = (1 - compare['omp_min_ms'] / compare['seq_max_ms']) * 100
    compare = compare.sort_values('reduction_pct', ascending=True)

    bars4 = ax4.barh([FUNC_NAMES.get(f, f"F{f}") for f in compare['func']],
                     compare['reduction_pct'],
                     color=sns.color_palette('coolwarm_r', len(compare)),
                     edgecolor='white', linewidth=0.5)
    ax4.set_xlabel('Time Reduction (%)')
    ax4.set_title('Best-Case Time Reduction', fontweight='bold')

    # 5. Top-5 best speedups
    ax5 = fig.add_subplot(gs[1, 1])
    top_speedups = merged.nlargest(10, 'speedup')[['func', 'coatis', 'iters', 'threads', 'speedup']]
    top_speedups['label'] = top_speedups.apply(
        lambda r: f"F{int(r['func'])} C{int(r['coatis'])} I{int(r['iters'])} T{int(r['threads'])}", axis=1)
    ax5.barh(top_speedups['label'], top_speedups['speedup'],
             color=sns.color_palette('viridis', len(top_speedups)),
             edgecolor='white', linewidth=0.5)
    ax5.set_xlabel('Speedup')
    ax5.set_title('Top 10 Speedup Configurations', fontweight='bold')

    # 6. Worst performers (speedup < 1 = slowdown)
    ax6 = fig.add_subplot(gs[1, 2])
    slowdowns = merged[merged['speedup'] < 1]
    if len(slowdowns) > 0:
        worst = slowdowns.nsmallest(10, 'speedup')
        worst['label'] = worst.apply(
            lambda r: f"F{int(r['func'])} C{int(r['coatis'])} I{int(r['iters'])} T{int(r['threads'])}", axis=1)
        ax6.barh(worst['label'], worst['speedup'],
                 color='#e74c3c', edgecolor='white', linewidth=0.5, alpha=0.85)
        ax6.axvline(x=1.0, color='black', linestyle='--', alpha=0.5)
        ax6.set_xlabel('Speedup (< 1 = Slowdown)')
        ax6.set_title('Top 10 Slowdown Configurations', fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'No slowdowns detected!',
                ha='center', va='center', transform=ax6.transAxes,
                fontsize=14, color='#2ecc71', fontweight='bold')
        ax6.set_title('Slowdown Analysis', fontweight='bold')

    return save_fig(fig, '09_overall_summary_dashboard')


def plot_10_convergence_comparison(records):
    """Convergence curves: Sequential vs OpenMP vs CUDA for selected configurations."""
    # Pick a representative function and workload
    target_configs = [
        (1, 1000, 1000),
        (5, 1000, 1000),
        (10, 1000, 1000),
    ]

    fig, axes = plt.subplots(1, len(target_configs), figsize=(7 * len(target_configs), 6))
    fig.suptitle('Convergence Curves: Sequential vs OpenMP vs CUDA',
                 fontsize=16, fontweight='bold', y=1.02)

    if len(target_configs) == 1:
        axes = [axes]

    for i, (func_id, coatis, iters) in enumerate(target_configs):
        ax = axes[i]
        func_name = FUNC_NAMES.get(func_id, f"F{func_id}")

        # Find matching records
        for rec in records:
            if rec['func'] == func_id and rec['coatis'] == coatis and rec['iters'] == iters:
                conv = rec['convergence']
                if not conv:
                    continue
                x_vals = [c[0] for c in conv]
                y_vals = [max(c[1], 1e-320) for c in conv]  # Floor for log scale

                if rec['impl'] == 'SEQ':
                    ax.plot(x_vals, y_vals, marker='o', markersize=4,
                            linewidth=2, color=COLORS['seq'], alpha=0.8,
                            label='Sequential')
                elif rec['impl'] == 'OpenMP' and rec.get('threads') == 16:
                    ax.plot(x_vals, y_vals, marker='s', markersize=4,
                            linewidth=2, color=COLORS['omp_16'], alpha=0.8,
                            label=f"OpenMP T={rec['threads']}")
                elif rec['impl'] == 'CUDA':
                    ax.plot(x_vals, y_vals, marker='*', markersize=6,
                            linewidth=2, color=COLORS['cuda'], alpha=0.8,
                            label='CUDA (GPU)')

        ax.set_yscale('symlog', linthresh=1e-300)
        ax.set_title(f'{func_name}\nC={coatis}, I={iters}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Best Score' if i == 0 else '')
        ax.legend(fontsize=8)

    fig.tight_layout()
    return save_fig(fig, '10_convergence_comparison')


def plot_11_amdahl_analysis(merged):
    """Amdahl's Law analysis: estimate serial fraction from speedup data."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("Amdahl's Law Analysis", fontsize=16, fontweight='bold', y=1.02)

    thread_counts = sorted(merged['threads'].unique())

    # Left: Actual vs Amdahl's predicted speedup for various serial fractions
    ax1 = axes[0]
    avg_speedup = merged.groupby('threads')['speedup'].mean()

    # Plot actual
    ax1.plot(avg_speedup.index, avg_speedup.values, 'ko-', markersize=8,
             linewidth=2.5, label='Actual (Avg)', zorder=5)

    # Plot Amdahl's curves for various serial fractions
    p_range = np.linspace(1, max(thread_counts), 100)
    for f_serial in [0.01, 0.05, 0.1, 0.2, 0.5]:
        amdahl = 1.0 / (f_serial + (1 - f_serial) / p_range)
        ax1.plot(p_range, amdahl, '--', alpha=0.6,
                 label=f'f={f_serial:.0%} serial')

    ax1.set_xlabel('Number of Threads (p)', fontsize=12)
    ax1.set_ylabel('Speedup S(p)', fontsize=12)
    ax1.set_title("Actual vs Amdahl's Predicted Speedup", fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.set_xlim(0.5, max(thread_counts) + 1)

    # Right: Estimated serial fraction per function
    ax2 = axes[1]
    serial_fractions = {}
    for func_id in range(1, 11):
        func_data = merged[(merged['func'] == func_id) & (merged['threads'] == max(thread_counts))]
        if func_data.empty:
            continue
        avg_s = func_data['speedup'].mean()
        p = max(thread_counts)
        # From Amdahl: S = 1/(f + (1-f)/p)  =>  f = (1/S - 1/p) / (1 - 1/p)
        if avg_s > 0 and p > 1:
            f_est = (1/avg_s - 1/p) / (1 - 1/p)
            serial_fractions[func_id] = max(0, min(1, f_est))

    if serial_fractions:
        funcs = list(serial_fractions.keys())
        fracs = [serial_fractions[f] for f in funcs]
        labels = [FUNC_NAMES.get(f, f"F{f}") for f in funcs]
        colors_sf = ['#e74c3c' if f > 0.5 else '#f39c12' if f > 0.2 else '#2ecc71' for f in fracs]
        ax2.barh(labels, fracs, color=colors_sf, edgecolor='white', linewidth=0.5)
        ax2.set_xlabel('Estimated Serial Fraction', fontsize=12)
        ax2.set_title(f'Serial Fraction Estimate (T={max(thread_counts)})', fontweight='bold')
        ax2.set_xlim(0, 1)

    fig.tight_layout()
    return save_fig(fig, '11_amdahl_analysis')


def plot_12_per_function_time_breakdown(merged, seq, cuda_merged=None):
    """Grouped bar chart: per-function time comparison across all thread counts."""
    has_cuda = cuda_merged is not None and len(cuda_merged) > 0
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.suptitle('Average Execution Time per Function (All Implementations)',
                 fontsize=16, fontweight='bold')

    funcs = list(range(1, 11))
    func_labels = [FUNC_NAMES.get(f, f"F{f}") for f in funcs]

    # Sequential
    seq_avg = seq.groupby('func')['time_ms'].mean()

    thread_counts = sorted(merged['threads'].unique())
    n_groups = 1 + len(thread_counts) + (1 if has_cuda else 0)
    x = np.arange(len(funcs))
    width = 0.8 / n_groups

    # Sequential bars
    ax.bar(x - 0.4 + width/2, [seq_avg.get(f, 0) for f in funcs], width,
           label='Sequential', color=COLORS['seq'], alpha=0.85, edgecolor='white', linewidth=0.5)

    # OpenMP bars
    for j, threads in enumerate(thread_counts):
        omp_avg = merged[merged['threads'] == threads].groupby('func')['time_ms'].mean()
        ax.bar(x - 0.4 + (j + 1.5) * width,
               [omp_avg.get(f, 0) for f in funcs], width,
               label=f'OpenMP T={threads}',
               color=THREAD_COLORS.get(threads, '#333'),
               alpha=0.85, edgecolor='white', linewidth=0.5)

    # CUDA bars
    if has_cuda:
        cuda_avg = cuda_merged.groupby('func')['time_ms'].mean()
        bar_offset = len(thread_counts) + 1.5
        ax.bar(x - 0.4 + bar_offset * width,
               [cuda_avg.get(f, 0) for f in funcs], width,
               label='CUDA (GPU)', color=COLORS['cuda'],
               alpha=0.85, edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(func_labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Average Execution Time (ms)', fontsize=12)
    ax.legend(fontsize=9, ncol=3, loc='upper left')

    fig.tight_layout()
    return save_fig(fig, '12_per_function_time_breakdown')


def plot_13_cuda_vs_cpu_speedup(cuda_merged, seq):
    """CUDA GPU speedup over Sequential and bar comparison per function."""
    if cuda_merged is None or len(cuda_merged) == 0:
        return None

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('CUDA GPU Performance Analysis',
                 fontsize=16, fontweight='bold', y=1.02)

    # Left: CUDA speedup (over sequential) per function
    ax1 = axes[0]
    avg_speedup = cuda_merged.groupby('func')['speedup'].mean().sort_values(ascending=True)
    colors_bar = ['#e74c3c' if v < 1 else '#00d2ff' for v in avg_speedup.values]
    ax1.barh([FUNC_NAMES.get(f, f"F{f}") for f in avg_speedup.index],
             avg_speedup.values, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax1.axvline(x=1.0, color='black', linestyle='--', alpha=0.5, label='Baseline (1x)')
    ax1.set_xlabel('CUDA Speedup over Sequential', fontsize=12)
    ax1.set_title('GPU Speedup by Function', fontweight='bold')
    ax1.legend(fontsize=9)

    # Right: Execution time comparison (Seq vs CUDA per function)
    ax2 = axes[1]
    funcs = sorted(cuda_merged['func'].unique())
    func_labels = [FUNC_NAMES.get(f, f"F{f}") for f in funcs]
    seq_avg = seq.groupby('func')['time_ms'].mean()
    cuda_avg = cuda_merged.groupby('func')['time_ms'].mean()

    x = np.arange(len(funcs))
    width = 0.35
    ax2.bar(x - width/2, [seq_avg.get(f, 0) for f in funcs], width,
            label='Sequential (CPU)', color=COLORS['seq'], alpha=0.85, edgecolor='white')
    ax2.bar(x + width/2, [cuda_avg.get(f, 0) for f in funcs], width,
            label='CUDA (GPU)', color=COLORS['cuda'], alpha=0.85, edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(func_labels, rotation=30, ha='right', fontsize=9)
    ax2.set_ylabel('Average Execution Time (ms)', fontsize=12)
    ax2.set_title('CPU vs GPU Execution Time', fontweight='bold')
    ax2.legend(fontsize=10)

    fig.tight_layout()
    return save_fig(fig, '13_cuda_vs_cpu_speedup')


def plot_14_all_implementations_speedup(merged, cuda_merged):
    """Combined speedup comparison: OpenMP (all thread counts) + CUDA vs Sequential."""
    if cuda_merged is None or len(cuda_merged) == 0:
        return None

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle('Speedup Comparison: OpenMP vs CUDA (over Sequential)',
                 fontsize=16, fontweight='bold')

    funcs = list(range(1, 11))
    func_labels = [FUNC_NAMES.get(f, f"F{f}") for f in funcs]

    # OpenMP speedup by thread count
    thread_counts = sorted(merged['threads'].unique())
    for threads in thread_counts:
        sub = merged[merged['threads'] == threads]
        avg = sub.groupby('func')['speedup'].mean()
        ax.plot([FUNC_NAMES.get(f, f"F{f}") for f in avg.index], avg.values,
                marker=THREAD_MARKERS.get(threads, 'o'), markersize=7,
                linewidth=1.5, alpha=0.6,
                color=THREAD_COLORS.get(threads, '#333'),
                label=f'OpenMP T={threads}')

    # CUDA speedup
    cuda_avg = cuda_merged.groupby('func')['speedup'].mean()
    ax.plot([FUNC_NAMES.get(f, f"F{f}") for f in cuda_avg.index], cuda_avg.values,
            marker='*', markersize=14, linewidth=2.5,
            color=COLORS['cuda'], label='CUDA (GPU)', zorder=5)

    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.3, label='Baseline (1x)')
    ax.set_xlabel('Benchmark Function', fontsize=12)
    ax.set_ylabel('Speedup over Sequential', fontsize=12)
    ax.legend(fontsize=9, ncol=2)
    plt.xticks(rotation=30, ha='right')

    fig.tight_layout()
    return save_fig(fig, '14_all_implementations_speedup')


# ── Report Generation ──────────────────────────────────────────────────────────

def generate_report(df, merged, seq, seq_baseline, warnings, cuda_merged=None):
    """Generate a comprehensive analysis report."""
    has_cuda = cuda_merged is not None and len(cuda_merged) > 0
    lines = []
    lines.append("# Coati Optimization Algorithm — Performance Analysis Report")
    lines.append(f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    cuda_info = f", `logs/cuda/` ({len(cuda_merged)} entries)" if has_cuda else ""
    lines.append(f"**Data Source**: `logs/sequential/` ({len(seq)} entries), `logs/openmp/` ({len(merged)} entries){cuda_info}")
    lines.append("")

    # ── Data summary
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

    # ── Speedup analysis
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
        if avg_s >= 10:
            assessment = "🟢 Excellent"
        elif avg_s >= 5:
            assessment = "🟡 Good"
        elif avg_s >= 2:
            assessment = "🟠 Moderate"
        elif avg_s >= 1:
            assessment = "🔵 Marginal"
        else:
            assessment = "🔴 Slowdown"
        lines.append(f"| {FUNC_NAMES.get(func_id, f'F{func_id}')} | {avg_s:.3f} | {assessment} |")
    lines.append("")

    # ── Scalability
    lines.append("## 3. Scalability Analysis")
    lines.append("")
    lines.append("### Does performance improve proportionally with thread count?")
    lines.append("")

    avg_by_threads = merged.groupby('threads')['speedup'].mean()
    for i, t in enumerate(thread_counts):
        if i == 0:
            continue
        prev_t = thread_counts[i - 1]
        marginal_gain = (avg_by_threads.get(t, 0) - avg_by_threads.get(prev_t, 0))
        expected_gain = avg_by_threads.get(prev_t, 0) * (t / prev_t - 1)
        ratio = marginal_gain / expected_gain if expected_gain > 0 else 0
        lines.append(f"- **T={prev_t} → T={t}**: Marginal speedup gain = {marginal_gain:.3f} "
                     f"(expected ~{expected_gain:.3f} for linear scaling, ratio: {ratio:.1%})")
    lines.append("")

    # Check if large workloads scale better
    lines.append("### Workload Size Impact on Scalability")
    lines.append("")
    small_wl = merged[merged['workload'] <= 200000]
    large_wl = merged[merged['workload'] > 500000]
    if len(small_wl) > 0 and len(large_wl) > 0:
        lines.append(f"- **Small workloads** (≤200K): Avg speedup = {small_wl['speedup'].mean():.3f}")
        lines.append(f"- **Large workloads** (>500K): Avg speedup = {large_wl['speedup'].mean():.3f}")
        if large_wl['speedup'].mean() > small_wl['speedup'].mean():
            lines.append("- ✅ Larger workloads benefit more from parallelization (as expected)")
        else:
            lines.append("- ⚠️ Larger workloads do NOT show better speedup — possible overhead dominance")
    lines.append("")

    # ── Efficiency
    lines.append("## 4. Parallel Efficiency (Amdahl's Law Perspective)")
    lines.append("")
    lines.append("| Thread Count | Avg Efficiency | Interpretation |")
    lines.append("|:---:|:---:|:---|")
    for t in thread_counts:
        sub = merged[merged['threads'] == t]
        avg_e = sub['efficiency'].mean()
        if avg_e >= 0.8:
            interp = "Excellent utilization"
        elif avg_e >= 0.5:
            interp = "Good utilization"
        elif avg_e >= 0.3:
            interp = "Moderate — overhead becoming significant"
        else:
            interp = "Poor — diminishing returns"
        lines.append(f"| {t} | {avg_e:.3f} | {interp} |")
    lines.append("")

    # Estimated serial fraction
    lines.append("### Estimated Serial Fraction (from Amdahl's Law)")
    lines.append("")
    max_t = max(thread_counts)
    lines.append("| Function | Estimated Serial Fraction | Max Theoretical Speedup |")
    lines.append("|:---|:---:|:---:|")
    for func_id in range(1, 11):
        func_data = merged[(merged['func'] == func_id) & (merged['threads'] == max_t)]
        if func_data.empty:
            continue
        avg_s = func_data['speedup'].mean()
        if avg_s > 0:
            f_est = max(0, min(1, (1/avg_s - 1/max_t) / (1 - 1/max_t)))
            max_speedup = 1.0 / f_est if f_est > 0 else float('inf')
            lines.append(f"| {FUNC_NAMES.get(func_id, f'F{func_id}')} | {f_est:.4f} "
                        f"| {max_speedup:.1f}x |")
    lines.append("")

    # ── Overhead/Anomalies
    lines.append("## 5. Overhead & Anomaly Analysis")
    lines.append("")

    # Cases where OpenMP is slower than sequential
    slowdowns = merged[merged['speedup'] < 1.0]
    lines.append(f"### Slowdowns (OpenMP slower than Sequential)")
    lines.append(f"- **Total slowdown cases**: {len(slowdowns)} out of {len(merged)} "
                 f"({100*len(slowdowns)/len(merged):.1f}%)")
    lines.append("")

    if len(slowdowns) > 0:
        lines.append("**Worst slowdowns:**")
        lines.append("")
        lines.append("| Config | Threads | Seq Time (ms) | OMP Time (ms) | Speedup |")
        lines.append("|:---|:---:|:---:|:---:|:---:|")
        worst = slowdowns.nsmallest(5, 'speedup')
        for _, row in worst.iterrows():
            lines.append(f"| F{int(row['func'])} C{int(row['coatis'])} I{int(row['iters'])} "
                        f"| {int(row['threads'])} | {row['seq_time_ms']:.0f} | {row['time_ms']:.0f} "
                        f"| {row['speedup']:.3f} |")
        lines.append("")

        # Analyze slowdown patterns
        sd_by_threads = slowdowns.groupby('threads').size()
        lines.append("**Slowdown distribution by thread count:**")
        for t, count in sd_by_threads.items():
            total_at_t = len(merged[merged['threads'] == t])
            lines.append(f"- T={int(t)}: {count} slowdowns ({100*count/total_at_t:.1f}% of T={int(t)} runs)")
        lines.append("")

    # Super-linear speedup (anomaly)
    superlinear = merged[merged['speedup'] > merged['threads']]
    if len(superlinear) > 0:
        lines.append(f"### Super-Linear Speedup (anomalous)")
        lines.append(f"- **Total super-linear cases**: {len(superlinear)} "
                     f"({100*len(superlinear)/len(merged):.1f}%)")
        lines.append("- This may indicate cache effects, measurement noise, or algorithmic differences.")
        lines.append("")

    # T=1 OpenMP vs Sequential (overhead measurement)
    t1_omp = merged[merged['threads'] == 1]
    if len(t1_omp) > 0:
        lines.append("### OpenMP Thread Management Overhead (T=1 vs Sequential)")
        lines.append("")
        avg_t1_speedup = t1_omp['speedup'].mean()
        lines.append(f"- Average speedup at T=1: **{avg_t1_speedup:.3f}**")
        if avg_t1_speedup < 1.0:
            overhead_pct = (1 - avg_t1_speedup) * 100
            lines.append(f"- OpenMP overhead at T=1: ~**{overhead_pct:.1f}%** slower than sequential")
            lines.append("- This represents the pure thread management/synchronization overhead")
        else:
            lines.append("- OpenMP at T=1 is faster than or equal to sequential (minimal overhead)")
        lines.append("")

    # ── Key findings
    lines.append("## 6. Key Findings & Recommendations")
    lines.append("")

    overall_speedup_16 = t16['speedup'].mean() if len(t16) > 0 else 0
    best_func = t16.groupby('func')['speedup'].mean().idxmax() if len(t16) > 0 else None
    worst_func = t16.groupby('func')['speedup'].mean().idxmin() if len(t16) > 0 else None

    lines.append(f"1. **Overall average speedup (T=16)**: {overall_speedup_16:.2f}x")
    if best_func:
        lines.append(f"2. **Best performing function**: {FUNC_NAMES.get(best_func, f'F{best_func}')} "
                    f"({t16[t16['func']==best_func]['speedup'].mean():.2f}x)")
    if worst_func:
        lines.append(f"3. **Worst performing function**: {FUNC_NAMES.get(worst_func, f'F{worst_func}')} "
                    f"({t16[t16['func']==worst_func]['speedup'].mean():.2f}x)")
    lines.append(f"4. **Slowdown rate**: {100*len(slowdowns)/len(merged):.1f}% of configurations")

    avg_eff_16 = t16['efficiency'].mean() if len(t16) > 0 else 0
    lines.append(f"5. **Average parallel efficiency (T=16)**: {avg_eff_16:.1%}")

    if has_cuda:
        avg_cuda_speedup = cuda_merged['speedup'].mean()
        max_cuda_speedup = cuda_merged['speedup'].max()
        best_cuda_func = cuda_merged.groupby('func')['speedup'].mean().idxmax()
        lines.append(f"6. **CUDA avg speedup over sequential**: {avg_cuda_speedup:.2f}x")
        lines.append(f"7. **CUDA max speedup**: {max_cuda_speedup:.2f}x")
        lines.append(f"8. **Best CUDA function**: {FUNC_NAMES.get(best_cuda_func, f'F{best_cuda_func}')} "
                    f"({cuda_merged[cuda_merged['func']==best_cuda_func]['speedup'].mean():.2f}x)")
    lines.append("")

    lines.append("### Recommendations")
    lines.append("")
    if avg_eff_16 < 0.5:
        lines.append("- ⚠️ Parallel efficiency is low at T=16. Consider reducing thread count to 4-8 for better efficiency.")
    if len(slowdowns) > 0 and len(slowdowns[slowdowns['threads'] == 1]) > 0:
        lines.append("- ⚠️ Slowdowns detected even at T=1, suggesting OpenMP runtime overhead. "
                    "Consider using `if` clause for small workloads.")
    lines.append("- For production use, profile with 4-8 threads as optimal trade-off between speed and efficiency.")
    lines.append("- Functions with high serial fraction could benefit from algorithmic restructuring.")
    if has_cuda:
        lines.append("- GPU acceleration is most effective for large population sizes and compute-heavy functions.")
    lines.append("")

    # Warnings
    if warnings:
        lines.append("## Warnings")
        lines.append("")
        for w in warnings:
            lines.append(f"- {w}")
        lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by `performance_analysis.py` at {datetime.now().isoformat()}*")

    return "\n".join(lines)


def generate_csv_summary(df, merged, seq, cuda_merged=None):
    """Generate a structured CSV summary of all metrics."""
    rows = []

    # Per-configuration summary
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

    # Add sequential entries
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

    # Add CUDA entries
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

    # Create output directory
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
        print(f"  Efficiency range: {merged['efficiency'].min():.3f} – {merged['efficiency'].max():.3f}")
    if len(cuda_merged) > 0:
        print(f"  CUDA dataset: {len(cuda_merged)} rows")
        print(f"  CUDA speedup range: {cuda_merged['speedup'].min():.3f} – {cuda_merged['speedup'].max():.3f}")

    # ── Step 3: Generate graphs
    print("\n[3/4] Generating graphs...")
    graphs = []
    try:
        graphs.append(plot_01_exec_time_comparison(merged, seq, cuda_merged))
    except Exception as e:
        print(f"  [WARN] Failed to generate exec_time_comparison: {e}")

    try:
        graphs.append(plot_02_speedup_by_threads(merged))
    except Exception as e:
        print(f"  [WARN] Failed to generate speedup_by_threads: {e}")

    try:
        graphs.append(plot_03_efficiency_curves(merged))
    except Exception as e:
        print(f"  [WARN] Failed to generate efficiency_curves: {e}")

    try:
        graphs.append(plot_04_scalability_heatmap(merged))
    except Exception as e:
        print(f"  [WARN] Failed to generate scalability_heatmap: {e}")

    try:
        graphs.append(plot_05_workload_vs_speedup(merged))
    except Exception as e:
        print(f"  [WARN] Failed to generate workload_vs_speedup: {e}")

    try:
        graphs.append(plot_06_time_by_population(merged, seq, cuda_merged))
    except Exception as e:
        print(f"  [WARN] Failed to generate time_by_population: {e}")

    try:
        graphs.append(plot_07_speedup_distribution(merged))
    except Exception as e:
        print(f"  [WARN] Failed to generate speedup_distribution: {e}")

    try:
        graphs.append(plot_08_efficiency_heatmap_by_workload(merged))
    except Exception as e:
        print(f"  [WARN] Failed to generate efficiency_heatmap: {e}")

    try:
        graphs.append(plot_09_overall_summary(merged, seq, seq_baseline))
    except Exception as e:
        print(f"  [WARN] Failed to generate overall_summary: {e}")

    try:
        graphs.append(plot_10_convergence_comparison(records))
    except Exception as e:
        print(f"  [WARN] Failed to generate convergence_comparison: {e}")

    try:
        graphs.append(plot_11_amdahl_analysis(merged))
    except Exception as e:
        print(f"  [WARN] Failed to generate amdahl_analysis: {e}")

    try:
        graphs.append(plot_12_per_function_time_breakdown(merged, seq, cuda_merged))
    except Exception as e:
        print(f"  [WARN] Failed to generate per_function_breakdown: {e}")

    try:
        result = plot_13_cuda_vs_cpu_speedup(cuda_merged, seq)
        if result:
            graphs.append(result)
    except Exception as e:
        print(f"  [WARN] Failed to generate cuda_vs_cpu_speedup: {e}")

    try:
        result = plot_14_all_implementations_speedup(merged, cuda_merged)
        if result:
            graphs.append(result)
    except Exception as e:
        print(f"  [WARN] Failed to generate all_implementations_speedup: {e}")

    print(f"  Generated {len(graphs)} graphs")

    # ── Step 4: Save results
    print("\n[4/4] Saving results...")

    # Save report
    report = generate_report(df, merged, seq, seq_baseline, warnings, cuda_merged)
    report_path = OUTPUT_DIR / "analysis_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  [SAVED] {report_path.name}")

    # Save CSV summary
    csv_df = generate_csv_summary(df, merged, seq, cuda_merged)
    csv_path = OUTPUT_DIR / "metrics_summary.csv"
    csv_df.to_csv(csv_path, index=False)
    print(f"  [SAVED] {csv_path.name}")

    # Save JSON summary
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

    # Per-function stats
    for func_id in range(1, 11):
        func_data = merged[(merged['func'] == func_id) & (merged['threads'] == 16)]
        if not func_data.empty:
            json_summary['per_function_speedup_T16'][FUNC_NAMES.get(func_id, f"F{func_id}")] = {
                'avg_speedup': round(float(func_data['speedup'].mean()), 4),
                'max_speedup': round(float(func_data['speedup'].max()), 4),
                'avg_efficiency': round(float(func_data['efficiency'].mean()), 4),
            }

    # Per-function CUDA stats
    if len(cuda_merged) > 0:
        for func_id in range(1, 11):
            func_data = cuda_merged[cuda_merged['func'] == func_id]
            if not func_data.empty:
                json_summary['per_function_cuda_speedup'][FUNC_NAMES.get(func_id, f"F{func_id}")] = {
                    'avg_speedup': round(float(func_data['speedup'].mean()), 4),
                    'max_speedup': round(float(func_data['speedup'].max()), 4),
                    'avg_time_ms': round(float(func_data['time_ms'].mean()), 2),
                }

    # Per-thread stats
    if len(merged) > 0:
        for t in sorted(merged['threads'].unique()):
            sub = merged[merged['threads'] == t]
            json_summary['per_thread_avg_speedup'][str(int(t))] = {
                'avg_speedup': round(float(sub['speedup'].mean()), 4),
                'avg_efficiency': round(float(sub['efficiency'].mean()), 4),
            }

    # CUDA overall entry
    if len(cuda_merged) > 0:
        json_summary['per_thread_avg_speedup']['CUDA'] = {
            'avg_speedup': round(float(cuda_merged['speedup'].mean()), 4),
            'avg_efficiency': None,
        }

    json_path = OUTPUT_DIR / "metrics_summary.json"
    with open(json_path, 'w') as f:
        json.dump(json_summary, f, indent=2)
    print(f"  [SAVED] {json_path.name}")

    # Create/update 'latest' symlink/copy
    if LATEST_DIR.exists():
        if LATEST_DIR.is_symlink() or LATEST_DIR.is_dir():
            shutil.rmtree(LATEST_DIR, ignore_errors=True)
    try:
        # Try symlink first (may fail on Windows without admin)
        os.symlink(OUTPUT_DIR, LATEST_DIR)
        print(f"  [LINK] results/latest -> {TIMESTAMP}")
    except (OSError, NotImplementedError):
        # Fall back to copy
        shutil.copytree(OUTPUT_DIR, LATEST_DIR)
        print(f"  [COPY] results/latest <- {TIMESTAMP}")

    print()
    print("=" * 70)
    print(f"  Analysis complete! Results saved to: results/{TIMESTAMP}/")
    print(f"  Also available at: results/latest/")
    print(f"  Files:")
    print(f"    - analysis_report.md")
    print(f"    - metrics_summary.csv  ({len(csv_df)} rows)")
    print(f"    - metrics_summary.json")
    print(f"    - {len(graphs)} PNG graphs")
    print("=" * 70)


if __name__ == '__main__':
    main()
