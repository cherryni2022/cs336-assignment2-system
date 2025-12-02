import os
from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt


DATA_FILE = Path("flash_attn_bwd1_benchmark_bfloat16.md")
OUT_DIR = Path("plots/flash_attn_bwd1_bf16")


def read_markdown_table(md_path: Path) -> pd.DataFrame:
    """Read the single markdown table into a DataFrame.

    The file uses a pipe-separated table with headers.
    """
    lines = md_path.read_text().strip().splitlines()
    # Filter table lines (start with '|') and skip the separator row of dashes
    table_lines = [ln for ln in lines if ln.strip().startswith('|')]
    # Remove the alignment row (contains only dashes and colons)
    clean_lines = []
    for ln in table_lines:
        if re.match(r"\|[-: ]+\|", ln):
            continue
        clean_lines.append(ln)

    # Convert pipe table to CSV-like rows
    rows = []
    for ln in clean_lines:
        parts = [p.strip() for p in ln.strip().strip('|').split('|')]
        rows.append(parts)

    header = rows[0]
    data_rows = rows[1:]
    df = pd.DataFrame(data_rows, columns=header)
    # Cast numerics
    for col in df.columns:
        if col in {"context_length", "d_model"}:
            df[col] = df[col].astype(int)
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def compute_speedups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['forward_speedup'] = df['regular_attn_forward_ms'] / df['flash_attn_triton_forward_ms']
    df['backward_speedup'] = df['regular_attn_backward_ms'] / df['flash_attn_backward_ms']
    df['total_speedup'] = df['regular_attn_total_ms'] / df['flash_attn_total_ms']
    return df


def plot_curves_for_d_model(df: pd.DataFrame, d_model: int, out_dir: Path):
    sub = df[df['d_model'] == d_model].sort_values('context_length')

    # Forward comparison curves
    plt.figure(figsize=(8, 5))
    plt.plot(sub['context_length'], sub['regular_attn_forward_ms'], label='Regular Forward', marker='o')
    plt.plot(sub['context_length'], sub['flash_attn_triton_forward_ms'], label='Flash Forward', marker='o')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('context_length (log2)')
    plt.ylabel('time (ms, log)')
    plt.title(f'Forward Time vs Context (bf16), d_model={d_model}')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / f'forward_curves_d{d_model}.png', dpi=150)
    plt.close()

    # Backward comparison curves
    plt.figure(figsize=(8, 5))
    plt.plot(sub['context_length'], sub['regular_attn_backward_ms'], label='Regular Backward', marker='o')
    plt.plot(sub['context_length'], sub['flash_attn_backward_ms'], label='Flash Backward', marker='o')
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('context_length (log2)')
    plt.ylabel('time (ms, log)')
    plt.title(f'Backward Time vs Context (bf16), d_model={d_model}')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / f'backward_curves_d{d_model}.png', dpi=150)
    plt.close()

    # Speedup curves
    plt.figure(figsize=(8, 5))
    plt.plot(sub['context_length'], sub['forward_speedup'], label='Forward speedup', marker='o')
    plt.plot(sub['context_length'], sub['backward_speedup'], label='Backward speedup', marker='o')
    plt.plot(sub['context_length'], sub['total_speedup'], label='Total speedup', marker='o')
    plt.xscale('log', base=2)
    plt.xlabel('context_length (log2)')
    plt.ylabel('Regular / Flash speedup (×)')
    plt.title(f'Speedup vs Context (bf16), d_model={d_model}')
    plt.legend()
    plt.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / f'speedup_curves_d{d_model}.png', dpi=150)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = read_markdown_table(DATA_FILE)
    df = compute_speedups(df)

    for d in sorted(df['d_model'].unique()):
        plot_curves_for_d_model(df, int(d), OUT_DIR)

    # Also export a CSV with computed speedups
    df.to_csv(OUT_DIR / 'bf16_bwd1_speedups.csv', index=False)
    print(f"Saved plots to {OUT_DIR} and speedups CSV.")


if __name__ == '__main__':
    main()

