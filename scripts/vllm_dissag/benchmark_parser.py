#!/usr/bin/env python3
"""
Benchmark Log Parser

Extracts performance metrics from the vLLM disaggregated benchmark logs produced
by benchmark_xPyD.sh and benchmark_long_context.sh (i.e. `vllm bench serve`
"Serving Benchmark Result" blocks) and displays them in a formatted table.
Optionally saves results to CSV.

Each measured cell in those logs is delimited by a [RUNNING] line in one of two
shapes:
  benchmark_long_context.sh:  [RUNNING] isl=1024 osl=1024 con=1 warmups=2 prompts=16 ...
  benchmark_xPyD.sh:          [RUNNING] prompts 16 isl 1024 osl 1024 con 1 ...
xP / yD / model are not on the [RUNNING] line; they are taken from the log
filename (..._xP<P>_yD<D>_<MODEL>_CONCURRENCY.log).
"""

import os
import re
import sys
import pandas as pd
import argparse

# Columns produced by parse_benchmark_log(), in display order. Single source of
# truth so format_dataframe()/--compact stay in sync with what's actually parsed.
COLUMNS = [
    'Model', 'xP_yD', 'ISL', 'OSL', 'Concurrency', 'Prompts',
    'Successful', 'Failed',
    'Total_Input_Tokens', 'Total_Output_Tokens',
    'Request_Throughput_req_s', 'Output_Token_Throughput_tok_s',
    'Total_Token_Throughput_tok_s',
    'Mean_TTFT_ms', 'Median_TTFT_ms',
    'Mean_ITL_ms', 'Median_ITL_ms',
    'Mean_TPOT_ms', 'Median_TPOT_ms',
]
FLOAT_COLS = [
    'Request_Throughput_req_s', 'Output_Token_Throughput_tok_s',
    'Total_Token_Throughput_tok_s', 'Mean_TTFT_ms', 'Median_TTFT_ms',
    'Mean_ITL_ms', 'Median_ITL_ms', 'Mean_TPOT_ms', 'Median_TPOT_ms',
]
INT_COMMA_COLS = ['Total_Input_Tokens', 'Total_Output_Tokens']
COMPACT_COLS = [
    'Model', 'xP_yD', 'ISL', 'OSL', 'Concurrency', 'Successful', 'Failed',
    'Output_Token_Throughput_tok_s', 'Median_TTFT_ms', 'Median_ITL_ms',
]


def _meta_from_filename(logfile):
    """Pull model / xP / yD from the log filename; None if absent."""
    name = os.path.basename(logfile)
    model = m.group(1) if (m := re.search(r'_xP\d+_yD\d+_(.+?)_CONCURRENCY', name)) else None
    xy = re.search(r'_xP(\d+)_yD(\d+)_', name)
    xp_yd = f"{xy.group(1)}p{xy.group(2)}d" if xy else None
    return model, xp_yd


def parse_benchmark_log(logfile):
    """Parse a benchmark log file and extract per-cell performance metrics."""
    with open(logfile, 'r') as f:
        content = f.read()

    model, xp_yd = _meta_from_filename(logfile)
    results = []

    # Each measured cell starts with a [RUNNING] line.
    runs = re.split(r'\[RUNNING\]', content)[1:]  # skip preamble before first cell

    for run in runs:
        header = run.splitlines()[0] if run else ''
        # Accept both harness formats: 'isl=N osl=N con=N' and 'isl N osl N con N'.
        isl_m = re.search(r'isl[=\s]+(\d+)', header)
        osl_m = re.search(r'osl[=\s]+(\d+)', header)
        con_m = re.search(r'con[=\s]+(\d+)', header)
        if not (isl_m and osl_m and con_m):
            continue
        isl, osl, concurrency = int(isl_m.group(1)), int(osl_m.group(1)), int(con_m.group(1))
        prompts_m = re.search(r'prompts[=\s]+(\d+)', header)
        prompts = int(prompts_m.group(1)) if prompts_m else None

        # Only emit a row if the cell actually produced a result block.
        if '============ Serving Benchmark Result ============' not in run:
            continue

        def extract(pattern):
            m = re.search(pattern, run)
            if not m:
                return None
            v = float(m.group(1).replace(',', ''))
            # counters/token totals are integers — return int so CSV/table don't show 16.0
            return int(v) if v.is_integer() else v

        results.append({
            'Model': model,
            'xP_yD': xp_yd,
            'ISL': isl,
            'OSL': osl,
            'Concurrency': concurrency,
            'Prompts': prompts,
            'Successful': extract(r'Successful requests:\s+([\d,]+)'),
            'Failed': extract(r'Failed requests:\s+([\d,]+)'),
            'Total_Input_Tokens': extract(r'Total input tokens:\s+([\d,]+)'),
            # vLLM labels decode tokens "Total generated tokens".
            'Total_Output_Tokens': extract(r'Total generated tokens:\s+([\d,]+)'),
            'Request_Throughput_req_s': extract(r'Request throughput \(req/s\):\s+([\d,\.]+)'),
            'Output_Token_Throughput_tok_s': extract(r'Output token throughput \(tok/s\):\s+([\d,\.]+)'),
            'Total_Token_Throughput_tok_s': extract(r'Total token throughput \(tok/s\):\s+([\d,\.]+)'),
            'Mean_TTFT_ms': extract(r'Mean TTFT \(ms\):\s+([\d,\.]+)'),
            'Median_TTFT_ms': extract(r'Median TTFT \(ms\):\s+([\d,\.]+)'),
            'Mean_ITL_ms': extract(r'Mean ITL \(ms\):\s+([\d,\.]+)'),
            'Median_ITL_ms': extract(r'Median ITL \(ms\):\s+([\d,\.]+)'),
            'Mean_TPOT_ms': extract(r'Mean TPOT \(ms\):\s+([\d,\.]+)'),
            'Median_TPOT_ms': extract(r'Median TPOT \(ms\):\s+([\d,\.]+)'),
        })

    return results

def format_dataframe(df):
    """Format numeric columns for readable display (keys match COLUMNS)."""
    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)
    for col in INT_COMMA_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else x)
    return df

def main():
    parser = argparse.ArgumentParser(
        description='Parse vLLM disaggregated benchmark logs (vllm bench serve) and extract performance metrics.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s benchmark.log                    # Display results on screen
  %(prog)s benchmark.log --csv results.csv # Save to CSV file
  %(prog)s benchmark.log --csv             # Save to auto-named CSV file

The tool extracts metrics including:
  - Model configuration (xP/yD)
  - Input/Output sequence lengths (ISL/OSL)
  - Concurrency levels and prompt counts
  - Token throughput (input/output/total)
  - Latency metrics (E2E, TTFT, ITL)
        """
    )

    # Required arguments
    parser.add_argument(
        'logfile',
        help='Path to the benchmark log file to parse'
    )

    # Optional arguments
    parser.add_argument(
        '--csv',
        nargs='?',
        const='benchmark_results.csv',
        metavar='FILE',
        help='Save results to CSV file. If no filename provided, uses "benchmark_results.csv"'
    )

    parser.add_argument(
        '--compact',
        action='store_true',
        help='Use compact output format (fewer columns)'
    )

    parser.add_argument(
        '--no-screen',
        action='store_true',
        help='Skip screen output, only save to CSV (requires --csv)'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.no_screen and not args.csv:
        parser.error("--no-screen requires --csv option")

    try:
        results = parse_benchmark_log(args.logfile)
    except FileNotFoundError:
        print(f"Error: File '{args.logfile}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error parsing log file: {e}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("No benchmark results found in the log file.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(results)

    # Select columns based on compact option (keys match COLUMNS / parsed output)
    if args.compact:
        df = df[[c for c in COMPACT_COLS if c in df.columns]]

    # Format for display
    display_df = format_dataframe(df.copy())

    # Screen output
    if not args.no_screen:
        print("Benchmark Results Summary:")
        print("=" * 120)
        print(display_df.to_string(index=False))
        print(f"\nTotal runs parsed: {len(results)}")

    # CSV output
    if args.csv:
        try:
            # Save original unformatted data to CSV for better data processing
            df.to_csv(args.csv, index=False)
            if not args.no_screen:
                print(f"\nResults saved to: {args.csv}")
        except Exception as e:
            print(f"Error saving CSV file: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
