#!/usr/bin/env python3
"""
Benchmark Log Parser

Extracts performance metrics from SGLang benchmark logs and displays results
in a formatted table. Optionally saves results to CSV file.
"""

import re
import sys
import pandas as pd
import argparse

def parse_benchmark_log(logfile):
    """Parse benchmark log file and extract performance metrics."""
    with open(logfile, 'r') as f:
        content = f.read()

    results = []

    # Split content by benchmark runs (each starts with [RUNNING])
    runs = re.split(r'\[RUNNING\]', content)[1:]  # Skip first empty split

    for run in runs:
        # Extract parameters from the [RUNNING] line
        match = re.search(r'prompts\s+isl\s+(\d+)\s+osl\s+(\d+)\s+con\s+(\d+)\s+model\s+([^\s]+)\s+xP=(\d+)\s+yD=(\d+)', run)
        if not match:
            continue

        question_len = int(match.group(1))
        output_len = int(match.group(2))
        concurrency = int(match.group(3))
        model = match.group(4)
        xp = int(match.group(5))
        yd = int(match.group(6))

        # Extract dataset statistics
        prompts_per_group = None
        total_prompts = None
        total_input_tokens = None
        total_output_tokens = None

        dataset_match = re.search(r'Prompts per group:\s+(\d+)', run)
        if dataset_match:
            prompts_per_group = int(dataset_match.group(1))

        total_prompts_match = re.search(r'Total prompts:\s+(\d+)', run)
        if total_prompts_match:
            total_prompts = int(total_prompts_match.group(1))

        input_tokens_match = re.search(r'Total input tokens:\s+([\d,]+)', run)
        if input_tokens_match:
            total_input_tokens = int(input_tokens_match.group(1).replace(',', ''))

        output_tokens_match = re.search(r'Total output tokens:\s+([\d,]+)', run)
        if output_tokens_match:
            total_output_tokens = int(output_tokens_match.group(1).replace(',', ''))

        # Find benchmark result block
        if '============ Serving Benchmark Result ============' in run:
            def extract(pattern):
                m = re.search(pattern, run)
                return float(m.group(1).replace(',', '')) if m else None

            # Extract all metrics
            successful_requests = extract(r'Successful requests:\s+([\d,]+)')
            duration = extract(r'Benchmark duration \(s\):\s+([\d\.]+)')
            req_throughput = extract(r'Request throughput \(req/s\):\s+([\d\.]+)')
            input_tok_throughput = extract(r'Input token throughput \(tok/s\):\s+([\d,\.]+)')
            output_tok_throughput = extract(r'Output token throughput \(tok/s\):\s+([\d,\.]+)')
            total_tok_throughput = extract(r'Total token throughput \(tok/s\):\s+([\d,\.]+)')
            mean_e2e = extract(r'Mean E2E Latency \(ms\):\s+([\d,\.]+)')
            mean_ttft = extract(r'Mean TTFT \(ms\):\s+([\d,\.]+)')
            mean_itl = extract(r'Mean ITL \(ms\):\s+([\d,\.]+)')

            results.append({
                'Model': model,
                'xP_yD': f"{xp}p{yd}d",
                'ISL': question_len,
                'OSL': output_len,
                'Concurrency': concurrency,
                'Prompts_Group': prompts_per_group,
                'Total_Prompts': total_prompts,
                'Total_Input_Tokens': total_input_tokens,
                'Total_Output_Tokens': total_output_tokens,
                'Request_Throughput_req_s': req_throughput,
                'Input_Token_Throughput_tok_s': input_tok_throughput,
                'Output_Token_Throughput_tok_s': output_tok_throughput,
                'Total_Token_Throughput_tok_s': total_tok_throughput,
                'Mean_E2E_Latency_ms': mean_e2e,
                'Mean_TTFT_ms': mean_ttft,
                'Mean_ITL_ms': mean_itl,
            })

    return results

def format_dataframe(df):
    """Format the dataframe for better readability."""
    # Format numeric columns
    numeric_cols = [
        'Request Throughput (req/s)', 'Input Token Throughput (tok/s)',
        'Output Token Throughput (tok/s)', 'Total Token Throughput (tok/s)',
        'Mean E2E Latency (ms)', 'Mean TTFT (ms)', 'Mean ITL (ms)'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}" if pd.notna(x) else x)

    # Format integer columns with commas
    integer_cols = ['Total Input Tokens', 'Total Output Tokens']
    for col in integer_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{int(x):,}" if pd.notna(x) else x)

    return df

def main():
    parser = argparse.ArgumentParser(
        description='Parse SGLang benchmark logs and extract performance metrics.',
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

    # Select columns based on compact option
    if args.compact:
        columns = [
            'Model', 'xP/yD', 'ISL', 'OSL', 'Concurrency',
            'Request Throughput (req/s)', 'Total Token Throughput (tok/s)',
            'Mean E2E Latency (ms)', 'Mean TTFT (ms)', 'Mean ITL (ms)'
        ]
        df = df[columns]

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
