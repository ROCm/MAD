#!/usr/bin/env python3
"""
Parse SGLang benchmark log file and save results to CSV.
Extracts: Concurrency, Input tokens, Output tokens, Total Token throughput (tok/s)
For each configuration, takes the MAX Total Token throughput across all iterations.
"""

import re
import csv
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


def parse_benchmark_log(log_file: str) -> Dict[Tuple[int, int, int], Dict]:
    """Parse benchmark log file and extract results, keeping max throughput per configuration."""
    results = defaultdict(lambda: {'concurrency': None, 'input_tokens': None, 
                                    'output_tokens': None, 'max_throughput': 0.0})
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find the start of the first iteration (ignore warmup)
    first_iter_match = re.search(r'RUNNING: the benchserving script for iter: 1', content)
    if not first_iter_match:
        print("Warning: No iteration 1 found. Processing entire file.")
        start_pos = 0
    else:
        start_pos = first_iter_match.start()
    
    # Process only from first iteration onwards
    content = content[start_pos:]
    
    # Split by benchmark result sections
    sections = re.split(r'============ Serving Benchmark Result ============', content)
    
    current_input_seq_len = None
    current_output_seq_len = None
    current_concurrency = None
    
    for i, section in enumerate(sections[1:], 1):  # Skip first empty section
        # Look for configuration in previous sections (from RUNNING line)
        if i > 1:
            prev_section = sections[i-1]
            
            # Extract config: prompts  isl <num> osl <num> con <num>
            config_match = re.search(r'RUNNING: prompts\s+isl\s+(\d+)\s+osl\s+(\d+)\s+con\s+(\d+)', prev_section)
            if config_match:
                current_input_seq_len = int(config_match.group(1))
                current_output_seq_len = int(config_match.group(2))
                current_concurrency = int(config_match.group(3))
        
        # Extract Total token throughput (tok/s) from benchmark result section
        throughput_match = re.search(r'Total token throughput \(tok/s\):\s+([\d.]+)', section)
        throughput = float(throughput_match.group(1)) if throughput_match else None
        
        # Only process if we have a valid configuration from RUNNING line and throughput
        if current_input_seq_len and current_output_seq_len and current_concurrency and throughput is not None:
            config_key = (current_input_seq_len, current_output_seq_len, current_concurrency)
            
            # Update results for this configuration
            # Always use values from RUNNING line (isl, osl, con)
            results[config_key]['concurrency'] = current_concurrency
            results[config_key]['input_tokens'] = current_input_seq_len
            results[config_key]['output_tokens'] = current_output_seq_len
            
            # Keep the maximum throughput
            if throughput > results[config_key]['max_throughput']:
                results[config_key]['max_throughput'] = throughput
    
    return results


def save_to_csv(results: Dict[Tuple[int, int, int], Dict], output_file: str):
    """Save results to CSV file with specified columns."""
    if not results:
        print("No results to save.")
        return
    
    # Define column order
    fieldnames = ['Concurrency', 'Input tokens', 'Output tokens', 'Total Token throughput (tok/s)']
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Sort by concurrency, then input tokens, then output tokens
        for (input_tokens, output_tokens, concurrency), data in sorted(results.items(), 
                                                                         key=lambda x: (x[0][2], x[0][0], x[0][1])):
            row = {
                'Concurrency': data['concurrency'],
                'Input tokens': data['input_tokens'],
                'Output tokens': data['output_tokens'],
                'Total Token throughput (tok/s)': f"{data['max_throughput']:.2f}"
            }
            writer.writerow(row)
    
    print(f"Saved {len(results)} benchmark configurations to {output_file}")


def main():
    """Main function."""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse SGLang benchmark log file and save results to CSV')
    parser.add_argument('log_file', type=str, help='Path to benchmark log file')
    parser.add_argument('-o', '--output', type=str, help='Output CSV file name (default: <log_file>_results.csv)')
    
    args = parser.parse_args()
    
    log_file = args.log_file
    
    # Check if file exists
    if not Path(log_file).exists():
        print(f"Error: Log file not found: {log_file}")
        sys.exit(1)
    
    print(f"Parsing log file: {log_file}")
    
    # Parse the log file
    results = parse_benchmark_log(log_file)
    
    if not results:
        print("No benchmark results found in log file.")
        return
    
    # Generate output filename
    if args.output:
        output_file = args.output
    else:
        output_file = Path(log_file).stem + '_results.csv'
    
    # Save to CSV
    save_to_csv(results, output_file)
    
    print(f"\nSummary:")
    print(f"  Total unique configurations: {len(results)}")
    print(f"  Output file: {output_file}")


if __name__ == '__main__':
    main()
