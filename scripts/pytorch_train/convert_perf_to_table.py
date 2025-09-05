#!/usr/bin/env python3
"""
Script to convert performance CSV data into a formatted table.
Converts all CSV files in the current directory (excluding _env.csv, perf.csv, perf_entry.csv)
into a table format similar to the benchmark results image.
"""

import pandas as pd
import numpy as np
from tabulate import tabulate
import argparse
import sys
import os
import glob

def get_csv_files():
    """
    Get all CSV files in the current directory, excluding specified patterns.
    
    Returns:
        list: List of CSV file paths
    """
    # Get all CSV files in current directory
    csv_files = glob.glob("*.csv")
    
    # Filter out excluded files
    excluded_patterns = ['_env.csv', 'perf.csv', 'perf_entry.csv']
    filtered_files = []
    
    for file in csv_files:
        should_exclude = False
        for pattern in excluded_patterns:
            if file.endswith(pattern):
                should_exclude = True
                break
        
        if not should_exclude:
            filtered_files.append(file)
    
    return filtered_files

def convert_csv_to_table(csv_file, output_format='grid', highlight_best=True):
    """
    Convert CSV performance data to a formatted table.
    
    Args:
        csv_file (str): Path to the CSV file
        output_format (str): Table format ('grid', 'simple', 'pipe', etc.)
        highlight_best (bool): Whether to highlight the best performing row
        
    Returns:
        str: Formatted table string
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Initialize results list
    results = []
    
    # Check if precision column exists in the CSV
    has_precision = 'precision' in df.columns
    
    # Group by mode to process each finetuning method
    for mode in df['mode'].unique():
        mode_data = df[df['mode'] == mode]
        
        # Determine finetune method based on mode
        if mode == 'finetune_fw':
            finetune_method = 'full weight (SFT)'
        elif mode == 'finetune_lora':
            finetune_method = 'lora'
        else:
            continue
        
        if has_precision:
            # If precision column exists, use it directly
            for _, row in mode_data.iterrows():
                if row['metric'] == 'avg_tokens_per_s_per_gpu':
                    tokens_per_gpu = row['performance']
                    precision = row['precision'].strip()  # Strip whitespace
                    
                    # Find corresponding memory data for the same precision
                    mem_row = mode_data[(mode_data['metric'] == 'max_memory_alloc') & 
                                       (mode_data['precision'].str.strip() == precision)]
                    if not mem_row.empty:
                        mem_alloc = mem_row.iloc[0]['performance']
                        
                        # Extract model family and size from model name
                        model_name = row['model']
                        model_family = model_name.split('-')[0].lower() if '-' in model_name else 'unknown'
                        model_size = '-'.join(model_name.split('-')[1:]) if '-' in model_name else model_name
                        
                        results.append({
                            'Model Family': model_family,
                            'Model Size': model_size,
                            'Finetune Method': finetune_method,
                            'Precision': precision.lower(),  # Convert to lowercase for consistency
                            'Token/sec/GPU': tokens_per_gpu,
                            'Max mem alloc (GB)': mem_alloc
                        })
        else:
            # Fallback to old logic if no precision column
            tokens_data = mode_data[mode_data['metric'] == 'avg_tokens_per_s_per_gpu']['performance'].values
            mem_data = mode_data[mode_data['metric'] == 'max_memory_alloc']['performance'].values
            
            # Based on the CSV structure, we have 2 pairs of data for each mode
            # The first pair is likely bf16, the second is fp8
            if len(tokens_data) == 2 and len(mem_data) == 2:
                # Get model info from the first row
                first_row = mode_data.iloc[0]
                model_name = first_row['model']
                model_family = model_name.split('-')[0].lower() if '-' in model_name else 'unknown'
                model_size = '-'.join(model_name.split('-')[1:]) if '-' in model_name else model_name
                
                # First pair (bf16)
                results.append({
                    'Model Family': model_family,
                    'Model Size': model_size,
                    'Finetune Method': finetune_method,
                    'Precision': 'bf16',
                    'Token/sec/GPU': tokens_data[0],
                    'Max mem alloc (GB)': mem_data[0]
                })
                
                # Second pair (fp8)
                results.append({
                    'Model Family': model_family,
                    'Model Size': model_size,
                    'Finetune Method': finetune_method,
                    'Precision': 'fp8',
                    'Token/sec/GPU': tokens_data[1],
                    'Max mem alloc (GB)': mem_data[1]
                })
            else:
                # Single entry case
                # Get model info from the first row
                first_row = mode_data.iloc[0]
                model_name = first_row['model']
                model_family = model_name.split('-')[0].lower() if '-' in model_name else 'unknown'
                model_size = '-'.join(model_name.split('-')[1:]) if '-' in model_name else model_name
                
                results.append({
                    'Model Family': model_family,
                    'Model Size': model_size,
                    'Finetune Method': finetune_method,
                    'Precision': 'bf16',  # Default assumption
                    'Token/sec/GPU': tokens_data[0],
                    'Max mem alloc (GB)': mem_data[0]
                })
    
    # Create DataFrame from results
    result_df = pd.DataFrame(results)
    
    # Sort by finetune method and precision for better presentation
    result_df = result_df.sort_values(['Finetune Method', 'Precision'], 
                                     key=lambda x: pd.Categorical(x, categories=['full weight (SFT)', 'lora']))
    
    # Format the numeric columns
    result_df['Token/sec/GPU'] = result_df['Token/sec/GPU'].apply(lambda x: f"{x:.2f}")
    result_df['Max mem alloc (GB)'] = result_df['Max mem alloc (GB)'].apply(lambda x: f"{x:.2f}")
    
    # Create formatted table
    table = tabulate(result_df, headers='keys', tablefmt=output_format, showindex=False)
    
    # Add highlighting if requested
    if highlight_best:
        # Find the row with highest token/sec/GPU
        max_tokens_idx = result_df['Token/sec/GPU'].astype(float).idxmax()
        table_lines = table.split('\n')
        
        # Add highlighting comment
        highlighted_row = max_tokens_idx + 2  # +2 for header and separator
        if highlighted_row < len(table_lines):
            table_lines[highlighted_row] = f"# HIGHLIGHTED: {table_lines[highlighted_row]}"
        
        table = '\n'.join(table_lines)
    
    return table

def extract_data_for_excel(csv_file):
    """
    Extract data from CSV file for Excel export.
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        list: List of dictionaries with extracted data
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Initialize results list
    results = []
    
    # Get model name from filename and remove "perf_" prefix
    model_name = os.path.splitext(os.path.basename(csv_file))[0]
    if model_name.startswith('perf_'):
        model_name = model_name[5:]  # Remove "perf_" prefix
    
    # Check if precision column exists in the CSV
    has_precision = 'precision' in df.columns
    
    # Group by mode to process each finetuning method
    for mode in df['mode'].unique():
        mode_data = df[df['mode'] == mode]
        
        # Determine finetune method based on mode
        if mode == 'finetune_fw':
            finetune_method = 'full weight (SFT)'
        elif mode == 'finetune_lora':
            finetune_method = 'lora'
        else:
            continue
        
        if has_precision:
            # If precision column exists, use it directly
            for _, row in mode_data.iterrows():
                if row['metric'] == 'avg_tokens_per_s_per_gpu':
                    tokens_per_gpu = row['performance']
                    precision = row['precision'].strip()  # Strip whitespace
                    
                    # Find corresponding memory data for the same precision
                    mem_row = mode_data[(mode_data['metric'] == 'max_memory_alloc') & 
                                       (mode_data['precision'].str.strip() == precision)]
                    if not mem_row.empty:
                        mem_alloc = mem_row.iloc[0]['performance']
                        
                        results.append({
                            'Model Name': model_name,
                            'Finetune Method': finetune_method,
                            'Precision': precision.lower(),  # Convert to lowercase for consistency
                            'Token/sec/GPU': round(tokens_per_gpu, 2),
                            'Max mem alloc (GB)': round(mem_alloc, 2)
                        })
        else:
            # Fallback to old logic if no precision column
            tokens_data = mode_data[mode_data['metric'] == 'avg_tokens_per_s_per_gpu']['performance'].values
            mem_data = mode_data[mode_data['metric'] == 'max_memory_alloc']['performance'].values
            
            # Based on the CSV structure, we have 2 pairs of data for each mode
            # The first pair is likely bf16, the second is fp8
            if len(tokens_data) == 2 and len(mem_data) == 2:
                # First pair (bf16)
                results.append({
                    'Model Name': model_name,
                    'Finetune Method': finetune_method,
                    'Precision': 'bf16',
                    'Token/sec/GPU': round(tokens_data[0], 2),
                    'Max mem alloc (GB)': round(mem_data[0], 2)
                })
                
                # Second pair (fp8)
                results.append({
                    'Model Name': model_name,
                    'Finetune Method': finetune_method,
                    'Precision': 'fp8',
                    'Token/sec/GPU': round(tokens_data[1], 2),
                    'Max mem alloc (GB)': round(mem_data[1], 2)
                })
            else:
                # Single entry case
                results.append({
                    'Model Name': model_name,
                    'Finetune Method': finetune_method,
                    'Precision': 'bf16',  # Default assumption
                    'Token/sec/GPU': round(tokens_data[0], 2),
                    'Max mem alloc (GB)': round(mem_data[0], 2)
                })
    
    return results

def save_to_markdown(result_df, output_file):
    """Save results to a markdown table format."""
    markdown_table = result_df.to_markdown(index=False)
    
    with open(output_file, 'w') as f:
        f.write("# Performance Benchmark Results\n\n")
        f.write(markdown_table)
        f.write("\n\n")
        
        # Add summary statistics
        f.write("## Summary\n\n")
        f.write(f"- **Best Performance**: {result_df['Token/sec/GPU'].astype(float).max():.2f} tokens/sec/GPU\n")
        f.write(f"- **Lowest Memory Usage**: {result_df['Max mem alloc (GB)'].astype(float).min():.2f} GB\n")
        f.write(f"- **Total Configurations**: {len(result_df)}\n")

def process_all_csv_files(output_format='grid', highlight_best=True, markdown_output=False, excel_output=False):
    """
    Process all CSV files in the current directory.
    
    Args:
        output_format (str): Table format
        highlight_best (bool): Whether to highlight best performance
        markdown_output (bool): Whether to save markdown files
        excel_output (bool): Whether to save Excel-compatible CSV
    """
    csv_files = get_csv_files()
    
    if not csv_files:
        print("No CSV files found matching the criteria.")
        return
    
    print(f"Found {len(csv_files)} CSV file(s) to process:")
    for file in csv_files:
        print(f"  - {file}")
    print()
    
    # For Excel output, collect all data
    all_excel_data = []
    
    for csv_file in csv_files:
        try:
            print(f"Processing: {csv_file}")
            
            # Convert to table (but don't save txt file)
            table = convert_csv_to_table(csv_file, output_format, highlight_best)
            
            print(f"  ✓ Processed '{csv_file}'")
            
            # Collect data for Excel if requested
            if excel_output:
                excel_data = extract_data_for_excel(csv_file)
                all_excel_data.extend(excel_data)
            
                            # Save markdown if requested
            if markdown_output:
                # Re-read the data to get the DataFrame for markdown
                df = pd.read_csv(csv_file)
                results = []
                
                # Check if precision column exists in the CSV
                has_precision = 'precision' in df.columns
                
                for mode in df['mode'].unique():
                    mode_data = df[df['mode'] == mode]
                    
                    if mode == 'finetune_fw':
                        finetune_method = 'full weight (SFT)'
                    elif mode == 'finetune_lora':
                        finetune_method = 'lora'
                    else:
                        continue
                    
                    if has_precision:
                        # If precision column exists, use it directly
                        for _, row in mode_data.iterrows():
                            if row['metric'] == 'avg_tokens_per_s_per_gpu':
                                tokens_per_gpu = row['performance']
                                precision = row['precision'].strip()  # Strip whitespace
                                
                                # Find corresponding memory data for the same precision
                                mem_row = mode_data[(mode_data['metric'] == 'max_memory_alloc') & 
                                                   (mode_data['precision'].str.strip() == precision)]
                                if not mem_row.empty:
                                    mem_alloc = mem_row.iloc[0]['performance']
                                    
                                    results.append({
                                        'Finetune Method': finetune_method,
                                        'Precision': precision.lower(),
                                        'Token/sec/GPU': f"{tokens_per_gpu:.2f}",
                                        'Max mem alloc (GB)': f"{mem_alloc:.2f}"
                                    })
                    else:
                        # Fallback to old logic if no precision column
                        tokens_data = mode_data[mode_data['metric'] == 'avg_tokens_per_s_per_gpu']['performance'].values
                        mem_data = mode_data[mode_data['metric'] == 'max_memory_alloc']['performance'].values
                        
                        if len(tokens_data) == 2 and len(mem_data) == 2:
                            results.extend([
                                {
                                    'Finetune Method': finetune_method,
                                    'Precision': 'bf16',
                                    'Token/sec/GPU': f"{tokens_data[0]:.2f}",
                                    'Max mem alloc (GB)': f"{mem_data[0]:.2f}"
                                },
                                {
                                    'Finetune Method': finetune_method,
                                    'Precision': 'fp8',
                                    'Token/sec/GPU': f"{tokens_data[1]:.2f}",
                                    'Max mem alloc (GB)': f"{mem_data[1]:.2f}"
                                }
                            ])
                
                result_df = pd.DataFrame(results)
                result_df = result_df.sort_values(['Finetune Method', 'Precision'])
                
                markdown_file = f"{base_name}_table.md"
                save_to_markdown(result_df, markdown_file)
                print(f"  ✓ Markdown saved to '{markdown_file}'")
            
        except Exception as e:
            print(f"  ✗ Error processing {csv_file}: {e}")
    
    # Save Excel-compatible CSV if requested
    if excel_output and all_excel_data:
        excel_df = pd.DataFrame(all_excel_data)
        excel_df = excel_df.sort_values(['Model Name', 'Finetune Method', 'Precision'])
        
        excel_file = 'all_benchmark_results.csv'
        excel_df.to_csv(excel_file, index=False)
        print(f"\n✓ Excel-compatible CSV saved to '{excel_file}'")
        print(f"  Total rows: {len(excel_df)}")
        print(f"  Columns: {', '.join(excel_df.columns)}")
    
    print(f"\nProcessing complete! Processed {len(csv_files)} file(s).")

def main():
    """Main function to run the conversion."""
    parser = argparse.ArgumentParser(description='Convert performance CSV files to formatted tables')
    parser.add_argument('--input', '-i', 
                       help='Input CSV file path (if not specified, processes all matching CSV files)')
    parser.add_argument('--output', '-o', 
                       help='Output file path (only used with --input)')
    parser.add_argument('--format', '-f', default='grid',
                       choices=['grid', 'simple', 'pipe', 'orgtbl', 'rst', 'mediawiki', 'html', 'latex'],
                       help='Table format')
    parser.add_argument('--markdown', '-m', action='store_true',
                       help='Save as markdown format')
    parser.add_argument('--excel', '-e', action='store_true',
                       help='Save all data to Excel-compatible CSV file')
    parser.add_argument('--no-highlight', action='store_true',
                       help='Disable highlighting of best performance row')
    
    args = parser.parse_args()
    
    try:
        if args.input:
            # Single file mode
            table = convert_csv_to_table(args.input, args.format, not args.no_highlight)
            
            print("Performance Benchmark Results")
            print("=" * 50)
            print(table)
            
            print(f"\nTable generated for '{args.input}'")
            
            # If markdown option is selected, also save as markdown
            if args.markdown:
                # Re-read the data to get the DataFrame for markdown
                df = pd.read_csv(args.input)
                results = []
                
                # Check if precision column exists in the CSV
                has_precision = 'precision' in df.columns
                
                for mode in df['mode'].unique():
                    mode_data = df[df['mode'] == mode]
                    
                    if mode == 'finetune_fw':
                        finetune_method = 'full weight (SFT)'
                    elif mode == 'finetune_lora':
                        finetune_method = 'lora'
                    else:
                        continue
                    
                    if has_precision:
                        # If precision column exists, use it directly
                        for _, row in mode_data.iterrows():
                            if row['metric'] == 'avg_tokens_per_s_per_gpu':
                                tokens_per_gpu = row['performance']
                                precision = row['precision'].strip()  # Strip whitespace
                                
                                # Find corresponding memory data for the same precision
                                mem_row = mode_data[(mode_data['metric'] == 'max_memory_alloc') & 
                                                   (mode_data['precision'].str.strip() == precision)]
                                if not mem_row.empty:
                                    mem_alloc = mem_row.iloc[0]['performance']
                                    
                                    results.append({
                                        'Finetune Method': finetune_method,
                                        'Precision': precision.lower(),
                                        'Token/sec/GPU': f"{tokens_per_gpu:.2f}",
                                        'Max mem alloc (GB)': f"{mem_alloc:.2f}"
                                    })
                    else:
                        # Fallback to old logic if no precision column
                        tokens_data = mode_data[mode_data['metric'] == 'avg_tokens_per_s_per_gpu']['performance'].values
                        mem_data = mode_data[mode_data['metric'] == 'max_memory_alloc']['performance'].values
                        
                        if len(tokens_data) == 2 and len(mem_data) == 2:
                            results.extend([
                                {
                                    'Finetune Method': finetune_method,
                                    'Precision': 'bf16',
                                    'Token/sec/GPU': f"{tokens_data[0]:.2f}",
                                    'Max mem alloc (GB)': f"{mem_data[0]:.2f}"
                                },
                                {
                                    'Finetune Method': finetune_method,
                                    'Precision': 'fp8',
                                    'Token/sec/GPU': f"{tokens_data[1]:.2f}",
                                    'Max mem alloc (GB)': f"{mem_data[1]:.2f}"
                                }
                            ])
                
                result_df = pd.DataFrame(results)
                result_df = result_df.sort_values(['Finetune Method', 'Precision'])
                
                markdown_file = args.output.replace('.txt', '.md')
                save_to_markdown(result_df, markdown_file)
                print(f"Markdown table saved to '{markdown_file}'")
        
        else:
            # Process all CSV files mode
            process_all_csv_files(args.format, not args.no_highlight, args.markdown, args.excel)
        
    except FileNotFoundError:
        if args.input:
            print(f"Error: File '{args.input}' not found.")
        else:
            print("Error: No CSV files found in current directory.")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
