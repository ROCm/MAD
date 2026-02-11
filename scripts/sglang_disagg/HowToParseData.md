# How to Parse Benchmark Data

This guide explains how to use the `parse_to_csv.py` script to extract benchmark results from SGLang log files.

## Overview

The `parse_to_csv.py` script parses SGLang benchmark log files and extracts key performance metrics into a CSV file. It extracts the following columns:

- **Concurrency**: Maximum request concurrency
- **Input tokens**: Input sequence length (from `isl` in RUNNING line)
- **Output tokens**: Output sequence length (from `osl` in RUNNING line)
- **Total Token throughput (tok/s)**: Maximum total token throughput across all iterations

## Requirements

- Python 3.x
- No external dependencies (uses only standard library)

## Usage

### Basic Usage

Run the script with the log file as an argument:

```bash
python3 parse_to_csv.py <log_file_name>
```

**Example:**
```bash
python3 parse_to_csv.py benchmark_xP1_yD1_CONCURRENCY.log
```

This will create a CSV file named `<log_file_name>_results.csv` in the same directory.

### Custom Output File

To specify a custom output file name:

```bash
python3 parse_to_csv.py <log_file_name> -o output.csv
```

or

```bash
python3 parse_to_csv.py <log_file_name> --output output.csv
```

**Example:**
```bash
python3 parse_to_csv.py benchmark.log -o my_results.csv
```

### Help

To see all available options:

```bash
python3 parse_to_csv.py --help
```

## How It Works

1. **Ignores Warmup Runs**: The script skips everything before the first `RUNNING: the benchserving script for iter: 1` line.

2. **Extracts Configuration**: For each benchmark run, it extracts:
   - Input tokens from `isl` value in `RUNNING: prompts isl X osl Y con Z`
   - Output tokens from `osl` value in `RUNNING: prompts isl X osl Y con Z`
   - Concurrency from `con` value in `RUNNING: prompts isl X osl Y con Z`

3. **Extracts Throughput**: Gets the Total Token throughput (tok/s) from the benchmark result section.

4. **Finds Maximum**: For each unique configuration (isl, osl, con), it keeps the maximum Total Token throughput across all iterations.

5. **Generates CSV**: Creates a CSV file with one row per unique configuration, sorted by concurrency, then input tokens, then output tokens.

## Output Format

The CSV file contains the following columns:

| Column | Description |
|--------|-------------|
| Concurrency | Maximum request concurrency |
| Input tokens | Input sequence length |
| Output tokens | Output sequence length |
| Total Token throughput (tok/s) | Maximum total token throughput (tokens per second) |

## Example Output

```csv
Concurrency,Input tokens,Output tokens,Total Token throughput (tok/s)
8,1024,1024,1741.40
8,8192,1024,6661.10
16,1024,1024,3438.62
16,8192,1024,11037.58
32,1024,1024,5852.66
32,8192,1024,16016.30
...
```

## Notes

- The script processes all iterations and takes the maximum throughput for each configuration.
- Only configurations that appear after the first iteration marker are processed.
- If a configuration appears multiple times across iterations, only the maximum throughput value is kept.

## Troubleshooting

**Error: Log file not found**
- Make sure the log file path is correct
- Use absolute path if the file is in a different directory

**No benchmark results found**
- Check that the log file contains benchmark result sections
- Verify the log file format matches the expected SGLang benchmark format

**Empty CSV file**
- Ensure the log file contains at least one `RUNNING: the benchserving script for iter: 1` line
- Check that benchmark result sections are present after the first iteration
