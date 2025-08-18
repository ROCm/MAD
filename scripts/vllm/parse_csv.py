"""
Parse the given CSV model files of latency and throughput and serving and combine them into a new CSV file.
"""
import pandas as pd
import os
import argparse


def parse_latency_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Create a new DataFrame
    df_new = pd.DataFrame()
    
    # Combine the columns of model, tp, batch_size, input_len, output_len, and dtype to a new column named 'model'
    df_new['model'] = df['model'] + '_' + 'latency' + '_' + df['tp'].astype(str) + '_' + df['bs'].astype(str) + '_' + df['in'].astype(str) + '_' + df['out'].astype(str) + '_' + df['dtype']
    
    # Put the column of 'avg_latency (sec)' to a new column named 'performance'
    df_new['performance'] = df['avg_latency (sec)'] * 1000
    
    # Add a new column named 'metric' and set the value to 'ms'
    df_new['metric'] = 'ms'
    
    return df_new

def parse_throughput_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Create a new DataFrame
    df_new = pd.DataFrame()
    
    for row in df.to_dict(orient="records"):
        model = row['model'] + '_' + 'throughput' + '_' + str(row['tp']) + '_' + str(row['num_prompts']) + '_' + str(row['in']) + '_' + str(row['out']) + '_' + str(row['dtype'])
        outputs = ['throughput_tot (tok/sec)', 'throughput_gen (tok/sec)']
        for output in outputs:
            row_new = pd.DataFrame(index=[0])
            row_new['model'] = model + '_' + output.split()[0]
            row_new['performance'] = row[output]
            row_new['metric'] = output.split()[1].strip('()')
            df_new = pd.concat([df_new, row_new], ignore_index=True)
    
    return df_new

def parse_serving_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Create a new DataFrame
    df_new = pd.DataFrame()

    # For serving, report multiple output metrics by appending the metric name to the model string and creating a new row for each metric
    for row in df.to_dict(orient="records"):
        model = row['model'] + '_' + 'serving' + '_' + str(row['tp']) + '_' + str(row['num_prompts']) + '_' + str(row['in']) + '_' + str(row['out']) + '_' + str(row['max_concurrency']) + '_' + str(row['dtype'])
        outputs = ['throughput_tot (tok/sec)', 'throughput_gen (tok/sec)', 'median_ttft (ms)', 'median_tpot (ms)', 'median_itl (ms)', 'median_e2el (ms)']
        for output in outputs:
            row_new = pd.DataFrame(index=[0])
            row_new['model'] = model + '_' + output.split()[0]
            row_new['performance'] = row[output]
            row_new['metric'] = output.split()[1].strip('()')
            df_new = pd.concat([df_new, row_new], ignore_index=True)

    return df_new


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--latency_csv', type=str)
    parser.add_argument('--throughput_csv', type=str)
    parser.add_argument('--serving_csv', type=str)
    parser.add_argument('--output_csv', type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()

    df_latency = pd.DataFrame()
    df_throughput = pd.DataFrame()
    df_serving = pd.DataFrame()
    
    # Check if the file exists
    if args.latency_csv and os.path.exists(args.latency_csv):
        # Parse the CSV file
        df_latency = parse_latency_csv(args.latency_csv)
        
        # Print the first 5 rows of the DataFrame
        print(df_latency.head())

    # Check if the file exists
    if args.throughput_csv and os.path.exists(args.throughput_csv):
        # Parse the CSV file
        df_throughput = parse_throughput_csv(args.throughput_csv)
        
        # Print the first 5 rows of the DataFrame
        print(df_throughput.head())
    
    # Check if the file exists
    if args.serving_csv and os.path.exists(args.serving_csv):
        # Parse the CSV file
        df_serving = parse_serving_csv(args.serving_csv)
        
        # Print the first 5 rows of the DataFrame
        print(df_serving.head())

    # Combine the DataFrames and write to a new CSV file
    df_combined = pd.concat([df_latency, df_throughput, df_serving], ignore_index=True)
    df_combined.to_csv(args.output_csv, index=False)
    print("Combined results have been written to", args.output_csv)
