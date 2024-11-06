"""
Parse the given CSV model files of latency and throughput and combine them into a new CSV file.
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
    df_new['model'] = df['model'] + '_' + df['tp'].astype(str) + '_' + df['batch_size'].astype(str) + '_' + df['input_len'].astype(str) + '_' + df['output_len'].astype(str) + '_' + df['dtype']
    
    # Put the column of 'latency (ms)' to a new column named 'performance'
    df_new['performance'] = df['latency (ms)']
    
    # Add a new column named 'metric' and set the value to 'ms'
    df_new['metric'] = 'ms'
    
    return df_new

def parse_throughput_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Create a new DataFrame
    df_new = pd.DataFrame()
    
    # Combine the columns of model, tp, batch_size, input_len, output_len, and dtype to a new column named 'model'
    df_new['model'] = df['model'] + '_' + df['tp'].astype(str) + '_' + df['requests'].astype(str) + '_' + df['input_len'].astype(str) + '_' + df['output_len'].astype(str) + '_' + df['dtype']
    
    # Put the column of 'throughput_tot (tok/sec)' to a new column named 'performance'
    df_new['performance'] = df['throughput_tot (tok/sec)']
    
    # Add a new column named 'metric' and set the value to 'samples/sec'
    df_new['metric'] = 'tok/sec'
    
    return df_new


def parse_args():
    parser = argparse.ArgumentParser(description='Parse the CSV file about latency and throughput.')
    parser.add_argument('--file_latency', type=str, help='The file name of the latency report')
    parser.add_argument('--file_throughput', type=str, help='The file name of the throughput report')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()
    file_latency = args.file_latency
    file_throughput = args.file_throughput

    # Extract the model name from the file name
    model_name = file_latency.split('/')[-1].split('_')[0]

    df_latency = pd.DataFrame()
    df_throughput = pd.DataFrame()
    
    # Check if the file exists
    if file_latency and os.path.exists(file_latency):
        # Parse the CSV file
        df_latency = parse_latency_csv(file_latency)
        
        # Print the first 5 rows of the DataFrame
        print(df_latency.head())
    else:
        print('The file of latency summary is not found.')

    # Check if the file exists
    if file_throughput and os.path.exists(file_throughput):
        # Parse the CSV file
        df_throughput = parse_throughput_csv(file_throughput)
        
        # Print the first 5 rows of the DataFrame
        print(df_throughput.head())
    else:
        print('The file of throughput summary is not found.')

    # Combine the DataFrames of latency and throughput and write to a new CSV file
    df_combined = pd.concat([df_latency, df_throughput], ignore_index=True)

    # Get the parent directory of the __file__
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    df_combined.to_csv(f'{parent_dir}/perf_{model_name}.csv', index=False)
    print('Parsing the multiple results and creating perf.csv have been done.')
