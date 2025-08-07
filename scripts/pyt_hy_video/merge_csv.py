import pandas as pd

# Input CSV file names
csv_files = ['/myworkspace/perf_pyt_hy_video_1280.csv', '/myworkspace/perf_pyt_hy_video_960.csv', '/myworkspace/perf_pyt_hy_video_720.csv']

# Output file name
output_file = '/myworkspace/perf_pyt_hy_video.csv'

# Read and concatenate all CSVs
merged_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Save the result
merged_df.to_csv(output_file, index=False)

print(f"Merged {len(csv_files)} files into '{output_file}'")
