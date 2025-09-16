import pandas as pd

# Input CSV file names
csv_files = ['/myworkspace/scripts/pyt_huggingface_diffusers/csvs/ROCM_SDXL_FINETUNE_bs24.csv', '/myworkspace/scripts/pyt_huggingface_diffusers/csvs/ROCM_SDXL_FINETUNE_bs3.csv']

# Output file name
output_file = '/myworkspace/results_pyt_huggingface_stable_diffusion_xl_lora._finetuning.csv'

# Read and concatenate all CSVs
merged_df = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

# Save the result
merged_df.to_csv(output_file, index=False)

print(f"Merged {len(csv_files)} files into '{output_file}'")
