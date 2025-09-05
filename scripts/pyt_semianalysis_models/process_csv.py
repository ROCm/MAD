import csv
import sys

#Reads CSV from llm-train-bench and updates the DLM CSV.
def process_csv(input_file, output_file):
    with open(input_file, 'r') as infile:
        reader = csv.reader(infile)
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(["model", "performance", "metric"])
            next(reader, None)

            for row in reader:
                row = [field.strip() for field in row]
                model, strategy, gpu, dtype, batch_size, tflops, mfu = row
                model_name = f"{model}_{strategy}_{gpu}_{dtype}_bsz{batch_size}"
                writer.writerow([model_name+"_TFLOPS", tflops, "TFLOP/s/GPU"])
                writer.writerow([model_name+"_MFU", mfu, "MFU"])

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_csv> <output_csv>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]

    # Process the CSV and generate output csv in required format
    process_csv(input_csv, output_csv)
