import sys
import re
import csv

''' log line to be parsed:
    [Benchmark] e2e training completed (sec): 244.07
'''

def write_csv_header(csv_file):
    with open(csv_file, 'w') as f_csv:
        csv_writer = csv.writer(f_csv)
        header = ['model', 'performance', 'metric']
        csv_writer.writerow(header)

def write_csv(csv_file, model, th):
    with open(csv_file, 'a') as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(['sdxl_'+model, th, 'seconds'])

def parse_log(log_file, model, csv_file):
    with open(log_file, 'r') as f_log:
        for line in f_log.readlines():
            if '[Benchmark]' in line:
                print(line)
                feature = "[Benchmark] e2e training completed (sec): "
                th = line.replace(feature, "")
                write_csv(csv_file, model, th)

write_csv_header(sys.argv[2])
parse_log(sys.argv[1], 'lora', sys.argv[2])
