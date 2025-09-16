import sys
import re
import csv

''' log line to be parsed:
    DLL 2024-03-03 04:51:55.337358 - () best_train_throughput : 7447588.619071551 samples/s best_eval_throughput : 146262009.73534802 samples/s mean_train_throughput : 7386178.477837743 samples/s mean_eval_throughput : 140314647.07790136 samples/s best_accuracy : 0.8359122843753836 None best_epoch : 1 None time_to_target : 28.232350826263428 s time_to_best_model : 28.232211112976074 s validation_loss : 0.10824 None train_loss : 0.21788 None
'''

def write_csv_header(csv_file):
    with open(csv_file, 'w') as f_csv:
        csv_writer = csv.writer(f_csv)
        header = ['model', 'performance', 'metric']
        csv_writer.writerow(header)

def write_csv(csv_file, model, th):
    with open(csv_file, 'a') as f_csv:
        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(['ncf_'+model, th, 'samples/s'])

def parse_log(log_file, model, csv_file):
    with open(log_file, 'r') as f_log:
        for line in f_log.readlines():
            if 'best_train_throughput' in line:
                match = re.search(r"best_train_throughput : (\d+\.\d+)", line)
                if match:
                    th = match.group(1)
                    #print(f'performance: {th} samples/s')
                    write_csv(csv_file, model, th)
                else:
                    print(f'ERROR: Incorrect line format.')

write_csv_header(sys.argv[3])
parse_log(sys.argv[1], 'fp16', sys.argv[3])
parse_log(sys.argv[2], 'fp32', sys.argv[3])
