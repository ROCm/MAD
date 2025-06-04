'''
Code adapted from: https://github.com/genmoai/mochi/blob/main/demos/cli.py 
Thanks to the the authors of Mochi.
'''
import torch
import csv
import argparse

torch.set_float32_matmul_precision("high")


from genmo.mochi_preview.pipelines import (
    DecoderModelFactory,
    DitModelFactory,
    MochiSingleGPUPipeline,
    T5ModelFactory,
    linear_quadratic_schedule,
)

# Set up command line argument parsing
parser = argparse.ArgumentParser(description="Benchmark the Mochi pipeline")
parser.add_argument("--model_dir", type=str, default="/data/mochi", 
                    help="Directory containing model files")
parser.add_argument("--warmup_steps", type=int, default=1, 
                    help="warmup steps")
parser.add_argument("--benchmark_steps", type=int, default=5, 
                    help="benchmark steps")

args = parser.parse_args()

MOCHI_DIR = args.model_dir

 
pipeline = MochiSingleGPUPipeline(
    text_encoder_factory=T5ModelFactory(),
    dit_factory=DitModelFactory(
        model_path=f"{MOCHI_DIR}/dit.safetensors", model_dtype="bf16"
    ),
    decoder_factory=DecoderModelFactory(
        model_path=f"{MOCHI_DIR}/decoder.safetensors",
    ),
    cpu_offload=False,
    decode_type="tiled_full",
)
 
def benchmark_pipeline():
    # Parameters for benchmarking
    height = 480
    width = 848
    num_frames = 31
    num_inference_steps = 64
    sigma_schedule = linear_quadratic_schedule(64, 0.025)
    cfg_schedule = [4.5] * 64
    batch_cfg = False
    prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."
    negative_prompt = ""
    seed = 12345
     
    # Prepare CUDA events for timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
     
    # Warmup (run the pipeline twice without timing)
    print("Starting warmup...")
    for _ in range(args.warmup_steps):
        video = pipeline(
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            sigma_schedule=sigma_schedule,
            cfg_schedule=cfg_schedule,
            batch_cfg=batch_cfg,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
        )
     
    # Now we will run the benchmarking loop for 10 repeats
    timings = []
    print("Starting benchmarking...")
    for i in range(args.benchmark_steps):
        # Start the CUDA event
        start_event.record()
         
        # Run the pipeline
        video = pipeline(
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            sigma_schedule=sigma_schedule,
            cfg_schedule=cfg_schedule,
            batch_cfg=batch_cfg,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
        )
         
        # End the CUDA event
        end_event.record()
         
        # Wait for the events to complete
        torch.cuda.synchronize()
         
        # Calculate the elapsed time
        elapsed_time = start_event.elapsed_time(end_event)  # Time in milliseconds
        timings.append(elapsed_time)
        print(f"Repeat {i+1}: {elapsed_time:.2f} ms")
 
    # Calculate the average time and standard deviation
    avg_time = sum(timings) / len(timings)
    stddev_time = (sum((x - avg_time) ** 2 for x in timings) / len(timings)) ** 0.5
     
    print(f"\nAverage time: {avg_time:.2f} ms")
    print(f"Standard deviation: {stddev_time:.2f} ms")

    filename = "mochi_latency_results.csv"

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Latency (ms)'])  # Header row
        writer.writerow([f"{avg_time:.2f}"])
 
# Run the benchmark
benchmark_pipeline()