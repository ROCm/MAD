# lint as: python3
###############################################################################
#
# MIT License
#
# Copyright (c) 2024 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################
"""MAD: Model Automation and Dashboarding

The script builds the Docker image, runs the Docker container, executes training or inference of the LLMs on the container, 
and logs the performance metrics. 

The script takes the following arguments:
    --tags: tags to run model.
    --timeout: Timeout for the application running model in seconds, default timeout of 14400 (4 hours).
    --live-output: Prints output in real-time directly on STDOUT.
    --clean-docker_cache: Rebuild docker image without using cache.
    --keep-alive: Keep the container alive after the application finishes running.
    --keep-model-dir: Keep the model directory after the application finishes running.
    -o, --output: Output file for the result.
    --log-level: Log level for the logger.

The script uses the following environment variables:
    MAD_MODEL_NAME: Model name to run the application.
    MAD_GPU_VENDOR: GPU vendor (NVIDIA or AMD).
    MAD_SYSTEM_NGPUS: Number of GPUs in the system.
    MAD_SYSTEM_GPU_ARCHITECTURE: GPU architecture in the system.

The script uses the following Docker options:
    --network host: Use the host network.
    --cap-add SYS_PTRACE: Add SYS_PTRACE capability

The script uses the following Docker volumes:
    /myworkspace: Workspace directory for the model.

The script uses the following Docker image:
    ci-{model_name}: Docker image for the model.

The script uses the following Docker container:
    container_ci-{model_name}: Docker container for the model.

The script uses the following model details:
    Model name: Model name to run the application.
    Model URL: Model URL to clone the model repository.
    Model Dockerfile: Dockerfile for the model.
    Model Scripts: Model scripts to run the model.
    Model Args: Model arguments to run the model.

The script uses the following run details:
    Host name: Host name of the system.
    Host OS: Host OS of the system.
    System GPU architecture: GPU architecture in the system.
    System number of GPUs: Number of GPUs in the system.
    Model name: Model name to run the application.
    Model tags: Model tags for the model.
    Model arguments: Model arguments to run the model.
    Model Dockerfile: Dockerfile for the model.
    Model Docker image: Docker image for the model.
    Model base Docker: Base Docker for the model.
    Model Docker SHA: Docker SHA for the model.
    Build duration: Duration to build the Docker image.
    Test duration: Duration to run the model.
    Status: Status of the run (SUCCESS or FAILURE).
    Performance: Performance for the model.
    Metric: Metric of performance.

The script uses the following performance metrics:
    Performance: Performance for the model.
    Metric: Metric of performance.

The script uses the following Docker commands:
    docker build: Build the Docker image.
    docker run: Run the Docker container.
    docker exec: Execute the command in the Docker container.
    docker cp: Copy files and directories to the Docker container.
    docker rm: Remove the Docker container.
    docker rmi: Remove the Docker image.
"""
import argparse
import os
import sys
import time
import re
import json
import typing

from utils import get_dockerfile_gpu_suffix, get_dockerfile_linux_suffix, load_models
from utils import get_env_docker_args, get_mount_docker_args, get_cpu_docker_args, get_gpu_docker_args
from utils import get_system_gpus, get_system_gpu_arch, get_gpu_vendor, get_host_name, get_host_os
from utils import get_base_docker, get_base_docker_sha
from utils import get_perf_metric, update_dict
from utils import update_perf_csv
from utils import Console, Docker, Timeout, RunDetails
from version import __version__
from logger import get_logger


def get_args() -> argparse.Namespace:
    """Get input arguments from command line

    Input arguments:
        --tags: tags to run model.
        --timeout: Timeout for the application running model in seconds, default timeout of 14400 (4 hours).
        --live-output: Prints output in real-time directly on STDOUT.
        --clean-docker_cache: Rebuild docker image without using cache.
        --keep-alive: Keep the container alive after the application finishes running.
        --keep-model-dir: Keep the model directory after the application finishes running.
        -o, --output: Output file for the result.
        --log-level: Log level for the logger.

    Returns:
        argparse.Namespace: The input arguments
    """
    parser = argparse.ArgumentParser(description="Run the application of MAD, Model Automation and Dashboarding v" + __version__ + ".")
    parser.add_argument(
        "--tags", nargs='+', default=[], help="tags to run model (can be multiple)."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout for the application running model in seconds, default timeout of 14400 (4 hours).",
        default=14400,
    )
    parser.add_argument(
        "--live-output",
        action="store_true",
        help="Prints output in real-time directly on STDOUT.",
    )
    parser.add_argument(
        "--clean-docker-cache",
        action="store_true",
        help="Rebuild docker image without using cache.",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep the container alive after the application finishes running.",
    )
    parser.add_argument(
        "--keep-model-dir",
        action="store_true",
        help="Keep the model directory after the application finishes running.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output file for the result.",
        default="perf.csv",
    )
    parser.add_argument(
        "--log-level", type=str, help="Log level for the logger.", default="INFO"
    )

    args = parser.parse_args()
    return args

def run_model(
        model: typing.Dict[str, typing.Any],
        args: argparse.Namespace,
        console: Console
    ) -> bool:
    """Run the model application.
    
    Args:
        model_info (dict): The model information
        args (argparse.Namespace): The input arguments
        console (Console): The console object
    
    Returns:
        bool: The status of the run (return code: True for success, False for failure)
    """
    # Parse the input arguments
    model_name = model['name']
    timeout = args.timeout
    keep_alive = args.keep_alive
    keep_model_dir = args.keep_model_dir
    log_level = args.log_level
    output = args.output    

    log_file = f"logs/{model_name}.live.log"
    # Check the log file exist in the directory or not, if not then create the log file, if exist then empty the log file.
    if os.path.exists(log_file):
        open(log_file, "w").close()
    else:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = get_logger("MAD", log_level, log_file)

    # Initialize the status
    status = 'FAILURE'
    return_code = False

    # Log the input arguments
    logger.info(f"Model name: {model_name}")
    logger.info(f"Output file: {output}")

    # Get the Dockerfile suffixes
    dockerfile_gpu_suffix = get_dockerfile_gpu_suffix()
    dockerfile_linux_suffix = get_dockerfile_linux_suffix()

    # Initialize the run details
    run_details = RunDetails()

    # Store the data for the run details
    run_details.machine_name = get_host_name()
    run_details.host_os = get_host_os()
    run_details.gpu_architecture = get_system_gpu_arch()
    run_details.n_gpus = get_system_gpus()    
    run_details.pipeline = os.environ.get('pipeline')

    # Parse the model dictionary
    model_url = model["url"] if "url" in model else None
    model_dockerfile_prefix = model["dockerfile"]
    model_dockerfile = f"./{model_dockerfile_prefix}{dockerfile_linux_suffix}{dockerfile_gpu_suffix}.Dockerfile"
    model_scripts = model["scripts"]
    model_tags = model["tags"]
    model_args = model["args"]
    training_precision = model["training_precision"]

    # Log the model details
    logger.info(f"Model name: {model_name}")
    logger.info(f"Model URL: {model_url}")
    logger.info(f"Model Dockerfile: {model_dockerfile}")
    logger.info(f"Model Scripts: {model_scripts}")
    logger.info(f"Model Args: {model_args}")

    # Build the Docker image
    use_cache_str = ""
    if args.clean_docker_cache:
        use_cache_str = "--no-cache"

    build_args = ""
    docker_context = "./docker"
    model_docker_image = f"ci-{model_name}"
    model_docker_container = f"container_ci-{model_name}"

    # Store the data for the run details
    run_details.model = model_name
    run_details.tags = model_tags
    run_details.args = model_args
    run_details.docker_file = model_dockerfile
    run_details.docker_image = model_docker_image
    run_details.training_precision = training_precision

    # Build the Docker image
    build_start_time = time.time()
    console.sh(
        "docker build "
        + use_cache_str
        + " --network=host "
        + " -t "
        + model_docker_image
        + " --pull -f "
        + model_dockerfile
        + " "
        + build_args
        + " "
        + docker_context,
        timeout=None,
    )
    build_duration = time.time() - build_start_time
    logger.info(f"Build duration: {build_duration} seconds")

    base_docker = get_base_docker(model_dockerfile)
    logger.info(f"Base Docker: {base_docker}")
    base_docker_sha = get_base_docker_sha(model_docker_image)
    logger.info(f"Base Docker SHA: {base_docker_sha}")

    # Store the data for the run details
    run_details.base_docker = get_base_docker(model_dockerfile)
    run_details.docker_sha = get_base_docker_sha(model_docker_image)
    run_details.build_duration = build_duration

    # Run the Docker container
    docker_opts = ""
    # Check 'AMD' or 'NVIDIA' string in the dockerfile_gpu_suffix or not
    if re.search("amd", dockerfile_gpu_suffix):
        docker_opts = "--network host -u root --group-add video --cap-add=SYS_PTRACE --cap-add SYS_ADMIN --device /dev/fuse --security-opt seccomp=unconfined --security-opt apparmor=unconfined --ipc=host "
    elif re.search("nvidia", dockerfile_gpu_suffix):
        docker_opts = "--cap-add=SYS_PTRACE --cap-add SYS_ADMIN --cap-add SYS_NICE --device /dev/fuse --security-opt seccomp=unconfined --security-opt apparmor=unconfined  --network host -u root --ipc=host "
    else:
        raise ValueError("Unknown GPU type")

    # Add environment variables
    # docker_opts += "--env MAD_MODEL_NAME='" + model_name + "' "

    run_envs = {
        "MAD_MODEL_NAME": model_name,
        "MAD_GPU_VENDOR": get_gpu_vendor(),
        "MAD_SYSTEM_NGPUS": get_system_gpus(),
        "MAD_SYSTEM_GPU_ARCHITECTURE": get_system_gpu_arch()
    }
    mad_secrets = {}
    for key in os.environ:
        if "MAD_SECRETS" in key:
            mad_secrets[key] = os.environ[key]
    if mad_secrets:
        update_dict(run_envs, mad_secrets)
    docker_opts += get_env_docker_args(run_envs)

    docker_opts += get_gpu_docker_args()
    # docker_opts += get_cpu_docker_args()        

    mount_data_paths = []
    docker_opts += get_mount_docker_args(mount_data_paths)

    logger.debug(f"Running Docker container with options: {docker_opts}")

    # Run the Docker container
    with Timeout(seconds=timeout):
        # Run the Docker container
        logger.info(
            f"Running Docker container: {model_docker_container}, image: {model_docker_image}, timeout: {timeout} seconds"
        )

        docker = Docker(
            image=model_docker_image, 
            container_name=model_docker_container, 
            docker_opts=docker_opts,
            keep_alive=keep_alive,
            console=console
        )

        logger.info(f"User is {docker.sh('whoami')}")

        # Echo GPU information
        if re.search("nvidia", dockerfile_gpu_suffix):
            docker.sh('/usr/bin/nvidia-smi || true')
        elif re.search("amd", dockerfile_gpu_suffix):
            docker.sh('/opt/rocm/bin/rocm-smi || true')
        else:
            logger.error("No GPU information available")
            raise ValueError("Unknown GPU type")

        # Clean up the previous model
        model_dir = 'run_directory'
        if model_url:
            model_dir = model_url.split("/")[-1]

        # Clone the model repository
        if model_url:
            docker.sh(f"git clone {model_url}")
        else:
            docker.sh(f"mkdir -p {model_dir}")

        # Update the submodules
        docker.sh(f"git config --global --add safe.directory /myworkspace")
        docker.sh(f"git config --global --add safe.directory /myworkspace/{model_dir}")

        # echo git commit
        run_details.git_commit = docker.sh(f"cd {model_dir} && git rev-parse HEAD")
        logger.info(f"MODEL GIT COMMIT is {run_details.git_commit}")
        
        if model_url:
            docker.sh(f"cd {model_dir} && git submodule update --init --recursive")

        # Check the model directory
        docker.sh(f"ls -la /myworkspace/{model_dir}")

        # Prepare the model scripts
        model_scripts = model["scripts"]
        model_scripts_dir_path = None
        model_scripts_execute = None
        if model_scripts.endswith(".sh"):
            model_scripts_dir_path = os.path.dirname(model_scripts)
            model_scripts_execute = f"bash {os.path.basename(model_scripts)}"
        elif model_scripts.endswith(".py"):
            model_scripts_dir_path = os.path.dirname(model_scripts)
            model_scripts_execute = f"python3 {os.path.basename(model_scripts)}"
        else:
            model_scripts_dir_path = model_scripts
            model_scripts_execute = "bash run.sh"

        # Copy the model scripts to the model directory in the Docker container
        # The process is to recursively copy files and directories, following symbolic links, while preserving all file attributes and displaying detailed information about the copying process.
        docker.sh(
            f"cp -vLR --preserve=all {model_scripts_dir_path}/. {model_dir}/"
        )

        # Keep the model_dir as universally read-write
        docker.sh(f"chmod -R a+rw {model_dir}")

        # Test the model
        test_start_time = time.time()
        try:
            docker.sh(f"cd /myworkspace/{model_dir} && {model_scripts_execute} {model_args}", timeout=None)
            status = 'SUCCESS'
        except Exception as e:
            logger.error(f"Failed to run the model: {e}")
            status = 'FAILURE'
            run_details.status = status
            # Generate the report of exception
            run_details.generate_json("perf_entry.json")
            update_perf_csv(exception_result="perf_entry.json", perf_csv=output)
            # Clean up the instance of docker
            del docker
            sys.exit(1)
        
        test_duration = time.time() - test_start_time
        logger.info(f"Test duration: {test_duration} seconds")

        # Store the data for the run details
        run_details.test_duration = test_duration

        # Keep the container alive
        if not keep_alive and not keep_model_dir:
            # Delete the model directory and stop the container
            docker.sh(f"rm -rf {model_dir}")
            logger.info(f"Model directory: {model_dir} is deleted.")
        else:
            # Keep the model directory and stop the container
            docker.sh(f"chmod -R a+rw {model_dir}")
            logger.info(f"Keeping the container alive after the application finishes running.")
            logger.info(f"Model directory: {model_dir} is not deleted.")

    # Store the data for the run details
    run_details.status = status
    logger.info(f"Status: {status}")

    if status == 'SUCCESS':
        logger.info(f"Model {model_name} ran successfully at container {model_docker_container}.")
    else:
        logger.error(f"Model {model_name} failed to run at container {model_docker_container}.")

    # Write the run details to the output file
    try:
        if run_details.status == 'SUCCESS':
            # Check if we are looking for a single result or multiple.
            multiple_results = model['multiple_results'] if 'multiple_results' in model and model["multiple_results"] else None

            # Get performance metric from log
            if multiple_results:
                # Parse the performance metrics for multiple results case
                logger.info(f"Multiple results: {multiple_results}")
                run_details.performance = multiple_results
                run_details.generate_json("common_info.json", multiple_results=True)
                update_perf_csv(
                    multiple_results=model["multiple_results"], 
                    perf_csv=output, 
                    model_name=run_details.model, 
                    common_info="common_info.json"
                )
            else:
                # Parse the performance metrics for single result case
                try:
                    # Get the performance metrics
                    run_details.performance, run_details.metric = get_perf_metric(log_file)
                    # Log the performance metrics
                    run_details.print_perf_metric()
                    run_details.generate_json("perf_entry.json")
                    # Update the performance metrics to the CSV file
                    logger.info(f"Single result: {run_details.performance} {run_details.metric}")
                    update_perf_csv(single_result="perf_entry.json", perf_csv=output)
                except Exception as e:
                    logger.error(f"Failed to parse the performance metrics: {e}")
                    run_details.performance = None
                    run_details.metric = None
        else:
            run_details.generate_json("perf_entry.json")
            update_perf_csv(exception_result="perf_entry.json", perf_csv=output)
        
    except Exception as e:
        logger.error(f"Failed to write the run details to the output file: {e}")

    # Clean up the instance of docker
    del docker

    return_code = True if run_details.status == 'SUCCESS' else False    

    return return_code


def main() -> bool:
    """Main function to run the MAD application.
    
    Returns:
        bool: The status of the run (return code: True for success, False for failure)
        
    Raises:
        ValueError: If the GPU type is unknown
    """
    # Initialize the return status
    return_status = True

    # Get input arguments
    args = get_args()

    # Get the console
    console = Console(live_output=args.live_output)

    # Load models.json file to list of dictionary.
    models = load_models()

    user_tags = None
    if args.tags:
        user_tags = {'tags': args.tags }
    else:
        with open('tags.json') as t:
            user_tags = json.load(t)

    print("Selected TAGS are " + ', '.join(user_tags["tags"]) + '.' )

    for model_info in models:
        if [x for x in user_tags["tags"] if x in model_info["tags"] or x == model_info["name"] ]:
            print("Selected MODEL, " + model_info["name"] + " :", model_info)
            try:
                return_status &= run_model(model_info, args, console)
            except Exception as e:
                print(f"Failed to run the model: {e}")
                return_status = False

    return return_status


if __name__ == "__main__":
    sys.exit(main() == False)
