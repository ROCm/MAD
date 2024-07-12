# MAD, Model Automation and Dashboarding

## What is this repository for? 

MAD (Model Automation and Dashboarding), is a combination of AI/ML model zoo, automation for running the models on various GPU architectures, a mechanism of maintaining historical performance data, and generating dashboards for tracking. 

## DISCLAIMER

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard versionchanges, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated.AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2024 Advanced Micro Devices, Inc. All Rights Reserved.

## How to run model

Install requirements: 
```
pip3 install -r requirements.txt
```

With tools/run_models.py script, all models in models.json can be run locally on a docker host, to collect performance results.

```
usage: tools/run_models.py [-h] [--model_name MODEL_NAME] [--timeout TIMEOUT] [--live_output] [--clean_docker_cache] [--keep_alive] [--keep_model_dir] [-o OUTPUT] [--log_level LOG_LEVEL]

Run the application of MAD, Model Automation and Dashboarding v1.0.0.

options:
  -h, --help            show this help message and exit
  --model_name MODEL_NAME
                        Model name to run the application.
  --timeout TIMEOUT     Timeout for the application running model in seconds, default timeout of 7200 (2 hours).
  --live_output         Prints output in real-time directly on STDOUT.
  --clean_docker_cache  Rebuild docker image without using cache.
  --keep_alive          Keep the container alive after the application finishes running.
  --keep_model_dir      Keep the model directory after the application finishes running.
  -o OUTPUT, --output OUTPUT
                        Output file for the result.
  --log_level LOG_LEVEL
                        Log level for the logger.
```

run_models.py is the main MAD CLI(Command Line Interface) for running models locally. While the tool has many options, running a singular model is very easy. To run any model simply look for its name or tag in the models.json and the command is of the form:

For each model in models.json, the script

* builds docker images associated with each model. The images are named 'ci-$(model_name)', and are not removed after the script completes.
* starts the docker container, with name, 'container_$(model_name)'. The container should automatically be stopped and removed whenever the script exits.
* clones the git 'url', and runs the 'script'
* compiles the final perf.csv and perf.html

### Tag functionality

With the tag functionality, the user can select a subset of the models, that have the corresponding tags matching user specified tags, to be run. User specified tags can be specified in 'tags.json' or with the --tags argument. If multiple tags are specified, all models that match any tag is selected. Each model name in models.json is automatically a tag that can be used to run that model. Tags are also supported in comma-separated form

"python3 tools/run_models.py --tags TAG" so for example to run the pyt_huggingface_bert model use "python3 tools/run_models.py --tags pyt_huggingface_bert" or to run all pytorch models "python3 tools/run_models.py --tags pyt".

### Custom timeouts

The default timeout for model run is 2 hrs. This can be overridden if the model in models.json contains a 'timeout' : TIMEOUT entry. Both the default timeout and/or timeout specified in models.json can be overridden using --timeout TIMEOUT command line argument. Having TIMEOUT set to 0 means that the model run will never timeout.

### Debugging

Some of the more useful flags to be aware of are "--liveOutput" and "–keepAlive". "--liveOutput" will show all the logs as MAD is running, otherwise they are saved to log files on the current directory. "–keepAlive" will prevent MAD from stopping and removing the container when it is done, which can be very useful for manual debugging or experimentation. Note that when running with the "–keepAlive" flag the user is responsible for stopping and deleting that container. The same MAD model cannot run again until that container is cleared. 

For a more details on the tool please look at the "–help" flag 

## To add a model to the MAD repo

0. create workload name. The names of the modules should follow a specfic format. First it should be the framework(tf_, tf2_, pyt_, ort_, ...) , the name of the project and finally the workload. For example

    ```
    tf2_huggingface_gpt2
    ```
    Use this name in the models.json, as the dockerfile name and the scripts folder name.

1. add the necessary info to the models.json file. Here is a sample model info entry for bert
    ```json
        {
            "name": "tf2_bert_large",
            "url": "https://github.com/ROCmSoftwarePlatform/bert",
            "dockerfile": "docker/tf2_bert_large",
            "scripts": "scripts/tf2_bert_large",
            "n_gpus": "4",
            "owner": "john.doe@amd.com",
            "training_precision": "fp32",
            "tags": [
                "per_commit",
                "tf2",
                "bert",
                "fp32"
            ],
            "args": ""
        }
    ```
   | Field               | Description                                                                |
   |---------------------| ---------------------------------------------------------------------------|
   | name                | a unique model name                                                        |
   | url                 | model url to clone                                                         |
   | dockerfile          | initial search path dockerfile collection                                  |
   | scripts             | model script to execute in dockerfile under cloned model directory         |
   | data                | Optional field denoting data for script                                    |
   | n_gpus              | number of gpus exposed inside docker container. '-1' => all available gpus |
   | timeout             | model specific timeout, default of 2 hrs                                   |
   | owner               | email address for model owner                                              |
   | training\_precision | precision, currently used only for reporting                               |
   | tags                | list of tags for selecting model. The model name is a default tag.         |
   | multiple\_results   | optional parameter for multiple results, pointing to csv that holds results|
   | args                | extra arguments passed to model scripts                                    | 

2. create a dockerfile, or reuse an existing dockerfile in the docker directory. Here is an example below that should serve as a template. 
    ```docker
    # CONTEXT {'gpu_vendor': 'AMD', 'guest_os': 'UBUNTU'}
    FROM rocm/tensorflow

    # Install dependencies
    RUN apt update && apt install -y \
        unzip 
    RUN pip3 install pandas

    # Download data
    RUN URL=https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip && \
        wget --directory-prefix=/data -c $URL && \
        ZIP_NAME=$(basename $URL) && \
        unzip /data/$ZIP_NAME -d /data
    ```
3. create a directory in the scripts directory that contains everything necessary to do a run and report performance. The contents of this directory will be copied to the model root directory. Make sure the directory has a run script. If the script name is not explicitly specified, MAD assumes that the script name is 'run.sh'. Here is a sample run.sh script for bert.
    ```bash
        # setup model
        MODEL_CONFIG_DIR=/data/uncased_L-24_H-1024_A-16
        BATCH=2
        SEQ=512
        TRAIN_DIR=bert_large_ba${BATCH}_seq${SEQ}
        TRAIN_STEPS=100
        TRAIN_WARM_STEPS=10
        LEARNING_RATE=1e-4
        DATA_SOURCE_FILE_PATH=sample_text.txt
        DATA_TFRECORD=sample_text_seq${SEQ}.tfrecord
        MASKED_LM_PROB=0.15
        calc_max_pred() {
            echo $(python3 -c "import math; print(math.ceil($SEQ*$MASKED_LM_PROB))")
        }
        MAX_PREDICTION_PER_SEQ=$(calc_max_pred)

        python3 create_pretraining_data.py \
            --input_file=$DATA_SOURCE_FILE_PATH \
            --output_file=$DATA_TFRECORD \
            --vocab_file=$MODEL_CONFIG_DIR/vocab.txt \
            --do_lower_case=True \
            --max_seq_length=$SEQ \
            --max_predictions_per_seq=$MAX_PREDICTION_PER_SEQ \
            --masked_lm_prob=$MASKED_LM_PROB \
            --random_seed=12345 \
            --dupe_factor=5

        # train model
        python3 run_pretraining.py \
            --input_file=$DATA_TFRECORD \
            --output_dir=$TRAIN_DIR \
            --do_train=True \
            --do_eval=True \
            --bert_config_file=$MODEL_CONFIG_DIR/bert_config.json \
            --train_batch_size=$BATCH \
            --max_seq_length=$SEQ \
            --max_predictions_per_seq=$MAX_PREDICTION_PER_SEQ \
            --num_train_steps=$TRAIN_STEPS \
            --num_warmup_steps=$TRAIN_WARM_STEPS \
            --learning_rate=$LEARNING_RATE \
            2>&1 | tee log.txt

        # report performance metric
        python3 get_bert_model_metrics.py $TRAIN_DIR
    ```
    Note that there is a python script for reporting performance that was also included in the model script directory. For single result reporting scripts, make sure that you print performance in the following format, `performance: PERFORMANCE_NUMBER PERFORMANCE_METRIC`. For example, the performance reporting script for bert, get_bert_model_metrics.py, prints `performance: 3.0637370347976685 examples/sec`.

    For scripts that report multiple results, signal MAD to expect multiple results with 'multiple\_results' field in models.json. This points to a csv generated by the script. The csv should have 3 columns, `models,performance,metric`, with different rows for different results. 

4. For a particular model, multiple tags such as the precision, the framework, workload may be given.  For example, the "tf2_mlperf_resnet50v1.nchw" could have the "tf2" and "resnet50" tag.  If this workload also specified the precision then this would be a valid tag as well (e.g. "fp16" or "fp32").  Also, MAD considers each model name to be a default tag, that need not be explicitly specified.


## Special environment variables 

MAD uses special environment variables to provide additional functionality within MAD. These environment variables always have a MAD_ prefix. These variables are accessible within the model scripts. 

  | Variable                    | Description                          |
  |-----------------------------|--------------------------------------|
  | MAD_SYSTEM_GPU_ARCHITECTURE | GPU Architecture for the host system |
  | MAD_RUNTIME_NGPUS           | Number of GPU available to the model |

### Model environment variables

MAD also exposes model-environment variables to allow for model tuning at runtime. These environment variables always have a MAD_MODEL_ prefix. These variables are accessible within the model scripts and are set to default values if not specified at runtime. 

   | Field                       | Description                                                                       |
   |-----------------------------| ----------------------------------------------------------------------------------|
   | MAD_MODEL_NAME              | model's name in `models.json`                                                     |
   | MAD_MODEL_NUM_EPOCHS        | number of epochs                                                                  |
   | MAD_MODEL_BATCH_SIZE        | batch-size                                                                        |
