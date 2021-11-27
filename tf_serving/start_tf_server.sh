#!/bin/bash
# check resouce limits
CPUs=$(expr $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us) / $(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us))
if [ "$CPUs" == 0 ]; then
    CPUs=$(nproc)
fi
echo "Number of avaiable cpus:" $CPUs

# create tensorflow serving config
echo "max_batch_size { value: 1000 }
batch_timeout_micros { value: 0 }
max_enqueued_batches { value: 10 }
num_batch_threads { value: $CPUs }
pad_variable_length_inputs: true" > batch
cat batch
echo "model_config_list {}" > model_config
cat model_config

/usr/bin/tensorflow_model_server \
--port=8500 \
--model_config_file=model_config \
--enable_batching \
--batching_parameters_file=batch