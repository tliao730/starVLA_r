#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=/usr/local/envs/starvla/bin/python
# your_ckpt=results/Checkpoints/Qwen2.5-VL-FAST-LIBERO-4in1/checkpoints/steps_30000_pytorch_model.pt
your_ckpt=results/Checkpoints/finetune_task2/final_model/pytorch_model.pt
gpu_id=0
port=5694
################# star Policy Server ######################

# export DEBUG=true
# CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
#     --ckpt_path ${your_ckpt} \
#     --port ${port} \
#     --use_bf16

# recode the server logs, we use nohup to run the server in the background.
nohup env CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 > server.log 2>&1 &

# #################################
