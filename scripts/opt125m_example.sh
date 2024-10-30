
# run AWQ search (optional; we provided the pre-computed results)
srun -p octave -N 1 -N 1 -G 1 python -m awq.entry --model_path /home/dataset/opt-125m \
    --w_bit 4 --q_group_size 128 \
    --run_awq --dump_awq /home/dataset/awq-model-zoo/opt-125m-w4-g128.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
srun -p octave -N 1 -N 1 -G 1 python -m awq.entry --model_path /home/dataset/opt-125m \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_awq /home/dataset/awq-model-zoo/opt-125m-w4-g128.pt \
    --q_backend fake

# generate real quantized weights (w4)
srun -p octave -N 1 -N 1 -G 1 python -m awq.entry --model_path /home/dataset/opt-125m \
    --w_bit 4 --q_group_size 128 \
    --load_awq /home/dataset/awq-model-zoo/opt-125m-w4-g128.pt \
    --q_backend real --dump_quant /home/dataset/awq-model-zoo/opt-125m-w4-g128-awq.pt

# load and evaluate the real quantized model (smaller gpu memory usage)
srun -p octave -N 1 -N 1 -G 1 python -m awq.entry --model_path /home/dataset/opt-125m \
    --tasks wikitext \
    --w_bit 4 --q_group_size 128 \
    --load_quant /home/dataset/awq-model-zoo/opt-125m-w4-g128-awq-v2.pt