export PYTHONPATH=.
python tools/svdp.py --ema_rate=0.999 --model_lr=0.0001 --prompt_lr=0.0001 --prompt_sparse_rate=0.001 --scale=0.1 
python tools/svdp.py --ema_rate=0.999 --model_lr=0.0003 --prompt_lr=0.01 --prompt_sparse_rate=0.0001 --scale=0.01

wait



