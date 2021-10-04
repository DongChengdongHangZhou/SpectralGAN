CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./datasets/fingerprint --name fingerprint --pool_size 50 --no_dropout
# --dataroot means the directory of training/testing datasets
# --name means the model, the log and the html files will be saved in ./checkpoints/name
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./datasets/progan --name progan --pool_size 50 --no_dropout
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot ./datasets/stylegan --name stylegan --pool_size 50 --no_dropout