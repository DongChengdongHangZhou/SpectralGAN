CUDA_VISIBLE_DEVICES=1 python val.py --dataroot ./datasets/cyclegan --name cycleganA --save_name cycleganA --phase test --no_dropout
# --dataroot: the directory of testing dataset. --name: the checkpoints directory. --save_name: save the generated spectrum to this directory.
