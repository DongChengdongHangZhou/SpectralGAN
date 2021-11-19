# --dataroot means the directory of training/testing datasets
# --name means the model, the log and the html files will be saved in ./checkpoints/name
# if loss_type == powerloss, we train the model using the proposed powerloss to enforce the power distribution more natural
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./datasets/biggan --name bigganA --pool_size 50 --no_dropout --loss_type powerloss
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./datasets/biggan --name bigganB --pool_size 50 --no_dropout --loss_type spectralloss
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot ./datasets/biggan --name bigganC --pool_size 50 --no_dropout --loss_type noloss

CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./datasets/crn --name crnA --pool_size 50 --no_dropout --loss_type powerloss
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./datasets/crn --name crnB --pool_size 50 --no_dropout --loss_type spectralloss
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot ./datasets/crn --name crnC --pool_size 50 --no_dropout --loss_type noloss

CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./datasets/cyclegan --name cycleganA --pool_size 50 --no_dropout --loss_type powerloss
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./datasets/cyclegan --name cycleganB --pool_size 50 --no_dropout --loss_type spectralloss
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot ./datasets/cyclegan --name cycleganC --pool_size 50 --no_dropout --loss_type noloss

CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./datasets/deepfake+whichfaceisreal --name deepfake+whichfaceisrealA --pool_size 50 --no_dropout --loss_type powerloss
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./datasets/deepfake+whichfaceisreal --name deepfake+whichfaceisrealB --pool_size 50 --no_dropout --loss_type spectralloss
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot ./datasets/deepfake+whichfaceisreal --name deepfake+whichfaceisrealC --pool_size 50 --no_dropout --loss_type noloss

CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./datasets/fingerprint --name fingerprintA --pool_size 50 --no_dropout --loss_type powerloss
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./datasets/fingerprint --name fingerprintB --pool_size 50 --no_dropout --loss_type spectralloss
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot ./datasets/fingerprint --name fingerprintC --pool_size 50 --no_dropout --loss_type noloss

CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./datasets/gaugan --name gauganA --pool_size 50 --no_dropout --loss_type powerloss
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./datasets/gaugan --name gauganB --pool_size 50 --no_dropout --loss_type spectralloss
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot ./datasets/gaugan --name gauganC --pool_size 50 --no_dropout --loss_type noloss

CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./datasets/imle --name imleA --pool_size 50 --no_dropout --loss_type powerloss
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./datasets/imle --name imleB --pool_size 50 --no_dropout --loss_type spectralloss
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot ./datasets/imle --name imleC --pool_size 50 --no_dropout --loss_type noloss

CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./datasets/progan --name proganA --pool_size 50 --no_dropout --loss_type powerloss
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./datasets/progan --name proganB --pool_size 50 --no_dropout --loss_type spectralloss
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot ./datasets/progan --name proganC --pool_size 50 --no_dropout --loss_type noloss

CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./datasets/stargan --name starganA --pool_size 50 --no_dropout --loss_type powerloss
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./datasets/stargan --name starganB --pool_size 50 --no_dropout --loss_type spectralloss
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot ./datasets/stargan --name starganC --pool_size 50 --no_dropout --loss_type noloss

CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./datasets/stylegan --name styleganA --pool_size 50 --no_dropout --loss_type powerloss
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./datasets/stylegan --name styleganB --pool_size 50 --no_dropout --loss_type spectralloss
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot ./datasets/stylegan --name styleganC --pool_size 50 --no_dropout --loss_type noloss

CUDA_VISIBLE_DEVICES=0 python train.py --dataroot ./datasets/stylegan2 --name stylegan2A --pool_size 50 --no_dropout --loss_type powerloss
CUDA_VISIBLE_DEVICES=1 python train.py --dataroot ./datasets/stylegan2 --name stylegan2B --pool_size 50 --no_dropout --loss_type spectralloss
CUDA_VISIBLE_DEVICES=2 python train.py --dataroot ./datasets/stylegan2 --name stylegan2C --pool_size 50 --no_dropout --loss_type noloss
