### training stage one
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101_MS.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101_mean_tri.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101_con.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101_3.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101_2.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnet2.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnet1.yml


CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage2/resnext101.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage2/resnext101_MS.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage2/resnext101_mean_tri.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage2/resnext101_con.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage2/resnext101_3.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage2/resnext101_2.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage2/resnet2.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage2/resnet1.yml