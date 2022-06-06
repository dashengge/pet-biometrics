### training stage one
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101_MS.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101_mean_tri.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101_con.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101_3.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101_2.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnet2.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnet1.yml

### training stage two
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage2/resnext101.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage2/resnext101_MS.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage2/resnext101_mean_tri.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage2/resnext101_con.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage2/resnext101_3.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage2/resnext101_2.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage2/resnet2.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage2/resnet1.yml


### training stage three
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage3/resnext101.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage3/resnext101_MS.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage3/resnext101_mean_tri.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage3/resnext101_con.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage3/resnext101_3.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage3/resnext101_2.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage3/resnet2.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage3/resnet1.yml


### test for training checkpoints
python tools/test_for_train.py --config-file    logs/resnet101_1_final/config.yaml --MODEL.WEIGHTS  ./logs/resnet101_1_validation/model_final.pth  --submit-file ./submit1.csv
python tools/test_for_train.py --config-file    logs/resnet101_2_final/config.yaml --MODEL.WEIGHTS  ./logs/resnet101_2_validation/model_final.pth  --submit-file ./submit2.csv
python tools/test_for_train.py --config-file   logs/resnext101_2_final/config.yaml --MODEL.WEIGHTS  ./logs/resnext101_2_validation/model_final.pth  --submit-file ./submit3.csv
python tools/test_for_train.py --config-file   logs/resnext101_3_final/config.yaml --MODEL.WEIGHTS  ./logs/resnext101_3_validation/model_final.pth  --submit-file ./submit4.csv
python tools/test_for_train.py --config-file   logs/resnext101_2_final/config.yaml --MODEL.WEIGHTS  ./logs/resnext101_2_validation/model_final.pth  --submit-file ./submit5.csv
python tools/test_for_train.py --config-file logs/resnext101_con_final/config.yaml --MODEL.WEIGHTS  ./logs/resnext101_con_validation/model_final.pth  --submit-file ./submit6.csv
python tools/test_for_train.py --config-file  logs/resnext101_ms_final/config.yaml --MODEL.WEIGHTS  ./logs/resnext101_ms_validation/model_final.pth  --submit-file ./submit7.csv
python tools/test_for_train.py --config-file   logs/resnext101_1_final/config.yaml --MODEL.WEIGHTS  ./logs/resnext101_1_validation/model_final.pth  --submit-file ./submit8.csv
python tools/ensemble.py 