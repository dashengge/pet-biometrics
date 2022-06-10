#train final
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file  configs/reproduce/resnext101_con.yml
CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file  configs/reproduce/resnext101.yml
python tools/average_checkpoints.py --log-dir logs_submit/resnext101_con
python tools/average_checkpoints.py --log-dir logs_submit/resnext101_1
python tools/test.py --config-file   logs_submit/resnext101_con/config.yaml   --submit-file ./resnext101_con.csv     MODEL.WEIGHTS  ./logs_submit/resnext101_con/avg_model.pth
python tools/test.py --config-file   logs_submit/resnext101_1/config.yaml   --submit-file ./resnext101_1.csv     MODEL.WEIGHTS  ./logs_submit/resnext101_1/avg_model.pth
# python tools/test_for_train.py --config-file   logs/resnext101_1_final/config.yaml   --submit-file ./submit1.csv   MODEL.WEIGHTS  ./logs/resnext101_1_final/model_final.pth 
# python tools/test_for_train.py --config-file   logs/resnext101_ms_final/config.yaml    --submit-file ./submit2.csv  MODEL.WEIGHTS ./logs/resnext101_ms_final/model_final.pth 

# ### training stage one
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101.yml
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101_MS.yml
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101_mean_tri.yml
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101_con.yml
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101_3.yml
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnext101_2.yml
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnet2.yml
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage1/resnet1.yml
# ### training stage two
# CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py --config-file configs/Stage2/resnext101.yml
# CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py --config-file configs/Stage2/resnext101_MS.yml
# CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py --config-file configs/Stage2/resnext101_mean_tri.yml
# CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py --config-file configs/Stage2/resnext101_con.yml
# CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py --config-file configs/Stage2/resnext101_3.yml
# CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py --config-file configs/Stage2/resnext101_2.yml
# CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py --config-file configs/Stage2/resnet2.yml
# CUDA_VISIBLE_DEVICES=0,1 python tools/train_net.py --config-file configs/Stage2/resnet1.yml
# ### training stage three
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage3/resnext101.yml
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage3/resnext101_MS.yml
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage3/resnext101_mean_tri.yml
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage3/resnext101_con.yml
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage3/resnext101_3.yml
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage3/resnext101_2.yml
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage3/resnet2.yml
# CUDA_VISIBLE_DEVICES=1 python tools/train_net.py --config-file configs/Stage3/resnet1.yml
# ### test for training checkpoints
# python tools/test.py --config-file   logs/resnext101_1_final/config.yaml   --submit-file ./submit1.csv   MODEL.WEIGHTS  ./logs/resnext101_1_final/model_final.pth 
# python tools/test.py --config-file   logs/resnext101_ms_final/config.yaml    --submit-file ./submit2.csv  MODEL.WEIGHTS ./logs/resnext101_ms_final/model_final.pth 
# python tools/test.py --config-file   logs/resnext101_con_final/config.yaml  --submit-file ./submit3.csv  MODEL.WEIGHTS   ./logs/resnext101_mean_tri_final/model_final.pth 
# python tools/test.py --config-file   logs/resnext101_2_final/config.yaml   --submit-file ./submit4.csv   MODEL.WEIGHTS  ./logs/resnext101_con_final/model_final.pth 
# python tools/test.py --config-file   logs/resnext101_2_final/config.yaml   --submit-file ./submit5.csv   MODEL.WEIGHTS  ./logs/resnext101_2_final/model_final.pth 
# python tools/test.py --config-file   logs/resnext101_3_final/config.yaml   --submit-file ./submit6.csv   MODEL.WEIGHTS  ./logs/resnext101_3_final/model_final.pth 
# python tools/test.py --config-file   logs/resnet101_2_final/config.yam   --submit-file ./submit7.csv     MODEL.WEIGHTS  ./logs/resnet101_2_final/model_final.pth 
# python tools/test.py --config-file   logs/resnet101_1_final/config.yam   --submit-file ./submit8.csv     MODEL.WEIGHTS  ./logs/resnet101_1_final/model_final.pth 
# python tools/ensemble.py