python tools/test.py --config-file logs/resnet1/config.yaml --submit-file ./submit1.csv
python tools/test.py --config-file logs/resnet2/config.yaml --submit-file ./submit2.csv
python tools/test.py --config-file logs/resnext_cont1/config.yaml --submit-file ./submit3.csv
python tools/test.py --config-file logs/resnext_cont2/config.yaml --submit-file ./submit4.csv
python tools/test.py --config-file logs/resnext_cont3/config.yaml --submit-file ./submit5.csv
python tools/test.py --config-file logs/resnext1/config.yaml --submit-file ./submit6.csv
python tools/test.py --config-file logs/resnext2/config.yaml --submit-file ./submit7.csv
python tools/test.py --config-file logs/resnext10/config.yaml --submit-file ./submit8.csv
python tools/ensemble.py 
