#!/bin/bash
python3 eval.py -p=0 -i=/mnt/ICDAR2013/test/image -p=1 -n=/mnt/rfcn/test_icdar2013_test_no_aug/save/model-10000 -o=res
cd res
zip res.zip *.txt
cd ../eval
mv ../res/res.zip .
python script.py -g=gt.zip -s=res.zip


