#!/bin/bash
python3 eval.py -p=0 -i=/mnt/ICDAR2013/test/image -n=/mnt/rfcn/test_icdar2013/save/model-30000 -o=res
cd res
zip res.zip *.txt
cd ../eval
mv ../res/res.zip .
python script.py -g=gt.zip -s=res.zip


