#!/bin/bash

SAVEDIR=/root/result/rfcn/result/icdar_2013_2017/save

for F in $(cd $SAVEDIR; ls model*.index) 
do 
    F=$(echo $F | awk -F '.' '{print $1}')
    echo evaluation $F
    rm -rf /tmp/res
    mkdir /tmp/res
    python3 eval.py -p=0 -i=/root/result/rfcn/data/ICDAR2013/test/image -n=$SAVEDIR/$F -o=/tmp/res
    pushd /tmp/res
    zip res.zip *.txt
    popd
    mv /tmp/res/res.zip eval/res20132017_$F.zip
done


