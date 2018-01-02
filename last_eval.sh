#!/bin/bash

TEST=$1
SAVEDIR=/mnt/rfcn/$TEST/save
RES_FILE=/tmp/${TEST}_result.txt
echo "SAVEDIR = $SAVEDIR"
echo "RES_FILE = $RES_FILE"

LAST=$( cat $SAVEDIR/checkpoint | head -n 1 | awk '{print $2}' | tr -d '"')
F=$(echo $LAST | awk -F '.' '{print $1}')
echo evaluation $F
rm -rf /tmp/res
mkdir /tmp/res
python3 eval.py -p=0 -i=/mnt/ICDAR2013/test/image -n=$F -o=/tmp/res
pushd /tmp/res
echo "zip /tmp/${TEST}_${F}.zip *.txt"
zip /tmp/${TEST}.zip *.txt
popd
pushd /Workspace/focused_scene_eval
EVAL=$( python script.py -g=gt.zip -s=/tmp/${TEST}.zip )
echo "$F $EVAL"
echo "$F $EVAL" >> $RES_FILE
popd
