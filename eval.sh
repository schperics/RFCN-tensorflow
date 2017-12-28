#!/bin/bash

TEST=$1
SAVEDIR=/mnt/rfcn/$TEST/save
TMP_DIR=/mnt/tmp
RES_FILE=$TMP_DIR/${TEST}_result.txt
echo "SAVEDIR = $SAVEDIR"
echo "RES_FILE = $RES_FILE"
rm -rf $RES_FILE

for F in $(cd $SAVEDIR; ls model-*.index)
do
    F=$(echo $F | awk -F '.' '{print $1}')
    echo evaluation $F
    rm -rf $TMP_DIR/res
    mkdir $TMP_DIR/res
    python3 eval.py -p=0 -i=/mnt/ICDAR2013/test/image -n=$SAVEDIR/$F -o=$TMP_DIR/res
    pushd $TMP_DIR/res
    zip $TMP_DIR/$TEST_$F.zip *.txt
    popd
    pushd /Workspace/focused_scene_eval
    EVAL=$( python script.py -g=gt.zip -s=$TMP_DIR/$TEST_$F.zip )
    echo "$F $EVAL"
    echo "$F $EVAL" >> $RES_FILE
    popd
done
