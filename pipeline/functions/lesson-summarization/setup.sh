#!/bin/bash

chmod +x src/eva_train.sh
chmod +x src/eva_test.sh
chmod +x src/eva_forecast.sh

wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip
export CLASSPATH=$(pwd)/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
echo "Please tokenize this text." | java edu.stanford.nlp.process.PTBTokenizer

