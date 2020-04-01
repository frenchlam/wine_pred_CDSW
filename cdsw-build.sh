#!/bin/bash

pip3 install -r requirements.txt

if [ -f "spark_rf.tar" ]
then
  tar -xf ./spark_rf.tar
fi 