#!/bin/bash

pip3 install sklearn

if [ -f "spark_rf.tar" ]
then
  tar -xf ./spark_rf.tar
fi 