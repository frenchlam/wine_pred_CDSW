#!/bin/bash


if [ -d "models" ] 
then
  rm -rf models
fi

if [ -f "spark_rf.tar" ]
then
  tar -xf ./spark_rf.tar
fi 
