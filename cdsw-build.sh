#!/bin/bash


if [ -d "models" ] 
then
  rm -rf models
fi

tar -xf ./spark_rf.tar