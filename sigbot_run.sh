#!/bin/sh  
while true  
do  
  docker run -v ${PWD}/sigbot:/sigbot sigbot
  sleep 60
  docker container prune --force
  sleep 60
done



