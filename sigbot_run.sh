#!/bin/sh  
while true  
do  
  docker run -v ${PWD}/sigbot:/sigbot sigbot
  sleep 3600 * 24
  docker stop $(docker ps -q)
  sleep 60
done



