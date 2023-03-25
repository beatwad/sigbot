#!/bin/sh  
while true  
do  
  docker run -v ${PWD}/sigbot:/sigbot sigbot
  sleep 86400
  docker stop $(docker ps -q)
  sleep 60
done



