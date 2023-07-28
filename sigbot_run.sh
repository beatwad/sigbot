#!/bin/sh  
while true  
do  
  docker run -v ${PWD}/sigbot:/sigbot sigbot
  sleep 86400
  docker stop $(docker ps -q  --filter ancestor=sigbot)
  sleep 60
done



