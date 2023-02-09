#!/bin/sh  
while true  
do  
  rm -f ${PWD}/sigbot/signal_stat/*.pkl
  sleep 1
  docker run -v ${PWD}/sigbot:/sigbot sigbot
  sleep 3600 * 24
  docker stop $(docker ps -q)
  sleep 60
done



