#!/bin/sh
while true
do
  docker run -v ${PWD}/sigbot:/sigbot sigbot
  sleep 60
  docker rm $(docker ps -q --filter ancestor=sigbot --filter status=exited)
  sleep 60
done
