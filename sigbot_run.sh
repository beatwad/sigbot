#!/bin/sh
while true
do
  docker run --rm -v ${PWD}/:/sigbot sigbot
  sleep 60
done
