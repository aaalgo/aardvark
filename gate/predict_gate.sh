#!/bin/bash

./predict_gate.py \
                --model ./model/80 \
                --image /shared/s2/users/wdong/football/testthumbs 
                $* 
