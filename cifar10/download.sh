#!/bin/bash

if [ -z "$1" ]; then exit; fi

BN=`basename $1`
if [ ! -e $BN ]
then
    wget "$1" -O $BN
fi

