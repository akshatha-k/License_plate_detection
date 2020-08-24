#!/bin/bash

set -e

mkdir data/lp-detector -p


wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/lp-detector/wpod-net_update1.h5   -P /content/alpr-unconstrained/data/lp-detector/
wget -c -N www.inf.ufrgs.br/~smsilva/alpr-unconstrained/data/lp-detector/wpod-net_update1.json -P /content/alpr-unconstrained/data/lp-detector/
