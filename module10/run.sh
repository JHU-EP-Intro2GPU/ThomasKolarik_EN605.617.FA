#!/bin/sh
make
echo "Testing execution size 1000"
./assignment.exe 1000
echo "Testing execution size 10000"
./assignment.exe 10000
echo "Testing execution size 100000"
./assignment.exe 100000