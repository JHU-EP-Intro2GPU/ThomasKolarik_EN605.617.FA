#!/bin/sh
make
echo "Testing execution with seed 12345"
./assignment.exe 12345
echo "Testing execution with seed 98765"
./assignment.exe 98765