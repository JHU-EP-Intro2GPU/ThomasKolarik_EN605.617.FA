#!/bin/sh
make
echo "Testing execution with Add, and Subtract kernals"
./assignment.exe add sub
echo "Testing execution with Multiply, and Divide kernals"
./assignment.exe mult div
echo "Testing execution with Modulo, and Average kernals"
./assignment.exe mod avg
echo "Testing execution with Add, Subtract, Multiply, and Divide kernals"
./assignment.exe add sub mult div
echo "Testing execution with Add, Subtract, Multiply, Divide, Modulo, and Average kernals"
./assignment.exe add sub mult div mod avg