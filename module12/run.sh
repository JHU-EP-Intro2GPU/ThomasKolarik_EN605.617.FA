#!/bin/sh
make
echo "Testing execution with values 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16"
./assignment.exe --values 16 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
echo "Testing execution with values 16 25 18 62 101 256 825 32 76 15 199 10003 12 8504 746758 635272 78594873"
./assignment.exe --values 16 25 18 62 101 256 825 32 76 15 199 10003 12 8504 746758 635272 78594873