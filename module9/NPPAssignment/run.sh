#!/bin/sh
make
echo "Testing execution NVGraph with size 6, weight size 10, and boxfilter with width of 5"                                                                                                                                                                                      
./assignment.exe pageRankSize=6 pageRankWeightSize=10 boxfilterSize=5
echo "Testing execution NVGraph with size 60, weight size 100, and boxfilter with width of 50"                                                                                                                                                                                      
./assignment.exe pageRankSize=60 pageRankWeightSize=100 boxfilterSize=50