#!/bin/sh
make
echo "Testing execution with Blank.pgm"                                                                                                                                                                                      
./GameOfLife.exe 3 1 1 Blank.pgm
echo "Testing execution with Glider.pgm"                                                                                                                                                                                    
./GameOfLife.exe 3 1 1 Glider.pgm
echo "Testing execution with 1920x1440.pgm"                                                                                                                                                                                    
./GameOfLife.exe 3 1 1 1920x1440.pgm