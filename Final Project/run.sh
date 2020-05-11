#!/bin/sh
make
echo "Testing execution with Blank.pgm"                                                                                                                                                                                      
./assignment.exe 3 1 1 Blank.pgm
echo "Testing execution with Glider.pgm"                                                                                                                                                                                    
./assignment.exe 3 1 1 Glider.pgm
echo "Testing execution with 1920x1440.pgm"                                                                                                                                                                                    
./assignment.exe 3 1 1 1920x1440.pgm