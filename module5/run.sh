#!/bin/sh
make
echo "Testing execution with 32 blocks of size 32"                                                                                                                                                                                      
./assignment.exe 32 32                                                                                                                                                                                                               
echo "Testing execution with 128 blocks of size 128"                                                                                                                                                                                    
./assignment.exe 128 128
echo "Testing execution with 1024 blocks of size 1024"                                                                                                                                                                                    
./assignment.exe 1024 1024