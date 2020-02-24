#!/bin/sh
make
echo "Testing execution with 256 blocks of size 256"                                                                                                                                                                                      
./assignment.exe 256 256                                                                                                                                                                                                               
echo "Testing execution with 4096 blocks of size 4096"                                                                                                                                                                                    
./assignment.exe 4096 4096
echo "Testing execution with 65536 blocks of size 65536"                                                                                                                                                                                    
./assignment.exe 65536 65536