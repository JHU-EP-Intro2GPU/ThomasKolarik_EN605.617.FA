#!/bin/sh
make
echo "Testing execution with 32 blocks of size 32"                                                                                                                                                                                      
./assignment.exe 32 32                                                                                                                                                                                                               
echo "Testing execution with 1024 blocks of size 1024"                                                                                                                                                                                    
./assignment.exe 1024 1024
echo "Testing execution with 8192 blocks of size 8192"                                                                                                                                                                                    
./assignment.exe 8192 8192