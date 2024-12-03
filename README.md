# CSE5242 Project 2
### Alexey Gaulke, Heng Cao, Cole Parker
To compile our code use the following command:  
`gcc -O3 -mavx2 -o db5242 db5242.c`  
  
To run the executable, use the following command:  
`./db5242 N X Y Z R`  
where  
N = the size of the query data (or outer table for band join),  
X = the size of the inner table,  
Y = the size of the results table for band join,  
Z = the bound for the band join,  
R = the number of times to repeat the binary searches.