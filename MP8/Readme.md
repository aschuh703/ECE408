## Important Note

If you have already checked out the repo before 11/10 8PM, you need to rerun `git pull` to make sure everything is up to date.

To submit your code, you need to run `rai -p ./MP8 --submit MP8`. To check your submission history, run `rai history -p ./MP8` (last 20 enties) or `rai l-history -p ./MP8` (last 100 enties). 


## Objective


The purpose of this lab is to implement a SpMV (Sparse Matrix Vector Multiplication) kernel for an input sparse matrix based
on the Jagged Diagonal Storage (JDS) transposed format. A diagram demonstrating the JDS transposed format is shown below:

![image](imgs/figure.png "thumbnail")

## Prerequisites

Before starting this lab, make sure that:

* You have completed all week 8 lecture videos
* You have completed MP-7

## Instructions

Edit the kernel and the host function in the file to implement sparse matrix-vector multiplication using the JDS format. The kernel shall
be launched so that each thread will generate one output Y element. The kernel should have each thread to use the appropriate elements of
the JDS data array, the JDS col index array, JDS row index array, and the JDS transposed col ptr array to generate one Y element.

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

