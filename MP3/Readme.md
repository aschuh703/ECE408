
## Important Note

If you have already checked out the repo before 9/15 8PM, you need to rerun `git pull` to make sure everything is up to date.

To submit your code, you need to run `rai -p ./MP3 --submit MP3`. To check your submission history, run `rai history -p ./MP3` (last 20 enties) or `rai l-history -p ./MP3` (last 100 enties). 

## Objective

The purpose of this lab is to implement a tiled dense matrix multiplication routine using shared memory.

## Prerequisites

Before starting this lab, make sure that:

* You have completed the "Basic Matrix Multiplication" MP (MP2)

* You have completed all week 3 lectures or videos

## Instruction

Edit the code in `template.cu` to perform the following:

- allocate device memory
- copy host memory to device
- initialize thread block and kernel grid dimensions
- invoke CUDA kernel
- copy results from device to host
- deallocate device memory
- implement the matrix-matrix multiplication routine using shared memory and tiling

Instructions about where to place each part of the code is
demarcated by the `//@@` comment lines.

You can test your code by running `rai -p ./MP3`. If your solution is 
correct, you should be able to see the following output for each of 
the 10 test datasets:
```
--------------
Dataset  X
The dimensions of A are X x X
The dimensions of B are X x X
...
Solution is correct
```