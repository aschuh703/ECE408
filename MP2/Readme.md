
## Important Note

If you have already checked out the repo before 9/8 8PM, you need to rerun `git pull` to make sure everything is up to date.

To submit your code, you need to run `rai -p ./MP2 --submit MP2`. To check your submission history, run `rai history -p ./MP2` (last 20 enties) or `rai l-history -p ./MP2` (last 100 enties). 

## Objective

The purpose of this lab is to implement a basic dense matrix multiplication routine.

## Prerequisites

Before starting this lab, make sure that:

* You have completed the "Vector Addition" MP (MP1)

* You have completed all week 2 lectures or videos

## Instruction

Edit the code in `template.cu` to perform the following:

- allocate device memory
- copy host memory to device
- initialize thread block and kernel grid dimensions
- invoke CUDA kernel
- copy results from device to host
- deallocate device memory

Instructions about where to place each part of the code is
demarcated by the `//@@` comment lines.

You can test your code by running `rai -p ./MP2`. If your solution is 
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

