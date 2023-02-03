
## Important Note

If you have already checked out the repo before 9/9 8PM, you need to rerun `git pull` to make sure everything is up to date.

To submit your code, you need to run `rai -p ./MP3 --submit MP3`. To check your submission history, run `rai history -p ./MP3` (last 20 enties) or `rai l-history -p ./MP3` (last 100 enties). 

## Objective

The purpose of this lab is for you to practice with using the CUDA API by implementing a tiled Matrix Multiply kernel and its associated host code as shown in the lectures.

## Prerequisites

Before starting this lab, make sure that:

* You have completed all week 2 lectures or videos

* You have completed Lab2 (MP2)

## Instruction

You should edit the code in `template.cu` to perform the following:

* Allocate device memory

* Copy host memory to device

* Initialize thread block and kernel grid dimensions

* Invoke CUDA kernel

* Copy results from device to host

* Free device memory

* Write the CUDA kernel

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

## WSL issue with run_datasets

Windows may convert newline character of our run_datasets script, which will cause execution failure (executing command $'\r') on the RAI server. To fix this, run the following commands to convert the script back to Unix:
```
sudo apt-get install dos2unix
dos2unix run_datasets
```
