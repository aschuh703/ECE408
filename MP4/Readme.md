
## Important Note

If you have already checked out the repo before 9/22 8PM, you need to rerun `git pull` to make sure everything is up to date.

To submit your code, you need to run `rai -p ./MP4 --submit MP4`. To check your submission history, run `rai history -p ./MP4` (last 20 enties) or `rai l-history -p ./MP4` (last 100 enties). 

## Objective
The purpose of this lab is to implement a 3D convolution using constant memory for the kernel and 3D shared memory tiling. 


## Prerequisite

Before starting this lab, make sure that:

* You have completed the "Tiled Matrix Multiplication" MP (MP3)

* You have completed all week 4 lectures or videos

## Instructions
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



