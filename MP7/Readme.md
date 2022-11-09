
## Objective


The purpose of this lab is to implement a SpMV (Sparse Matrix Vector Multiplication) kernel for an input sparse matrix based
on the Jagged Diagonal Storage (JDS) transposed format.

## Prerequisites

Before starting this lab, make sure that:

* You have completed all week 8 lecture videos
* You have completed MP-6

## Instructions

Edit the kernel and the host function in the file to implement sparse matrix-vector multiplication using the JDS format. The kernel shall
be launched so that each thread will generate one output Y element. The kernel should have each thread to use the appropriate elements of
the JDS data array, the JDS col index array, JDS row index array, and the JDS transposed col ptr array to generate one Y element.

Instructions about where to place each part of the code is demarcated by the `//@@` comment lines.

