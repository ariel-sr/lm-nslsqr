// Definition for general utility kernels

#ifndef _KERNEL_H
#define _KERNEL_H

// Modify a single value
__global__
void kernel_mod_value(double *dst, int idx, double val);
// Set an array with a fixed value
__global__
void kernel_set_value(double *dst, int M, double val);

#endif