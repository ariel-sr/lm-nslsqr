// bintable.h
// This file only contains an array representing a bidimensional matrix.
// This matrix contains the mapping between and integer and its binary 
// representation up to 8 bits.
// The coefficient in the position (i, j) corresponds to the coefficient
// corresponding to the value 2^j of the number i.
// This table is used by our implementation of the quantization matrix
// approximation.

#ifndef _BINTABLE_HPP
#define _BINTABLE_HPP

// Binary table
static int bintable[256*8] = {
	 0, 0, 0, 0, 0, 0, 0, 0,
	 0, 0, 0, 0, 0, 0, 0, 1,
	 0, 0, 0, 0, 0, 0, 1, 0,
	 0, 0, 0, 0, 0, 0, 1, 1,
	 0, 0, 0, 0, 0, 1, 0, 0,
	 0, 0, 0, 0, 0, 1, 0, 1,
	 0, 0, 0, 0, 0, 1, 1, 0,
	 0, 0, 0, 0, 0, 1, 1, 1,
	 0, 0, 0, 0, 1, 0, 0, 0,
	 0, 0, 0, 0, 1, 0, 0, 1,
	 0, 0, 0, 0, 1, 0, 1, 0,
	 0, 0, 0, 0, 1, 0, 1, 1,
	 0, 0, 0, 0, 1, 1, 0, 0,
	 0, 0, 0, 0, 1, 1, 0, 1,
	 0, 0, 0, 0, 1, 1, 1, 0,
	 0, 0, 0, 0, 1, 1, 1, 1,
	 0, 0, 0, 1, 0, 0, 0, 0,
	 0, 0, 0, 1, 0, 0, 0, 1,
	 0, 0, 0, 1, 0, 0, 1, 0,
	 0, 0, 0, 1, 0, 0, 1, 1,
	 0, 0, 0, 1, 0, 1, 0, 0,
	 0, 0, 0, 1, 0, 1, 0, 1,
	 0, 0, 0, 1, 0, 1, 1, 0,
	 0, 0, 0, 1, 0, 1, 1, 1,
	 0, 0, 0, 1, 1, 0, 0, 0,
	 0, 0, 0, 1, 1, 0, 0, 1,
	 0, 0, 0, 1, 1, 0, 1, 0,
	 0, 0, 0, 1, 1, 0, 1, 1,
	 0, 0, 0, 1, 1, 1, 0, 0,
	 0, 0, 0, 1, 1, 1, 0, 1,
	 0, 0, 0, 1, 1, 1, 1, 0,
	 0, 0, 0, 1, 1, 1, 1, 1,
	 0, 0, 1, 0, 0, 0, 0, 0,
	 0, 0, 1, 0, 0, 0, 0, 1,
	 0, 0, 1, 0, 0, 0, 1, 0,
	 0, 0, 1, 0, 0, 0, 1, 1,
	 0, 0, 1, 0, 0, 1, 0, 0,
	 0, 0, 1, 0, 0, 1, 0, 1,
	 0, 0, 1, 0, 0, 1, 1, 0,
	 0, 0, 1, 0, 0, 1, 1, 1,
	 0, 0, 1, 0, 1, 0, 0, 0,
	 0, 0, 1, 0, 1, 0, 0, 1,
	 0, 0, 1, 0, 1, 0, 1, 0,
	 0, 0, 1, 0, 1, 0, 1, 1,
	 0, 0, 1, 0, 1, 1, 0, 0,
	 0, 0, 1, 0, 1, 1, 0, 1,
	 0, 0, 1, 0, 1, 1, 1, 0,
	 0, 0, 1, 0, 1, 1, 1, 1,
	 0, 0, 1, 1, 0, 0, 0, 0,
	 0, 0, 1, 1, 0, 0, 0, 1,
	 0, 0, 1, 1, 0, 0, 1, 0,
	 0, 0, 1, 1, 0, 0, 1, 1,
	 0, 0, 1, 1, 0, 1, 0, 0,
	 0, 0, 1, 1, 0, 1, 0, 1,
	 0, 0, 1, 1, 0, 1, 1, 0,
	 0, 0, 1, 1, 0, 1, 1, 1,
	 0, 0, 1, 1, 1, 0, 0, 0,
	 0, 0, 1, 1, 1, 0, 0, 1,
	 0, 0, 1, 1, 1, 0, 1, 0,
	 0, 0, 1, 1, 1, 0, 1, 1,
	 0, 0, 1, 1, 1, 1, 0, 0,
	 0, 0, 1, 1, 1, 1, 0, 1,
	 0, 0, 1, 1, 1, 1, 1, 0,
	 0, 0, 1, 1, 1, 1, 1, 1,
	 0, 1, 0, 0, 0, 0, 0, 0,
	 0, 1, 0, 0, 0, 0, 0, 1,
	 0, 1, 0, 0, 0, 0, 1, 0,
	 0, 1, 0, 0, 0, 0, 1, 1,
	 0, 1, 0, 0, 0, 1, 0, 0,
	 0, 1, 0, 0, 0, 1, 0, 1,
	 0, 1, 0, 0, 0, 1, 1, 0,
	 0, 1, 0, 0, 0, 1, 1, 1,
	 0, 1, 0, 0, 1, 0, 0, 0,
	 0, 1, 0, 0, 1, 0, 0, 1,
	 0, 1, 0, 0, 1, 0, 1, 0,
	 0, 1, 0, 0, 1, 0, 1, 1,
	 0, 1, 0, 0, 1, 1, 0, 0,
	 0, 1, 0, 0, 1, 1, 0, 1,
	 0, 1, 0, 0, 1, 1, 1, 0,
	 0, 1, 0, 0, 1, 1, 1, 1,
	 0, 1, 0, 1, 0, 0, 0, 0,
	 0, 1, 0, 1, 0, 0, 0, 1,
	 0, 1, 0, 1, 0, 0, 1, 0,
	 0, 1, 0, 1, 0, 0, 1, 1,
	 0, 1, 0, 1, 0, 1, 0, 0,
	 0, 1, 0, 1, 0, 1, 0, 1,
	 0, 1, 0, 1, 0, 1, 1, 0,
	 0, 1, 0, 1, 0, 1, 1, 1,
	 0, 1, 0, 1, 1, 0, 0, 0,
	 0, 1, 0, 1, 1, 0, 0, 1,
	 0, 1, 0, 1, 1, 0, 1, 0,
	 0, 1, 0, 1, 1, 0, 1, 1,
	 0, 1, 0, 1, 1, 1, 0, 0,
	 0, 1, 0, 1, 1, 1, 0, 1,
	 0, 1, 0, 1, 1, 1, 1, 0,
	 0, 1, 0, 1, 1, 1, 1, 1,
	 0, 1, 1, 0, 0, 0, 0, 0,
	 0, 1, 1, 0, 0, 0, 0, 1,
	 0, 1, 1, 0, 0, 0, 1, 0,
	 0, 1, 1, 0, 0, 0, 1, 1,
	 0, 1, 1, 0, 0, 1, 0, 0,
	 0, 1, 1, 0, 0, 1, 0, 1,
	 0, 1, 1, 0, 0, 1, 1, 0,
	 0, 1, 1, 0, 0, 1, 1, 1,
	 0, 1, 1, 0, 1, 0, 0, 0,
	 0, 1, 1, 0, 1, 0, 0, 1,
	 0, 1, 1, 0, 1, 0, 1, 0,
	 0, 1, 1, 0, 1, 0, 1, 1,
	 0, 1, 1, 0, 1, 1, 0, 0,
	 0, 1, 1, 0, 1, 1, 0, 1,
	 0, 1, 1, 0, 1, 1, 1, 0,
	 0, 1, 1, 0, 1, 1, 1, 1,
	 0, 1, 1, 1, 0, 0, 0, 0,
	 0, 1, 1, 1, 0, 0, 0, 1,
	 0, 1, 1, 1, 0, 0, 1, 0,
	 0, 1, 1, 1, 0, 0, 1, 1,
	 0, 1, 1, 1, 0, 1, 0, 0,
	 0, 1, 1, 1, 0, 1, 0, 1,
	 0, 1, 1, 1, 0, 1, 1, 0,
	 0, 1, 1, 1, 0, 1, 1, 1,
	 0, 1, 1, 1, 1, 0, 0, 0,
	 0, 1, 1, 1, 1, 0, 0, 1,
	 0, 1, 1, 1, 1, 0, 1, 0,
	 0, 1, 1, 1, 1, 0, 1, 1,
	 0, 1, 1, 1, 1, 1, 0, 0,
	 0, 1, 1, 1, 1, 1, 0, 1,
	 0, 1, 1, 1, 1, 1, 1, 0,
	 0, 1, 1, 1, 1, 1, 1, 1,
	 1, 0, 0, 0, 0, 0, 0, 0,
	 1, 0, 0, 0, 0, 0, 0, 1,
	 1, 0, 0, 0, 0, 0, 1, 0,
	 1, 0, 0, 0, 0, 0, 1, 1,
	 1, 0, 0, 0, 0, 1, 0, 0,
	 1, 0, 0, 0, 0, 1, 0, 1,
	 1, 0, 0, 0, 0, 1, 1, 0,
	 1, 0, 0, 0, 0, 1, 1, 1,
	 1, 0, 0, 0, 1, 0, 0, 0,
	 1, 0, 0, 0, 1, 0, 0, 1,
	 1, 0, 0, 0, 1, 0, 1, 0,
	 1, 0, 0, 0, 1, 0, 1, 1,
	 1, 0, 0, 0, 1, 1, 0, 0,
	 1, 0, 0, 0, 1, 1, 0, 1,
	 1, 0, 0, 0, 1, 1, 1, 0,
	 1, 0, 0, 0, 1, 1, 1, 1,
	 1, 0, 0, 1, 0, 0, 0, 0,
	 1, 0, 0, 1, 0, 0, 0, 1,
	 1, 0, 0, 1, 0, 0, 1, 0,
	 1, 0, 0, 1, 0, 0, 1, 1,
	 1, 0, 0, 1, 0, 1, 0, 0,
	 1, 0, 0, 1, 0, 1, 0, 1,
	 1, 0, 0, 1, 0, 1, 1, 0,
	 1, 0, 0, 1, 0, 1, 1, 1,
	 1, 0, 0, 1, 1, 0, 0, 0,
	 1, 0, 0, 1, 1, 0, 0, 1,
	 1, 0, 0, 1, 1, 0, 1, 0,
	 1, 0, 0, 1, 1, 0, 1, 1,
	 1, 0, 0, 1, 1, 1, 0, 0,
	 1, 0, 0, 1, 1, 1, 0, 1,
	 1, 0, 0, 1, 1, 1, 1, 0,
	 1, 0, 0, 1, 1, 1, 1, 1,
	 1, 0, 1, 0, 0, 0, 0, 0,
	 1, 0, 1, 0, 0, 0, 0, 1,
	 1, 0, 1, 0, 0, 0, 1, 0,
	 1, 0, 1, 0, 0, 0, 1, 1,
	 1, 0, 1, 0, 0, 1, 0, 0,
	 1, 0, 1, 0, 0, 1, 0, 1,
	 1, 0, 1, 0, 0, 1, 1, 0,
	 1, 0, 1, 0, 0, 1, 1, 1,
	 1, 0, 1, 0, 1, 0, 0, 0,
	 1, 0, 1, 0, 1, 0, 0, 1,
	 1, 0, 1, 0, 1, 0, 1, 0,
	 1, 0, 1, 0, 1, 0, 1, 1,
	 1, 0, 1, 0, 1, 1, 0, 0,
	 1, 0, 1, 0, 1, 1, 0, 1,
	 1, 0, 1, 0, 1, 1, 1, 0,
	 1, 0, 1, 0, 1, 1, 1, 1,
	 1, 0, 1, 1, 0, 0, 0, 0,
	 1, 0, 1, 1, 0, 0, 0, 1,
	 1, 0, 1, 1, 0, 0, 1, 0,
	 1, 0, 1, 1, 0, 0, 1, 1,
	 1, 0, 1, 1, 0, 1, 0, 0,
	 1, 0, 1, 1, 0, 1, 0, 1,
	 1, 0, 1, 1, 0, 1, 1, 0,
	 1, 0, 1, 1, 0, 1, 1, 1,
	 1, 0, 1, 1, 1, 0, 0, 0,
	 1, 0, 1, 1, 1, 0, 0, 1,
	 1, 0, 1, 1, 1, 0, 1, 0,
	 1, 0, 1, 1, 1, 0, 1, 1,
	 1, 0, 1, 1, 1, 1, 0, 0,
	 1, 0, 1, 1, 1, 1, 0, 1,
	 1, 0, 1, 1, 1, 1, 1, 0,
	 1, 0, 1, 1, 1, 1, 1, 1,
	 1, 1, 0, 0, 0, 0, 0, 0,
	 1, 1, 0, 0, 0, 0, 0, 1,
	 1, 1, 0, 0, 0, 0, 1, 0,
	 1, 1, 0, 0, 0, 0, 1, 1,
	 1, 1, 0, 0, 0, 1, 0, 0,
	 1, 1, 0, 0, 0, 1, 0, 1,
	 1, 1, 0, 0, 0, 1, 1, 0,
	 1, 1, 0, 0, 0, 1, 1, 1,
	 1, 1, 0, 0, 1, 0, 0, 0,
	 1, 1, 0, 0, 1, 0, 0, 1,
	 1, 1, 0, 0, 1, 0, 1, 0,
	 1, 1, 0, 0, 1, 0, 1, 1,
	 1, 1, 0, 0, 1, 1, 0, 0,
	 1, 1, 0, 0, 1, 1, 0, 1,
	 1, 1, 0, 0, 1, 1, 1, 0,
	 1, 1, 0, 0, 1, 1, 1, 1,
	 1, 1, 0, 1, 0, 0, 0, 0,
	 1, 1, 0, 1, 0, 0, 0, 1,
	 1, 1, 0, 1, 0, 0, 1, 0,
	 1, 1, 0, 1, 0, 0, 1, 1,
	 1, 1, 0, 1, 0, 1, 0, 0,
	 1, 1, 0, 1, 0, 1, 0, 1,
	 1, 1, 0, 1, 0, 1, 1, 0,
	 1, 1, 0, 1, 0, 1, 1, 1,
	 1, 1, 0, 1, 1, 0, 0, 0,
	 1, 1, 0, 1, 1, 0, 0, 1,
	 1, 1, 0, 1, 1, 0, 1, 0,
	 1, 1, 0, 1, 1, 0, 1, 1,
	 1, 1, 0, 1, 1, 1, 0, 0,
	 1, 1, 0, 1, 1, 1, 0, 1,
	 1, 1, 0, 1, 1, 1, 1, 0,
	 1, 1, 0, 1, 1, 1, 1, 1,
	 1, 1, 1, 0, 0, 0, 0, 0,
	 1, 1, 1, 0, 0, 0, 0, 1,
	 1, 1, 1, 0, 0, 0, 1, 0,
	 1, 1, 1, 0, 0, 0, 1, 1,
	 1, 1, 1, 0, 0, 1, 0, 0,
	 1, 1, 1, 0, 0, 1, 0, 1,
	 1, 1, 1, 0, 0, 1, 1, 0,
	 1, 1, 1, 0, 0, 1, 1, 1,
	 1, 1, 1, 0, 1, 0, 0, 0,
	 1, 1, 1, 0, 1, 0, 0, 1,
	 1, 1, 1, 0, 1, 0, 1, 0,
	 1, 1, 1, 0, 1, 0, 1, 1,
	 1, 1, 1, 0, 1, 1, 0, 0,
	 1, 1, 1, 0, 1, 1, 0, 1,
	 1, 1, 1, 0, 1, 1, 1, 0,
	 1, 1, 1, 0, 1, 1, 1, 1,
	 1, 1, 1, 1, 0, 0, 0, 0,
	 1, 1, 1, 1, 0, 0, 0, 1,
	 1, 1, 1, 1, 0, 0, 1, 0,
	 1, 1, 1, 1, 0, 0, 1, 1,
	 1, 1, 1, 1, 0, 1, 0, 0,
	 1, 1, 1, 1, 0, 1, 0, 1,
	 1, 1, 1, 1, 0, 1, 1, 0,
	 1, 1, 1, 1, 0, 1, 1, 1,
	 1, 1, 1, 1, 1, 0, 0, 0,
	 1, 1, 1, 1, 1, 0, 0, 1,
	 1, 1, 1, 1, 1, 0, 1, 0,
	 1, 1, 1, 1, 1, 0, 1, 1,
	 1, 1, 1, 1, 1, 1, 0, 0,
	 1, 1, 1, 1, 1, 1, 0, 1,
	 1, 1, 1, 1, 1, 1, 1, 0,
	 1, 1, 1, 1, 1, 1, 1, 1
};

#endif