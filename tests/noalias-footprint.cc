/*
** Copyright 2024,2025 INRIA
**
** This software is governed by the CeCILL-C license under French law and
** abiding by the rules of distribution of free software. See the LICENCE file.
**/

/* The CGIR_NOALIAS=prove analysis must over-approximate every footprint it
 * bounds (a smaller footprint would let it prove a false noalias). The cases
 * and the checks live in the library (see cgir_noalias_self_test in
 * src/noalias-llvmir.cc), so that this test needs no LLVM of its own. */

# include <stdio.h>

extern "C" int cgir_noalias_self_test(void);

int
main(void)
{
    const int failures = cgir_noalias_self_test();
    if (failures)
        fprintf(stderr, "%d noalias footprint case(s) failed\n", failures);
    return failures ? 1 : 0;
}
