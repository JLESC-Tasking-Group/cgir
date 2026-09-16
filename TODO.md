# TODO list
1. Add fastmath to JIT
1. Add options to passes. Notably, for JIT to enable/disable features (O3, fastmath, etc.); or for recursively apply passes to command-graph nodes (a 'recurse' parameter)
1. Add other pattern detection; currently only having sequence (sequence of nodes) and batch (island on same device) -- maybe other pattern can be used by runtime systems.
1. Verify and implement what is missing for 2 and 3 dimensional copies.
1. Replace the 'JIT' cache with an explicit 'format' object, that the user passes, and that is filled by CGIR, to effectively cache compilation of the same program without the need to hash source and everything.
