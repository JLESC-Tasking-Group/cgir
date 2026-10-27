# TODO list
1. Add optimization levels (ala -O0, -O1, -O2, -O3) and flags (ala --fast-maths)
1. "noalias" is currently achieved via simple pointer deduplication. This is wrong. Instead, we should do a dataflow analysis. Romain suggests having it only enabled at the highest level of optimization, because this analysis is costly.
1. Add options to passes. Notably, for JIT to enable/disable features (O3, fastmath, etc.); or for recursively apply passes to command-graph nodes (a 'recurse' parameter)
1. Add other pattern detection; currently only having sequence (sequence of nodes) and batch (island on same device) -- maybe other pattern can be used by runtime systems.
1. Verify and implement what is missing for 2 and 3 dimensional copies.
1. Replace the 'JIT' cache with an explicit 'format' object, that the user passes, and that is filled by CGIR, to effectively cache compilation of the same program without the need to hash source and everything.
1. Add a cache for the fusion pass.
