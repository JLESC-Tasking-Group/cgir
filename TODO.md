# TODO

Replace the 'JIT' cache with an explicit 'format' object, that the user passes,
and that is filled by CGIR, to effectively cache compilation of the same
program without the need to hash source and everything.

Writing the paper:
- MLIR is outdated, liekly not compiling anymore
- prog-fuse not always generates a PACKED version; it should.
- in CGIR implementation, there is actually more function arguments convention, but they could be simplified to the one presented in the paper
