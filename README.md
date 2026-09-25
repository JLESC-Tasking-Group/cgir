[![docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://JLESC-Tasking-Group.github.io/cgir/)

# CGIR - Open Command Graphs

CGIR is a library that defines **commands graph** (**CG**) - a **Vendor-agnostic**, **Multi-devices** and **General** Intermediate Representation (IR) for programming [**Command Processor**](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/command-processor.html
) available on modern GPUs. CGIR provides **Optimization passes** of its IR.

In a command graph
- Nodes hold a **device unique identifier** and are either
  - a **command** (e.g., kernel launch, H2D copy, etc.)
  - a **command graph** (i.e., the structure is recursive)
  - a **condition** (demuxer or loop, to conditionally execute associated commands)
- Edges represent **precedence constraints** for execution

A few examples of **commands**:
- 1-dimensional copy: copy(src, dst, ptr, size)
- 2-dimensional copy(m, n, src, src_ld, dst, dst_ld)
- etc. See [include/cgir/command.hpp](include/cgir/command.hpp) for available commands  

A few examples of **optimization passes**:
- Transitive reduction, to minimize the graph complexity
- Batching, typically to minimize overheads by mapping to a vendor-specific batching structure (CUDA/HIP graphs, Level Zero command queues, etc.)
- etc. See [include/cgir/command-graph-pass.hpp](include/cgir/command-graph-pass.hpp) for available passes

# Installation
Using simple `cmake`
```
mkdir build-debug
cd build-debug
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=/path/to/install ..
make
```

# Runtime Configuration
You may set the following environment variables
- `CGIR_OPTIMIZER=[mlir|pod]` to switch the optimizer representation.
- `CGIR_JIT_DUMP=[0|1]` to dump the LLVM IR before and after JIT passes.
- `CGIR_PROG_FUSE_DUMP=[0|1]` to dump the LLVM IR of each prog-fusion for debugging, to `~/.cgir/`
- `CGIR_OPTIMIZE_DUMP=[0|1]` to dump the CGIR graph as Graphviz `.dot` before and after every optimization pass.
- `CGIR_STATS_CSV=[/path/to/csv/file]` to append stats of individual optimization passes to the given CSV file.
- `CGIR_STATS_TAG` an arbitrary string written verbatim in the `tag` column of every `CGIR_STATS_CSV` row.

JIT compilation caching and profiling (the `jit` pass compiles each program's
LLVM IR to a host function or device PTX):
- `CGIR_JIT_HOST_CODE_MODEL=[small|large]` selects the LLVM's code model for host programs: `small`(default) or `large`.
- `CGIR_JIT_DEVICE_NOALIAS=[0|1]` set to make the `jit` pass assume that pointer parameters of a device kernel do not overlap (default: 1).
- `CGIR_JIT_DEVICE_MINCTASM=[0|1]` control either the device JIT declares recorded occupancy (`command_prog_t::blocks_per_sm`) to the PTX assembler as `.minnctapersm` (default: 1).
- `CGIR_JIT_DEVICE_LTO=[0|1]` set to `0` makes the device JIT run a single per-module O3 after linking the DeviceRTL, instead of the two-phase pipeline clang use   under `-foffload-lto` (pre-link O3, link, post-link `lto<O3>`).
- `CGIR_JIT_AOT_DEVICE=[0|1]` set to `0` makes the `jit` pass leave alone any device program that still carries its ahead-of-time compiled kernel, i.e. recompile only what `prog-fuse` synthesized (default: 1)
- `CGIR_JIT_CACHE=[0|1]` caches JIT results (default: 1)
- `CGIR_JIT_CACHE_MODE=[r|w|rw]` restricts what the on-disk cache may do: `rw` (default), `w` write-only, `r` read-only. It can be used to enforce re-populatation of the cache.
- `CGIR_JIT_CACHE_DIR=[/path/to/cache/dir]` enables the persistent on-disk cache.
- `CGIR_JIT_TIMING=[0|1]` set to print, at process exit, a per-phase wall-clock breakdown of JIT compilation accumulated across all compiles.
- `CGIR_JIT_CACHE_STATS=[0|1]` set to to print, at process exit, statistics about caching (hit, etc.)
- `CGIR_JIT_STATS_CSV=[0|1]` set to a file path to append, at process exit, the full JIT breakdown (mirrors `CGIR_STATS_CSV`)

# Example
CGIR is integrated into the [XKRT](https://github.com/anlsys/xkrt) runtime system.
It serves as its abstraction for representing commands, notably to record and replay.
The [XKOMP](https://github.com/anlsys/xkomp) support for the `taskgraph` construct eventually fallbacks to CGIR.
See the glue here:
- The [graph instanciation](https://github.com/anlsys/xkrt/blob/master/src/command/command-graph.cc#L97-L101) for a task dependence graph IR.
- The [replay executionner](https://github.com/anlsys/xkrt/blob/master/src/command/command-graph.cc#L418)
- The [CUDA driver](https://github.com/anlsys/xkrt/blob/master/src/driver/driver_cu.cc#L618) to execute command graphs
