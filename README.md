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
- `CGIR_NOALIAS=[prove|distinct-base|trusted|none]` what the `prog-fuse` and `jit` passes may tell LLVM about aliasing (default: `prove`, or `trusted` with a warning when caching is explicitly requested, i.e. `CGIR_JIT_CACHE=1` or `CGIR_JIT_CACHE_DIR` set). `prove` marks a pair of pointers non-aliasing only when an analysis of the final code proves their accessed byte ranges disjoint for the actual argument values (direct accesses only; sound). `distinct-base` assumes pointers with different runtime values never overlap, so pointer deduplication is sufficient (an assumption about the program). `trusted` assumes it everywhere, without any footprint analysis: also in the grid-barrier decisions of fused device kernels, which then only order accesses through the same pointer value. Every decision then depends on which pointers are equal, not on their values, so `prog-fuse` results are reused across instances and runs (a stronger assumption: wrong for a kernel given two views of one buffer, such as `A` and `A+n`). `none` marks nothing.
- `CGIR_NOALIAS_VERIFY=[0|1|abort]` check every `distinct-base` assumption with the `prove` analysis and report each as proven or not proven (unbounded footprint, or bounded footprints that overlap: a possible violation), with a summary at exit (`abort` stops at the first possible violation). Under `prove`, reports how many candidates were proven. Under `trusted`, the analyses then see the real pointer values and the `prog-fuse` cache is not used.
- `CGIR_JIT_DEVICE_MINCTASM=[0|1]` control either the device JIT declares recorded occupancy (`command_prog_t::blocks_per_sm`) to the PTX assembler as `.minnctapersm` (default: 1).
- `CGIR_JIT_DEVICE_LTO=[0|1]` set to `0` makes the device JIT run a single per-module O3 after linking the DeviceRTL, instead of the two-phase pipeline clang use   under `-foffload-lto` (pre-link O3, link, post-link `lto<O3>`).
- `CGIR_JIT_VARIANT=[auto|jit|aot]` what the `jit` pass does with a program that already has an ahead-of-time implementation (default: `auto`). `jit` replaces it, `aot` keeps it (only programs without one, e.g. fused ones, are compiled), `auto` compiles them while keeping the ahead-of-time implementation (device kernel or host task routine), and the runtime measures both and launches the faster (see XKRT `XKRT_PROG_VARIANT_SAMPLES`).
- `CGIR_PROG_FUSE_GRID_BARRIER=[0|1]` whether `prog-fuse` may fuse device programs that need a grid-wide barrier between them (default: 1). Such a kernel is a cooperative launch at its constituents' lowest occupancy; `0` fuses only when the barrier is proven unnecessary.
- `CGIR_JIT_CACHE=[0|1]` caches JIT and `prog-fuse` results (default: 1). A `prog-fuse` result (the fused program, or the refusal to fuse) is keyed by the programs' code, launch, deduplication and argument values; under `CGIR_NOALIAS=trusted` the pointer values enter only as which ones are equal, so it is reused across instances and, with `CGIR_JIT_CACHE_DIR`, across runs. Under the other modes it is reused only for identical argument values.
- `CGIR_JIT_CACHE_MODE=[r|w|rw]` restricts what the on-disk cache may do: `rw` (default), `w` write-only, `r` read-only. It can be used to enforce re-populatation of the cache.
- `CGIR_JIT_CACHE_DIR=[/path/to/cache/dir]` enables the persistent on-disk cache.
- `CGIR_JIT_TIMING=[0|1]` set to print, at process exit, a per-phase wall-clock breakdown of JIT compilation accumulated across all compiles.
- `CGIR_JIT_CACHE_STATS=[0|1]` set to to print, at process exit, statistics about caching (hit, etc.)
- `CGIR_JIT_STATS_CSV=[0|1]` set to a file path to append, at process exit, the full JIT breakdown (mirrors `CGIR_STATS_CSV`)

# Passes
Passes are requested as a set (e.g. XKOMP's `OMP_TASKGRAPH_OPT=reduce-node,transitive-reduction,jit`) and always run in this order:
`copy-fuse`, `reduce-node`, `transitive-reduction`, `prog-fuse`, `jit`, `critical-path`, `sequence`, `pack`.

- `critical-path` does not transform the graph: it sets each command's launch priority (`command_t::priority`) to the duration-weighted length of the longest path from it to the graph's exit, from the commands' measured execution times (`command_t::duration`, ns). Durations come from the runtime: while one is missing, the pass sets `command_graph_t::durations_wanted`, XKRT times the commands during the next replays (`XKRT_PROG_VARIANT_SAMPLES` timed replays after a warm-up), then runs the pass again. When it runs again on the replayed graph, it recurses into packs: a pack weighs its measured duration (or its inner critical path), and the commands inside it get the priority of their inner chain plus what follows the pack. XKRT launches the ready command with the highest priority first, and gives each kernel node of a CUDA graph a node priority from it.

# Example
CGIR is integrated into the [XKRT](https://github.com/anlsys/xkrt) runtime system.
It serves as its abstraction for representing commands, notably to record and replay.
The [XKOMP](https://github.com/anlsys/xkomp) support for the `taskgraph` construct eventually fallbacks to CGIR.
See the glue here:
- The [graph instanciation](https://github.com/anlsys/xkrt/blob/master/src/command/command-graph.cc#L97-L101) for a task dependence graph IR.
- The [replay executionner](https://github.com/anlsys/xkrt/blob/master/src/command/command-graph.cc#L418)
- The [CUDA driver](https://github.com/anlsys/xkrt/blob/master/src/driver/driver_cu.cc#L618) to execute command graphs
