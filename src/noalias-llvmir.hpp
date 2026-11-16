/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
**
** This software is governed by the CeCILL-C license under French law and
** abiding by the rules of distribution of free software. See the LICENCE file.
**/

/*
 *  The aliasing and barrier analysis shared by the prog-fuse and jit passes.
 *
 *  A compiler cannot tell LLVM that two pointer arguments of a kernel do not
 *  overlap (an `omp target`'s mapped list items need not be distinct objects).
 *  A task runtime knows the actual values a recorded program is replayed with,
 *  so CGIR proves it, per instance, on the program's final code: every access is
 *  attributed to a base pointer whose runtime value is known, its byte range is
 *  bounded with ScalarEvolution evaluated over the actual scalar arguments and
 *  launch dimensions, and two bases are declared disjoint only when their ranges
 *  do not overlap (or neither side writes). Only direct accesses are bounded: an
 *  access through a pointer loaded from memory is "unknown" and defeats the
 *  proof for any base it might conflict with. Nothing is assumed: sound.
 *
 *  What it is used for:
 *   - `noalias` on the pointer parameters of a JIT'd device kernel, and
 *     scoped-noalias metadata between the captures of a fused host program
 *     (what enables vectorization, loop fusion, load forwarding, and the
 *     read-only data path);
 *   - the synchronization a fused device kernel needs between its bodies
 *     (barrier_requirements: none, block, or grid-wide), and whether it is
 *     thread-independent (its block barriers can go and its loops merge).
 *
 *    CGIR_NOALIAS_DEBUG=1        print, per analysed kernel, each pointer's
 *                                footprint and each fused kernel's barrier plan;
 *                                =2 also each offset that could not be bounded
 *                                and each call reaching a base.
 *    CGIR_NOALIAS_DUMP=<dir>     keep the specialized copies the analysis ran on.
 *
 *  The former policy knobs (CGIR_NOALIAS, CGIR_NOALIAS_MODE,
 *  CGIR_PROG_FUSE_BARRIER_MODE, CGIR_NOALIAS_VERIFY) were removed on 2026-11-15:
 *  set, they are ignored with a warning (warn_removed_knobs).
 *
 *  Internal to src/; not installed.
 */

# ifndef __CGIR_NOALIAS_LLVMIR_HPP__
# define __CGIR_NOALIAS_LLVMIR_HPP__

# include <cgir/namespace.hpp>
# include <cgir/support.h>

# if CGIR_SUPPORT_LLVM

#  include <llvm/ADT/DenseMap.h>
#  include <llvm/ADT/SmallVector.h>
#  include <llvm/IR/Instruction.h>
#  include <llvm/IR/Function.h>

#  include <cstdint>
#  include <string>
#  include <vector>

CGIR_NAMESPACE_BEGIN
namespace noalias {

/* Warn, once, about the removed policy knobs if any is set. */
void warn_removed_knobs(void);

/* A pointer of the analysed function whose runtime value is known: a kernel
 * parameter, or a captured base loaded from a fused program's args buffer. */
struct base_t
{
    const llvm::Value * v;
    uint64_t            addr;
};

/* What the analysis may assume about the function's other values. */
struct env_t
{
    /* runtime values of integer SSA values (scalar parameters, slot loads) */
    llvm::DenseMap<const llvm::Value *, int64_t> consts;

    /* launch dimensions, 0 when unknown (host) */
    unsigned grid[3]  = {0, 0, 0};
    unsigned block[3] = {0, 0, 0};

    /* The OpenMP kernel's `dyn_ptr` (its launch environment) -- several in a
     * fused kernel, one per constituent. Pointers loaded through one address
     * runtime-allocated scratch (e.g. the teams reduction buffer), which the
     * runtime never hands out as user data. */
    llvm::SmallVector<const llvm::Value *, 4> runtime_envs;

    /* Whether accesses to the module's global variables are private to the
     * program. True for a JIT'd device module: its globals are fresh objects of
     * the loaded PTX module, so no recorded argument can point into them. */
    bool globals_private = false;
};

/* The bytes a base is accessed at, relative to its runtime value. */
struct footprint_t
{
    bool    mod     = false;
    bool    ref     = false;
    bool    bounded = true;          /* false: anywhere */
    int64_t lo      = INT64_MAX;     /* [lo, hi), empty while lo >= hi */
    int64_t hi      = INT64_MIN;

    bool accessed(void) const { return mod || ref; }
};

struct result_t
{
    std::vector<footprint_t> base;   /* one per base_t, same order */
    footprint_t              unknown;/* accesses not attributable to a base */
    unsigned                 accesses = 0;
};

/* Attribute every memory access of `F` to a base (or to `unknown`) and bound
 * its byte range. `F` is not modified. */
result_t analyze(llvm::Function & F, const std::vector<base_t> & bases, const env_t & env);

/* Whether accesses `a` (to base value `va`) and `b` (to `vb`) may conflict:
 * both accessed, one of them written, and their absolute ranges may overlap. */
bool may_conflict(const footprint_t & a, uint64_t va, const footprint_t & b, uint64_t vb);

/* Whether the conflict is certain to the extent the ranges are: both bounded
 * and overlapping with a write. Used by the verifier to tell a violation from
 * an unverifiable assumption. */
bool bounded_conflict(const footprint_t & a, uint64_t va, const footprint_t & b, uint64_t vb);

/* ---------------------------------------------------------------------------
 * Grid barriers of fused device kernels.
 *
 * prog-fuse separates the bodies it fuses with a call to this marker (an
 * opaque, convergent declaration the optimizer can neither move nor drop),
 * and the jit pass lowers each one, once the kernel is in its final form, to
 * the weakest synchronization that is still correct:
 *
 *   NONE   every pair of conflicting accesses across it (same memory, one of
 *          them a write) is made by the same thread -- program order already
 *          orders them;
 *   BLOCK  the remaining conflicts are on the block's shared memory -- a
 *          block barrier orders them;
 *   GRID   otherwise: the device-wide barrier, which needs every block of the
 *          grid to be resident (a cooperative launch).
 *
 * "Same thread" is proven, per pair, when both accesses go through the same
 * base pointer at offsets that are the SAME exact affine function
 *     c + sum_i coef_i * v_i
 * of the thread/block indices and loop counters (counters matched by
 * coefficient), and that function is injective: every thread index of a
 * dimension larger than one appears, and sorted by coefficient,
 *     coef_0 >= access size,   coef_{i+1} >= coef_i * n_i
 * (n_i the number of values v_i takes) -- a mixed-radix numbering, so equal
 * addresses imply equal (thread, iteration) tuples. Disjoint footprints of two
 * base pointers, a thread's own stack, and shared-vs-global accesses never
 * conflict; unknown memory, runtime scratch and module globals written on both
 * sides always need the grid barrier. Elementwise chains (y = a*x + y; then
 * dot(y, z)) pass; stencils, gathers (SpMV), and a reduction followed by a use
 * of its result do not.
 * ------------------------------------------------------------------------- */

# define CGIR_GRID_BARRIER_MARKER "__cgir_grid_barrier"
/* Function attribute of a fused device kernel: the index of its parameter that
 * holds the grid barriers' state (see command_prog_t::grid_barrier_state). */
# define CGIR_GRID_BARRIER_STATE_ATTR "cgir-grid-barrier-state"

enum barrier_t { BARRIER_NONE = 0, BARRIER_BLOCK = 1, BARRIER_GRID = 2 };

/* The synchronization each of `markers` (calls to the marker in `F`, in any
 * order) needs, in the same order. */
std::vector<barrier_t> barrier_requirements(llvm::Function & F,
                                            const std::vector<const llvm::Instruction *> & markers,
                                            const std::vector<base_t> & bases, const env_t & env);

/* Whether no two threads of `F` ever communicate through memory: every pair of
 * conflicting accesses in the kernel (an access with itself included, as two
 * threads execute it) is provably made by one thread, by the same rule as a
 * NONE barrier above, and no shared memory, runtime scratch, module global or
 * unknown memory is written. Then no block barrier in `F` orders anything
 * another thread can observe, and all of them may go -- which, between two
 * fused bodies, lets their loops merge. */
bool thread_independent(llvm::Function & F, const std::vector<base_t> & bases, const env_t & env);

} /* namespace noalias */
CGIR_NAMESPACE_END

# endif /* CGIR_SUPPORT_LLVM */

# endif /* __CGIR_NOALIAS_LLVMIR_HPP__ */
