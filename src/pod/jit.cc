/*
** Copyright 2024,2025 INRIA
**
** Contributors :
** Romain PEREIRA, rpereira@anl.gov
**
** This software is a computer program whose purpose is to execute
** blas subroutines on multi-GPUs system.
**
** This software is governed by the CeCILL-C license under French law and
** abiding by the rules of distribution of free software.  You can  use,
** modify and/ or redistribute the software under the terms of the CeCILL-C
** license as circulated by CEA, CNRS and INRIA at the following URL
** "http://www.cecill.info".

** As a counterpart to the access to the source code and  rights to copy,
** modify and redistribute granted by the license, users are provided only
** with a limited warranty  and the software's author,  the holder of the
** economic rights,  and the successive licensors  have only  limited
** liability.

** In this respect, the user's attention is drawn to the risks associated
** with loading,  using,  modifying and/or developing or reproducing the
** software by the user in light of its specific status of free software,
** that may mean  that it is complicated to manipulate,  and  that  also
** therefore means  that it is reserved for developers  and  experienced
** professionals having in-depth computer knowledge. Users are therefore
** encouraged to load and test the software's suitability as regards their
** requirements in conditions enabling the security of their systems and/or
** data to be ensured and,  more generally, to use and operate it in the
** same conditions as regards security.

** The fact that you are presently reading this means that you have had
** knowledge of the CeCILL-C license and that you accept its terms.
**/

// Legacy (POD) jit pass: JIT-compiles every PROG node carrying a non-null
// LLVM-IR source (e.g. a program produced by prog-fuse) and installs the
// resulting function pointer, via the shared core command_graph_jit_llvmir()
// (see ../prog-fuse-llvmir.cc).

# include <cgir/namespace.hpp>
# include <cgir/support.h>
# include <cgir/command.hpp>
# include <cgir/command-graph.hpp>

# include "prog-fuse-llvmir.hpp"
# include "jit-support.hpp"

# include <stdlib.h>
# include <string.h>

CGIR_NAMESPACE_USE;

/* true iff `u` is a command node holding a non-null LLVM-IR program source */
static inline bool
node_has_llvmir_source(const command_graph_node_t * u)
{
    return u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND
        && u->command != nullptr
        && u->command->type == COMMAND_TYPE_PROG
        && u->command->prog.source.type == COMMAND_PROG_SOURCE_TYPE_LLVMIR
        && u->command->prog.source.content.llvmir.raw != nullptr;
}


/* What the pass does with a program that already has an ahead-of-time
 * implementation is chosen by CGIR_JIT_VARIANT (see jit-support.hpp): replace
 * it (`jit`), keep it (`aot`), or compile and keep both so the runtime picks the
 * faster by measurement (`auto`, the default).
 *
 * Why recompiling is not simply always right, and why the runtime must hold the
 * occupancy whichever variant runs: on a CSR SpMV `omp target teams distribute
 * parallel for`, the recompiled inner loop is instruction-for-instruction
 * identical to the ahead-of-time one; but ahead-of-time the kernel keeps a
 * runtime `IsSPMDMode` dispatch and a dead duplicate of the loop nest for
 * generic mode, and with it 9 named barriers -- and the barrier count was what
 * limited the kernel to 7 blocks/SM. This pass runs OpenMPOpt, which proves the
 * kernel SPMD-only and deletes that path, leaving 1 barrier: the device then
 * co-schedules 16 blocks/SM, they compete for the same cache, and DRAM traffic
 * goes from 168MB to 625MB for a 6x slower kernel. Better code, three times
 * slower. See command_prog_t::blocks_per_sm. And even at equal occupancy the
 * two pipelines win on different kernels -- hence `auto`, which measures both
 * and keeps the faster. */

/* true iff `prog` still holds an ahead-of-time implementation the pass would
 * replace: a device kernel handle, or a host OpenMP task routine. Programs
 * synthesized by prog-fuse have neither -- the pass clears the launcher
 * precisely to mean "must be JIT'd". */
static inline bool
prog_has_aot(const command_prog_t * prog)
{
    if (prog->source.content.llvmir.triple != nullptr)               /* device */
        return prog->launcher.variadic.fn != nullptr;
    return prog->prototype == CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_KMP /* host */
        && prog->launcher.kmp.fn != nullptr;
}

static void
jit_prog(command_prog_t * prog)
{
# if !CGIR_SUPPORT_LLVM
    command_graph_jit_llvmir(prog);     /* reports and aborts */
# else
    const jit::jit_variant_mode_t mode = jit::jit_variant_mode();
    const bool has_aot = prog_has_aot(prog);
    const bool device  = prog->source.content.llvmir.triple != nullptr;

    if (has_aot && mode == jit::JIT_VARIANT_AOT)
        return ;

    /* Keep the ahead-of-time implementation for the runtime to measure
     * against: the kernel handle of a device program, the task routine of a
     * host one (whose JIT'd replacement changes the calling convention but not
     * the task, so the routine stays valid). */
    const bool keep = has_aot && mode == jit::JIT_VARIANT_AUTO;
    void (*aot_fn)(void **)        = (keep && device)  ? prog->launcher.variadic.fn : nullptr;
    int  (*aot_kmp_fn)(int, void *) = (keep && !device) ? prog->launcher.kmp.fn      : nullptr;
    void * aot_kmp_task             = (keep && !device) ? prog->launcher.kmp.task    : nullptr;
    const unsigned int aot_dyn = prog->dyn_shmem;

    command_graph_jit_llvmir(prog);

    /* The JIT may decline a program (entry not externally resolvable) and leave
     * it untouched; only a program whose code it actually replaced has two. */
    const bool replaced = device
        ? (prog->source.type == COMMAND_PROG_SOURCE_TYPE_PTX)
        : (prog->prototype != CGIR_COMMAND_PROG_FUNCTION_PROTOTYPE_KMP);
    if (!replaced)
    {
        aot_fn     = nullptr;
        aot_kmp_fn = nullptr;
    }

    prog->aot_variant.fn        = aot_fn;
    prog->aot_variant.dyn_shmem = aot_dyn;
    prog->aot_variant.kmp_fn    = aot_kmp_fn;
    prog->aot_variant.kmp_task  = aot_kmp_fn ? aot_kmp_task : nullptr;
    prog->aot_variant.selected  = CGIR_COMMAND_PROG_VARIANT_UNDECIDED;
# endif
}

void
command_graph_t::pass_jit(void)
{
    this->walk([&] (command_graph_node_t * node)
    {
        if (node_has_llvmir_source(node))
            jit_prog(&node->command->prog);
    });
}
