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

# include <cgir/namespace.hpp>
# include <cgir/command.hpp>
# include <cgir/command-graph.hpp>
# include <cgir/min-max.h>

# include <algorithm>

# include <assert.h>

CGIR_NAMESPACE_USE;

/*
 * CRITICAL-PATH PASS - annotate each command with a launch priority: the length
 * of the longest path from it to the graph's exit, where a command weighs its
 * measured execution time (command_t::duration, in ns) and every other node
 * nothing:
 *
 *     priority(u) = duration(u) + max over successors v of priority(v)
 *
 * A runtime that launches ready commands one at a time (one host thread
 * feeding several streams) starts the highest-priority one first: the command
 * with the most work left behind it, i.e. on the critical path. Launching it
 * later delays the whole graph by as much; launching an off-path command later
 * costs nothing as long as it stays off the path.
 *
 * Packs. The pass runs before `sequence`/`pack`, and the priorities travel
 * with the commands into the packs. When the runtime runs it again on the
 * replayed (packed) graph, it recurses: a PACK command weighs its measured
 * duration -- or, before it has one, the critical path of its sub-graph -- and
 * each command inside gets its inner bottom level plus the priority of what
 * follows the pack, so priorities compare across nesting levels. A runtime
 * representation of the pack (e.g. a CUDA graph) uses them per node.
 *
 * Durations are a property of a run, not of the program: they are measured by
 * the runtime while it replays the graph. So when some command has none yet,
 * the pass computes what it can (an unknown duration weighs 0) and sets
 * command_graph_t::durations_wanted, asking the runtime to time the commands
 * and to run the pass again once it has. The graph itself is not modified.
 */

/* per-node pass storage */
struct critical_path_pls_t
{
    uint64_t level;     /* bottom level of the node, local to its graph */
    critical_path_pls_t(void) : level(0) {}
    ~critical_path_pls_t(void) {}
};

/* Set the priority of every command of 'cg' to its bottom level in 'cg' plus
 * 'offset', the priority of what follows 'cg' in its parent graph (0 at the
 * top level), recursing into packs. Sets 'missing' if a command has no
 * measured duration. Returns the critical path length of 'cg'. */
static uint64_t
command_graph_pass_critical_path_rec(
    command_graph_t * cg,
    const uint64_t offset,
    bool & missing
) {
    /* DFS post-order: every node comes after all of its successors. Entry/exit
     * are included: a sequence's sub-graph uses its first/last commands as such */
    constexpr bool include_entry_exit = true;
    auto nodes = cg->create_node_iterators<include_entry_exit, critical_path_pls_t,
                                           COMMAND_GRAPH_WALK_SEARCH_DFS,
                                           COMMAND_GRAPH_WALK_ORDER_POST>();
    const int n = (int) nodes.size();

    uint64_t length = 0;
    for (int i = 0 ; i < n ; ++i)
    {
        command_graph_node_t * u = nodes[i].node;
        assert(u);

        /* the longest path after 'u' */
        uint64_t after = 0;
        for (command_graph_node_t * v : u->successors)
        {
            assert(v->iterator_index < (command_graph_node_index_t) n && nodes[v->iterator_index].node == v);
            after = MAX(after, nodes[v->iterator_index].data.level);
        }

        uint64_t weight = 0;
        if (u->type == COMMAND_GRAPH_NODE_TYPE_COMMAND && u->command)
        {
            command_t * c = u->command;
            weight = c->duration;

            /* a pack: what follows it in 'cg' also follows its commands. The
             * sub-graph is walked after 'cg' was, so the walks do not nest. */
            if (c->type == COMMAND_TYPE_PACK && c->pack.cg)
            {
                const uint64_t inner = command_graph_pass_critical_path_rec(c->pack.cg, offset + after, missing);
                if (weight == 0)
                    weight = inner;
            }
            /* an empty command does nothing: it weighs nothing */
            else if (weight == 0 && c->type != COMMAND_TYPE_EMPTY)
                missing = true;

            c->priority = offset + after + weight;
        }

        nodes[i].data.level = after + weight;
        length = MAX(length, nodes[i].data.level);
    }
    return length;
}

void
command_graph_t::pass_critical_path(void)
{
    bool missing = false;
    command_graph_pass_critical_path_rec(this, 0, missing);
    this->durations_wanted = missing;
}
