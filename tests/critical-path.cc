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
 *  Tests for the `critical-path` pass (command_graph_t::pass_critical_path):
 *  each command's priority is the duration-weighted length of the longest path
 *  from it to the exit, and the graph asks for durations while one is missing.
 */

# include <stdint.h>
# include <stdlib.h>
# include <stdio.h>

# if NDEBUG
#  define assert(X) X
# endif

# include "cgir-tests.cc"

static command_graph_node_t *
make_node(command_graph_t * cg, uint64_t duration)
{
    command_t * cmd = command_new(cg, COMMAND_TYPE_PROG);
    assert(cmd);
    cmd->duration = duration;
    command_graph_node_t * node = command_graph_node_new(cg, 1, COMMAND_GRAPH_NODE_TYPE_COMMAND);
    assert(node);
    node->command = cmd;
    return node;
}

static command_graph_t *
make_graph(command_graph_node_t ** entry, command_graph_node_t ** exit)
{
    command_graph_t * cg = command_graph_new();
    assert(cg);
    cg->init(command_new, command_graph_node_new, command_graph_new);
    *entry = cg->node_get_entry();
    *exit  = cg->node_get_exit();
    (*entry)->successors.clear();
    (*exit)->predecessors.clear();
    return cg;
}

static int
check(const char * test, const char * what, uint64_t got, uint64_t expected)
{
    if (got == expected)
        return 0;
    fprintf(stderr, "FAIL [%s]: %s: priority %llu, expected %llu\n", test, what,
            (unsigned long long) got, (unsigned long long) expected);
    return 1;
}

/*
 *  The CG shape that motivated the pass: after `div` three commands are ready,
 *  and only one of them leads to the rest of the iteration.
 *
 *      entry -> div(3) -> axpy(5) ------------------------> exit
 *                      -> dotp(2) ------------------------> exit
 *                      -> fused(10) -> dot(26) -> xpby(4) -> exit
 *
 *  Depth alone ranks axpy and dotp like fused's first step; durations put fused
 *  far ahead (40 vs 5 and 2).
 */
static int
test_weights_not_depth(void)
{
    command_graph_node_t * entry, * exit;
    command_graph_t * cg = make_graph(&entry, &exit);
    command_graph_node_t * div   = make_node(cg, 3);
    command_graph_node_t * axpy  = make_node(cg, 5);
    command_graph_node_t * dotp  = make_node(cg, 2);
    command_graph_node_t * fused = make_node(cg, 10);
    command_graph_node_t * dot   = make_node(cg, 26);
    command_graph_node_t * xpby  = make_node(cg, 4);
    entry->precedes(div);
    div->precedes(axpy);  axpy->precedes(exit);
    div->precedes(dotp);  dotp->precedes(exit);
    div->precedes(fused); fused->precedes(dot); dot->precedes(xpby); xpby->precedes(exit);

    cg->pass_critical_path();

    int rc = 0;
    rc |= check("weights", "xpby",  xpby->command->priority,  4);
    rc |= check("weights", "dot",   dot->command->priority,   30);
    rc |= check("weights", "fused", fused->command->priority, 40);
    rc |= check("weights", "axpy",  axpy->command->priority,  5);
    rc |= check("weights", "dotp",  dotp->command->priority,  2);
    rc |= check("weights", "div",   div->command->priority,   43);
    if (cg->durations_wanted)
    {
        fprintf(stderr, "FAIL [weights]: every duration is known, yet durations are wanted\n");
        rc = 1;
    }
    if (rc == 0)
        fprintf(stdout, "PASS [weights]: priorities follow durations, not depth\n");
    return rc;
}

/*
 *  A diamond: the priority takes the longer branch, and a node reached twice
 *  is counted once.
 *
 *      entry -> a(1) -> b(10) -> d(1) -> exit
 *                    -> c(2)  -> d
 */
static int
test_diamond(void)
{
    command_graph_node_t * entry, * exit;
    command_graph_t * cg = make_graph(&entry, &exit);
    command_graph_node_t * a = make_node(cg, 1);
    command_graph_node_t * b = make_node(cg, 10);
    command_graph_node_t * c = make_node(cg, 2);
    command_graph_node_t * d = make_node(cg, 1);
    entry->precedes(a);
    a->precedes(b); a->precedes(c);
    b->precedes(d); c->precedes(d);
    d->precedes(exit);

    cg->pass_critical_path();

    int rc = 0;
    rc |= check("diamond", "d", d->command->priority, 1);
    rc |= check("diamond", "b", b->command->priority, 11);
    rc |= check("diamond", "c", c->command->priority, 3);
    rc |= check("diamond", "a", a->command->priority, 12);
    if (rc == 0)
        fprintf(stdout, "PASS [diamond]: longest branch wins\n");
    return rc;
}

/*
 *  An unmeasured command weighs nothing, and makes the graph ask for durations
 *  until it has one.
 */
static int
test_missing_duration(void)
{
    command_graph_node_t * entry, * exit;
    command_graph_t * cg = make_graph(&entry, &exit);
    command_graph_node_t * a = make_node(cg, 7);
    command_graph_node_t * b = make_node(cg, 0);
    entry->precedes(a); a->precedes(b); b->precedes(exit);

    cg->pass_critical_path();
    int rc = check("missing", "a (b unknown)", a->command->priority, 7);
    if (!cg->durations_wanted)
    {
        fprintf(stderr, "FAIL [missing]: a duration is missing, yet none is wanted\n");
        rc = 1;
    }

    b->command->duration = 5;
    cg->pass_critical_path();
    rc |= check("missing", "a (b measured)", a->command->priority, 12);
    if (cg->durations_wanted)
    {
        fprintf(stderr, "FAIL [missing]: durations still wanted once all are known\n");
        rc = 1;
    }
    if (rc == 0)
        fprintf(stdout, "PASS [missing]: unknown durations are requested, then used\n");
    return rc;
}

/*
 *  A pack: inner commands get their inner chain plus what follows the pack, and
 *  the pack weighs its inner critical path until it has a duration of its own.
 *
 *      entry -> P[ x(4) -> y(6) ] -> z(10) -> exit
 */
static int
test_pack(void)
{
    command_graph_node_t * entry, * exit;
    command_graph_t * cg = make_graph(&entry, &exit);
    command_graph_node_t * sentry, * sexit;
    command_graph_t * sub = make_graph(&sentry, &sexit);
    command_graph_node_t * x = make_node(sub, 4);
    command_graph_node_t * y = make_node(sub, 6);
    sentry->precedes(x); x->precedes(y); y->precedes(sexit);

    command_t * pc = command_new(cg, COMMAND_TYPE_PACK);
    assert(pc);
    pc->pack.cg = sub;
    command_graph_node_t * P = command_graph_node_new(cg, 1, COMMAND_GRAPH_NODE_TYPE_COMMAND);
    P->command = pc;
    command_graph_node_t * z = make_node(cg, 10);
    entry->precedes(P); P->precedes(z); z->precedes(exit);

    cg->pass_critical_path();

    int rc = 0;
    rc |= check("pack", "z", z->command->priority, 10);
    rc |= check("pack", "P (inner path)", pc->priority, 20);
    rc |= check("pack", "y", y->command->priority, 16);
    rc |= check("pack", "x", x->command->priority, 20);
    if (cg->durations_wanted)
    {
        fprintf(stderr, "FAIL [pack]: every inner duration is known, yet durations are wanted\n");
        rc = 1;
    }

    /* measured, the pack weighs its own duration (here more than its path) */
    pc->duration = 14;
    cg->pass_critical_path();
    rc |= check("pack", "P (measured)", pc->priority, 24);
    rc |= check("pack", "x (measured pack)", x->command->priority, 20);
    if (rc == 0)
        fprintf(stdout, "PASS [pack]: priorities propagate into packs\n");
    return rc;
}

int
main(void)
{
    int rc = 0;
    rc |= test_weights_not_depth();
    rc |= test_diamond();
    rc |= test_missing_duration();
    rc |= test_pack();
    return rc;
}
