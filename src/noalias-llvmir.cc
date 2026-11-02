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
 *  Aliasing policy and analysis -- see noalias-llvmir.hpp.
 *
 *  The analysis answers one question about a function whose pointer arguments
 *  have known runtime values: which bytes, relative to each of them, may it
 *  read or write? It walks every instruction that touches memory and
 *
 *    1. attributes it to a base: getUnderlyingObjects() must lead to one of the
 *       known pointers, to private memory (stack, GPU shared/local memory, the
 *       module's own globals, runtime scratch reached through the OpenMP
 *       launch environment), or else the access is "unknown";
 *    2. bounds its offset from that base: ScalarEvolution expresses the address
 *       as base + f(loop counters, scalars, thread ids), and f is evaluated
 *       with interval arithmetic over the values the program is actually
 *       replayed with -- its scalar arguments are constants, its thread and
 *       block indices range over the launch grid, and each loop counter over
 *       [0, max backedge-taken count]. Anything the evaluator does not
 *       understand makes the footprint unbounded.
 *
 *  It is deliberately simple and conservative: it never claims a range it
 *  cannot derive, so a proof of disjointness built on it is sound, while a
 *  failure to prove only costs an optimization.
 */

# include "noalias-llvmir.hpp"

# if CGIR_SUPPORT_LLVM

#  include <llvm/Analysis/AssumptionCache.h>
#  include <llvm/Analysis/LoopInfo.h>
#  include <llvm/Analysis/ScalarEvolution.h>
#  include <llvm/Analysis/ScalarEvolutionExpressions.h>
#  include <llvm/Analysis/TargetLibraryInfo.h>
#  include <llvm/Analysis/ValueTracking.h>
#  include <llvm/IR/Dominators.h>
#  include <llvm/IR/InstIterator.h>
#  include <llvm/IR/Instructions.h>
#  include <llvm/IR/IntrinsicInst.h>
#  include <llvm/IR/IntrinsicsNVPTX.h>
#  include <llvm/IR/Module.h>
#  include <llvm/Support/raw_ostream.h>
#  include <llvm/TargetParser/Triple.h>

#  include <algorithm>
#  include <functional>
#  include <cstdio>
#  include <cstdlib>
#  include <cstring>
#  include <mutex>

CGIR_NAMESPACE_USE;
using namespace CGIR_NAMESPACE::noalias;

/* ---------------------------------------------------------------------------
 * Knobs
 * ------------------------------------------------------------------------- */

policy_t
CGIR_NAMESPACE::noalias::mode(void)
{
    static const policy_t value = [] {
        for (const char * gone : { "CGIR_JIT_DEVICE_NOALIAS", "CGIR_JIT_AOT_DEVICE" })
            if (getenv(gone))
                fprintf(stderr, "cgir: %s was removed and is ignored; see CGIR_NOALIAS "
                                "and CGIR_JIT_VARIANT\n", gone);
        const char * e = getenv("CGIR_NOALIAS");
        if (e == nullptr || e[0] == '\0' || strcmp(e, "prove") == 0)
            return MODE_PROVE;
        if (strcmp(e, "none") == 0)
            return MODE_NONE;
        if (strcmp(e, "distinct-base") == 0)
            return MODE_DISTINCT_BASE;
        fprintf(stderr, "cgir: CGIR_NOALIAS='%s' is not one of none, prove, "
                        "distinct-base; using prove\n", e);
        return MODE_PROVE;
    }();
    return value;
}

const char *
CGIR_NAMESPACE::noalias::mode_name(policy_t m)
{
    switch (m)
    {
        case MODE_NONE:          return "none";
        case MODE_PROVE:         return "prove";
        case MODE_DISTINCT_BASE: return "distinct-base";
    }
    return "?";
}

static int
verify_level(void)
{
    static const int value = [] {
        const char * e = getenv("CGIR_NOALIAS_VERIFY");
        if (e == nullptr || e[0] == '\0' || strcmp(e, "0") == 0)
            return 0;
        return strcmp(e, "abort") == 0 ? 2 : 1;
    }();
    return value;
}

bool
CGIR_NAMESPACE::noalias::verify(void)
{
    return verify_level() > 0;
}

/* ---------------------------------------------------------------------------
 * Interval arithmetic over ScalarEvolution expressions
 * ------------------------------------------------------------------------- */

namespace {

typedef __int128 wide_t;

/* Anything beyond this magnitude is treated as unbounded: no buffer is that
 * large, and it keeps every product of two bounds inside 128 bits. */
static const wide_t WIDE_LIMIT = ((wide_t) 1) << 62;

struct interval_t
{
    bool   ok;
    wide_t lo, hi;

    static interval_t none(void) { return {false, 0, 0}; }
    static interval_t of(wide_t a, wide_t b)
    {
        if (a > b || a < -WIDE_LIMIT || b > WIDE_LIMIT)
            return none();
        return {true, a, b};
    }
    static interval_t point(wide_t a) { return of(a, a); }
};

static interval_t
iadd(interval_t a, interval_t b)
{
    if (!a.ok || !b.ok) return interval_t::none();
    return interval_t::of(a.lo + b.lo, a.hi + b.hi);
}

static interval_t
imul(interval_t a, interval_t b)
{
    if (!a.ok || !b.ok) return interval_t::none();
    const wide_t c[4] = { a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi };
    return interval_t::of(*std::min_element(c, c + 4), *std::max_element(c, c + 4));
}

/* Rewrites min/max expressions whose operands provably dominate one another
 * into the dominating operand, so that ScalarEvolution can re-simplify what
 * surrounds them. Whether `a >= b` is decided on SE's own difference a - b,
 * where correlated terms cancel -- interval arithmetic on each operand alone
 * cannot see that ctaid*B + tid - ctaid*B is just tid. An OpenMP chunk loop's
 * trip count, (umax(ctaid*B + B, ctaid*B + tid + B) - ctaid*B - tid - 1) / B,
 * folds this way to exactly 0. */
struct minmax_simplifier_t : public llvm::SCEVRewriteVisitor<minmax_simplifier_t>
{
    std::function<interval_t(const llvm::SCEV *)> eval;

    minmax_simplifier_t(llvm::ScalarEvolution & se, std::function<interval_t(const llvm::SCEV *)> e)
        : llvm::SCEVRewriteVisitor<minmax_simplifier_t>(se), eval(std::move(e)) {}

    /* a >= b provably (signed, or unsigned when both are non-negative) */
    bool geq(const llvm::SCEV * a, const llvm::SCEV * b, bool uns)
    {
        if (uns)
        {
            const interval_t ia = eval(a), ib = eval(b);
            if (!ia.ok || !ib.ok || ia.lo < 0 || ib.lo < 0)
                return false;
        }
        const llvm::SCEV * d = SE.getMinusSCEV(a, b);
        if (llvm::isa<llvm::SCEVCouldNotCompute>(d))
            return false;
        const interval_t id = eval(d);
        return id.ok && id.lo >= 0;
    }

    template <typename E>
    llvm::SmallVector<llvm::SCEVUse, 2> prune(const E * Expr, bool is_max, bool uns)
    {
        llvm::SmallVector<const llvm::SCEV *, 4> ops;
        for (const llvm::SCEV * Op : Expr->operands())
            ops.push_back(this->visit(Op));
        llvm::SmallVector<llvm::SCEVUse, 2> keep;
        for (size_t i = 0 ; i < ops.size() ; ++i)
        {
            bool dominated = false;
            for (size_t j = 0 ; j < ops.size() && !dominated ; ++j)
                if (j != i && (is_max ? geq(ops[j], ops[i], uns) : geq(ops[i], ops[j], uns)) &&
                    (ops[i] != ops[j] || j < i))
                    dominated = true;
            if (!dominated)
                keep.push_back(ops[i]);
        }
        if (keep.empty())
            keep.push_back(ops[0]);
        return keep;
    }

    const llvm::SCEV * one_or(llvm::SmallVector<llvm::SCEVUse, 2> & v,
                              const llvm::SCEV * (*build)(llvm::ScalarEvolution &, llvm::SmallVector<llvm::SCEVUse, 2> &))
    {
        return v.size() == 1 ? (const llvm::SCEV *) v[0] : build(SE, v);
    }

    const llvm::SCEV * visitSMaxExpr(const llvm::SCEVSMaxExpr * E)
    { auto v = prune(E, true, false);  return one_or(v, [] (llvm::ScalarEvolution & S, llvm::SmallVector<llvm::SCEVUse, 2> & o) { return S.getSMaxExpr(o); }); }
    const llvm::SCEV * visitUMaxExpr(const llvm::SCEVUMaxExpr * E)
    { auto v = prune(E, true, true);   return one_or(v, [] (llvm::ScalarEvolution & S, llvm::SmallVector<llvm::SCEVUse, 2> & o) { return S.getUMaxExpr(o); }); }
    const llvm::SCEV * visitSMinExpr(const llvm::SCEVSMinExpr * E)
    { auto v = prune(E, false, false); return one_or(v, [] (llvm::ScalarEvolution & S, llvm::SmallVector<llvm::SCEVUse, 2> & o) { return S.getSMinExpr(o); }); }
    const llvm::SCEV * visitUMinExpr(const llvm::SCEVUMinExpr * E)
    { auto v = prune(E, false, true);  return one_or(v, [] (llvm::ScalarEvolution & S, llvm::SmallVector<llvm::SCEVUse, 2> & o) { return S.getUMinExpr(o); }); }
};

struct evaluator_t
{
    llvm::ScalarEvolution &                            SE;
    const env_t &                                      env;
    llvm::DenseMap<const llvm::SCEV *, interval_t>     memo;
    unsigned                                           depth = 0;

    /* the access being bounded (null: none), for the loop exit-test refinement */
    const llvm::Instruction *                          ctx = nullptr;
    llvm::DominatorTree *                              DT  = nullptr;

    evaluator_t(llvm::ScalarEvolution & se, const env_t & e) : SE(se), env(e) {}

    /* the loop's symbolic max backedge-taken count, min/max-simplified */
    const llvm::SCEV * btc(const llvm::Loop * L)
    {
        const llvm::SCEV * B = SE.getSymbolicMaxBackedgeTakenCount(L);
        if (llvm::isa<llvm::SCEVCouldNotCompute>(B))
            return B;
        minmax_simplifier_t simp(SE, [this] (const llvm::SCEV * x) { return eval(x); });
        return simp.visit(B);
    }

    void set_context(const llvm::Instruction * I)
    {
        if (I != ctx)
            memo.clear();
        ctx = I;
    }

    /* Upper bound on an increasing affine recurrence `AR` of loop L at the
     * context access, from L's exit tests -- interval arithmetic alone loses
     * the correlation between a grid-stride loop's start (the thread index)
     * and its trip count, and overestimates the footprint by a whole stride.
     * Two shapes are sound:
     *   - the test is in L's header and the access is dominated by the
     *     header's in-loop successor: in the body, AR < UB;
     *   - the test is in L's latch on the post-increment value: every
     *     iteration after the first passed it, so AR < UB except for the
     *     first value, which is `start`.
     * Returns the bound (exclusive), or none. */
    interval_t exit_bound(const llvm::SCEVAddRecExpr * AR, interval_t start)
    {
        using namespace llvm;
        const Loop * L = AR->getLoop();
        if (ctx == nullptr || DT == nullptr || !L->contains(ctx) || !AR->isAffine())
            return interval_t::none();
        const auto * SA = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
        if (SA == nullptr || SA->getAPInt().getSignificantBits() > 63)
            return interval_t::none();
        const wide_t step_a = SA->getAPInt().getSExtValue();

        SmallVector<BasicBlock *, 4> exiting;
        L->getExitingBlocks(exiting);
        interval_t best = interval_t::none();
        for (BasicBlock * E : exiting)
        {
            auto * Br = dyn_cast<BranchInst>(E->getTerminator());
            if (Br == nullptr || !Br->isConditional())
                continue ;
            auto * Cmp = dyn_cast<ICmpInst>(Br->getCondition());
            if (Cmp == nullptr)
                continue ;
            const bool true_stays = L->contains(Br->getSuccessor(0));
            if (true_stays == L->contains(Br->getSuccessor(1)))
                continue ;
            ICmpInst::Predicate P = true_stays ? Cmp->getPredicate() : Cmp->getInversePredicate();
            const SCEV * A = SE.getSCEV(Cmp->getOperand(0));
            const SCEV * B = SE.getSCEV(Cmp->getOperand(1));
            if (!isa<SCEVAddRecExpr>(A) || cast<SCEVAddRecExpr>(A)->getLoop() != L)
            {
                std::swap(A, B);
                P = ICmpInst::getSwappedPredicate(P);
            }
            const auto * IVc = dyn_cast<SCEVAddRecExpr>(A);
            if (IVc == nullptr || IVc->getLoop() != L || !IVc->isAffine() || !SE.isLoopInvariant(B, L))
                continue ;
            const bool strict = (P == ICmpInst::ICMP_ULT || P == ICmpInst::ICMP_SLT);
            if (!strict && P != ICmpInst::ICMP_ULE && P != ICmpInst::ICMP_SLE)
                continue ;
            const bool uns = (P == ICmpInst::ICMP_ULT || P == ICmpInst::ICMP_ULE);

            /* The compared recurrence is either the loop's counter as tested in
             * the header (pre-increment), or as tested in the latch after the
             * increment. Recover the pre-increment counter `IV` either way. */
            const auto * SI = dyn_cast<SCEVConstant>(IVc->getStepRecurrence(SE));
            if (SI == nullptr || SI->getAPInt().isNonPositive() || SI->getAPInt().getSignificantBits() > 63)
                continue ;
            const wide_t step_i = SI->getAPInt().getSExtValue();
            /* Classify the test by what it guards, not by the block it is in
             * (in a single-block loop the header is also the latch):
             *   backedge test -- staying goes back to the header, so the tested
             *     value bounds the NEXT iteration: relate the access to the
             *     previous iteration's tested value (IVc - step), and cover the
             *     first iteration, which no test preceded, by its start;
             *   in-body test  -- staying continues into the body: the bound
             *     holds for the accesses strictly after it in the same
             *     iteration, i.e. dominated by the in-loop successor and not in
             *     the testing block itself. */
            BasicBlock * stay = Br->getSuccessor(true_stays ? 0 : 1);
            const SCEVAddRecExpr * IV = IVc;
            bool post = false;
            if (E == L->getLoopLatch() && stay == L->getHeader())
            {
                IV = dyn_cast<SCEVAddRecExpr>(SE.getMinusSCEV(IVc, SI));
                if (IV == nullptr || IV->getLoop() != L)
                    continue ;
                post = true;
            }
            else if (stay != L->getHeader())
            {
                if (ctx->getParent() == E || !DT->dominates(stay, ctx->getParent()))
                    continue ;
            }
            else
                continue ;

            /* AR = D + m * ext(IV) with m constant and D loop-invariant? */
            if (step_a % step_i != 0)
                continue ;
            const wide_t m = step_a / step_i;
            Type * Ta = AR->getType();
            const SCEV * ext = IV;
            if (IV->getType() != Ta)
            {
                if (!IV->getType()->isIntegerTy() || !Ta->isIntegerTy() ||
                    IV->getType()->getIntegerBitWidth() > Ta->getIntegerBitWidth())
                    continue ;
                ext = uns ? SE.getZeroExtendExpr(IV, Ta) : SE.getSignExtendExpr(IV, Ta);
            }
            const SCEV * Dlt = SE.getMinusSCEV(AR, SE.getMulExpr(SE.getConstant(Ta, (uint64_t) m, true), ext));
            if (isa<SCEVCouldNotCompute>(Dlt) || !SE.isLoopInvariant(Dlt, L))
                continue ;

            const interval_t ub = eval(B);
            const interval_t iv0 = eval(IV->getStart());
            const interval_t d = eval(Dlt);
            if (!ub.ok || !iv0.ok || !d.ok || (uns && (iv0.lo < 0 || ub.lo < 0)))
                continue ;
            const wide_t lim = strict ? ub.hi : ub.hi + 1;   /* tested value < lim */
            /* The largest counter value reached in the body. Header test: the
             * body only runs for counters that passed it. Latch test: the
             * incremented value that passes it IS the next iteration's
             * counter, so every counter after the first is < lim -- and the
             * first is the start. */
            const wide_t iv_hi = post ? std::max(iv0.hi, lim - 1) : lim - 1;
            if (iv_hi < iv0.lo || m <= 0)
                continue ;
            const interval_t b = interval_t::of(start.lo, d.hi + m * iv_hi + 1);
            if (b.ok && (!best.ok || b.hi < best.hi))
                best = b;
        }
        return best;
    }

    /* range of a launch-dimension register read, if it is one */
    interval_t sreg(const llvm::Value * V)
    {
        const auto * II = llvm::dyn_cast<llvm::IntrinsicInst>(V);
        if (II == nullptr)
            return interval_t::none();
        auto dim = [&] (const unsigned * d, unsigned i, bool index) -> interval_t {
            if (d[i] == 0) return interval_t::none();
            return index ? interval_t::of(0, (wide_t) d[i] - 1) : interval_t::point(d[i]);
        };
        switch (II->getIntrinsicID())
        {
            case llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x:    return dim(env.block, 0, true);
            case llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y:    return dim(env.block, 1, true);
            case llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z:    return dim(env.block, 2, true);
            case llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x:   return dim(env.block, 0, false);
            case llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_y:   return dim(env.block, 1, false);
            case llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_z:   return dim(env.block, 2, false);
            case llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x:  return dim(env.grid,  0, true);
            case llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y:  return dim(env.grid,  1, true);
            case llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z:  return dim(env.grid,  2, true);
            case llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_x: return dim(env.grid,  0, false);
            case llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_y: return dim(env.grid,  1, false);
            case llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_z: return dim(env.grid,  2, false);
            default: return interval_t::none();
        }
    }

    /* what ScalarEvolution itself knows (range metadata, known bits, ...) */
    interval_t signed_range(const llvm::SCEV * S)
    {
        const llvm::ConstantRange R = SE.getSignedRange(S);
        if (R.isFullSet() || R.isEmptySet() || R.getBitWidth() > 64)
            return interval_t::none();
        return interval_t::of(R.getSignedMin().getSExtValue(), R.getSignedMax().getSExtValue());
    }

    interval_t eval(const llvm::SCEV * S)
    {
        if (auto it = memo.find(S); it != memo.end())
            return it->second;
        if (++depth > 64)
        {
            --depth;
            return interval_t::none();
        }
        interval_t r = eval_uncached(S);
        --depth;
        /* A narrower range SCEV already knows can only help. */
        if (r.ok && S->getType()->isIntegerTy())
        {
            const interval_t k = signed_range(S);
            if (k.ok)
                r = interval_t::of(std::max(r.lo, k.lo), std::min(r.hi, k.hi));
        }
        else if (!r.ok && S->getType()->isIntegerTy())
            r = signed_range(S);
        memo[S] = r;
        return r;
    }

    interval_t eval_uncached(const llvm::SCEV * S)
    {
        using namespace llvm;
        switch (S->getSCEVType())
        {
            case scConstant:
            {
                const APInt & C = cast<SCEVConstant>(S)->getAPInt();
                if (C.getSignificantBits() > 63) return interval_t::none();
                return interval_t::point(C.getSExtValue());
            }

            case scUnknown:
            {
                const Value * V = cast<SCEVUnknown>(S)->getValue();
                if (auto it = env.consts.find(V); it != env.consts.end())
                    return interval_t::point(it->second);
                return sreg(V);
            }

            case scTruncate:
            {
                const auto * T = cast<SCEVTruncateExpr>(S);
                const interval_t o = eval(T->getOperand());
                const unsigned w = T->getType()->getIntegerBitWidth();
                if (!o.ok || w >= 64) return o.ok && w >= 64 ? o : interval_t::none();
                const wide_t half = ((wide_t) 1) << (w - 1);
                return (o.lo >= -half && o.hi < half) ? o : interval_t::none();
            }

            case scZeroExtend:
            {
                const interval_t o = eval(cast<SCEVZeroExtendExpr>(S)->getOperand());
                return (o.ok && o.lo >= 0) ? o : interval_t::none();
            }

            case scSignExtend:
                return eval(cast<SCEVSignExtendExpr>(S)->getOperand());

            case scAddExpr:
            {
                interval_t acc = interval_t::point(0);
                for (const SCEV * Op : cast<SCEVAddExpr>(S)->operands())
                    acc = iadd(acc, eval(Op));
                return acc;
            }

            case scMulExpr:
            {
                interval_t acc = interval_t::point(1);
                for (const SCEV * Op : cast<SCEVMulExpr>(S)->operands())
                    acc = imul(acc, eval(Op));
                return acc;
            }

            case scUDivExpr:
            {
                const auto * D = cast<SCEVUDivExpr>(S);
                const interval_t a = eval(D->getLHS()), b = eval(D->getRHS());
                if (!a.ok || !b.ok || a.lo < 0 || b.lo <= 0) return interval_t::none();
                return interval_t::of(a.lo / b.hi, a.hi / b.lo);
            }

            case scSMaxExpr: case scUMaxExpr:
            case scSMinExpr: case scUMinExpr:
            case scSequentialUMinExpr:
            {
                const SCEVTypes K = S->getSCEVType();
                const bool is_max = (K == scSMaxExpr || K == scUMaxExpr);
                const bool is_uns = (K == scUMaxExpr || K == scUMinExpr || K == scSequentialUMinExpr);
                interval_t acc = interval_t::none();
                bool first = true;
                for (const SCEV * Op : cast<SCEVNAryExpr>(S)->operands())
                {
                    const interval_t o = eval(Op);
                    /* unsigned order agrees with signed order on non-negatives */
                    if (!o.ok || (is_uns && o.lo < 0)) return interval_t::none();
                    if (first) { acc = o; first = false; continue; }
                    acc = is_max ? interval_t::of(std::max(acc.lo, o.lo), std::max(acc.hi, o.hi))
                                 : interval_t::of(std::min(acc.lo, o.lo), std::min(acc.hi, o.hi));
                }
                return acc;
            }

            case scAddRecExpr:
            {
                const auto * AR = cast<SCEVAddRecExpr>(S);
                if (!AR->isAffine()) return interval_t::none();
                const interval_t start = eval(AR->getStart());
                const interval_t step  = eval(AR->getOperand(1));
                const SCEV * BTC = btc(AR->getLoop());
                if (isa<SCEVCouldNotCompute>(BTC)) return interval_t::none();
                const interval_t n = eval(BTC);
                if (!start.ok || !step.ok || !n.ok || n.hi < 0) return interval_t::none();
                /* the counter takes the values 0..BTC in the loop and BTC+1 in an
                 * LCSSA use after it: cover both */
                const wide_t k = n.hi + 1;
                interval_t r = interval_t::of(start.lo + std::min<wide_t>(0, step.lo * k),
                                              start.hi + std::max<wide_t>(0, step.hi * k));
                /* tighten an increasing recurrence with the loop's exit test */
                if (r.ok && step.lo > 0)
                {
                    const interval_t b = exit_bound(AR, start);
                    if (b.ok && b.hi - 1 < r.hi)
                        r = interval_t::of(r.lo, std::max(r.lo, (wide_t) (b.hi - 1)));
                }
                return r;
            }

            default:   /* vscale, pointer casts, could-not-compute */
                return interval_t::none();
        }
    }
};

/* ---------------------------------------------------------------------------
 * Access attribution
 * ------------------------------------------------------------------------- */

/* Who owns the memory an access may touch. For aliasing, every kind but BASE
 * and UNKNOWN is private: no recorded argument can point into it. For grid
 * barriers the private kinds differ by who else can see them: a thread's stack,
 * its block's shared memory, or the whole grid (module globals, runtime
 * scratch). Ordered by visibility, so that merging takes the max. */
enum class owner_kind_t { STACK, SHARED, SYSTEM, BASE, UNKNOWN };

static inline bool
owner_private(owner_kind_t k)
{
    return k == owner_kind_t::STACK || k == owner_kind_t::SHARED || k == owner_kind_t::SYSTEM;
}

struct owner_t
{
    owner_kind_t kind;
    unsigned     base;   /* when kind == BASE */
};

/* An access offset as an exact affine function of the thread indices and the
 * loop counters: c + sum(coef * var). Variables are keyed by an NVPTX
 * special-register intrinsic id (thread/block index) or by their Loop. */
struct affine_t
{
    bool                                       ok = false;
    wide_t                                     c  = 0;
    std::vector<std::pair<const void *, wide_t>> terms;   /* var -> coef */
    std::vector<std::pair<const void *, wide_t>> range;   /* var -> number of values */

    void add(const void * var, wide_t coef, wide_t n)
    {
        for (auto & t : terms)
            if (t.first == var) { t.second += coef; return ; }
        terms.push_back({var, coef});
        range.push_back({var, n});
    }
    wide_t coef(const void * var) const
    {
        for (const auto & t : terms) if (t.first == var) return t.second;
        return 0;
    }
};

/* One recorded access, for the grid-barrier analysis. */
struct access_rec_t
{
    unsigned     region;
    owner_t      owner;
    bool         mod, ref;
    bool         bounded;
    wide_t       lo, hi;     /* byte footprint relative to the base (BASE only) */
    wide_t       size;       /* bytes per access point */
    affine_t     form;       /* offset, when exactly affine (BASE only) */
};

struct analysis_t
{
    llvm::Function &                               F;
    const std::vector<base_t> &                    bases;
    const env_t &                                  env;
    llvm::DenseMap<const llvm::Value *, unsigned>  base_of;
    llvm::ScalarEvolution &                        SE;
    llvm::LoopInfo &                               LI;
    evaluator_t                                    ev;
    result_t                                       res;

    /* grid-barrier analysis: the markers separating the regions of a fused
     * kernel, and every access recorded with its region */
    const std::vector<const llvm::Instruction *> * markers = nullptr;
    llvm::DominatorTree *                          DTp     = nullptr;
    std::vector<access_rec_t> *                    recs    = nullptr;
    unsigned                                       region  = 0;

    /* memo of runtime_owned() */
    llvm::DenseMap<const llvm::Value *, bool>      env_memo;
    /* module-private globals only ever assigned launch-environment pointers */
    llvm::DenseMap<const llvm::Value *, bool>      env_global;

    analysis_t(llvm::Function & f, const std::vector<base_t> & b, const env_t & e,
               llvm::ScalarEvolution & se, llvm::LoopInfo & li, llvm::DominatorTree & dt)
        : F(f), bases(b), env(e), SE(se), LI(li), ev(se, e)
    {
        ev.DT = &dt;
        DTp   = &dt;
        for (unsigned i = 0 ; i < bases.size() ; ++i)
            base_of[bases[i].v] = i;
        res.base.resize(bases.size());
    }

    /* Whether every store into the private global `G`, anywhere in the module,
     * stores a runtime-owned pointer. The OpenMP device runtime keeps the launch
     * environment pointer in such a (shared-memory) global after
     * __kmpc_target_init, and reloads it wherever it needs the reduction
     * counters or buffer. */
    bool env_carrying(const llvm::GlobalVariable * G)
    {
        if (auto it = env_global.find(G); it != env_global.end())
            return it->second;
        env_global[G] = false;           /* break cycles pessimistically */
        bool ok = true;
        for (const llvm::Function & H : *F.getParent())
            for (const llvm::Instruction & I : llvm::instructions(H))
                if (const auto * S = llvm::dyn_cast<llvm::StoreInst>(&I))
                    if (llvm::getUnderlyingObject(S->getPointerOperand()) == G)
                    {
                        const llvm::Value * V = S->getValueOperand();
                        if (!V->getType()->isPointerTy() ||
                            !(llvm::isa<llvm::ConstantPointerNull>(V) ||
                              runtime_owned(llvm::getUnderlyingObject(V))))
                            ok = false;
                    }
        env_global[G] = ok;
        return ok;
    }

    /* Whether `U` (an underlying object) is memory the runtime allocated for its
     * own use: the launch environment itself, or what a pointer read out of it,
     * directly or through an env-carrying global, addresses. */
    bool runtime_owned(const llvm::Value * U)
    {
        if (env.runtime_envs.empty() || U == nullptr)
            return false;
        for (const llvm::Value * R : env.runtime_envs)
            if (U == R)
                return true;
        const auto * L = llvm::dyn_cast<llvm::LoadInst>(U);
        if (L == nullptr)
            return false;
        if (auto it = env_memo.find(U); it != env_memo.end())
            return it->second;
        env_memo[U] = false;
        const llvm::Value * from = llvm::getUnderlyingObject(L->getPointerOperand());
        bool owned = runtime_owned(from);
        if (!owned && env.globals_private)
            if (const auto * G = llvm::dyn_cast<llvm::GlobalVariable>(from))
                owned = env_carrying(G);
        env_memo[U] = owned;
        return owned;
    }

    /* the private kind of an address space, if it is one */
    static bool space_kind(const llvm::Value * P, owner_kind_t & k)
    {
        const unsigned as = P->getType()->getPointerAddressSpace();
        if (as == 3) { k = owner_kind_t::SHARED; return true; }   /* NVPTX/AMDGPU shared */
        if (as == 5) { k = owner_kind_t::STACK;  return true; }   /* local/private */
        return false;
    }

    /* The single owner of the memory `P` may point to. */
    owner_t owner(const llvm::Value * P)
    {
        owner_kind_t k;
        if (space_kind(P, k))
            return {k, 0};

        llvm::SmallVector<const llvm::Value *, 4> objs;
        llvm::getUnderlyingObjects(P, objs, &LI);
        owner_t o = {owner_kind_t::STACK, 0};
        bool have_base = false;
        for (const llvm::Value * U : objs)
        {
            owner_t u;
            if (auto it = base_of.find(U); it != base_of.end())
                u = {owner_kind_t::BASE, it->second};
            else if (llvm::isa<llvm::AllocaInst>(U))
                u = {owner_kind_t::STACK, 0};
            else if (space_kind(U, k))
                u = {k, 0};
            else if (env.globals_private && llvm::isa<llvm::GlobalVariable>(U))
                u = {owner_kind_t::SYSTEM, 0};
            else if (runtime_owned(U))
                u = {owner_kind_t::SYSTEM, 0};    /* runtime scratch */
            else
                return {owner_kind_t::UNKNOWN, 0};

            if (u.kind == owner_kind_t::BASE)
            {
                if (have_base && o.base != u.base)
                    return {owner_kind_t::UNKNOWN, 0};   /* may be either: give up */
                o = u;
                have_base = true;
            }
            else if (!have_base && (int) u.kind > (int) o.kind)
                o = u;
        }
        return o;
    }

    /* Decompose `S` (times `scale`) into `f`; false when it is not exactly an
     * affine function of thread indices and loop counters. Narrowing casts are
     * refused; widening ones are accepted when the evaluator shows the narrow
     * value cannot have wrapped (then the wide value equals the math value). */
    bool decompose(const llvm::SCEV * S, wide_t scale, affine_t & f, unsigned depth = 0)
    {
        using namespace llvm;
        if (depth > 32)
            return false;
        switch (S->getSCEVType())
        {
            case scConstant:
            {
                const APInt & C = cast<SCEVConstant>(S)->getAPInt();
                if (C.getSignificantBits() > 63) return false;
                f.c += scale * (wide_t) C.getSExtValue();
                return true;
            }
            case scAddExpr:
                for (const SCEV * Op : cast<SCEVAddExpr>(S)->operands())
                    if (!decompose(Op, scale, f, depth + 1)) return false;
                return true;
            case scMulExpr:
            {
                wide_t k = scale;
                const SCEV * var = nullptr;
                for (const SCEV * Op : cast<SCEVMulExpr>(S)->operands())
                {
                    if (const auto * C = dyn_cast<SCEVConstant>(Op))
                    {
                        if (C->getAPInt().getSignificantBits() > 63) return false;
                        k *= (wide_t) C->getAPInt().getSExtValue();
                    }
                    else if (var == nullptr)
                        var = Op;
                    else
                        return false;   /* product of two variables */
                }
                if (k > WIDE_LIMIT || k < -WIDE_LIMIT) return false;
                return var ? decompose(var, k, f, depth + 1) : (f.c += k, true);
            }
            case scAddRecExpr:
            {
                const auto * AR = cast<SCEVAddRecExpr>(S);
                if (!AR->isAffine()) return false;
                const auto * St = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE));
                if (St == nullptr || St->getAPInt().getSignificantBits() > 63) return false;
                const SCEV * BTC = ev.btc(AR->getLoop());
                if (isa<SCEVCouldNotCompute>(BTC)) return false;
                const interval_t n = ev.eval(BTC);
                if (!n.ok || n.hi < 0) return false;
                if (!decompose(AR->getStart(), scale, f, depth + 1)) return false;
                /* the body sees counters 0..BTC; a use after the loop also BTC+1 */
                const bool inside = ev.ctx && AR->getLoop()->contains(ev.ctx);
                f.add(AR->getLoop(), scale * (wide_t) St->getAPInt().getSExtValue(),
                      n.hi + (inside ? 1 : 2));
                return true;
            }
            case scZeroExtend:
            case scSignExtend:
            {
                const SCEV * Op = cast<SCEVCastExpr>(S)->getOperand();
                const interval_t r = ev.eval(Op);
                const unsigned w = Op->getType()->getIntegerBitWidth();
                if (!r.ok || w >= 64) return false;
                const bool zext = S->getSCEVType() == scZeroExtend;
                const wide_t lo = zext ? 0 : -(((wide_t) 1) << (w - 1));
                const wide_t hi = zext ? (((wide_t) 1) << w) - 1 : (((wide_t) 1) << (w - 1)) - 1;
                if (r.lo < lo || r.hi > hi) return false;
                return decompose(Op, scale, f, depth + 1);
            }
            case scUnknown:
            {
                const Value * V = cast<SCEVUnknown>(S)->getValue();
                if (auto it = env.consts.find(V); it != env.consts.end())
                {
                    f.c += scale * (wide_t) it->second;
                    return true;
                }
                const auto * II = dyn_cast<IntrinsicInst>(V);
                if (II == nullptr) return false;
                const unsigned * dims = nullptr; unsigned d = 0;
                switch (II->getIntrinsicID())
                {
                    case Intrinsic::nvvm_read_ptx_sreg_tid_x:   dims = env.block; d = 0; break;
                    case Intrinsic::nvvm_read_ptx_sreg_tid_y:   dims = env.block; d = 1; break;
                    case Intrinsic::nvvm_read_ptx_sreg_tid_z:   dims = env.block; d = 2; break;
                    case Intrinsic::nvvm_read_ptx_sreg_ctaid_x: dims = env.grid;  d = 0; break;
                    case Intrinsic::nvvm_read_ptx_sreg_ctaid_y: dims = env.grid;  d = 1; break;
                    case Intrinsic::nvvm_read_ptx_sreg_ctaid_z: dims = env.grid;  d = 2; break;
                    default: return false;
                }
                if (dims[d] == 0) return false;
                f.add(thread_var(II->getIntrinsicID()), scale, dims[d]);
                return true;
            }
            default:
                return false;
        }
    }

    /* the variable key of a thread/block index (small integers, never a Loop*) */
    static const void * thread_var(unsigned intrinsic_id)
    {
        return reinterpret_cast<const void *>((uintptr_t) intrinsic_id + 1);
    }

    /* Exact affine byte offset of `P` from `base`, through pointer casts. */
    bool affine_offset(const llvm::Value * P, const llvm::Value * base, affine_t & f, unsigned depth = 0)
    {
        if (P == base)
            return true;
        if (depth > 8)
            return false;
        const llvm::SCEV * S = SE.getSCEV(const_cast<llvm::Value *>(P));
        const llvm::SCEV * PB = SE.getPointerBase(S);
        const llvm::SCEV * off = SE.removePointerBase(S);
        const auto * U = llvm::dyn_cast<llvm::SCEVUnknown>(PB);
        if (U == nullptr || llvm::isa<llvm::SCEVCouldNotCompute>(off) || !decompose(off, 1, f))
            return false;
        const llvm::Value * V = U->getValue();
        if (V == base)
            return true;
        if (const auto * C = llvm::dyn_cast<llvm::Operator>(V))
            if (C->getOpcode() == llvm::Instruction::AddrSpaceCast ||
                C->getOpcode() == llvm::Instruction::BitCast)
                return affine_offset(C->getOperand(0), base, f, depth + 1);
        return false;
    }

    /* Record an access for the grid-barrier analysis (no-op otherwise). */
    void record(const owner_t & o, bool mod, bool ref, bool bounded, wide_t lo, wide_t hi,
                wide_t size, const llvm::Value * P)
    {
        if (recs == nullptr)
            return ;
        access_rec_t r;
        r.region = region; r.owner = o; r.mod = mod; r.ref = ref;
        r.bounded = bounded; r.lo = lo; r.hi = hi; r.size = size;
        if (P && o.kind == owner_kind_t::BASE)
        {
            r.form.ok = affine_offset(P, bases[o.base].v, r.form);
            if (!r.form.ok) r.form = affine_t();
        }
        recs->push_back(std::move(r));
    }

    void widen(footprint_t & f, bool mod, bool ref, bool bounded, wide_t lo, wide_t hi)
    {
        f.mod |= mod;
        f.ref |= ref;
        /* An interval this wide only reflects the width of an index type
         * (e.g. a loaded 32-bit index scaled by 8): treat it as unbounded,
         * which it is for every purpose here. */
        if (!bounded || hi - lo > (((wide_t) 1) << 33))
        {
            f.bounded = false;
            return ;
        }
        if (lo < f.lo) f.lo = (int64_t) lo;
        if (hi > f.hi) f.hi = (int64_t) hi;
    }

    /* Record an access of `size` bytes (an interval, for memory intrinsics) at
     * pointer `P`. */
    void access(const llvm::Value * P, interval_t size, bool mod, bool ref)
    {
        ++res.accesses;
        const owner_t o = owner(P);
        if (owner_private(o.kind))
        {
            record(o, mod, ref, false, 0, 0, size.ok ? size.hi : 0, nullptr);
            return ;
        }
        if (o.kind == owner_kind_t::UNKNOWN)
        {
            widen(res.unknown, mod, ref, false, 0, 0);
            record(o, mod, ref, false, 0, 0, 0, nullptr);
            return ;
        }

        footprint_t & f = res.base[o.base];
        const llvm::SCEV * D = nullptr;
        interval_t off = offset_from(P, bases[o.base].v, &D, 0);
        if (!off.ok || !size.ok || size.lo < 0)
        {
            /* CGIR_NOALIAS_DEBUG=2: the offset the evaluator could not bound */
            static const bool debug2 = [] {
                const char * e = getenv("CGIR_NOALIAS_DEBUG");
                return e && atoi(e) >= 2;
            }();
            if (debug2)
            {
                std::string str;
                llvm::raw_string_ostream os(str);
                D->print(os);
                fprintf(stderr, "[cgir noalias]     unbounded offset of base %u in %s: %s\n",
                        o.base, F.getName().str().c_str(), str.c_str());
            }
            widen(f, mod, ref, false, 0, 0);
            record(o, mod, ref, false, 0, 0, size.ok ? size.hi : 0, P);
            return ;
        }
        widen(f, mod, ref, true, off.lo, off.hi + size.hi);
        record(o, mod, ref, true, off.lo, off.hi + size.hi, size.hi, P);
    }

    /* Byte offset of pointer `P` from `base`, bounded. ScalarEvolution splits
     * `P` into a pointer base and an integer offset; when the base is `base`
     * itself we are done, and when it is a cast of another pointer -- GPU code
     * reaches its data through addrspacecasts from the generic parameters to
     * the global address space, which ScalarEvolution cannot subtract across --
     * the offset of the cast's operand is added on. `D` gets the last offset
     * expression, for diagnostics. */
    interval_t offset_from(const llvm::Value * P, const llvm::Value * base,
                           const llvm::SCEV ** D, unsigned depth)
    {
        if (P == base)
            return interval_t::point(0);
        if (depth > 8)
            return interval_t::none();
        const llvm::SCEV * S = SE.getSCEV(const_cast<llvm::Value *>(P));
        const llvm::SCEV * PB = SE.getPointerBase(S);
        const llvm::SCEV * off = SE.removePointerBase(S);
        *D = off;
        const auto * U = llvm::dyn_cast<llvm::SCEVUnknown>(PB);
        if (U == nullptr || llvm::isa<llvm::SCEVCouldNotCompute>(off))
            return interval_t::none();
        const interval_t here = ev.eval(off);
        const llvm::Value * V = U->getValue();
        if (V == base)
            return here;
        if (const auto * C = llvm::dyn_cast<llvm::Operator>(V))
            if (C->getOpcode() == llvm::Instruction::AddrSpaceCast ||
                C->getOpcode() == llvm::Instruction::BitCast)
                return iadd(here, offset_from(C->getOperand(0), base, D, depth + 1));
        return interval_t::none();
    }

    interval_t value_interval(const llvm::Value * V)
    {
        if (const auto * C = llvm::dyn_cast<llvm::ConstantInt>(V))
            return C->getValue().getSignificantBits() <= 63
                ? interval_t::point(C->getSExtValue()) : interval_t::none();
        if (!V->getType()->isIntegerTy())
            return interval_t::none();
        return ev.eval(SE.getSCEV(const_cast<llvm::Value *>(V)));
    }

    /* Intrinsics that order or observe but do not access user memory. */
    static bool ignorable(const llvm::CallBase * CB)
    {
        const llvm::Function * callee = CB->getCalledFunction();
        if (callee == nullptr || !callee->isIntrinsic())
            return false;
        switch (callee->getIntrinsicID())
        {
            case llvm::Intrinsic::lifetime_start:
            case llvm::Intrinsic::lifetime_end:
            case llvm::Intrinsic::assume:
            case llvm::Intrinsic::experimental_noalias_scope_decl:
            case llvm::Intrinsic::invariant_start:
            case llvm::Intrinsic::invariant_end:
            case llvm::Intrinsic::prefetch:
            case llvm::Intrinsic::sideeffect:
            case llvm::Intrinsic::pseudoprobe:
                return true;
            default:
                break ;
        }
        const llvm::StringRef name = callee->getName();
        return name.starts_with("llvm.dbg.")
            || name.starts_with("llvm.nvvm.barrier")
            || name.starts_with("llvm.nvvm.bar.")
            || name.starts_with("llvm.nvvm.membar")
            || name.starts_with("llvm.nvvm.fence")
            || name.starts_with("llvm.nvvm.read.ptx.sreg")
            || name.starts_with("llvm.nvvm.vote")
            || name.starts_with("llvm.nvvm.shfl")
            || name.starts_with("llvm.nvvm.activemask")
            || name.starts_with("llvm.nvvm.match")
            || name.starts_with("llvm.amdgcn.s.barrier")
            || name.starts_with("llvm.amdgcn.workitem")
            || name.starts_with("llvm.amdgcn.workgroup");
    }

    void call(const llvm::CallBase * CB)
    {
        if (ignorable(CB))
            return ;
        if (const llvm::Function * callee = CB->getCalledFunction())
            if (callee->getName() == CGIR_GRID_BARRIER_MARKER)
                return ;

        if (const auto * MI = llvm::dyn_cast<llvm::MemIntrinsic>(CB))
        {
            const interval_t len = value_interval(MI->getLength());
            access(MI->getRawDest(), len, true, false);
            if (const auto * MT = llvm::dyn_cast<llvm::MemTransferInst>(MI))
                access(MT->getRawSource(), len, false, true);
            return ;
        }

        const llvm::MemoryEffects me = CB->getMemoryEffects();
        if (me.doesNotAccessMemory())
            return ;

        /* through its pointer arguments: anywhere relative to each */
        const llvm::ModRefInfo argmr = me.getModRef(llvm::IRMemLocation::ArgMem);
        if (llvm::isModOrRefSet(argmr))
            for (const llvm::Use & A : CB->args())
                if (A->getType()->isPointerTy())
                {
                    ++res.accesses;
                    const owner_t o = owner(A.get());
                    record(o, llvm::isModSet(argmr), llvm::isRefSet(argmr), false, 0, 0, 0, nullptr);
                    if (getenv("CGIR_NOALIAS_DEBUG") && atoi(getenv("CGIR_NOALIAS_DEBUG")) >= 2 &&
                        !owner_private(o.kind))
                    {
                        std::string str;
                        llvm::raw_string_ostream os(str);
                        CB->print(os);
                        fprintf(stderr, "[cgir noalias]     call reaching %s in %s: %s\n",
                                o.kind == owner_kind_t::BASE ? "a base" : "unknown memory",
                                F.getName().str().c_str(), str.c_str());
                    }
                    if (o.kind == owner_kind_t::BASE)
                        widen(res.base[o.base], llvm::isModSet(argmr), llvm::isRefSet(argmr), false, 0, 0);
                    else if (o.kind == owner_kind_t::UNKNOWN)
                        widen(res.unknown, llvm::isModSet(argmr), llvm::isRefSet(argmr), false, 0, 0);
                }

        /* anything else it may touch, except memory only it can see */
        const llvm::MemoryEffects other =
            me.getWithoutLoc(llvm::IRMemLocation::ArgMem)
              .getWithoutLoc(llvm::IRMemLocation::InaccessibleMem);
        if (!other.doesNotAccessMemory())
        {
            const llvm::ModRefInfo mr = other.getModRef();
            ++res.accesses;
            widen(res.unknown, llvm::isModSet(mr), llvm::isRefSet(mr), false, 0, 0);
            record({owner_kind_t::UNKNOWN, 0}, llvm::isModSet(mr), llvm::isRefSet(mr), false, 0, 0, 0, nullptr);
        }
    }

    void run(void)
    {
        const llvm::DataLayout & DL = F.getDataLayout();
        auto bytes = [&] (llvm::Type * T) {
            return interval_t::point((wide_t) DL.getTypeStoreSize(T).getKnownMinValue());
        };
        for (llvm::Instruction & I : llvm::instructions(F))
        {
            ev.set_context(&I);
            if (markers)
            {
                region = 0;
                for (const llvm::Instruction * M : *markers)
                    if (M != &I && DTp->dominates(M, &I))
                        ++region;
            }
            if (auto * L = llvm::dyn_cast<llvm::LoadInst>(&I))
                access(L->getPointerOperand(), bytes(L->getType()), false, true);
            else if (auto * S = llvm::dyn_cast<llvm::StoreInst>(&I))
                access(S->getPointerOperand(), bytes(S->getValueOperand()->getType()), true, false);
            else if (auto * A = llvm::dyn_cast<llvm::AtomicRMWInst>(&I))
                access(A->getPointerOperand(), bytes(A->getValOperand()->getType()), true, true);
            else if (auto * X = llvm::dyn_cast<llvm::AtomicCmpXchgInst>(&I))
                access(X->getPointerOperand(), bytes(X->getNewValOperand()->getType()), true, true);
            else if (auto * CB = llvm::dyn_cast<llvm::CallBase>(&I))
                call(CB);
            else if (I.mayReadOrWriteMemory() && !llvm::isa<llvm::FenceInst>(&I))
            {
                ++res.accesses;   /* va_arg and friends: anything */
                widen(res.unknown, I.mayWriteToMemory(), I.mayReadFromMemory(), false, 0, 0);
                record({owner_kind_t::UNKNOWN, 0}, I.mayWriteToMemory(), I.mayReadFromMemory(),
                       false, 0, 0, 0, nullptr);
            }
        }
    }
};

} /* anonymous namespace */

result_t
CGIR_NAMESPACE::noalias::analyze(llvm::Function & F, const std::vector<base_t> & bases, const env_t & env)
{
    llvm::TargetLibraryInfoImpl TLII(F.getParent()->getTargetTriple());
    llvm::TargetLibraryInfo     TLI(TLII);
    llvm::AssumptionCache       AC(F);
    llvm::DominatorTree         DT(F);
    llvm::LoopInfo              LI(DT);
    llvm::ScalarEvolution       SE(F, TLI, AC, DT, LI);

    analysis_t a(F, bases, env, SE, LI, DT);
    a.run();
    return std::move(a.res);
}

namespace {

/* Whether accesses `a` and `b` (same base pointer value) are made by the same
 * thread whenever they touch the same bytes -- see noalias-llvmir.hpp. */
static bool
same_thread(const access_rec_t & a, const access_rec_t & b, const env_t & env,
            const void * const thread_vars[6], const unsigned thread_dims[6])
{
    const affine_t & fa = a.form, & fb = b.form;
    if (!fa.ok || !fb.ok || fa.c != fb.c)
        return false;

    std::vector<std::pair<wide_t, wide_t>> terms;   /* (coef, n) */
    auto is_thread = [&] (const void * v) {
        for (int i = 0 ; i < 6 ; ++i) if (thread_vars[i] == v) return i;
        return -1;
    };
    /* thread indices: same coefficient on both sides, and every dimension of
     * size > 1 present, or two threads could share an address */
    for (int i = 0 ; i < 6 ; ++i)
    {
        const wide_t ca = fa.coef(thread_vars[i]), cb = fb.coef(thread_vars[i]);
        if (ca != cb)
            return false;
        if (thread_dims[i] > 1)
        {
            if (ca == 0)
                return false;
            terms.push_back({ca, thread_dims[i]});
        }
    }
    /* loop counters: matched by coefficient, ranges merged */
    std::vector<std::pair<wide_t, wide_t>> la, lb;
    for (size_t i = 0 ; i < fa.terms.size() ; ++i)
        if (is_thread(fa.terms[i].first) < 0 && fa.terms[i].second != 0)
            la.push_back({fa.terms[i].second, fa.range[i].second});
    for (size_t i = 0 ; i < fb.terms.size() ; ++i)
        if (is_thread(fb.terms[i].first) < 0 && fb.terms[i].second != 0)
            lb.push_back({fb.terms[i].second, fb.range[i].second});
    if (la.size() != lb.size())
        return false;
    std::sort(la.begin(), la.end());
    std::sort(lb.begin(), lb.end());
    for (size_t i = 0 ; i < la.size() ; ++i)
    {
        if (la[i].first != lb[i].first)
            return false;
        terms.push_back({la[i].first, std::max(la[i].second, lb[i].second)});
    }

    /* injectivity: a mixed-radix numbering with points at least an access apart */
    std::vector<std::pair<wide_t, wide_t>> t;
    for (const auto & x : terms)
        if (x.second > 1)
            t.push_back(x);
    std::sort(t.begin(), t.end());
    if (t.empty())
        return true;            /* a single thread: trivially the same */
    if (t[0].first < std::max(a.size, b.size) || a.size <= 0 || b.size <= 0)
        return false;
    for (size_t i = 0 ; i + 1 < t.size() ; ++i)
        if (t[i].first <= 0 || t[i + 1].first < t[i].first * t[i].second)
            return false;
    (void) env;
    return true;
}

/* The synchronization a pair of accesses across a marker needs. */
static barrier_t
pair_requirement(const access_rec_t & a, const access_rec_t & b,
                 const std::vector<base_t> & bases, const env_t & env,
                 const void * const thread_vars[6], const unsigned thread_dims[6])
{
    if (!(a.mod || b.mod))
        return BARRIER_NONE;
    const owner_kind_t ka = a.owner.kind, kb = b.owner.kind;
    if (ka == owner_kind_t::STACK || kb == owner_kind_t::STACK)
        return BARRIER_NONE;                         /* a thread's own memory */
    if (ka == owner_kind_t::SHARED || kb == owner_kind_t::SHARED)
        return (ka == kb) ? BARRIER_BLOCK : BARRIER_NONE;   /* shared vs global */
    if (ka == owner_kind_t::UNKNOWN || kb == owner_kind_t::UNKNOWN)
        return BARRIER_GRID;
    if (ka == owner_kind_t::SYSTEM || kb == owner_kind_t::SYSTEM)
        return (ka == kb) ? BARRIER_GRID : BARRIER_NONE;    /* never user data */

    /* two user buffers */
    const uint64_t va = bases[a.owner.base].addr, vb = bases[b.owner.base].addr;
    if (a.bounded && b.bounded)
    {
        const wide_t alo = (wide_t) va + a.lo, ahi = (wide_t) va + a.hi;
        const wide_t blo = (wide_t) vb + b.lo, bhi = (wide_t) vb + b.hi;
        if (std::max(alo, blo) >= std::min(ahi, bhi))
            return BARRIER_NONE;
    }
    if (va == vb && same_thread(a, b, env, thread_vars, thread_dims))
        return BARRIER_NONE;
    return BARRIER_GRID;
}

} /* anonymous namespace */

std::vector<barrier_t>
CGIR_NAMESPACE::noalias::barrier_requirements(llvm::Function & F,
                                              const std::vector<const llvm::Instruction *> & markers,
                                              const std::vector<base_t> & bases, const env_t & env)
{
    llvm::TargetLibraryInfoImpl TLII(F.getParent()->getTargetTriple());
    llvm::TargetLibraryInfo     TLI(TLII);
    llvm::AssumptionCache       AC(F);
    llvm::DominatorTree         DT(F);
    llvm::LoopInfo              LI(DT);
    llvm::ScalarEvolution       SE(F, TLI, AC, DT, LI);

    /* markers in program order: by how many of the others dominate each */
    std::vector<std::pair<unsigned, unsigned>> order;   /* (rank, index) */
    for (unsigned i = 0 ; i < markers.size() ; ++i)
    {
        unsigned rank = 0;
        for (const llvm::Instruction * M : markers)
            if (M != markers[i] && DT.dominates(M, markers[i]))
                ++rank;
        order.push_back({rank, i});
    }
    std::sort(order.begin(), order.end());

    std::vector<access_rec_t> recs;
    analysis_t a(F, bases, env, SE, LI, DT);
    a.markers = &markers;
    a.recs    = &recs;
    a.run();

    const unsigned id[6] = {
        llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x,   llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y,
        llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z,   llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x,
        llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y, llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z };
    const void * thread_vars[6];
    for (int i = 0 ; i < 6 ; ++i)
        thread_vars[i] = analysis_t::thread_var(id[i]);
    const unsigned thread_dims[6] = { env.block[0], env.block[1], env.block[2],
                                      env.grid[0],  env.grid[1],  env.grid[2] };
    for (int i = 0 ; i < 6 ; ++i)
        if (thread_dims[i] == 0)                 /* unknown launch: assume the worst */
            return std::vector<barrier_t>(markers.size(), BARRIER_GRID);

    /* Greedy, in program order: a removed (or block-level) barrier merges its
     * two regions, so a later barrier must also order everything since the
     * last grid barrier that was kept. */
    std::vector<barrier_t> out(markers.size(), BARRIER_GRID);
    unsigned open_from = 0;                       /* first region since the last kept grid barrier */
    for (unsigned r = 0 ; r < order.size() ; ++r)
    {
        const unsigned after = r + 1;             /* the region the marker opens */
        barrier_t need = BARRIER_NONE;
        for (const access_rec_t & x : recs)
        {
            if (x.region < open_from || x.region > r)
                continue ;
            for (const access_rec_t & y : recs)
            {
                if (y.region != after)
                    continue ;
                const barrier_t q = pair_requirement(x, y, bases, env, thread_vars, thread_dims);
                if (q > need) need = q;
                static const bool debug2 = [] {
                    const char * e = getenv("CGIR_NOALIAS_DEBUG");
                    return e && atoi(e) >= 2;
                }();
                if (debug2 && q == BARRIER_GRID)
                {
                    auto desc = [&] (const access_rec_t & r) {
                        static const char * kinds[] = { "stack", "shared", "system", "base", "unknown" };
                        char b[256];
                        int k = snprintf(b, sizeof(b), "%s%s %s", r.ref ? "R" : "", r.mod ? "W" : "",
                                         kinds[(int) r.owner.kind]);
                        if (r.owner.kind == owner_kind_t::BASE)
                        {
                            k += snprintf(b + k, sizeof(b) - k, "#%u size=%lld form=", r.owner.base, (long long) r.size);
                            if (!r.form.ok) k += snprintf(b + k, sizeof(b) - k, "none");
                            else
                            {
                                k += snprintf(b + k, sizeof(b) - k, "%lld", (long long) r.form.c);
                                for (size_t t = 0 ; t < r.form.terms.size() && k < 200 ; ++t)
                                    k += snprintf(b + k, sizeof(b) - k, " + %lld*v%llx[%lld]",
                                                  (long long) r.form.terms[t].second,
                                                  (unsigned long long) (uintptr_t) r.form.terms[t].first & 0xffff,
                                                  (long long) r.form.range[t].second);
                            }
                        }
                        return std::string(b);
                    };
                    fprintf(stderr, "[cgir barriers]   marker %u needs grid: region %u %s vs region %u %s\n",
                            r, x.region, desc(x).c_str(), y.region, desc(y).c_str());
                }
                if (need == BARRIER_GRID) break;
            }
            if (need == BARRIER_GRID) break;
        }
        out[order[r].second] = need;
        if (need == BARRIER_GRID)
            open_from = after;
    }
    return out;
}

bool
CGIR_NAMESPACE::noalias::may_conflict(const footprint_t & a, uint64_t va, const footprint_t & b, uint64_t vb)
{
    if (!a.accessed() || !b.accessed() || !(a.mod || b.mod))
        return false;
    if (!a.bounded || !b.bounded)
        return true;
    if (a.lo >= a.hi || b.lo >= b.hi)
        return false;   /* an empty range accesses nothing */
    const wide_t alo = (wide_t) va + a.lo, ahi = (wide_t) va + a.hi;
    const wide_t blo = (wide_t) vb + b.lo, bhi = (wide_t) vb + b.hi;
    return std::max(alo, blo) < std::min(ahi, bhi);
}

bool
CGIR_NAMESPACE::noalias::bounded_conflict(const footprint_t & a, uint64_t va, const footprint_t & b, uint64_t vb)
{
    return a.bounded && b.bounded && may_conflict(a, va, b, vb);
}

/* ---------------------------------------------------------------------------
 * Reporting
 * ------------------------------------------------------------------------- */

namespace {

struct report_t
{
    std::mutex mtx;
    unsigned   proven = 0, candidates = 0;           /* prove */
    unsigned   assumed = 0, verified = 0, unbounded = 0, overlapping = 0;  /* distinct-base */

    ~report_t()
    {
        if (!verify_level())
            return ;
        fprintf(stderr, "[cgir noalias] mode=%s", mode_name(mode()));
        if (candidates)
            fprintf(stderr, ": %u of %u candidates proven noalias", proven, candidates);
        if (assumed)
            fprintf(stderr, "%s %u assumed disjoint: %u proven, %u not proven (footprint unbounded), "
                            "%u not proven (bounded footprints overlap: possible violations)",
                    candidates ? ";" : ":", assumed, verified, unbounded, overlapping);
        if (!candidates && !assumed)
            fprintf(stderr, ": nothing to check");
        fputc('\n', stderr);
    }
};

static report_t &
report(void)
{
    static report_t r;
    return r;
}

} /* anonymous namespace */

void
CGIR_NAMESPACE::noalias::report_proven(unsigned proven, unsigned candidates)
{
    if (!verify_level())
        return ;
    std::lock_guard<std::mutex> lk(report().mtx);
    report().proven     += proven;
    report().candidates += candidates;
}

void
CGIR_NAMESPACE::noalias::report_assumption(const std::string & what,
                                           const footprint_t & a, uint64_t va,
                                           const footprint_t & b, uint64_t vb)
{
    if (!verify_level())
        return ;
    /* only a pair that could matter: both accessed, one of them written */
    if (!a.accessed() || !b.accessed() || !(a.mod || b.mod))
        return ;
    /* Footprints are over-approximations, so an overlap of bounds never
     * certifies an actual overlap: an assumption is either proven, or not. The
     * bounded-but-overlapping case is the one worth a look (a possible
     * violation); an unbounded footprint just could not be analysed. */
    const bool proven   = !may_conflict(a, va, b, vb);
    const bool overlap  = !proven && bounded_conflict(a, va, b, vb);
    {
        std::lock_guard<std::mutex> lk(report().mtx);
        report().assumed++;
        if (proven)       report().verified++;
        else if (overlap) report().overlapping++;
        else              report().unbounded++;
    }
    if (overlap)
    {
        fprintf(stderr, "cgir: noalias assumption not proven in %s (possible violation): "
                        "[%#llx+%lld, +%lld) %s vs [%#llx+%lld, +%lld) %s overlap\n", what.c_str(),
                (unsigned long long) va, (long long) a.lo, (long long) a.hi, a.mod ? "written" : "read",
                (unsigned long long) vb, (long long) b.lo, (long long) b.hi, b.mod ? "written" : "read");
        if (verify_level() > 1)
            abort();
    }
}

# endif /* CGIR_SUPPORT_LLVM */

/* ---------------------------------------------------------------------------
 * Self-test (tests/noalias-footprint.cc): small kernels whose exact footprints
 * are known, one per loop shape the evaluator reasons about. Every computed
 * footprint must CONTAIN the true one (soundness) and, for these shapes, equal
 * it (precision). Kept in the library so the test needs no LLVM of its own.
 * ------------------------------------------------------------------------- */

# if CGIR_SUPPORT_LLVM
#  include <llvm/AsmParser/Parser.h>
#  include <llvm/Support/SourceMgr.h>

namespace {

struct footprint_case_t
{
    const char * name;
    const char * ir;           /* defines @k */
    int64_t      scalar;       /* value of the i32/i64 parameter #1, if any */
    unsigned     grid, block;  /* launch dimensions (x) */
    int64_t      lo, hi;       /* exact byte footprint of pointer parameter #0 */
    bool         exact;        /* also require equality, not just containment */
};

static const footprint_case_t footprint_cases[] = {
    /* single-block loop, latch test on the incremented counter (the shape of
     * a specialized OpenMP chunk loop): i = ctaid*B + tid; do { a[i] } while
     * ((i += B) <= ctaid*B + B-1) -- one element per thread */
    { "latch-post-increment",
      "target triple = \"nvptx64-nvidia-cuda\"\n"
      "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
      "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n"
      "define void @k(ptr %a, i64 %unused) {\n"
      "e:\n"
      "  %t = call range(i32 0, 512) i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
      "  %c = call range(i32 0, 512) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n"
      "  %cb = shl nuw nsw i32 %c, 9\n"
      "  %ub = or disjoint i32 %cb, 511\n"
      "  %s = or disjoint i32 %t, %cb\n"
      "  br label %l\n"
      "l:\n"
      "  %i = phi i32 [ %s, %e ], [ %n, %l ]\n"
      "  %z = zext nneg i32 %i to i64\n"
      "  %p = getelementptr inbounds nuw [8 x i8], ptr %a, i64 %z\n"
      "  %q = addrspacecast ptr %p to ptr addrspace(1)\n"
      "  store double 0.0, ptr addrspace(1) %q\n"
      "  %n = add nuw nsw i32 %i, 512\n"
      "  %x = icmp samesign ult i32 %ub, %n\n"
      "  br i1 %x, label %out, label %l\n"
      "out:\n"
      "  ret void\n"
      "}\n",
      0, 512, 512, 0, 512 * 512 * 8, true },

    /* header-tested grid-stride loop: for (i = gtid; i < n; i += G*B) a[i] */
    { "header-grid-stride",
      "target triple = \"nvptx64-nvidia-cuda\"\n"
      "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
      "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n"
      "define void @k(ptr %a, i64 %n) {\n"
      "e:\n"
      "  %t = call range(i32 0, 128) i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
      "  %c = call range(i32 0, 64) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n"
      "  %cb = mul nuw nsw i32 %c, 128\n"
      "  %g = add nuw nsw i32 %cb, %t\n"
      "  %s = zext i32 %g to i64\n"
      "  br label %h\n"
      "h:\n"
      "  %i = phi i64 [ %s, %e ], [ %nx, %b ]\n"
      "  %x = icmp slt i64 %i, %n\n"
      "  br i1 %x, label %b, label %out\n"
      "b:\n"
      "  %p = getelementptr inbounds double, ptr %a, i64 %i\n"
      "  %v = load double, ptr %p\n"
      "  %nx = add nsw i64 %i, 8192\n"
      "  br label %h\n"
      "out:\n"
      "  ret void\n"
      "}\n",
      100000, 64, 128, 0, 100000 * 8, true },

    /* rotated (guarded do-while) loop over [0, n) */
    { "rotated-guarded",
      "define void @k(ptr %a, i64 %n) {\n"
      "e:\n"
      "  %g = icmp sgt i64 %n, 0\n"
      "  br i1 %g, label %l, label %out\n"
      "l:\n"
      "  %i = phi i64 [ 0, %e ], [ %nx, %l ]\n"
      "  %p = getelementptr inbounds i32, ptr %a, i64 %i\n"
      "  store i32 1, ptr %p\n"
      "  %nx = add nuw nsw i64 %i, 1\n"
      "  %x = icmp slt i64 %nx, %n\n"
      "  br i1 %x, label %l, label %out\n"
      "out:\n"
      "  ret void\n"
      "}\n",
      1000, 0, 0, 0, 1000 * 4, true },

    /* indirect access a[idx[0]]: the offset is a loaded value -> unbounded */
    { "indirect",
      "define void @k(ptr %a, ptr %idx) {\n"
      "e:\n"
      "  %j = load i64, ptr %idx\n"
      "  %p = getelementptr inbounds double, ptr %a, i64 %j\n"
      "  store double 0.0, ptr %p\n"
      "  ret void\n"
      "}\n",
      0, 0, 0, INT64_MIN, INT64_MAX, false },
};

} /* anonymous namespace */

extern "C" int
cgir_noalias_self_test(void)
{
    int failures = 0;
    for (const footprint_case_t & c : footprint_cases)
    {
        llvm::LLVMContext ctx;
        llvm::SMDiagnostic diag;
        std::unique_ptr<llvm::Module> M = llvm::parseAssemblyString(c.ir, diag, ctx);
        if (!M)
        {
            fprintf(stderr, "[%s] parse error: %s\n", c.name, diag.getMessage().str().c_str());
            ++failures;
            continue ;
        }
        llvm::Function * F = M->getFunction("k");
        env_t env;
        env.grid[0] = c.grid;   env.grid[1] = env.grid[2] = 1;
        env.block[0] = c.block; env.block[1] = env.block[2] = 1;
        if (F->arg_size() > 1 && F->getArg(1)->getType()->isIntegerTy())
            env.consts[F->getArg(1)] = c.scalar;
        std::vector<base_t> bases = { { F->getArg(0), 0x100000000ull } };
        const result_t r = analyze(*F, bases, env);
        const footprint_t & f = r.base[0];

        bool ok;
        if (c.lo == INT64_MIN)       /* expected unbounded */
            ok = f.accessed() && !f.bounded;
        else
        {
            const bool sound = f.accessed() && (!f.bounded || (f.lo <= c.lo && f.hi >= c.hi));
            const bool tight = f.bounded && f.lo == c.lo && f.hi == c.hi;
            ok = sound && (!c.exact || tight);
        }
        fprintf(stderr, "[%s] %s: footprint %s[%lld, %lld), expected [%lld, %lld)\n",
                ok ? "PASS" : "FAIL", c.name, f.bounded ? "" : "unbounded ",
                (long long) f.lo, (long long) f.hi, (long long) c.lo, (long long) c.hi);
        failures += !ok;
    }

    /* grid barriers between two regions of a fused kernel (grid 64, block 128) */
    struct barrier_case_t { const char * name; const char * body0; const char * body1; barrier_t expect; };
    static const char * head =
        "target triple = \"nvptx64-nvidia-cuda\"\n"
        "@sh = internal addrspace(3) global [128 x double] undef\n"
        "declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
        "declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n"
        "declare void @" CGIR_GRID_BARRIER_MARKER "() convergent nounwind\n"
        "define void @k(ptr %y, ptr %z, ptr %w) {\n"
        "e:\n"
        "  %t = call range(i32 0, 128) i32 @llvm.nvvm.read.ptx.sreg.tid.x()\n"
        "  %c = call range(i32 0, 64) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()\n"
        "  %cb = mul nuw nsw i32 %c, 128\n"
        "  %g32 = add nuw nsw i32 %cb, %t\n"
        "  %g = zext i32 %g32 to i64\n"
        "  %py = getelementptr inbounds double, ptr %y, i64 %g\n"
        "  %pz = getelementptr inbounds double, ptr %z, i64 %g\n"
        "  %t64 = zext i32 %t to i64\n"
        "  %ps = getelementptr inbounds [128 x double], ptr addrspace(3) @sh, i64 0, i64 %t64\n";
    static const barrier_case_t bcases[] = {
        { "elementwise", "  store double 1.0, ptr %py\n",
          "  %v = load double, ptr %py\n  store double %v, ptr %pz\n", BARRIER_NONE },
        { "neighbour", "  store double 1.0, ptr %py\n",
          "  %g1 = add i64 %g, 1\n  %pn = getelementptr inbounds double, ptr %y, i64 %g1\n"
          "  %v = load double, ptr %pn\n  store double %v, ptr %pz\n", BARRIER_GRID },
        { "broadcast", "  store double 1.0, ptr %py\n",
          "  %v = load double, ptr %y\n  store double %v, ptr %pz\n", BARRIER_GRID },
        { "shared", "  store double 1.0, ptr addrspace(3) %ps\n",
          "  %r = sub i64 127, %t64\n  %pr = getelementptr inbounds [128 x double], ptr addrspace(3) @sh, i64 0, i64 %r\n"
          "  %v = load double, ptr addrspace(3) %pr\n  store double %v, ptr %pz\n", BARRIER_BLOCK },
        { "disjoint", "  store double 1.0, ptr %py\n",
          "  %pw = getelementptr inbounds double, ptr %w, i64 %g\n  %v = load double, ptr %pw\n"
          "  store double %v, ptr %pz\n", BARRIER_NONE },
    };
    for (const barrier_case_t & c : bcases)
    {
        const std::string ir = std::string(head) + c.body0 +
            "  call void @" CGIR_GRID_BARRIER_MARKER "()\n" + c.body1 + "  ret void\n}\n";
        llvm::LLVMContext ctx;
        llvm::SMDiagnostic diag;
        std::unique_ptr<llvm::Module> M = llvm::parseAssemblyString(ir, diag, ctx);
        if (!M)
        {
            fprintf(stderr, "[FAIL] barrier %s: parse error: %s\n", c.name, diag.getMessage().str().c_str());
            ++failures;
            continue ;
        }
        llvm::Function * F = M->getFunction("k");
        std::vector<const llvm::Instruction *> ms;
        for (llvm::Instruction & I : llvm::instructions(*F))
            if (auto * CI = llvm::dyn_cast<llvm::CallInst>(&I))
                if (CI->getCalledFunction() && CI->getCalledFunction()->getName() == CGIR_GRID_BARRIER_MARKER)
                    ms.push_back(CI);
        env_t env;
        env.grid[0] = 64;   env.grid[1] = env.grid[2] = 1;
        env.block[0] = 128; env.block[1] = env.block[2] = 1;
        env.globals_private = true;
        /* y and z and w: 1 MiB apart, 64*128 doubles each */
        std::vector<base_t> bases = { { F->getArg(0), 0x10000000ull },
                                      { F->getArg(1), 0x10100000ull },
                                      { F->getArg(2), 0x10200000ull } };
        const std::vector<barrier_t> r = barrier_requirements(*F, ms, bases, env);
        const bool ok = r.size() == 1 && r[0] == c.expect;
        fprintf(stderr, "[%s] barrier %s: %d, expected %d\n", ok ? "PASS" : "FAIL", c.name,
                r.empty() ? -1 : (int) r[0], (int) c.expect);
        failures += !ok;
    }

    /* conflicts: two arrays one byte apart conflict, far apart they do not */
    footprint_t w; w.mod = true; w.lo = 0; w.hi = 4096;
    footprint_t rd; rd.ref = true; rd.lo = 0; rd.hi = 4096;
    if (may_conflict(w, 0x1000, rd, 0x2000) || !may_conflict(w, 0x1000, rd, 0x1FFF) ||
        may_conflict(rd, 0x1000, rd, 0x1000))
    {
        fprintf(stderr, "[FAIL] may_conflict\n");
        ++failures;
    }
    else
        fprintf(stderr, "[PASS] may_conflict\n");
    return failures;
}
# endif /* CGIR_SUPPORT_LLVM */
