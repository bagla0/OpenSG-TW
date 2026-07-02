"""Independent SymPy derivation of the expanded RM shell curvature strains
(kappa_11, kappa_22, kappa_12+kappa_21) for the tapered 3-D SG, by substituting
omega_3 (thesis eq 4.19) into the curvature rows (thesis eq 4.18/4.20):

    row_i(alpha) = kappa_{1i} x_{1;a} + [ d/dzeta_a ]_total omega_i
                 = kappa_{1i} x_{1;a} + omega_i' x_{1;a} + (micro contour partial)_a omega_i

    kappa_11^s        = x_{i;2} row_i(1)
    kappa_22^s        = -x_{i;1} row_i(2)
    kappa_12+kappa_21 = x_{i;2} row_i(2) - x_{i;1} row_i(1)

Conventions: prime = d/dx1 (macro); |a = micro contour partial at fixed x1;
total contour derivative = micro + prime * x_{1;a}.  w'' and kappa'' dropped.
Also builds the "handwritten variant" (omega3' x_{1;a} + TOTAL omega3|a, i.e.
the p5 total substituted into the compact slot) and prints the difference.
"""
import sympy as sp

S = sp.Symbol
x2, x3 = S('x_2'), S('x_3')
I3, A2 = (1, 2, 3), (1, 2)

Xg = {(i, a): S('X%d%d' % (i, a)) for i in I3 for a in A2}          # x_{i;a}
X2g = {}                                                             # x_{i;a;b} (sym in a,b)
for i in I3:
    for a in A2:
        for b in A2:
            aa, bb = sorted((a, b))
            X2g[(i, a, b)] = S('X%d%d_%d' % (i, aa, bb))

k1 = {1: S('kap1'), 2: S('kap2'), 3: S('kap3')}                      # beam curvatures kappa_{1i}
k1p = S('kap1p')                                                     # kappa_1' (torsion rate)
w = {i: S('w%d' % i) for i in I3}
wp = {i: S('w%dp' % i) for i in I3}                                  # w_i'
wm = {(i, a): S('w%d_%d' % (i, a)) for i in I3 for a in A2}          # w_{i|a}
wpm = {(i, a): S('w%dp_%d' % (i, a)) for i in I3 for a in A2}        # w'_{i|a}
wmm = {}
for i in I3:
    for a in A2:
        for b in A2:
            aa, bb = sorted((a, b))
            wmm[(i, a, b)] = S('w%d_%d%d' % (i, aa, bb))             # w_{i|a|b}
om = {b: S('om%d' % b) for b in A2}
omp = {b: S('om%dp' % b) for b in A2}
omm = {(b, a): S('om%d_%d' % (b, a)) for b in A2 for a in A2}

# ---- derivative rule tables ----
def rules_total(a):
    r = {x2: Xg[(2, a)], x3: Xg[(3, a)], k1[1]: k1p * Xg[(1, a)], k1p: 0}
    for i in I3:
        for c in A2:
            r[Xg[(i, c)]] = X2g[(i, c, a)]
        r[w[i]] = wm[(i, a)] + wp[i] * Xg[(1, a)]
        r[wp[i]] = wpm[(i, a)]                       # + w'' X1a dropped (eps^3)
        for c in A2:
            r[wm[(i, c)]] = wmm[(i, c, a)] + wpm[(i, c)] * Xg[(1, a)]
            r[wpm[(i, c)]] = 0
    for b in A2:
        r[om[b]] = omm[(b, a)] + omp[b] * Xg[(1, a)]
        r[omp[b]] = 0
        for c in A2:
            r[omm[(b, c)]] = 0
    return r

def rules_micro(a):
    r = rules_total(a).copy()
    r[k1[1]] = 0
    for i in I3:
        r[w[i]] = wm[(i, a)]
        for c in A2:
            r[wm[(i, c)]] = wmm[(i, c, a)]
    for b in A2:
        r[om[b]] = omm[(b, a)]
    return r

def rules_prime():
    r = {x2: 0, x3: 0, k1[1]: k1p, k1p: 0}
    for i in I3:
        for c in A2:
            r[Xg[(i, c)]] = 0
            r[wm[(i, c)]] = wpm[(i, c)]
            r[wpm[(i, c)]] = 0
        r[w[i]] = wp[i]
        r[wp[i]] = 0                                  # w'' dropped
    for key in X2g:
        r[X2g[key]] = 0
    for b in A2:
        r[om[b]] = omp[b]
        r[omp[b]] = 0
        for c in A2:
            r[omm[(b, c)]] = 0
    for i in I3:
        for a in A2:
            for b in A2:
                r[wmm[(i, min(a, b), max(a, b))]] = 0
    return r

def D(expr, r):
    expr = sp.expand(expr)
    out = 0
    for s in expr.free_symbols:
        if s in r:
            out += sp.diff(expr, s) * r[s]
    return sp.expand(out)

# ---- geometry composites ----
C31 = Xg[(2, 1)] * Xg[(3, 2)] - Xg[(3, 1)] * Xg[(2, 2)]
C32 = Xg[(1, 2)] * Xg[(3, 1)] - Xg[(1, 1)] * Xg[(3, 2)]
C33 = Xg[(1, 1)] * Xg[(2, 2)] - Xg[(2, 1)] * Xg[(1, 2)]
C3 = {1: C31, 2: C32}
Rn1 = x2 * Xg[(3, 1)] - x3 * Xg[(2, 1)]
Rn2 = x2 * Xg[(3, 2)] - x3 * Xg[(2, 2)]

# ---- omega_3 (thesis eq 4.19) ----
om3 = (k1[1] * (Xg[(1, 1)] * Rn2 - Xg[(1, 2)] * Rn1)
       + sum(wp[i] * (Xg[(1, 1)] * Xg[(i, 2)] - Xg[(1, 2)] * Xg[(i, 1)]) for i in I3)
       + sum(wm[(i, 1)] * Xg[(i, 2)] - wm[(i, 2)] * Xg[(i, 1)] for i in I3)) / (2 * C33) \
      - (C31 * om[1] + C32 * om[2]) / C33

rp = rules_prime()
om3p = D(om3, rp)
om3m = {a: D(om3, rules_micro(a)) for a in A2}
om3t = {a: D(om3, rules_total(a)) for a in A2}

# sanity: total = micro + prime * X1a
for a in A2:
    chk = sp.simplify(om3t[a] - om3m[a] - om3p * Xg[(1, a)])
    assert chk == 0, ("total/micro/prime split broken", a, chk)
print("[ok] om3 total = micro + om3' * X1a   (a=1,2)")

# ---- curvature rows ----
def row(i, a, om3_slot):
    if i == 3:
        fluct = om3_slot[a]
    else:
        fluct = omp[i] * Xg[(1, a)] + omm[(i, a)]
    return k1[i] * Xg[(1, a)] + fluct

slot_correct = {a: om3p * Xg[(1, a)] + om3m[a] for a in A2}   # = total, single count
slot_hand = {a: om3p * Xg[(1, a)] + om3t[a] for a in A2}      # p5 total + extra om3' X1a

def kappas(slot):
    k11s = sum(Xg[(i, 2)] * row(i, 1, slot) for i in I3)
    k22s = -sum(Xg[(i, 1)] * row(i, 2, slot) for i in I3)
    k1221 = sum(Xg[(i, 2)] * row(i, 2, slot) for i in I3) \
        - sum(Xg[(i, 1)] * row(i, 1, slot) for i in I3)
    return k11s, k22s, k1221

k11_c, k22_c, k1221_c = kappas(slot_correct)
k11_h, k22_h, k1221_h = kappas(slot_hand)

print("\n[diff] handwritten-variant minus correct:")
print("  k11 :", sp.factor(sp.simplify(k11_h - k11_c)))
print("  k22 :", sp.factor(sp.simplify(k22_h - k22_c)))
print("  k12+k21:", sp.factor(sp.simplify(k1221_h - k1221_c)))

# ---- collect correct expansions by field symbol ----
fields = ([k1[i] for i in I3] + [k1p]
          + [wp[i] for i in I3]
          + [wpm[(i, a)] for i in I3 for a in A2]
          + [wm[(i, a)] for i in I3 for a in A2]
          + sorted(set(wmm.values()), key=str)
          + [om[b] for b in A2] + [omp[b] for b in A2]
          + [omm[(b, a)] for b in A2 for a in A2])

def report(name, expr):
    print("\n===== %s =====" % name)
    expr = sp.expand(expr)
    resid = expr
    for f in fields:
        c = expr.coeff(f, 1)
        if c != 0:
            c2 = sp.simplify(sp.factor(c))
            print("  %-8s : %s" % (str(f), c2))
            resid = sp.expand(resid - f * c)
    resid = sp.simplify(resid)
    print("  residual :", resid)
    assert resid == 0

report("kappa_11^s  (correct)", k11_c)
report("kappa_22^s  (correct)", k22_c)
report("kappa_12^s+kappa_21^s  (correct)", k1221_c)

# ---- prismatic reduction sanity vs thesis eq 4.23 ----
xd2, xd3 = S('xd2'), S('xd3')
prism = {Xg[(1, 1)]: 1, Xg[(1, 2)]: 0, Xg[(2, 1)]: 0, Xg[(3, 1)]: 0,
         Xg[(2, 2)]: xd2, Xg[(3, 2)]: xd3}
for i in I3:
    for a in A2:
        for b in A2:
            aa, bb = sorted((a, b))
            if (i, aa, bb) in X2g:
                pass
prism2 = {}
for key, s in X2g.items():
    i, a, b = key
    if i == 1:
        prism2[s] = 0
    elif (a, b) == (2, 2):
        prism2[s] = S('xdd%d' % i)
    else:
        prism2[s] = 0
# micro-1 fields vanish for the 2-D SG (contour only along zeta2)
zero1 = {}
for i in I3:
    zero1[wm[(i, 1)]] = 0
    zero1[wpm[(i, 1)]] = 0
    zero1[wmm[(i, 1, 1)]] = 0
    zero1[wmm[(i, 1, 2)]] = 0
for b in A2:
    zero1[omm[(b, 1)]] = 0

k11_pr = sp.simplify(sp.expand(k11_c.subs(prism).subs(prism2).subs(zero1)))
k11_pr = sp.collect(k11_pr, [k1p, wpm[(1, 2)], omp[2]])
print("\n===== prismatic kappa_11^s (correct convention) =====")
print(sp.simplify(k11_pr))
print("thesis 4.23:  xd_eta*kap_eta + om2'/xd2 - (xd3/(2 xd2)) w1'|2 + (xd3/xd2) kap1' Rn   [arc-length: xd2^2+xd3^2=1]")
arc = sp.Symbol('one')  # substitute xd3**2 -> 1 - xd2**2 to compare
k11_pr_arc = sp.simplify(k11_pr.subs(xd3**2, 1 - xd2**2))
print("with arc-length substitution xd3^2 = 1 - xd2^2:")
print(sp.expand(k11_pr_arc))
