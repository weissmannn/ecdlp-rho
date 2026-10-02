#!/usr/bin/env python3
"""Generate params.h for the ECDLP Pollard-rho solver.

The solver works in a prime-order subgroup of E'(F_p^2) : y^2 = x^3 + (b0 + b1 i),
i^2 = -1.  This script:

  * searches for a random prime-order instance (a "sample" curve and a small
    "toy" curve) using the CM method for j = 0 curves,
  * emits the curve / generator / target constants as little-endian 32-bit limbs,
  * emits cross-check vectors for F_p, F_p^2 and Jacobian point arithmetic,
  * (toy only) emits the known scalar so the solver can self-test in seconds.

Run it to (re)generate params.h; the seed makes the output reproducible.
"""
import math
import os
import random

random.seed(0x504F4C4C415244)  # "POLLARD"

# ======================================================================
#  prime-field helpers
# ======================================================================
def is_prime(n):
    if n < 2:
        return False
    for q in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % q == 0:
            return n == q
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(30):
        a = random.randrange(2, n - 1)
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def sqrt_mod(a, p):
    return pow(a, (p + 1) // 4, p)          # p % 4 == 3


def cm_reps(p):
    """Return (x, y) with x^2 + 3 y^2 = p (j = 0 CM)."""
    u = sqrt_mod(p - 3, p)
    if (u * u + 3) % p:
        return None
    a, b = p, min(u, p - u)
    lim = math.isqrt(p)
    while b > lim:
        a, b = b, a % b
    x = b
    t = p - x * x
    if t % 3:
        return None
    y2 = t // 3
    y = math.isqrt(y2)
    if y * y != y2:
        return None
    return x, y


def cm_candidates(p):
    rep = cm_reps(p)
    if not rep:
        return None
    x, y = rep
    t = 2 * x
    v = 2 * y
    t2 = t * t - 2 * p
    v2 = t * v
    return [p * p + 1 - t2, p * p + 1 + t2,
            p * p + 1 - (t2 + 3 * v2) // 2, p * p + 1 + ((t2 + 3 * v2) // 2),
            p * p + 1 - (t2 - 3 * v2) // 2, p * p + 1 + ((t2 - 3 * v2) // 2)]


# ======================================================================
#  F_p^2 arithmetic (i^2 = -1), parameterised by p
# ======================================================================
def gmul(A, B, p):
    a0, a1 = A
    b0, b1 = B
    return ((a0 * b0 - a1 * b1) % p, (a0 * b1 + a1 * b0) % p)


def gadd(A, B, p):
    return ((A[0] + B[0]) % p, (A[1] + B[1]) % p)


def gsub(A, B, p):
    return ((A[0] - B[0]) % p, (A[1] - B[1]) % p)


def gneg(A, p):
    return ((-A[0]) % p, (-A[1]) % p)


def gsqr(A, p):
    return gmul(A, A, p)


def gpow(a, e, p):
    r = (1, 0)
    while e > 0:
        if e & 1:
            r = gmul(r, a, p)
        a = gmul(a, a, p)
        e >>= 1
    return r


def tsqrt(z, p):
    """Tonelli-Shanks square root in F_p^2."""
    if z == (0, 0):
        return (0, 0)
    order = p * p
    if gpow(z, (order - 1) // 2, p) != (1, 0):
        return None
    s = 0
    Q = order - 1
    while Q % 2 == 0:
        Q //= 2
        s += 1
    k = 1
    while True:
        n = (k % p, 1)
        if gpow(n, (order - 1) // 2, p) == (p - 1, 0):
            break
        n = (1, k % p)
        if gpow(n, (order - 1) // 2, p) == (p - 1, 0):
            break
        k += 1
    M = s
    c = gpow(n, Q, p)
    t = gpow(z, Q, p)
    r = gpow(z, (Q + 1) // 2, p)
    while t != (1, 0):
        i = 0
        tt = t
        while tt != (1, 0):
            tt = gmul(tt, tt, p)
            i += 1
        b = c
        for _ in range(M - i - 1):
            b = gmul(b, b, p)
        M = i
        c = gmul(b, b, p)
        t = gmul(t, c, p)
        r = gmul(r, b, p)
    return r


def random_point(p, B):
    while True:
        x = (random.randrange(p), random.randrange(p))
        y3 = gmul(gsqr(x, p), x, p)
        rhs = ((y3[0] + B[0]) % p, (y3[1] + B[1]) % p)
        y = tsqrt(rhs, p)
        if y is not None:
            return (x, y)


# ======================================================================
#  affine group law over F_p^2, curve y^2 = x^3 + B
# ======================================================================
def aadd(Pt, Qt, p, B):
    if Pt is None:
        return Qt
    if Qt is None:
        return Pt
    x1, y1 = Pt
    x2, y2 = Qt
    if x1 == x2 and (y1[0] + y2[0]) % p == 0 and (y1[1] + y2[1]) % p == 0:
        return None
    if Pt == Qt:
        num = ((3 * (x1[0] * x1[0] - x1[1] * x1[1])) % p, (6 * x1[0] * x1[1]) % p)
        den = ((2 * y1[0]) % p, (2 * y1[1]) % p)
    else:
        num = ((y2[0] - y1[0]) % p, (y2[1] - y1[1]) % p)
        den = ((x2[0] - x1[0]) % p, (x2[1] - x1[1]) % p)
    dd = (den[0] * den[0] + den[1] * den[1]) % p
    inv = pow(dd, -1, p)
    dinv = ((den[0] * inv) % p, (-den[1] * inv) % p)
    lam = gmul(num, dinv, p)
    x3 = ((lam[0] * lam[0] - lam[1] * lam[1] - x1[0] - x2[0]) % p,
          (2 * lam[0] * lam[1] - x1[1] - x2[1]) % p)
    y3 = ((lam[0] * (x1[0] - x3[0]) - lam[1] * (x1[1] - x3[1]) - y1[0]) % p,
          (lam[0] * (x1[1] - x3[1]) + lam[1] * (x1[0] - x3[0]) - y1[1]) % p)
    return (x3, y3)


def amul(k, Pt, p, B):
    R = None
    while k > 0:
        if k & 1:
            R = aadd(R, Pt, p, B)
        Pt = aadd(Pt, Pt, p, B)
        k >>= 1
    return R


def oncurve(aff, p, B):
    x, y = aff
    return gsub(gsqr(y, p), gadd(gmul(gsqr(x, p), x, p), B, p), p) == (0, 0)


# ======================================================================
#  random prime-order instance
# ======================================================================
def find_curve(bits, B=None):
    """Find (p, order, B, generator) with a prime-order subgroup of E'(F_p^2)."""
    while True:
        p = random.getrandbits(bits) | (1 << (bits - 1)) | 3
        if p % 4 != 3 or p % 3 != 1 or not is_prime(p):
            continue
        cands = cm_candidates(p)
        if not cands:
            continue
        BB = B or (random.randrange(1, p), random.randrange(0, p))
        Q = random_point(p, BB)
        for N in cands:
            if N > 1 and amul(N, Q, p, BB) is None:
                if is_prime(N):
                    return p, N, BB, Q
                break


SAMPLE_BITS = 44
TOY_BITS = 20

p_s, ord_s, B_s, C1_s = find_curve(SAMPLE_BITS)
sk_s = random.randrange(1, ord_s)
C2_s = amul(sk_s, C1_s, p_s, B_s)

p_t, ord_t, B_t, C1_t = find_curve(TOY_BITS)
sk_t = random.randrange(1, ord_t)
C2_t = amul(sk_t, C1_t, p_t, B_t)

assert oncurve(C1_s, p_s, B_s) and oncurve(C2_s, p_s, B_s)
assert oncurve(C1_t, p_t, B_t) and oncurve(C2_t, p_t, B_t)
assert amul(ord_s, C1_s, p_s, B_s) is None and amul(ord_s, C2_s, p_s, B_s) is None
assert amul(ord_t, C1_t, p_t, B_t) is None and amul(ord_t, C2_t, p_t, B_t) is None

print("# sample p=%d bits=%d  order=%d bits=%d"
      % (p_s, p_s.bit_length(), ord_s, ord_s.bit_length()))
print("# toy    p=%d bits=%d  order=%d bits=%d  sk=%#x"
      % (p_t, p_t.bit_length(), ord_t, ord_t.bit_length(), sk_t))


# ======================================================================
#  emit params.h
# ======================================================================
def limbs(v, n=3):
    return [(v >> (32 * i)) & 0xFFFFFFFF for i in range(n)]


def macs(name, val, n=3):
    L = limbs(val, n)
    return "".join("#define %s_%d 0x%08xu\n" % (name, i, L[i]) for i in range(n))


def p0inv_of(mod):
    return (-pow(mod & 0xFFFFFFFF, -1, 1 << 32)) & 0xFFFFFFFF


def r2_of(mod):
    return pow(2, 192, mod)


def emit_curve(mod, order, r2, B, c1, c2, sk=None):
    s = []
    s.append(macs("P", mod))
    s.append("#define P0INV 0x%08xu\n" % p0inv_of(mod))
    s.append(macs("ORD", order))
    s.append(macs("M2", r2))
    s.append(macs("BRE", B[0]) + macs("BIM", B[1]))
    for nm, cc in (("C1", c1), ("C2", c2)):
        s.append(macs(nm + "X0", cc[0]) + macs(nm + "X1", cc[1]) +
                 macs(nm + "Y0", cc[2]) + macs(nm + "Y1", cc[3]))
    if sk is not None:
        s.append(macs("SK", sk))
    return "".join(s)


out = []
out.append("/* AUTO-GENERATED by gen.py -- do not edit */\n")
out.append("#pragma once\n#include <stdint.h>\n\n")

flat = lambda t: (t[0][0], t[0][1], t[1][0], t[1][1])   # ((x0,x1),(y0,y1)) -> 4

out.append("#ifdef USE_TOY\n#define IS_TOY 1\n")
out.append(emit_curve(p_t, ord_t, r2_of(p_t), B_t,
                      flat(C1_t), flat(C2_t), sk=sk_t))
out.append("#else\n")
out.append(emit_curve(p_s, ord_s, r2_of(p_s), B_s, flat(C1_s), flat(C2_s)))
out.append("#endif\n\n")


# ---- cross-check vectors (sample curve) ----
def vectors():
    v = []
    fm = []
    for _ in range(64):
        a = random.randrange(1, p_s)
        b = random.randrange(1, p_s)
        c = a * b % p_s
        fm += limbs(a) + limbs(b) + limbs(c)
    v.append("\n/* field mul vectors: a,b,c=a*b (3x3 limbs each) */\n")
    v.append("static const uint32_t VEC_FMUL[%d] = {%s};\n"
             % (len(fm), ", ".join("0x%08x" % x for x in fm)))

    gi = []
    for _ in range(16):
        a = random.randrange(1, p_s)
        gi += limbs(a) + limbs(pow(a, -1, p_s))
    v.append("static const uint32_t VEC_FINV[%d] = {%s};\n"
             % (len(gi), ", ".join("0x%08x" % x for x in gi)))

    gm = []
    for _ in range(32):
        a = (random.randrange(1, p_s), random.randrange(1, p_s))
        b = (random.randrange(1, p_s), random.randrange(1, p_s))
        c = gmul(a, b, p_s)
        gm += limbs(a[0]) + limbs(a[1]) + limbs(b[0]) + limbs(b[1]) + \
              limbs(c[0]) + limbs(c[1])
    v.append("static const uint32_t VEC_GMUL[%d] = {%s};\n"
             % (len(gm), ", ".join("0x%08x" % x for x in gm)))

    pv = []
    for _ in range(16):
        k = random.randrange(1, ord_s)
        aff = amul(k, C1_s, p_s, B_s)
        pv += limbs(k) + limbs(aff[0][0]) + limbs(aff[0][1]) + \
              limbs(aff[1][0]) + limbs(aff[1][1])
    v.append("static const uint32_t VEC_PMUL[%d] = {%s};\n"
             % (len(pv), ", ".join("0x%08x" % x for x in pv)))
    return "".join(v)


out.append(vectors())
hdr = "".join(out)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "params.h"), "w") as f:
    f.write(hdr)
print("wrote params.h (%d bytes)" % len(hdr))
