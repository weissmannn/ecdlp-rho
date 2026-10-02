/* rho.cu -- parallel Pollard-rho discrete logarithm in a prime-order subgroup.
 *
 * Group : E'(F_p^2) : y^2 = x^3 + (b0 + b1 i),  i^2 = -1,  order ORD (prime).
 * Target: find k with  C2 = [k] C1.
 *
 * One source, three builds:
 *   clang++ -O3 -DNO_CUDA rho.cu -o rho_cpu     (validate on any host)
 *   nvcc    -O3 -DUSE_TOY  rho.cu -o rho_toy    (seconds-long self-test)
 *   nvcc    -O3            rho.cu -o rho        (full-size run)
 *
 * Usage: rho selftest | rho cpu [D] | rho gpu [D]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
#define RHO_MKDIR(d) _mkdir(d)
#else
#include <sys/stat.h>
#include <sys/types.h>
#define RHO_MKDIR(d) mkdir((d),0777)
#endif

#include "params.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <thread>
#include <atomic>
#include <math.h>
#define HD __host__ __device__
#else
#define HD
#endif

#define NL 3
#ifdef USE_TOY
#define WALK_MAX 1000000000LL
#else
#define WALK_MAX 20000000
#endif

/* ================================================================== */
/* F_p : 3 x 32-bit limbs, Montgomery form, R = 2^96                   */
/* ================================================================== */
typedef struct { uint32_t v[NL]; } fp;
typedef struct { fp a, b; } fp2;        /* a + b*i, i^2 = -1           */
typedef struct { uint32_t v[NL]; } sc;  /* scalar mod ORD              */

HD static inline void fp_zero(fp *x){ x->v[0]=x->v[1]=x->v[2]=0; }
HD static inline void fp_one (fp *x){ x->v[0]=1; x->v[1]=0; x->v[2]=0; }
HD static inline int  fp_is0(const fp *x){ return (x->v[0]|x->v[1]|x->v[2])==0; }
HD static inline int  fp_eq (const fp *a,const fp *b){
    return a->v[0]==b->v[0] && a->v[1]==b->v[1] && a->v[2]==b->v[2]; }

HD static inline int fp_ge_p(const fp *a){
    if (a->v[2] != P_2) return a->v[2] > P_2;
    if (a->v[1] != P_1) return a->v[1] > P_1;
    return a->v[0] >= P_0;
}
HD static inline void fp_sub_p(fp *a){
    uint64_t b=0,s;
    s=(uint64_t)a->v[0]-P_0;      a->v[0]=(uint32_t)s; b=(s>>63)&1;
    s=(uint64_t)a->v[1]-P_1-b;    a->v[1]=(uint32_t)s; b=(s>>63)&1;
    s=(uint64_t)a->v[2]-P_2-b;    a->v[2]=(uint32_t)s;
}
HD static inline void fp_add(fp *r,const fp *a,const fp *b){
    uint64_t c=0,s;
    s=(uint64_t)a->v[0]+b->v[0];       r->v[0]=(uint32_t)s; c=s>>32;
    s=(uint64_t)a->v[1]+b->v[1]+c;     r->v[1]=(uint32_t)s; c=s>>32;
    s=(uint64_t)a->v[2]+b->v[2]+c;     r->v[2]=(uint32_t)s;
    if (fp_ge_p(r)) fp_sub_p(r);
}
HD static inline void fp_sub(fp *r,const fp *a,const fp *b){
    uint64_t bb=0,s;
    s=(uint64_t)a->v[0]-b->v[0];       r->v[0]=(uint32_t)s; bb=(s>>63)&1;
    s=(uint64_t)a->v[1]-b->v[1]-bb;    r->v[1]=(uint32_t)s; bb=(s>>63)&1;
    s=(uint64_t)a->v[2]-b->v[2]-bb;    r->v[2]=(uint32_t)s; bb=(s>>63)&1;
    if (bb){
        uint64_t c=0;
        s=(uint64_t)r->v[0]+P_0;   r->v[0]=(uint32_t)s; c=s>>32;
        s=(uint64_t)r->v[1]+P_1+c; r->v[1]=(uint32_t)s; c=s>>32;
        s=(uint64_t)r->v[2]+P_2+c; r->v[2]=(uint32_t)s;
    }
}
HD static inline void fp3(fp *r,const fp *a){ fp t; fp_add(&t,a,a); fp_add(r,&t,a); }

/* Montgomery multiply: r = a*b*R^-1 mod P */
HD static inline void fp_mul(fp *r,const fp *a,const fp *b){
    uint32_t t[2*NL+4];
    for (int i=0;i<2*NL+4;i++) t[i]=0;
    uint32_t ai, m; uint64_t carry, cc, s; int i, j, k, q;
    for (i=0;i<NL;i++){
        ai=a->v[i]; carry=0;
        for (j=0;j<NL;j++){
            s=(uint64_t)t[i+j]+(uint64_t)ai*b->v[j]+carry;
            t[i+j]=(uint32_t)s; carry=s>>32;
        }
        k=i+NL; { uint32_t c=(uint32_t)carry;
            for (q=0;q<5;q++){ s=(uint64_t)t[k+q]+c; t[k+q]=(uint32_t)s; c=(uint32_t)(s>>32); } }
    }
    for (i=0;i<NL;i++){
        m=(uint32_t)((uint64_t)t[i]*P0INV);
        cc=0;
        for (j=0;j<NL;j++){
            uint64_t pj = (j==0)?P_0:((j==1)?P_1:P_2);
            s=(uint64_t)t[i+j]+(uint64_t)m*pj+cc;
            t[i+j]=(uint32_t)s; cc=s>>32;
        }
        k=i+NL; { uint32_t c=(uint32_t)cc;
            for (q=0;q<5;q++){ s=(uint64_t)t[k+q]+c; t[k+q]=(uint32_t)s; c=(uint32_t)(s>>32); } }
    }
    r->v[0]=t[NL]; r->v[1]=t[NL+1]; r->v[2]=t[NL+2];
    if (fp_ge_p(r)) fp_sub_p(r);
}
HD static inline void to_mont(fp *r,const fp *a){
    fp m2; m2.v[0]=M2_0; m2.v[1]=M2_1; m2.v[2]=M2_2;
    fp_mul(r,a,&m2);
}
HD static inline void from_mont(fp *r,const fp *a){
    fp one; fp_one(&one); fp_mul(r,a,&one);
}
/* Montgomery power: returns mont(x^k) for a = mont(x). acc starts at mont(1)=R. */
HD static inline void fp_pow(fp *r,const fp *a,const uint32_t e[NL]){
    fp base=*a, acc, oneRaw;
    fp_one(&oneRaw); to_mont(&acc,&oneRaw);        /* acc = R */
    int started=0;
    for (int bit=NL*32-1; bit>=0; --bit){
        if (started) fp_mul(&acc,&acc,&acc);
        if ((e[bit>>5]>>(bit&31))&1u){
            if (started) fp_mul(&acc,&acc,&base);
            else { acc=base; started=1; }
        }
    }
    *r=acc;
}
HD static inline void fp_inv_eea(fp *r,const fp *a){
    if (fp_is0(a)){ fp_zero(r); return; }
    fp u=*a, v, x1, x2, Pv;
    v.v[0]=P_0; v.v[1]=P_1; v.v[2]=P_2;
    Pv=v;
    fp_one(&x1); fp_zero(&x2);
    while (!(u.v[0]==1 && u.v[1]==0 && u.v[2]==0) &&
           !(v.v[0]==1 && v.v[1]==0 && v.v[2]==0)){
        while ((u.v[0]&1u)==0){
            u.v[0]=(u.v[0]>>1)|(u.v[1]<<31);
            u.v[1]=(u.v[1]>>1)|(u.v[2]<<31);
            u.v[2]=(u.v[2]>>1);
            if ((x1.v[0]&1u)==0){
                x1.v[0]=(x1.v[0]>>1)|(x1.v[1]<<31);
                x1.v[1]=(x1.v[1]>>1)|(x1.v[2]<<31);
                x1.v[2]=(x1.v[2]>>1);
            } else {
                uint64_t c=0,s;
                s=(uint64_t)x1.v[0]+P_0;   x1.v[0]=(uint32_t)s; c=s>>32;
                s=(uint64_t)x1.v[1]+P_1+c; x1.v[1]=(uint32_t)s; c=s>>32;
                s=(uint64_t)x1.v[2]+P_2+c; x1.v[2]=(uint32_t)s;
                x1.v[0]=(x1.v[0]>>1)|(x1.v[1]<<31);
                x1.v[1]=(x1.v[1]>>1)|(x1.v[2]<<31);
                x1.v[2]=(x1.v[2]>>1);
            }
        }
        while ((v.v[0]&1u)==0){
            v.v[0]=(v.v[0]>>1)|(v.v[1]<<31);
            v.v[1]=(v.v[1]>>1)|(v.v[2]<<31);
            v.v[2]=(v.v[2]>>1);
            if ((x2.v[0]&1u)==0){
                x2.v[0]=(x2.v[0]>>1)|(x2.v[1]<<31);
                x2.v[1]=(x2.v[1]>>1)|(x2.v[2]<<31);
                x2.v[2]=(x2.v[2]>>1);
            } else {
                uint64_t c=0,s;
                s=(uint64_t)x2.v[0]+P_0;   x2.v[0]=(uint32_t)s; c=s>>32;
                s=(uint64_t)x2.v[1]+P_1+c; x2.v[1]=(uint32_t)s; c=s>>32;
                s=(uint64_t)x2.v[2]+P_2+c; x2.v[2]=(uint32_t)s;
                x2.v[0]=(x2.v[0]>>1)|(x2.v[1]<<31);
                x2.v[1]=(x2.v[1]>>1)|(x2.v[2]<<31);
                x2.v[2]=(x2.v[2]>>1);
            }
        }
        int uge;
        if (u.v[2]!=v.v[2]) uge=u.v[2]>v.v[2];
        else if (u.v[1]!=v.v[1]) uge=u.v[1]>v.v[1];
        else uge=u.v[0]>=v.v[0];
        if (uge){ fp_sub(&u,&u,&v); fp_sub(&x1,&x1,&x2); }
        else    { fp_sub(&v,&v,&u); fp_sub(&x2,&x2,&x1); }
    }
    if (u.v[0]==1 && u.v[1]==0 && u.v[2]==0) *r=x1;
    else *r=x2;
    {   /* convert plain inverse to Montgomery inverse: multiply by R^2 */
        fp m2,tmp;
        m2.v[0]=M2_0; m2.v[1]=M2_1; m2.v[2]=M2_2;
        fp_mul(&tmp,r,&m2);
        fp_mul(r,&tmp,&m2);
    }
    (void)Pv;
}
/* Fermat inversion: returns mont(a^-1) given mont(a). Branchless (fixed exponent). */
HD static inline void fp_inv_fermat(fp *r,const fp *a){
    uint32_t e[NL]; uint64_t bb=0,s;
    s=(uint64_t)P_0-2;   e[0]=(uint32_t)s; bb=(s>>63)&1;
    s=(uint64_t)P_1-bb;  e[1]=(uint32_t)s; bb=(s>>63)&1;
    s=(uint64_t)P_2-bb;  e[2]=(uint32_t)s;
    if (fp_is0(a)){ fp_zero(r); return; }
    fp_pow(r,a,e);
}
#define INV_EEA 1
#define INV_FERMAT 2
#ifndef INV_MODE
#define INV_MODE INV_FERMAT
#endif
HD static inline void fp_inv(fp *r,const fp *a){
#if INV_MODE==INV_FERMAT
    fp_inv_fermat(r,a);
#else
    fp_inv_eea(r,a);
#endif
}

/* ---- F_p2 ---- */
HD static inline void fp2_zero(fp2 *x){ fp_zero(&x->a); fp_zero(&x->b); }
HD static inline int  fp2_eq (const fp2 *x,const fp2 *y){ return fp_eq(&x->a,&y->a)&&fp_eq(&x->b,&y->b); }
HD static inline void fp2_add(fp2 *r,const fp2 *x,const fp2 *y){ fp_add(&r->a,&x->a,&y->a); fp_add(&r->b,&x->b,&y->b); }
HD static inline void fp2_sub(fp2 *r,const fp2 *x,const fp2 *y){ fp_sub(&r->a,&x->a,&y->a); fp_sub(&r->b,&x->b,&y->b); }
HD static inline void fp2_neg(fp2 *r,const fp2 *x){
    fp z; fp_zero(&z);
    fp_sub(&r->a,&z,&x->a); fp_sub(&r->b,&z,&x->b);
}
HD static inline void fp2_3  (fp2 *r,const fp2 *x){ fp3(&r->a,&x->a); fp3(&r->b,&x->b); }
HD static inline void fp2_mul(fp2 *r,const fp2 *x,const fp2 *y){
    fp t0,t1,t2,sa,sb,re;
    fp_mul(&t0,&x->a,&y->a);          /* a0b0 */
    fp_mul(&t1,&x->b,&y->b);          /* a1b1 */
    fp_add(&sa,&x->a,&x->b);
    fp_add(&sb,&y->a,&y->b);
    fp_mul(&t2,&sa,&sb);              /* (a0+a1)(b0+b1) */
    fp_sub(&t2,&t2,&t0);
    fp_sub(&t2,&t2,&t1);
    fp_sub(&re,&t0,&t1);              /* real = a0b0 - a1b1 */
    r->a=re; r->b=t2;                 /* imag = a0b1 + a1b0 */
}
HD static inline void fp2_sqr(fp2 *r,const fp2 *x){ fp2_mul(r,x,x); }
HD static inline void fp2_inv(fp2 *r,const fp2 *x){
    fp nrm,t;
    fp2 nb;
    fp_mul(&nrm,&x->a,&x->a);
    fp_mul(&t,&x->b,&x->b);
    fp_add(&nrm,&nrm,&t);             /* norm = a0^2 + a1^2 */
    fp_inv(&nrm,&nrm);
    fp_mul(&r->a,&x->a,&nrm);
    fp_mul(&nb.b,&x->b,&nrm);
    fp_zero(&nb.a);
    fp2_neg(&nb,&nb);
    r->b=nb.b;                        /* -a1/norm */
}

/* ================================================================== */
/* Scalar mod ORD (3 limbs)                                             */
/* ================================================================== */
HD static inline int sc_ge_ord(const uint32_t a[NL]){
    if (a[2]!=ORD_2) return a[2]>ORD_2;
    if (a[1]!=ORD_1) return a[1]>ORD_1;
    return a[0]>=ORD_0;
}
HD static inline void sc_sub_ord(uint32_t a[NL]){
    uint64_t b=0,s;
    s=(uint64_t)a[0]-ORD_0;   a[0]=(uint32_t)s; b=(s>>63)&1;
    s=(uint64_t)a[1]-ORD_1-b; a[1]=(uint32_t)s; b=(s>>63)&1;
    s=(uint64_t)a[2]-ORD_2-b; a[2]=(uint32_t)s;
}
HD static inline void sc_add(sc *r,const sc *x,const sc *y){
    uint64_t c=0,s;
    s=(uint64_t)x->v[0]+y->v[0];       r->v[0]=(uint32_t)s; c=s>>32;
    s=(uint64_t)x->v[1]+y->v[1]+c;     r->v[1]=(uint32_t)s; c=s>>32;
    s=(uint64_t)x->v[2]+y->v[2]+c;     r->v[2]=(uint32_t)s;
    if (sc_ge_ord(r->v)) sc_sub_ord(r->v);
}
HD static inline void sc_sub(sc *r,const sc *x,const sc *y){
    uint64_t bb=0,s;
    s=(uint64_t)x->v[0]-y->v[0];       r->v[0]=(uint32_t)s; bb=(s>>63)&1;
    s=(uint64_t)x->v[1]-y->v[1]-bb;    r->v[1]=(uint32_t)s; bb=(s>>63)&1;
    s=(uint64_t)x->v[2]-y->v[2]-bb;    r->v[2]=(uint32_t)s; bb=(s>>63)&1;
    if (bb){
        uint64_t c=0;
        s=(uint64_t)r->v[0]+ORD_0;   r->v[0]=(uint32_t)s; c=s>>32;
        s=(uint64_t)r->v[1]+ORD_1+c; r->v[1]=(uint32_t)s; c=s>>32;
        s=(uint64_t)r->v[2]+ORD_2+c; r->v[2]=(uint32_t)s;
    }
}
HD static inline void sc_reduce(uint32_t a[NL]){
    uint32_t res[NL]; res[0]=res[1]=res[2]=0;
    for (int bit=NL*32-1; bit>=0; --bit){
        uint32_t bitv=(a[bit>>5]>>(bit&31))&1u;
        res[2]=(res[2]<<1)|(res[1]>>31);
        res[1]=(res[1]<<1)|(res[0]>>31);
        res[0]=(res[0]<<1)|bitv;
        if (sc_ge_ord(res)) sc_sub_ord(res);
    }
    a[0]=res[0]; a[1]=res[1]; a[2]=res[2];
}

static void mulmod_ord(uint32_t r[NL],const uint32_t a[NL],const uint32_t b[NL]){
    uint32_t t[2*NL+2]; int i,j;
    for (i=0;i<2*NL+2;i++) t[i]=0;
    for (i=0;i<NL;i++){
        uint64_t carry=0;
        for (j=0;j<NL;j++){
            uint64_t s=(uint64_t)t[i+j]+(uint64_t)a[i]*b[j]+carry;
            t[i+j]=(uint32_t)s; carry=s>>32;
        }
        uint64_t c=carry; int k=i+NL;
        while (c){ uint64_t s=(uint64_t)t[k]+c; t[k]=(uint32_t)s; c=s>>32; k++; }
    }
    uint32_t res[NL]; res[0]=res[1]=res[2]=0;
    for (int bit=(2*NL+2)*32-1; bit>=0; --bit){
        uint32_t bitv=(t[bit>>5]>>(bit&31))&1u;
        res[2]=(res[2]<<1)|(res[1]>>31);
        res[1]=(res[1]<<1)|(res[0]>>31);
        res[0]=(res[0]<<1)|bitv;
        if (sc_ge_ord(res)) sc_sub_ord(res);
    }
    r[0]=res[0]; r[1]=res[1]; r[2]=res[2];
}
static void sc_inv(sc *r,const sc *a){
    uint32_t e[NL]; uint64_t bb=0,s;
    s=(uint64_t)ORD_0-2;  e[0]=(uint32_t)s; bb=(s>>63)&1;
    s=(uint64_t)ORD_1-bb; e[1]=(uint32_t)s; bb=(s>>63)&1;
    s=(uint64_t)ORD_2-bb; e[2]=(uint32_t)s;
    uint32_t acc[NL], base[NL];
    acc[0]=1; acc[1]=0; acc[2]=0;
    base[0]=a->v[0]; base[1]=a->v[1]; base[2]=a->v[2];
    for (int bit=NL*32-1; bit>=0; --bit){
        mulmod_ord(acc,acc,acc);
        if ((e[bit>>5]>>(bit&31))&1u) mulmod_ord(acc,acc,base);
    }
    r->v[0]=acc[0]; r->v[1]=acc[1]; r->v[2]=acc[2];
}

/* ================================================================== */
/* Affine points over F_p2                                              */
/* ================================================================== */
typedef struct { fp2 x, y; int inf; } pt;

HD static inline void pt_inf(pt *r){ fp2_zero(&r->x); fp2_zero(&r->y); r->inf=1; }

HD static inline void pt_add(pt *r,const pt *p,const pt *q){
    fp2 lam,num,den,t,x3,y3,ny;
    if (p->inf){ *r=*q; return; }
    if (q->inf){ *r=*p; return; }
    if (fp2_eq(&p->x,&q->x)){
        fp2_zero(&ny); fp2_sub(&ny,&ny,&q->y);
        if (fp2_eq(&p->y,&ny)){ pt_inf(r); return; }   /* never in prime order */
        fp2_sqr(&num,&p->x); fp2_3(&num,&num);         /* 3x^2 */
        fp2_add(&den,&p->y,&p->y);                     /* 2y   */
    } else {
        fp2_sub(&num,&q->y,&p->y);
        fp2_sub(&den,&q->x,&p->x);
    }
    fp2_inv(&t,&den);
    fp2_mul(&lam,&num,&t);
    fp2_sqr(&t,&lam);
    fp2_sub(&t,&t,&p->x);
    fp2_sub(&x3,&t,&q->x);
    fp2_sub(&t,&p->x,&x3);
    fp2_mul(&t,&lam,&t);
    fp2_sub(&y3,&t,&p->y);
    r->x=x3; r->y=y3; r->inf=0;
}
HD static inline void pt_dbl(pt *r,const pt *p){ pt t=*p; pt_add(r,&t,&t); }

/* r = k * p, k as 3 little-endian limbs (normal) */
HD static inline void pt_mul(pt *r,const uint32_t k[NL],const pt *p){
    pt acc; pt_inf(&acc);
    for (int bit=NL*32-1; bit>=0; --bit){
        if (!acc.inf) pt_dbl(&acc,&acc);
        if ((k[bit>>5]>>(bit&31))&1u) pt_add(&acc,&acc,p);
    }
    *r=acc;
}
HD static inline int pt_eq(const pt *p,const pt *q){
    if (p->inf||q->inf) return p->inf&&q->inf;
    return fp2_eq(&p->x,&q->x)&&fp2_eq(&p->y,&q->y);
}
HD static inline uint32_t mix64(uint64_t z){
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;
    return (uint32_t)(z^(z>>31));
}
HD static inline uint32_t pt_hash(const pt *p){
    uint64_t h=0x9E3779B97F4A7C15ULL;
    h^=((uint64_t)p->x.a.v[0]<<32)|p->x.a.v[1]; h*=0xff51afd7ed558ccdULL;
    h^=((uint64_t)p->x.a.v[2]<<32)|p->x.b.v[0]; h*=0xc4ceb9fe1a85ec53ULL;
    h^=((uint64_t)p->x.b.v[1]<<32)|p->x.b.v[2];
    return mix64(h);
}

/* ================================================================== */
/* constants in Montgomery form                                         */
/* ================================================================== */
static pt C1m, C2m;
static fp2 mk2(uint32_t a0,uint32_t a1,uint32_t a2,uint32_t b0,uint32_t b1,uint32_t b2){
    fp2 r;
    r.a.v[0]=a0; r.a.v[1]=a1; r.a.v[2]=a2;
    r.b.v[0]=b0; r.b.v[1]=b1; r.b.v[2]=b2;
    return r;
}
static void load_consts(void){
    fp2 c1x=mk2(C1X0_0,C1X0_1,C1X0_2,C1X1_0,C1X1_1,C1X1_2);
    fp2 c1y=mk2(C1Y0_0,C1Y0_1,C1Y0_2,C1Y1_0,C1Y1_1,C1Y1_2);
    fp2 c2x=mk2(C2X0_0,C2X0_1,C2X0_2,C2X1_0,C2X1_1,C2X1_2);
    fp2 c2y=mk2(C2Y0_0,C2Y0_1,C2Y0_2,C2Y1_0,C2Y1_1,C2Y1_2);
    to_mont(&C1m.x.a,&c1x.a); to_mont(&C1m.x.b,&c1x.b);
    to_mont(&C1m.y.a,&c1y.a); to_mont(&C1m.y.b,&c1y.b);
    to_mont(&C2m.x.a,&c2x.a); to_mont(&C2m.x.b,&c2x.b);
    to_mont(&C2m.y.a,&c2y.a); to_mont(&C2m.y.b,&c2y.b);
    C1m.inf=C2m.inf=0;
}

static uint64_t rng_state;
static uint64_t rng_next(void){
    uint64_t z=(rng_state+=0x9E3779B97F4A7C15ULL);
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;
    return z^(z>>31);
}
static void sc_rand(sc *r){
    r->v[0]=(uint32_t)rng_next(); r->v[1]=(uint32_t)rng_next(); r->v[2]=(uint32_t)rng_next();
    sc_reduce(r->v);
}

/* ================================================================== */
/* selftest                                                             */
/* ================================================================== */
static int selftest(void){
#ifdef USE_TOY
    int fails=0;
    load_consts();
    {
        uint32_t o[NL]; o[0]=ORD_0; o[1]=ORD_1; o[2]=ORD_2;
        pt P; pt_mul(&P,o,&C1m);
        if(!P.inf){ printf("toy: ORD*C1 != O\n"); fails++; }
    }
    {
        uint32_t s[NL]; s[0]=SK_0; s[1]=SK_1; s[2]=SK_2;
        pt P; pt_mul(&P,s,&C1m);
        if(!pt_eq(&P,&C2m)){ printf("toy: SK*C1 != C2\n"); fails++; }
    }
    printf("selftest(toy): %s (%d failures)\n", fails?"FAIL":"ok", fails);
    return fails?1:0;
#else
    int fails=0;
    for (int i=0;i<64;i++){
        const uint32_t *A=&VEC_FMUL[i*9], *B=&VEC_FMUL[i*9+3], *C=&VEC_FMUL[i*9+6];
        fp a,b,am,bm,rm,r;
        a.v[0]=A[0];a.v[1]=A[1];a.v[2]=A[2];
        b.v[0]=B[0];b.v[1]=B[1];b.v[2]=B[2];
        to_mont(&am,&a); to_mont(&bm,&b);
        fp_mul(&rm,&am,&bm); from_mont(&r,&rm);
        if (r.v[0]!=C[0]||r.v[1]!=C[1]||r.v[2]!=C[2]){ if(fails<3) printf("FMUL %d fail\n",i); fails++; }
    }
    for (int i=0;i<16;i++){
        const uint32_t *A=&VEC_FINV[i*6], *C=&VEC_FINV[i*6+3];
        fp a,am,im,r;
        a.v[0]=A[0];a.v[1]=A[1];a.v[2]=A[2];
        to_mont(&am,&a); fp_inv(&im,&am); from_mont(&r,&im);
        if (r.v[0]!=C[0]||r.v[1]!=C[1]||r.v[2]!=C[2]){ if(fails<3) printf("FINV %d fail\n",i); fails++; }
    }
    for (int i=0;i<32;i++){
        const uint32_t *A=&VEC_GMUL[i*18];
        fp2 a,b,am,bm,rm,r;
        a.a.v[0]=A[0];a.a.v[1]=A[1];a.a.v[2]=A[2];
        a.b.v[0]=A[3];a.b.v[1]=A[4];a.b.v[2]=A[5];
        b.a.v[0]=A[6];b.a.v[1]=A[7];b.a.v[2]=A[8];
        b.b.v[0]=A[9];b.b.v[1]=A[10];b.b.v[2]=A[11];
        to_mont(&am.a,&a.a); to_mont(&am.b,&a.b);
        to_mont(&bm.a,&b.a); to_mont(&bm.b,&b.b);
        fp2_mul(&rm,&am,&bm);
        from_mont(&r.a,&rm.a); from_mont(&r.b,&rm.b);
        int ok=1;
        for (int k=0;k<3;k++){ if(r.a.v[k]!=A[12+k]) ok=0; if(r.b.v[k]!=A[15+k]) ok=0; }
        if(!ok){ if(fails<3) printf("GMUL %d fail\n",i); fails++; }
    }
    load_consts();
    for (int i=0;i<16;i++){
        const uint32_t *K=&VEC_PMUL[i*15];
        uint32_t k[NL]; k[0]=K[0]; k[1]=K[1]; k[2]=K[2];
        pt P; pt_mul(&P,k,&C1m);
        fp rxa,rxb,rya,ryb;
        from_mont(&rxa,&P.x.a); from_mont(&rxb,&P.x.b);
        from_mont(&rya,&P.y.a); from_mont(&ryb,&P.y.b);
        int ok=1;
        for (int t=0;t<3;t++){
            if (rxa.v[t]!=K[3+t]) ok=0;
            if (rxb.v[t]!=K[6+t]) ok=0;
            if (rya.v[t]!=K[9+t]) ok=0;
            if (ryb.v[t]!=K[12+t]) ok=0;
        }
        if(!ok){ if(fails<3) printf("PMUL %d fail\n",i); fails++; }
    }
    printf("selftest: %s (%d failures)\n", fails?"FAIL":"ok", fails);
    return fails?1:0;
#endif
}

/* ================================================================== */
/* distinguished-point store (host)                                     */
/* ================================================================== */
typedef struct { fp2 x; uint32_t A[NL], B[NL]; } dprec;

typedef struct {
    dprec *dp; uint64_t cap, n;
    uint64_t *ht; uint64_t htcap;
} store;

static uint64_t store_hash(const fp2 *x){
    uint64_t h=0x9E3779B97F4A7C15ULL;
    h^=((uint64_t)x->a.v[0]<<32)|x->a.v[1]; h*=0xff51afd7ed558ccdULL;
    h^=((uint64_t)x->a.v[2]<<32)|x->b.v[0]; h*=0xc4ceb9fe1a85ec53ULL;
    h^=((uint64_t)x->b.v[1]<<32)|x->b.v[2];
    return h^(h>>29);
}
/* 1 = found (sk_out), 0 = inserted, -1 = full/degenerate */
static int store_insert(store *S,const fp2 *x,const sc *A,const sc *B,sc *sk_out){
    uint64_t h=store_hash(x)&(S->htcap-1);
    for(;;){
        uint64_t e=S->ht[h];
        if (!e){
            uint64_t idx=S->n;
            if (idx>=S->cap) return -1;
            S->ht[h]=idx+1;
            S->dp[idx].x=*x;
            S->dp[idx].A[0]=A->v[0]; S->dp[idx].A[1]=A->v[1]; S->dp[idx].A[2]=A->v[2];
            S->dp[idx].B[0]=B->v[0]; S->dp[idx].B[1]=B->v[1]; S->dp[idx].B[2]=B->v[2];
            S->n++;
            return 0;
        }
        dprec *o=&S->dp[e-1];
        if (fp2_eq(&o->x,x)){
            sc oA,oB,dA,dB,dBi,sk;
            oA.v[0]=o->A[0]; oA.v[1]=o->A[1]; oA.v[2]=o->A[2];
            oB.v[0]=o->B[0]; oB.v[1]=o->B[1]; oB.v[2]=o->B[2];
            sc_sub(&dA,A,&oA);
            sc_sub(&dB,&oB,B);
            if (dB.v[0]|dB.v[1]|dB.v[2]){
                sc_inv(&dBi,&dB);
                uint32_t prod[NL];
                mulmod_ord(prod,dA.v,dBi.v);
                sk.v[0]=prod[0]; sk.v[1]=prod[1]; sk.v[2]=prod[2];
                *sk_out=sk;
                return 1;
            }
            return -1;
        }
        h=(h+1)&(S->htcap-1);
    }
}

/* ================================================================== */
/* result output: print + persist to disk immediately                   */
/* ================================================================== */
static void emit_sk(const sc *sk){
    printf("\nsk = 0x%08x%08x%08x\n",sk->v[2],sk->v[1],sk->v[0]);
    fflush(stdout);
    FILE *f=fopen("sk.txt","w");
    if (f){
        fprintf(f,"sk = 0x%08x%08x%08x\n",sk->v[2],sk->v[1],sk->v[0]);
        fclose(f);
        printf("saved to sk.txt\n");
    } else {
        printf("WARNING: could not write sk.txt\n");
    }
    fflush(stdout);
}

/* ================================================================== */
/* multi-instance / shared distinguished-point pool                     */
/*                                                                      */
/* Run N independent instances (one per GPU / machine) with distinct    */
/* RHO_SEED values, all pointing RHO_POOL_DIR at a shared directory.    */
/* Each instance appends the DPs it finds to pool_<seed>.bin and then   */
/* scans the union to catch collisions between instances.               */
/*   RHO_SEED      instance id (default 0)                              */
/*   RHO_POOL      1 = enable shared pool (default 0)                   */
/*   RHO_POOL_MAX  number of pool_<i>.bin files to scan (default 1)     */
/*   RHO_POOL_DIR  directory holding the pool files (default ".")       */
/*   RHO_BUDGET    real steps per thread per round (gpu; default 2^25)  */
/*   RHO_STEP      walk cap before restart       (gpu; default 2^25)    */
/* Smaller RHO_BUDGET = shorter rounds = earlier collision detection but */
/* more truncated (DP-less) walks. For N pooled GPUs, ~2^22-2^24 works.  */
/* ================================================================== */
static unsigned long long g_seed=0;
static int      g_pool=0;
static int      g_pool_max=1;
static char     g_pool_dir[512]=".";
static long long g_env_budget=0;   /* RHO_BUDGET: real steps/thread/round (gpu)   */
static long long g_env_step=0;     /* RHO_STEP:   walk cap (gpu)                  */

static uint64_t seed_mix(unsigned long long x){
    uint64_t z=(uint64_t)x+0x9E3779B97F4A7C15ULL;
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;
    return z^(z>>31);
}
static void env_init(void){
    const char *s;
    if ((s=getenv("RHO_SEED")))     g_seed=strtoull(s,NULL,0);
    if ((s=getenv("RHO_POOL")))     g_pool=atoi(s);
    if ((s=getenv("RHO_POOL_MAX"))) g_pool_max=atoi(s);
    if ((s=getenv("RHO_POOL_DIR"))){
        strncpy(g_pool_dir,s,sizeof(g_pool_dir)-1);
        g_pool_dir[sizeof(g_pool_dir)-1]=0;
    }
    if ((s=getenv("RHO_BUDGET")))   g_env_budget=atoll(s);
    if ((s=getenv("RHO_STEP")))     g_env_step=atoll(s);
    if (g_pool_max<1) g_pool_max=1;
}

static int scan_dps(dprec *dp,uint64_t n){
    if (!n) return 0;
    uint64_t htcap=1; while (htcap<2*(n+1)) htcap<<=1;
    uint64_t *ht=(uint64_t*)calloc(htcap,sizeof(uint64_t));
    if (!ht) return 0;
    for (uint64_t i=0;i<n;i++){
        sc A,B; A.v[0]=dp[i].A[0];A.v[1]=dp[i].A[1];A.v[2]=dp[i].A[2];
        B.v[0]=dp[i].B[0];B.v[1]=dp[i].B[1];B.v[2]=dp[i].B[2];
        uint64_t h=store_hash(&dp[i].x)&(htcap-1);
        for(;;){
            uint64_t e=ht[h];
            if (!e){ ht[h]=i+1; break; }
            dprec *o=&dp[e-1];
            if (fp2_eq(&o->x,&dp[i].x)){
                sc oA,oB,dA,dB,dBi,sk;
                oA.v[0]=o->A[0];oA.v[1]=o->A[1];oA.v[2]=o->A[2];
                oB.v[0]=o->B[0];oB.v[1]=o->B[1];oB.v[2]=o->B[2];
                sc_sub(&dA,&A,&oA); sc_sub(&dB,&oB,&B);
                if (dB.v[0]|dB.v[1]|dB.v[2]){
                    sc_inv(&dBi,&dB);
                    uint32_t prod[NL]; mulmod_ord(prod,dA.v,dBi.v);
                    sk.v[0]=prod[0]; sk.v[1]=prod[1]; sk.v[2]=prod[2];
                    pt chk; pt_mul(&chk,sk.v,&C1m);
                    if (pt_eq(&chk,&C2m)){
                        emit_sk(&sk);
                        free(ht); return 1;
                    }
                }
                break;
            }
            h=(h+1)&(htcap-1);
        }
    }
    free(ht);
    return 0;
}

/* Cheap, GPU-free correctness gate on one round's distinguished points.
   A healthy rho round must have an exact-duplicate rate near zero (distinct
   walks never repeat an (x,A,B) triple); exact duplicates mean the start
   generator is collapsing trajectories.  same-x-different-(A,B) is a genuine
   collision (the thing we want) and is reported when present. */
static void dup_stats(const dprec *dp,uint64_t n){
    if (!n){ printf("dup-gate: no records\n"); return; }
    uint64_t htcap=1; while (htcap<2*n) htcap<<=1;
    uint64_t *ht=(uint64_t*)calloc(htcap,sizeof(uint64_t));
    if (!ht) return;
    uint64_t exact=0,samex=0;
    for (uint64_t i=0;i<n;i++){
        uint64_t h=store_hash(&dp[i].x)&(htcap-1);
        for(;;){
            uint64_t e=ht[h];
            if (!e){ ht[h]=i+1; break; }
            const dprec *o=&dp[e-1];
            if (fp2_eq(&o->x,&dp[i].x)){
                int sameAB = o->A[0]==dp[i].A[0]&&o->A[1]==dp[i].A[1]&&o->A[2]==dp[i].A[2]&&
                             o->B[0]==dp[i].B[0]&&o->B[1]==dp[i].B[1]&&o->B[2]==dp[i].B[2];
                if (sameAB) exact++; else samex++;
                break;
            }
            h=(h+1)&(htcap-1);
        }
    }
    free(ht);
    printf("dup-gate: records=%llu exact-dup=%llu (%.2f%%) same-x-diff-AB=%llu\n",
           (unsigned long long)n,(unsigned long long)exact,
           100.0*(double)exact/(double)n,(unsigned long long)samex);
    if (exact*1000 > n)
        printf("dup-gate: WARNING exact-dup rate >0.1%% -> start generator aliasing; do NOT rent\n");
    fflush(stdout);
}

static int pool_path(int id,char *out,size_t cap){
    return snprintf(out,cap,"%s/pool_%d.bin",g_pool_dir,id);
}
static int pool_append(int id,const dprec *recs,uint64_t n){
    if (!n) return 0;
    char path[600]; pool_path(id,path,sizeof(path));
    FILE *f=fopen(path,"ab");
    if (!f){ printf("pool: cannot open %s for append\n",path); return -1; }
    size_t w=fwrite(recs,sizeof(dprec),(size_t)n,f);
    fclose(f);
    if (w!=(size_t)n){ printf("pool: short write to %s\n",path); return -1; }
    return 0;
}
/* Load the union of pool_0..pool_(g_pool_max-1) and look for collisions. */
static int pool_scan(void){
    uint64_t total=0;
    for (int i=0;i<g_pool_max;i++){
        char path[600]; pool_path(i,path,sizeof(path));
        FILE *f=fopen(path,"rb");
        if (!f) continue;
        if (fseek(f,0,SEEK_END)==0){
            long sz=ftell(f);
            if (sz>0) total+=(uint64_t)sz/sizeof(dprec);
        }
        fclose(f);
    }
    if (!total) return 0;
    dprec *buf=(dprec*)malloc(sizeof(dprec)*(size_t)total);
    if (!buf){ printf("pool: OOM loading %llu records\n",(unsigned long long)total); return 0; }
    uint64_t k=0;
    for (int i=0;i<g_pool_max && k<total;i++){
        char path[600]; pool_path(i,path,sizeof(path));
        FILE *f=fopen(path,"rb");
        if (!f) continue;
        uint64_t got=(uint64_t)fread(buf+k,sizeof(dprec),(size_t)(total-k),f);
        fclose(f); k+=got;
    }
    printf("pool: merged %llu record(s) from up to %d file(s)\n",
           (unsigned long long)k,g_pool_max);
    fflush(stdout);
    int r=scan_dps(buf,k);
    free(buf);
    return r;
}

/* CPU-only self-test of the pool file/merge/collision path.             */
static int pooltest(void){
    load_consts();
    RHO_MKDIR("pooltest_dir");
    strcpy(g_pool_dir,"pooltest_dir");
    g_pool=1; g_pool_max=2;
    int N0=20000, N1=20000;
    dprec *r0=(dprec*)malloc(sizeof(dprec)*N0);
    dprec *r1=(dprec*)malloc(sizeof(dprec)*N1);
    if (!r0||!r1){ printf("pooltest: OOM\n"); return 1; }
    rng_state=0x5EED5000ULL^(g_seed*0x9E3779B97F4A7C15ULL);
    for (int i=0;i<N0;i++){
        for (int k=0;k<3;k++){ r0[i].x.a.v[k]=(uint32_t)rng_next(); r0[i].x.b.v[k]=(uint32_t)rng_next(); }
        sc A,B; sc_rand(&A); sc_rand(&B);
        r0[i].A[0]=A.v[0];r0[i].A[1]=A.v[1];r0[i].A[2]=A.v[2];
        r0[i].B[0]=B.v[0];r0[i].B[1]=B.v[1];r0[i].B[2]=B.v[2];
    }
    for (int i=0;i<N1;i++){
        for (int k=0;k<3;k++){ r1[i].x.a.v[k]=(uint32_t)rng_next(); r1[i].x.b.v[k]=(uint32_t)rng_next(); }
        sc A,B; sc_rand(&A); sc_rand(&B);
        r1[i].A[0]=A.v[0];r1[i].A[1]=A.v[1];r1[i].A[2]=A.v[2];
        r1[i].B[0]=B.v[0];r1[i].B[1]=B.v[1];r1[i].B[2]=B.v[2];
    }
#ifdef USE_TOY
    /* Plant a genuine cross-file collision so scan_dps can recover SK:
       A1 = A2 + (B2-B1)*SK  =>  (A1,B1) and (A2,B2) map to the same X.   */
    {
        sc B1,B2,A2,dB,A1,prod; uint32_t pr[NL],sK[NL];
        sc_rand(&B1); sc_rand(&B2); sc_rand(&A2);
        sc_sub(&dB,&B2,&B1);
        if (!(dB.v[0]|dB.v[1]|dB.v[2])) dB.v[0]=1;
        sK[0]=SK_0; sK[1]=SK_1; sK[2]=SK_2;
        mulmod_ord(pr,dB.v,sK);
        prod.v[0]=pr[0]; prod.v[1]=pr[1]; prod.v[2]=pr[2];
        sc_add(&A1,&A2,&prod);
        fp2 X; for (int k=0;k<3;k++){ X.a.v[k]=(uint32_t)rng_next(); X.b.v[k]=(uint32_t)rng_next(); }
        r0[0].x=X; r1[0].x=X;
        r0[0].A[0]=A1.v[0];r0[0].A[1]=A1.v[1];r0[0].A[2]=A1.v[2];
        r0[0].B[0]=B1.v[0];r0[0].B[1]=B1.v[1];r0[0].B[2]=B1.v[2];
        r1[0].A[0]=A2.v[0];r1[0].A[1]=A2.v[1];r1[0].A[2]=A2.v[2];
        r1[0].B[0]=B2.v[0];r1[0].B[1]=B2.v[1];r1[0].B[2]=B2.v[2];
    }
#endif
    { char p[600]; pool_path(0,p,sizeof(p)); remove(p); pool_path(1,p,sizeof(p)); remove(p); }
    if (pool_append(0,r0,N0)||pool_append(1,r1,N1)){
        printf("pooltest: append failed\n"); free(r0); free(r1); return 1;
    }
    free(r0); free(r1);
    remove("sk.txt.bak");
    int had_sk=(rename("sk.txt","sk.txt.bak")==0);
    int found=pool_scan();
    int rc=0;
#ifdef USE_TOY
    if (!found){ printf("pooltest: FAIL (planted collision not recovered)\n"); rc=1; }
    else printf("pooltest: PASS (toy key recovered across pool files)\n");
    if (found) remove("sk.txt");
#else
    printf("pooltest: %s (I/O round-trip; no key plant on real curve)\n",
           found?"found-key":"ok");
#endif
    if (had_sk) rename("sk.txt.bak","sk.txt");
    return rc;
}

/* ================================================================== */
/* CPU Pollard rho                                                      */
/* ================================================================== */
static int rho_cpu(int D){
    load_consts();
    int W=32;
    pt *steps=(pt*)malloc(sizeof(pt)*W);
    sc *su=(sc*)malloc(sizeof(sc)*W), *sv=(sc*)malloc(sizeof(sc)*W);
    for (int j=0;j<W;j++){
        sc u,v;
        sc_rand(&u); sc_rand(&v);
        su[j]=u; sv[j]=v;
        pt a,b,t;
        pt_mul(&a,u.v,&C1m); pt_mul(&b,v.v,&C2m); pt_add(&t,&a,&b);
        steps[j]=t;
    }
#ifdef USE_TOY
    uint64_t cap=1ULL<<20;
#else
    uint64_t cap=1ULL<<24;
#endif
    store S;
    S.cap=cap; S.n=0;
    S.dp=(dprec*)malloc(sizeof(dprec)*cap);
    S.htcap=1; while (S.htcap<2*cap) S.htcap<<=1;
    S.ht=(uint64_t*)calloc(S.htcap,sizeof(uint64_t));
    uint32_t mask=(D>=32)?0xffffffffu:((1u<<D)-1u);
    rng_state=seed_mix(0x123456789abcdefULL^(uint64_t)time(NULL)
                       ^(g_seed*0x9E3779B97F4A7C15ULL));
    long long cnt=0;
    time_t t0=time(NULL), tlast=t0;
    sc a,b,A,B,sk;
    pt X,t;
    sc_rand(&a); sc_rand(&b);
    pt_mul(&X,a.v,&C1m); pt_mul(&t,b.v,&C2m); pt_add(&X,&X,&t);
    A=a; B=b;
    long long wsteps=0, walks=0;
    for (;;){
        uint32_t j=pt_hash(&X)%(uint32_t)W;
        pt_add(&X,&X,&steps[j]);
        sc_add(&A,&A,&su[j]); sc_add(&B,&B,&sv[j]);
        cnt++; wsteps++;
        if ((X.x.a.v[0]&mask)==0 && (X.y.a.v[0]&1u)==0){
            int r=store_insert(&S,&X.x,&A,&B,&sk);
            if (r==1){
                pt chk; pt_mul(&chk,sk.v,&C1m);
                if (pt_eq(&chk,&C2m)){
                    emit_sk(&sk);
                    printf("steps=%lld dps=%llu walks=%lld %.1fs\n",cnt,S.n,walks,difftime(time(NULL),t0));
                    return 0;
                }
            }
            if (S.n>=S.cap){
                printf("\nstore full\n");
                return 1;
            }
        }
        if (wsteps>=WALK_MAX){           /* DP-free cycle escape: new start, keep store */
            walks++; wsteps=0;
            sc_rand(&a); sc_rand(&b);
            pt_mul(&X,a.v,&C1m); pt_mul(&t,b.v,&C2m); pt_add(&X,&X,&t);
            A=a; B=b;
        }
        if ((cnt & 0x3FFFFF)==0){
            time_t now=time(NULL);
            if (now-tlast>=5){
                tlast=now;
                printf("  dps=%llu steps=%lld walks=%lld %.0fs\n",S.n,cnt,walks,difftime(now,t0));
                fflush(stdout);
            }
        }
    }
}

/* ================================================================== */
/* micro-benchmark of the hot step                                      */
/* ================================================================== */
static int bench(int iters){
    load_consts();
    int W=32;
    pt *steps=(pt*)malloc(sizeof(pt)*W);
    rng_state=0xC0FFEEULL;
    for (int j=0;j<W;j++){
        sc u,v;
        sc_rand(&u); sc_rand(&v);
        pt p,q,r;
        pt_mul(&p,u.v,&C1m); pt_mul(&q,v.v,&C2m); pt_add(&r,&p,&q);
        steps[j]=r;
    }
    sc a,b;
    sc_rand(&a); sc_rand(&b);
    pt X,t; pt_mul(&X,a.v,&C1m); pt_mul(&t,b.v,&C2m); pt_add(&X,&X,&t);
    for (int i=0;i<2000;i++){ uint32_t j=pt_hash(&X)%W; pt_add(&X,&X,&steps[j]); }
    clock_t c0=clock();
    for (int i=0;i<iters;i++){ uint32_t j=pt_hash(&X)%W; pt_add(&X,&X,&steps[j]); }
    clock_t c1=clock();
    double el=(double)(c1-c0)/CLOCKS_PER_SEC;
    printf("bench(inv=%s): %d pt_add in %.3fs = %.4g steps/s, %.1f ns/step\n",
           (INV_MODE==INV_FERMAT)?"fermat":"eea", iters, el, iters/el, 1e9*el/iters);
    return 0;
}

/* ================================================================== */
/* GPU Pollard rho (CUDA)                                               */
/* ================================================================== */
#ifdef __CUDACC__
__global__ void rho_kernel(const pt *steps,const sc *su,const sc *sv,int W,
                           const pt *C1,const pt *C2,
                           dprec *dp,unsigned long long *cnt,unsigned long long cap,
                           uint32_t mask,uint64_t seed,long long budget,int step_per_walk,
                           volatile int *stop,unsigned long long *realsteps){
    uint64_t tid=(uint64_t)blockIdx.x*blockDim.x+threadIdx.x;
    /* Start scalar must be a function of (seed, tid, walk-index) that is
       INJECTIVE in (tid, walk).  The old scheme advanced a single per-thread
       LCG by exactly 6 steps per walk starting from seed+tid*PHI, so thread
       tid's walk w landed on the identical stream position as thread tid+6's
       walk w-1 -> every walk after the first re-walked an existing trajectory
       (proven on host: distinct starts saturated at N+6*W no matter how many
       walks).  That silently wasted ~63% of all GPU work and made a collision
       ~2.5x less likely per unit time.  Pack (tid, walk) injectively instead. */
    uint64_t base=seed+((uint64_t)tid<<32);
    unsigned long long walk=0;
    long long left=budget;
    unsigned long long mydone=0;
    while (left>0){
        sc a,b;
        uint64_t k=base+(uint64_t)walk; walk++;
        a.v[0]=mix64(k^0x243F6A8885A308D3ULL);
        a.v[1]=mix64(k^0x13198A2E03707344ULL);
        a.v[2]=mix64(k^0xA4093822299F31D0ULL);
        b.v[0]=mix64(k^0x082EFA98EC4E6C89ULL);
        b.v[1]=mix64(k^0x452821E638D01377ULL);
        b.v[2]=mix64(k^0xBE5466CF34E90C6CULL);
        sc_reduce(a.v); sc_reduce(b.v);
        pt X,t;
        pt_mul(&X,a.v,C1); pt_mul(&t,b.v,C2); pt_add(&X,&X,&t);
        sc A=a,B=b;
        long long w=(left<step_per_walk)?left:step_per_walk;
        long long did=0;
        for (long long s=0;s<w;s++){
            uint32_t j=pt_hash(&X)%(uint32_t)W;
            pt_add(&X,&X,&steps[j]);
            sc_add(&A,&A,&su[j]); sc_add(&B,&B,&sv[j]);
            mydone++; did++;
            if ((X.x.a.v[0]&mask)==0 && (X.y.a.v[0]&1u)==0){
                unsigned long long idx=atomicAdd(cnt,1ULL);
                if (idx<cap){
                    dp[idx].x=X.x;
                    dp[idx].A[0]=A.v[0]; dp[idx].A[1]=A.v[1]; dp[idx].A[2]=A.v[2];
                    dp[idx].B[0]=B.v[0]; dp[idx].B[1]=B.v[1]; dp[idx].B[2]=B.v[2];
                }
                break;                         /* walk ends at a distinguished point */
            }
            if ((s&1023)==0 && *stop){ atomicAdd(realsteps,mydone); return; }
        }
        /* charge the real number of steps performed, not the walk cap: a walk
           that ends early at a distinguished point must not make this thread
           retire ahead of its neighbours, or the block idles on its slowest
           lane (the previous `left-=w` cost ~2-2.7x throughput). */
        left-=did;
    }
    atomicAdd(realsteps,mydone);
}
static int g_ov_blocks=0, g_ov_threads=0, g_ov_quick=0;
static long long g_ov_budget=0;
static int rho_gpu(int D){
    load_consts();
    rng_state=seed_mix(0xA5C31D57E9B4620FULL);   /* GLOBAL step table: every instance and every round must use the SAME walk function f, or distinguished-point pooling cannot detect cross-walk collisions. Start points are still diversified via g_seed/round in the kernel seed below. */
    int W=32;
    pt *hsteps=(pt*)malloc(sizeof(pt)*W);
    sc *hsu=(sc*)malloc(sizeof(sc)*W), *hsv=(sc*)malloc(sizeof(sc)*W);
    for (int j=0;j<W;j++){
        sc u,v; sc_rand(&u); sc_rand(&v);
        hsu[j]=u; hsv[j]=v;
        pt a,b,t; pt_mul(&a,u.v,&C1m); pt_mul(&b,v.v,&C2m); pt_add(&t,&a,&b);
        hsteps[j]=t;
    }
#ifdef USE_TOY
    uint64_t cap=1ULL<<22;
#else
    uint64_t cap=1ULL<<24;
#endif
    pt *dsteps,*dC1,*dC2; sc *dsu,*dsv; dprec *ddp; unsigned long long *dcnt,*dreal; int *dstop;
    cudaMalloc(&dsteps,sizeof(pt)*W); cudaMalloc(&dsu,sizeof(sc)*W); cudaMalloc(&dsv,sizeof(sc)*W);
    cudaMalloc(&dC1,sizeof(pt)); cudaMalloc(&dC2,sizeof(pt));
    cudaMalloc(&ddp,sizeof(dprec)*cap); cudaMalloc(&dcnt,sizeof(unsigned long long));
    cudaMalloc(&dreal,sizeof(unsigned long long)); cudaMalloc(&dstop,sizeof(int));
    if (cudaGetLastError()!=cudaSuccess){ printf("cudaMalloc failed (need ~%llu MB for dp store)\n",
           (unsigned long long)(sizeof(dprec)*cap>>20)); return 1; }
    cudaMemcpy(dsteps,hsteps,sizeof(pt)*W,cudaMemcpyHostToDevice);
    cudaMemcpy(dsu,hsu,sizeof(sc)*W,cudaMemcpyHostToDevice);
    cudaMemcpy(dsv,hsv,sizeof(sc)*W,cudaMemcpyHostToDevice);
    cudaMemcpy(dC1,&C1m,sizeof(pt),cudaMemcpyHostToDevice);
    cudaMemcpy(dC2,&C2m,sizeof(pt),cudaMemcpyHostToDevice);
    unsigned long long zero=0; cudaMemcpy(dcnt,&zero,sizeof(zero),cudaMemcpyHostToDevice);
    int izero=0; cudaMemcpy(dstop,&izero,sizeof(izero),cudaMemcpyHostToDevice);
    uint32_t mask=(D>=32)?0xffffffffu:((1u<<D)-1u);
    int blocks=4096, threads=256;
#ifdef USE_TOY
    if (D>14) D=14;                    /* keep toy DP spacing (and runtime) small */
    blocks=64; threads=64;
    long long budget=1LL<<21;          /* steps per thread per round */
    int step_per_walk=1<<22;
#else
    long long budget=1LL<<25;          /* walk cap: ~4x mean DP spacing (2^23) -> few truncated walks */
    int step_per_walk=1<<25;
#endif
    if (g_env_budget>0) budget=g_env_budget;                       /* RHO_BUDGET */
    if (g_env_step>0)   step_per_walk=(g_env_step>0x7fffffffLL)?0x7fffffff:(int)g_env_step;
    if (g_ov_blocks) blocks=g_ov_blocks;
    if (g_ov_threads) threads=g_ov_threads;
    if (g_ov_budget) budget=g_ov_budget;   /* step_per_walk stays at its default:
                                              budget is now real steps, not walk slots */
    dprec *hdp=(dprec*)malloc(sizeof(dprec)*cap);
    if (!hdp){ printf("host OOM for dp store\n"); return 1; }
    time_t t0=time(NULL);
    printf("gpu: %d x %d threads, cap=%llu, D=%d, budget/thread=%lld\n",
           blocks,threads,(unsigned long long)cap,D,budget);
    double ordd=(double)ORD_0+(double)ORD_1*4294967296.0+(double)ORD_2*1.8446744073709552e19;
    double expected=1.25*sqrt(ordd);
    printf("expected ~ %.3g point-adds total\n", expected);

    std::atomic<int> hb_stop(0);
    std::thread hb([&](){
        while (!hb_stop.load()){
            for (int i=0;i<30 && !hb_stop.load();++i)
                std::this_thread::sleep_for(std::chrono::seconds(1));
            if (hb_stop.load()) break;
            printf("    [alive] %.0fs GPU busy\n", difftime(time(NULL),t0)); fflush(stdout);
        }
    });

    long long total=0;
    int rc=1;
    if (g_pool) printf("pool: seed=%llu dir=%s max=%d\n",
                       g_seed,g_pool_dir,g_pool_max);
    for (int round=0; round<100000; round++){
        time_t r0=time(NULL);
        printf("round %d: launching...\n",round); fflush(stdout);
        cudaMemcpy(dreal,&zero,sizeof(zero),cudaMemcpyHostToDevice);
        /* In pool mode each round's DPs are flushed to our own file and the
           union is re-scanned, so the device store is reset every round. In
           local mode it accumulates across rounds (walks merge in the store). */
        if (g_pool) cudaMemcpy(dcnt,&zero,sizeof(zero),cudaMemcpyHostToDevice);
        rho_kernel<<<blocks,threads>>>(dsteps,dsu,dsv,W,dC1,dC2,ddp,dcnt,cap,mask,
                                       0xABCDEF0123456789ULL
                                         ^((uint64_t)g_seed*0x9E3779B97F4A7C15ULL)
                                         ^((uint64_t)round*0x9E3779B97F4A7C15ULL),
                                       budget,step_per_walk,dstop,dreal);
        cudaDeviceSynchronize();
        unsigned long long n; cudaMemcpy(&n,dcnt,sizeof(n),cudaMemcpyDeviceToHost);
        if (n>cap) n=cap;
        if (n) cudaMemcpy(hdp,ddp,sizeof(dprec)*n,cudaMemcpyDeviceToHost);
        unsigned long long rs; cudaMemcpy(&rs,dreal,sizeof(rs),cudaMemcpyDeviceToHost);
        total += (long long)blocks*threads*budget;
        double el=difftime(time(NULL),t0);
        double rel=difftime(time(NULL),r0);
        printf("round %d done: dps=%llu  real=%.4g (%.3g real steps/s)  nominal=%.4g  round=%.0fs  total=%.0fs\n",
               round,n,(double)rs,(double)rs/(rel>0?rel:1),(double)total,rel,el);
        if (round==0 && rs>0){
            printf("    real/thread = %.4g ; DPs = %llu ; avg walk/DP = %.4g ; DP rate = %.3g\n",
                   (double)rs/((double)blocks*threads),n,(double)rs/(double)(n?n:1),
                   (double)n/(double)rs);
        }
        if (round==0 && n) dup_stats(hdp,n);   /* pre-rental start-diversity gate */
        fflush(stdout);
        if (g_pool){
            if (n) pool_append((int)g_seed,hdp,n);   /* this round's DPs */
            if (pool_scan()){ rc=0; break; }
        } else {
            if (n && scan_dps(hdp,n)){ rc=0; break; }
        }
        if (g_ov_quick){ rc=0; break; }
    }
    hb_stop=1; hb.join();
    if (rc) printf("\nno sk after max rounds\n");
    return rc;
}
#endif

/* ================================================================== */
int main(int argc,char**argv){
    env_init();
    if (argc>=2 && !strcmp(argv[1],"selftest")) return selftest();
    if (argc>=2 && !strcmp(argv[1],"pooltest")) return pooltest();
    if (argc>=2 && !strcmp(argv[1],"poolscan")){   /* re-scan the full union of pool_*.bin */
        load_consts();
        printf("poolscan: dir=%s max=%d\n",g_pool_dir,g_pool_max);
        return pool_scan()?0:1;
    }
#ifdef USE_TOY
    int D=14;
#else
    int D=22;
#endif
    if (argc>=3 && (!strcmp(argv[1],"cpu")||!strcmp(argv[1],"gpu"))) D=atoi(argv[2]);
    if (argc>=2 && !strcmp(argv[1],"cpu")) return rho_cpu(D);
    if (argc>=2 && !strcmp(argv[1],"bench")) return bench(argc>=3?atoi(argv[2]):200000);
#ifdef __CUDACC__
    if (argc>=2 && !strcmp(argv[1],"gpu")) return rho_gpu(D);
    if (argc>=2 && !strcmp(argv[1],"speed")){
        g_ov_blocks=2048; g_ov_threads=256; g_ov_budget=1LL<<16; g_ov_quick=1;
        if (argc>=3) g_ov_blocks=atoi(argv[2]);
        if (argc>=4) g_ov_threads=atoi(argv[3]);
        if (argc>=5) g_ov_budget=atoll(argv[4]);
        return rho_gpu(D);
    }
    if (argc>=2 && !strcmp(argv[1],"diag")){
        int dD=(argc>=3)?atoi(argv[2]):D;
        g_ov_blocks=(argc>=4)?atoi(argv[3]):1024;
        g_ov_threads=(argc>=5)?atoi(argv[4]):256;
        g_ov_budget=(argc>=6)?atoll(argv[5]):(1LL<<20);
        g_ov_quick=1;
        return rho_gpu(dD);
    }
#else
    if (argc>=2 && !strcmp(argv[1],"gpu")){ printf("gpu: build with nvcc\n"); return 0; }
#endif
    fprintf(stderr,"usage: %s selftest | pooltest | poolscan | cpu [D] | gpu [D] | speed [blocks threads budget] | diag [D blocks threads budget]\n",argv[0]);
    return 2;
}
