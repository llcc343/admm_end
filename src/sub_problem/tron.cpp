//
// Created by zjh on 2019/11/25.
//

#include "sub_problem/tron.h"


#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif

#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif

static double dnrm2_(int *n, double *x, int *incx)
{
    long int ix, nn, iincx;
    double norm, scale, absxi, ssq, temp;

/*  DNRM2 returns the euclidean norm of a vector via the function
    name, so that

       DNRM2 := sqrt( x'*x )

    -- This version written on 25-October-1982.
       Modified on 14-October-1993 to inline the call to SLASSQ.
       Sven Hammarling, Nag Ltd.   */

    /* Dereference inputs */
    nn = *n;
    iincx = *incx;

    if( nn > 0 && iincx > 0 )
    {
        if (nn == 1)
        {
            norm = fabs(x[0]);
        }
        else
        {
            scale = 0.0;
            ssq = 1.0;

            /* The following loop is equivalent to this call to the LAPACK
               auxiliary routine:   CALL SLASSQ( N, X, INCX, SCALE, SSQ ) */

            for (ix=(nn-1)*iincx; ix>=0; ix-=iincx)
            {
                if (x[ix] != 0.0)
                {
                    absxi = fabs(x[ix]);
                    if (scale < absxi)
                    {
                        temp = scale / absxi;
                        ssq = ssq * (temp * temp) + 1.0;
                        scale = absxi;
                    }
                    else
                    {
                        temp = absxi / scale;
                        ssq += temp * temp;
                    }
                }
            }
            norm = scale * sqrt(ssq);
        }
    }
    else
        norm = 0.0;

    return norm;

} /* dnrm2_ */

static double ddot_(int *n, double *sx, int *incx, double *sy, int *incy)
{
    long int i, m, nn, iincx, iincy;
    double stemp;
    long int ix, iy;

    /* forms the dot product of two vectors.
       uses unrolled loops for increments equal to one.
       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*) */

    /* Dereference inputs */
    nn = *n;
    iincx = *incx;
    iincy = *incy;

    stemp = 0.0;
    if (nn > 0)
    {
        if (iincx == 1 && iincy == 1) /* code for both increments equal to 1 */
        {
            m = nn-4;
            for (i = 0; i < m; i += 5)
                stemp += sx[i] * sy[i] + sx[i+1] * sy[i+1] + sx[i+2] * sy[i+2] +
                         sx[i+3] * sy[i+3] + sx[i+4] * sy[i+4];

            for ( ; i < nn; i++)        /* clean-up loop */
                stemp += sx[i] * sy[i];
        }
        else /* code for unequal increments or equal increments not equal to 1 */
        {
            ix = 0;
            iy = 0;
            if (iincx < 0)
                ix = (1 - nn) * iincx;
            if (iincy < 0)
                iy = (1 - nn) * iincy;
            for (i = 0; i < nn; i++)
            {
                stemp += sx[ix] * sy[iy];
                ix += iincx;
                iy += iincy;
            }
        }
    }

    return stemp;
} /* ddot_ */

static int daxpy_(int *n, double *sa, double *sx, int *incx, double *sy,
                  int *incy)
{
    long int i, m, ix, iy, nn, iincx, iincy;
    register double ssa;

    /* constant times a vector plus a vector.
       uses unrolled loop for increments equal to one.
       jack dongarra, linpack, 3/11/78.
       modified 12/3/93, array(1) declarations changed to array(*) */

    /* Dereference inputs */
    nn = *n;
    ssa = *sa;
    iincx = *incx;
    iincy = *incy;

    if( nn > 0 && ssa != 0.0 )
    {
        if (iincx == 1 && iincy == 1) /* code for both increments equal to 1 */
        {
            m = nn-3;
            for (i = 0; i < m; i += 4)
            {
                sy[i] += ssa * sx[i];
                sy[i+1] += ssa * sx[i+1];
                sy[i+2] += ssa * sx[i+2];
                sy[i+3] += ssa * sx[i+3];
            }
            for ( ; i < nn; ++i) /* clean-up loop */
                sy[i] += ssa * sx[i];
        }
        else /* code for unequal increments or equal increments not equal to 1 */
        {
            ix = iincx >= 0 ? 0 : (1 - nn) * iincx;
            iy = iincy >= 0 ? 0 : (1 - nn) * iincy;
            for (i = 0; i < nn; i++)
            {
                sy[iy] += ssa * sx[ix];
                ix += iincx;
                iy += iincy;
            }
        }
    }

    return 0;
} /* daxpy_ */

static int dscal_(int *n, double *sa, double *sx, int *incx)
{
    long int i, m, nincx, nn, iincx;
    double ssa;

    /* scales a vector by a constant.
       uses unrolled loops for increment equal to 1.
       jack dongarra, linpack, 3/11/78.
       modified 3/93 to return if incx .le. 0.
       modified 12/3/93, array(1) declarations changed to array(*) */

    /* Dereference inputs */
    nn = *n;
    iincx = *incx;
    ssa = *sa;

    if (nn > 0 && iincx > 0)
    {
        if (iincx == 1) /* code for increment equal to 1 */
        {
            m = nn-4;
            for (i = 0; i < m; i += 5)
            {
                sx[i] = ssa * sx[i];
                sx[i+1] = ssa * sx[i+1];
                sx[i+2] = ssa * sx[i+2];
                sx[i+3] = ssa * sx[i+3];
                sx[i+4] = ssa * sx[i+4];
            }
            for ( ; i < nn; ++i) /* clean-up loop */
                sx[i] = ssa * sx[i];
        }
        else /* code for increment not equal to 1 */
        {
            nincx = nn * iincx;
            for (i = 0; i < nincx; i += iincx)
                sx[i] = ssa * sx[i];
        }
    }

    return 0;
} /* dscal_ */

static void default_print(const char *buf)
{
    fputs(buf,stdout);
    fflush(stdout);
}

static double uTMv(int n, double *u, double *M, double *v)
{
    const int m = n-4;
    double res = 0;
    int i;
    for (i=0; i<m; i+=5)
        res += u[i]*M[i]*v[i]+u[i+1]*M[i+1]*v[i+1]+u[i+2]*M[i+2]*v[i+2]+
               u[i+3]*M[i+3]*v[i+3]+u[i+4]*M[i+4]*v[i+4];
    for (; i<n; i++)
        res += u[i]*M[i]*v[i];
    return res;
}

void TRON::info(const char *fmt,...)
{
    char buf[BUFSIZ];
    va_list ap;
    va_start(ap,fmt);
    vsprintf(buf,fmt,ap);
    va_end(ap);
    (*tron_print_string)(buf);
}

TRON::TRON(const function *fun_obj, double eps, double eps_cg, int max_iter)
{
    this->fun_obj=const_cast<function *>(fun_obj);
    this->eps=eps;
    this->eps_cg=eps_cg;
    this->max_iter=max_iter;
    tron_print_string = default_print;
}

TRON::~TRON()
{
}

void TRON::tron(double *w,double *yi,double *zi)
{
    // Parameters for updating the iterates.
    double eta0 = 1e-4, eta1 = 0.25, eta2 = 0.75;

    // Parameters for updating the trust region size delta.
    double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;

    int n = fun_obj->get_nr_variable();
    int i, cg_iter;
    double delta=0, sMnorm, one=1.0;
    double alpha, f, fnew, prered, actred, gs;
    int search = 1, iter = 1, inc = 1;
    double *s = new double[n];
    double *r = new double[n];
    double *g = new double[n];

    const double alpha_pcg = 0.01;
    double *M = new double[n];

    // calculate gradient norm at w=0 for stopping condition.
    double *w0 = new double[n];
    for (i=0; i<n; i++)
        w0[i] = 0;
    fun_obj->fun(w0,yi,zi);
    fun_obj->grad(w0, g,yi,zi);
    double gnorm0 = dnrm2_(&n, g, &inc);
    delete [] w0;

    f = fun_obj->fun(w,yi,zi);
    fun_obj->grad(w, g,yi,zi);
    double gnorm = dnrm2_(&n, g, &inc);

    if (gnorm <= eps*gnorm0)
        search = 0;

    fun_obj->get_diagH(M);
    for(i=0; i<n; i++)
        M[i] = (1-alpha_pcg) + alpha_pcg*M[i];
    delta = sqrt(uTMv(n, g, M, g));

    double *w_new = new double[n];
    bool reach_boundary;
    bool delta_adjusted = false;
    while (iter <= max_iter && search)
    {
        cg_iter = trpcg(delta, g, M, s, r, &reach_boundary);

        memcpy(w_new, w, sizeof(double)*n);
        daxpy_(&n, &one, s, &inc, w_new, &inc);

        gs = ddot_(&n, g, &inc, s, &inc);
        prered = -0.5*(gs-ddot_(&n, s, &inc, r, &inc));
        fnew = fun_obj->fun(w_new,yi,zi);

        // Compute the actual reduction.
        actred = f - fnew;

        // On the first iteration, adjust the initial step bound.
        sMnorm = sqrt(uTMv(n, s, M, s));
        if (iter == 1 && !delta_adjusted)
        {
            delta = min(delta, sMnorm);
            delta_adjusted = true;
        }

        // Compute prediction alpha*sMnorm of the step.
        if (fnew - f - gs <= 0)
            alpha = sigma3;
        else
            alpha = max(sigma1, -0.5*(gs/(fnew - f - gs)));

        // Update the trust region bound according to the ratio of actual to predicted reduction.
        if (actred < eta0*prered)
            delta = min(alpha*sMnorm, sigma2*delta);
        else if (actred < eta1*prered)
            delta = max(sigma1*delta, min(alpha*sMnorm, sigma2*delta));
        else if (actred < eta2*prered)
            delta = max(sigma1*delta, min(alpha*sMnorm, sigma3*delta));
        else
        {
            if (reach_boundary)
                delta = sigma3*delta;
            else
                delta = max(delta, min(alpha*sMnorm, sigma3*delta));
        }

//        info("iter %2d act %5.3e pre %5.3e delta %5.3e f %5.3e |g| %5.3e CG %3d\n", iter, actred, prered, delta, f, gnorm, cg_iter);

        if (actred > eta0*prered)
        {
            iter++;
            memcpy(w, w_new, sizeof(double)*n);
            f = fnew;
            fun_obj->grad(w, g,yi,zi);
            fun_obj->get_diagH(M);
            for(i=0; i<n; i++)
                M[i] = (1-alpha_pcg) + alpha_pcg*M[i];

            gnorm = dnrm2_(&n, g, &inc);
            if (gnorm <= eps*gnorm0)
                break;
        }
        if (f < -1.0e+32)
        {
            info("WARNING: f < -1.0e+32\n");
            break;
        }
        if (prered <= 0)
        {
            info("WARNING: prered <= 0\n");
            break;
        }
        if (fabs(actred) <= 1.0e-12*fabs(f) &&
            fabs(prered) <= 1.0e-12*fabs(f))
        {
            info("WARNING: actred and prered too small\n");
            break;
        }
    }

    delete[] g;
    delete[] r;
    delete[] w_new;
    delete[] s;
    delete[] M;
}

int TRON::trpcg(double delta, double *g, double *M, double *s, double *r, bool *reach_boundary){
    int i, inc = 1;
    int n = fun_obj->get_nr_variable();
    double one = 1;
    double *d = new double[n];
    double *Hd = new double[n];
    double zTr, znewTrnew, alpha, beta, cgtol;
    double *z = new double[n];

    *reach_boundary = false;
    for (i=0; i<n; i++){
        s[i] = 0;
        r[i] = -g[i];
        z[i] = r[i] / M[i];
        d[i] = z[i];
    }

    zTr = ddot_(&n, z, &inc, r, &inc);
    cgtol = eps_cg*sqrt(zTr);
    int cg_iter = 0;

    while (1){
        if (sqrt(zTr) <= cgtol)
            break;
        cg_iter++;
        fun_obj->Hv(d, Hd);

        alpha = zTr/ddot_(&n, d, &inc, Hd, &inc);
        daxpy_(&n, &alpha, d, &inc, s, &inc);

        double sMnorm = sqrt(uTMv(n, s, M, s));
        if (sMnorm > delta)
        {
            info("cg reaches trust region boundary\n");
            *reach_boundary = true;
            alpha = -alpha;
            daxpy_(&n, &alpha, d, &inc, s, &inc);

            double sTMd = uTMv(n, s, M, d);
            double sTMs = uTMv(n, s, M, s);
            double dTMd = uTMv(n, d, M, d);
            double dsq = delta*delta;
            double rad = sqrt(sTMd*sTMd + dTMd*(dsq-sTMs));
            if (sTMd >= 0)
                alpha = (dsq - sTMs)/(sTMd + rad);
            else
                alpha = (rad - sTMd)/dTMd;
            daxpy_(&n, &alpha, d, &inc, s, &inc);
            alpha = -alpha;
            daxpy_(&n, &alpha, Hd, &inc, r, &inc);
            break;
        }
        alpha = -alpha;
        daxpy_(&n, &alpha, Hd, &inc, r, &inc);

        for (i=0; i<n; i++)
            z[i] = r[i] / M[i];
        znewTrnew = ddot_(&n, z, &inc, r, &inc);
        beta = znewTrnew/zTr;
        dscal_(&n, &beta, d, &inc);
        daxpy_(&n, &one, z, &inc, d, &inc);
        zTr = znewTrnew;
    }

    delete[] d;
    delete[] Hd;
    delete[] z;

    return(cg_iter);
}

double TRON::norm_inf(int n, double *x){
    double dmax = fabs(x[0]);
    for (int i=1; i<n; i++)
        if (fabs(x[i]) >= dmax)
            dmax = fabs(x[i]);
    return(dmax);
}

void TRON::set_print_string(void (*print_string) (const char *buf)){
    tron_print_string = print_string;
}