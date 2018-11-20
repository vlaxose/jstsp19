//
//  mex_srp.cpp
//
//  Created by He Jun on 12-8-11.
//
//  Mex Function for 
//      function [ s, w, y ] = mex_srp( U, v, OPTS)

#include <mex.h>
#include <matrix.h>
#include <iostream>
#include <vector>
#include <string.h>
#include "armadillo"

using namespace arma;
using namespace std;

void matlab2arma(mat& A, const mxArray *mxdata)
{
    // delete [] A.mem; // don't do this!
    arma::access::rw(A.mem)=mxGetPr(mxdata);
    arma::access::rw(A.n_rows)=mxGetM(mxdata); // transposed!
    arma::access::rw(A.n_cols)=mxGetN(mxdata);
    arma::access::rw(A.n_elem)=A.n_rows*A.n_cols;
};

void freeVar(mat& A, const double *ptr)
{
    arma::access::rw(A.mem)=ptr;
    arma::access::rw(A.n_rows)=1; // transposed!
    arma::access::rw(A.n_cols)=1;
    arma::access::rw(A.n_elem)=1;
};

inline void mex2cpp(const mxArray*& md, vector<double>& cd) 
{
    int m = mxGetM(md);
    int n = mxGetN(md); 
    double* xr = mxGetPr(md);
    cd.resize(m*n);
    for(int i=0; i<m*n; i++)
        cd[i] = xr[i];
    return;
}

inline void vector2mxarray(const vector<double> & vec, mxArray *& md)
{    
    int m = vec.size();
    int n = 1;
    md = mxCreateDoubleMatrix(m, n, mxREAL);
    double* xr = mxGetPr(md);
    for(int i=0; i<m*n; i++)
        xr[i] = vec[i];
    return;    
};

mat shrinkage_max(const mat & a)
{
    mat b(a.n_rows,a.n_cols);
    b.zeros();    
    for (int i=0; i<a.n_rows; i++)
    {
        if (a(i,0)>0) b(i,0)=a(i,0);
    }
    return b;
};

mat shrinkage(const mat &a, double kappa)
{
    mat y;
    
    // as matlab :y = max(0, a-kappa) - max(0, -a-kappa);
    y= shrinkage_max( a-kappa) - shrinkage_max(-a-kappa);
    
    return y;
}

void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    // Get U v and OPTS fro prhs[]
#define U_IN        prhs[0]
#define v_IN        prhs[1]
#define OPTS_IN     prhs[2]
    
#define s_OUT       plhs[0]
#define w_OUT       plhs[1]
#define y_OUT       plhs[2]
    
    /* Check for proper number of input and output arguments */    
    if (nrhs < 2 || nlhs!=3) 
    {
        mexErrMsgTxt("Usage: [s, w, dual] = mex_srp(U, v, OPTS)");
    } 
    
    
    mat     U_cpp(1,1);
    mat     v_cpp(1,1);
    mat     w_cpp(1,1), w2_cpp(1,1);
    mat     s_cpp(1,1), s2_cpp(1,1);
    mat     y_cpp(1,1), y2_cpp(1,1);
    mat     P_cpp(1,1), P;
    
    const double *Umem = arma::access::rw(U_cpp.mem);
    const double *vmem = arma::access::rw(v_cpp.mem);
    const double *wmem = arma::access::rw(w_cpp.mem);
    const double *smem = arma::access::rw(s_cpp.mem); 
    const double *ymem = arma::access::rw(y_cpp.mem);
    const double *w2mem = arma::access::rw(w2_cpp.mem);
    const double *s2mem = arma::access::rw(s2_cpp.mem); 
    const double *y2mem = arma::access::rw(y2_cpp.mem);
        
    double rho, TOL, mu;
    int    QUIET, MAX_ITER, M, N; 

    int      ndim = 2, dims[2] = {1, 1};
    
    // Default values
    rho = 1.8; TOL = 1e-7;
    QUIET = 1; MAX_ITER = 100;
    
    matlab2arma(U_cpp, U_IN);        
    matlab2arma(v_cpp, v_IN); 
    
    s_OUT = mxCreateDoubleMatrix(U_cpp.n_rows,1, mxREAL);
    y_OUT = mxCreateDoubleMatrix(U_cpp.n_rows,1, mxREAL);        
    w_OUT = mxCreateDoubleMatrix(U_cpp.n_cols,1, mxREAL);  
    
    matlab2arma(w_cpp, w_OUT);
    matlab2arma(s_cpp, s_OUT);
    matlab2arma(y_cpp, y_OUT); 
    
    w_cpp.zeros(); s_cpp.zeros(); y_cpp.zeros();
    
    // Parse OPTS 
    int number_of_fields, field_num;    
    const char *fld_name;
    
    mu = 1.25/norm(v_cpp,2);
    
    number_of_fields = mxGetNumberOfFields(OPTS_IN);
    
    for (field_num=0; field_num<number_of_fields; field_num++)
    {
        mxArray *pa;
        pa = mxGetFieldByNumber(OPTS_IN, 0, field_num); // only one struct
        fld_name = mxGetFieldNameByNumber(OPTS_IN, field_num);
        
        if (strcmp(fld_name,"RHO") ==0 )
        {
            rho = mxGetScalar(pa);
        }
        else if (strcmp(fld_name,"TOL") ==0 )
        {
            TOL = mxGetScalar(pa);
        }
        else if (strcmp(fld_name,"QUIET") ==0 )
        {
            QUIET = (int)mxGetScalar(pa);
        }
        else if (strcmp(fld_name,"MAX_ITER") ==0 )
        {
            MAX_ITER = (int)mxGetScalar(pa);
        }
    }
    

    // main algorithm
    mat  s_old, Uw_hat, h;
    mat  UtU = trans(U_cpp) * U_cpp;
    bool bRet = solve(P, UtU, trans(U_cpp));
    
    if (!bRet)
    {
        mexErrMsgTxt("Can not solve P = (U^T*U)\\U^T!\\n");
    }

    double max_err, objval, r_norm, s_norm, dual_norm, eps_pri, eps_dual;
    
    for (int k=0; k<MAX_ITER; k++)
    {
        // w update
        w_cpp = P * (v_cpp -s_cpp - y_cpp/mu);
                
        // s update
        Uw_hat = U_cpp*w_cpp;
        s_cpp  = shrinkage( v_cpp-Uw_hat - y_cpp/mu, 1/mu);
        
        // y update
        h = Uw_hat + s_cpp - v_cpp;
        y_cpp  = y_cpp + mu * h;
        
        // diagnostics, reporting, termination checks
        
        mu = mu * rho;
        
        if (norm(h,2) < TOL)
        {
            break;
        }
    }
    
    
    freeVar(U_cpp, Umem);
    freeVar(v_cpp, vmem);
    freeVar(w_cpp, wmem);
    freeVar(s_cpp, smem);
    freeVar(y_cpp, ymem);
    
    return;
}



