/*This function carries out the multiplication of two dense matrices and
 *returns a subset of the product as a spare matrix. The calling structure
 *is sparseMult(A,X,rLoc,cLoc). A and X are the dense matrices to multiply.
 *rLoc and cLoc are vectors containing the desired (row,col) locations to
 *compute from the matrix Z = A^H*X.
 *Note the transpose on the first matrix term.
 * You must mex this with:
 * mex sparseMult2.c -largeArrayDims
 * to properly use the new mwIndex and mwSize variables.*/



/* Include statements */
#include "mex.h" /* MUST include this one */
#include "math.h"





/* The actual mex function */
void mexFunction(
        int nlhs,       mxArray *plhs[],
        int nrhs, const mxArray *prhs[]
        )
{
    
    /* Declare variables */
    mwSize M,N,L,nzmax; /*Z is MxL, A is MxN, X is NxL. nzmax is output counter */
    
    /*Pointer to X components */
    double *pXr,*pXi;
    
    /* Point to A components */
    double *pAr,*pAi;
    
    /* Pointer for Z components */
    double *pZr,*pZi;
    
    /* Pointers for row and column values */
    double *pR;
    double *pC;
    
    
    /*Temp variables */
    mwIndex k,Aoffset,Xoffset,counter;
    
    /*Get sizes */
    M  = mxGetN(prhs[0]);/*from A*/
    N  = mxGetM(prhs[0]);/*from A*/
    L  = mxGetN(prhs[1]);/*from X*/
    nzmax = mxGetM(prhs[2]); /*from rLoc*/
    
    
    /*Get pointers*/
    pAr = mxGetPr(prhs[0]);
    pAi = mxGetPi(prhs[0]);
    pXr = mxGetPr(prhs[1]);
    pXi = mxGetPi(prhs[1]);
    pR = mxGetPr(prhs[2]);
    pC = mxGetPr(prhs[3]);
    
    
    /*Check if data is complex*/
    if (!mxIsComplex(prhs[0]))
    {
        
        /*Allocate Z*/
        plhs[0] = mxCreateDoubleMatrix(1,nzmax,mxREAL);;
        pZr  = mxGetPr(plhs[0]);
        
        
        
        /*Compute the desired entries from Z*/
        for (k=0; (k < nzmax); k++)
        {
            /* Determine offset locations in the two dense matrices*/
            Aoffset = N*(pR[k] - 1.0);
            Xoffset = N*(pC[k] - 1.0);
            
            /*Assign the value*/
            pZr[k] = 0.0;
            for (counter=0; counter < N; counter++)
            {
                pZr[k] += pAr[counter+Aoffset]* pXr[counter+Xoffset];
            }
        }
    }
    else
    {
        /*Allocate Z*/
        plhs[0] = mxCreateDoubleMatrix(1,nzmax,mxCOMPLEX);;
        pZr  = mxGetPr(plhs[0]);
        pZi  = mxGetPi(plhs[0]);
        
        
        /*Compute the desired entries from Z*/
        for (k=0; (k < nzmax); k++)
        {
            /* Determine offset locations in the two dense matrices*/
            Aoffset = N*(pR[k] - 1.0);
            Xoffset = N*(pC[k] - 1.0);
            
            /*Assign the value*/
            pZr[k] = 0.0;
            pZi[k] = 0.0;
            for (counter=0; counter < N; counter++)
            {
                pZr[k] += pAr[counter+Aoffset]* pXr[counter+Xoffset];
                pZr[k] -= pAi[counter+Aoffset]* pXi[counter+Xoffset];
                pZi[k] += pAr[counter+Aoffset]* pXi[counter+Xoffset];
                pZi[k] += pAi[counter+Aoffset]* pXr[counter+Xoffset];
            }
        }
        
    }
    
}