/*
Fast mex Fisher vectors of data X, GMM parameters and GMM probabilities p
(mex-interface modified from the original yael package https://gforge.inria.fr/projects/yael)

- Accept single/double precision input
- Support of BLAS/OpenMP for multi-core computation

Usage
-----

F                                       =    yael_fisher_gmm(X , M , S , w , p , [options] , [mF] , [sF]);

Inputs
------

X                                        Input data matrix (d x N) in single/double format 
M                                        Means matrix of GMM (d x K) in single/double format 
S                                        Variance matrix of GMM (d x K) in single/double format
w                                        Weights vector of GMM (1 x K) in single/double format
p                                       ¨GMM output probabilities of X (K x N) in single/double format

options
 	   fv_flags_w                        Include weights in Fisher vector (default fv_flags_w = 0)
	   fv_flags_M                        Include means in Fisher vector (default fv_flags_M = 1)
	   fv_flags_S                        Include variance in Fisher vector (default gmm_flag_S = 1) 
	   palpha                            alpha-power normalization if alpha > 0.0 and alpha < 1.0 (default palpha = 0.5) 
	   norm_l2                           L2-normalization of fisher vector (default norm_l2 = 0)

If compiled with the "OMP" compilation flag

       num_threads                       Number of threads   (default num_threads = max number of core)

mF                                       Initial mean Fisher vector (d x 1). When mF is given in input, F output represents variance of Fisher vectors 
sF                                       Initial variance of Fisher vector (d x 1). When both mF & sF are given, F output represents centered/standardized Fisher vector


Output
------

F                                        Fisher vector (d x 1) in single/double format where d = K*(options.fv_flags_w + options.fv_flags_M*d + options.fv_flags_S*d)

Example 1
---------

clear

d                                    = 128;                 % dimensionality of the vectors
N                                    = 2000;                % number of vectors

X                                    = randn(d , N , 'single'); % random set of vectors 

options.K                            = 100;
options.max_ite_kmeans               = 10;
options.max_ite_gmm                  = 3;
options.init_random_mode             = 0;
options.redo                         = 1;
options.normalize_sophisticated_mode = 0;
options.BLOCK_N1                     = 1024;
options.BLOCK_N2                     = 1024;
options.gmm_1sigma                   = 0;
options.gmm_flags_norm_l2            = 0;
options.fv_flags_w                   = 1;
options.fv_flags_M                   = 1;
options.fv_flags_S                   = 1;
options.palpha                       = 0.5;
options.norm_l2                      = 1;

options.seed                         = 1234543;
options.num_threads                  = 2;
options.verbose                      = 1;


[M , S , w]                          = yael_gmm(X , options);
p                                    = yael_proba_gmm(X , M , S , w , options);
F                                    = yael_fisher_gmm(X , M , S , w , p , options);


To compile
----------

mex  -g   yael_fisher_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex  -g -DBLAS  yael_fisher_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


mex  -g -DOMP -DBLAS  yael_fisher_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex -v -g -DBLAS -DOMP yael_fisher_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


mex  -f mexopts_intel10.bat -DBLAS -DOMP  yael_fisher_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex  -f mexopts_intel10.bat -DBLAS yael_fisher_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

If compiled with OMP option, OMP support

mex -v -DOMP  yael_fisher_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex -v -DOMP -f mexopts_intel10.bat yael_fisher_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

If compiled with BLAS & OMP options

mex -v -DBLAS -DOMP  yael_fisher_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex -v -DBLAS -DOMP -f mexopts_intel10.bat yael_fisher_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


mex -v -DBLAS -f mexopts_intel10.bat yael_fisher_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


References  [1] Hervé Jégou, Matthijs Douze and Cordelia Schmid, 
----------      "Product quantization for nearest neighbor search"
                IEEE Transactions on Pattern Analysis and Machine Intelligence

            [2] Florent Perronnin, Jorge Sánchez, Thomas Mensink,
		        "Improving the Fisher Kernel for Large-Scale Image Classification", ECCV' 10


Author : Sébastien PARIS : sebastien.paris@lsis.org
------  

Changelog : 
---------
            v 1.0 Initial release 10/09/2012

*/

#include <math.h>
#include <malloc.h>
#include <mex.h>

#ifdef OMP
 #include <omp.h>
#endif

#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

#if defined(__OS2__)  || defined(__WINDOWS__) || defined(WIN32) || defined(WIN64) || defined(_MSC_VER)
#define BLASCALL(f) f
#else
#define BLASCALL(f) f ##_
#endif

struct opts
{
	int    fv_flags_w;
	int    fv_flags_M;
	int    fv_flags_S;
	int    gmm_flags_no_norm;
	double palpha;
	int    norm_l2;
	int    f;
#ifdef OMP 
    int    num_threads;
#endif
};

/* ------------------------------------------------------------------------------------------------------------------------------------ */

extern void  BLASCALL(sgemm)(char *, char *, int*, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *);
extern void  BLASCALL(dgemm)(char *, char *, int*, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);

void  sfisher_gmm(float *, float *, float * , float * , float *, int , int , int , struct opts , float * , float * , float *);
float svec_normalize(float * , int , float);

void   dfisher_gmm(double *, double *, double * , double * , double *, int , int , int , struct opts , double * , double * , double *);
double dvec_normalize(double * , int , double);

/* ------------------------------------------------------------------------------------------------------------------------------------ */
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
	int d , N , K , issingle = 0;
	int f;
	float  *sX , *sM , *sS , *sw , *sp , *sF , *smF=NULL , *ssF=NULL;
	double *dX , *dM , *dS , *dw , *dp , *dF , *dmF=NULL , *dsF=NULL;

#ifdef OMP 
	struct opts options = {0 , 1 , 1 , 0  , 0.5 , 0 , 4112 , -1};
#else
	struct opts options = {0 , 1 , 1 , 0  , 0.5 , 0 , 4112};
#endif
	mxArray *mxtemp;
	double *tmp;
	double temp;
	int tempint;

	if ((nrhs < 5) || (nrhs > 8)) 
	{
		mexPrintf(
			"\n"
			"\n"
			"Fast mex Fisher vectors of data X, GMM parameters and GMM probabilities p \n"
			"(mex-interface modified from the original yael package https://gforge.inria.fr/projects/yael)\n"
			"\n"
			"- Accept single/double precision input\n"
			"- Support of BLAS/OpenMP for multi-core computation\n"
			"\n"
			"Usage\n"
			"-----\n"
			"\n"
			"F                                        = yael_fisher_gmm(X , M , S , w , p , [options] , [mF] , [sF]);\n"
			"\n"
			"\n"
			"Inputs\n"
			"------\n"
			"\n"
			"X                                        Input data matrix (d x N) in single/double format\n"
            "M                                        Means matrix of GMM (d x K) in single/double format\n" 
            "S                                        Variance matrix of GMM (d x K) in single/double format\n"
            "w                                        Weights vector of GMM (1 x K) in single/double format\n"
            "p                                       ¨GMM output probabilities of X (K x N) in single/double format\n"
			"\n"
			"options\n"
            "       fv_flags_w                       Include weights in Fisher vector (default fv_flags_w = 0)\n"
            "       fv_flags_M                       Include means in Fisher vector (default fv_flags_M = 1)\n"
            "       gmm_flag_S                        Include variance in Fisher vector (default gmm_flag_S = 1) \n"
			"       palpha                            alpha-power normalization if alpha > 0.0 and alpha < 1.0 (default palpha = 0.5)\n"
			"       norm_l2                           Don't normalize Fisher vector (default norm_l2 = 0)\n"
#ifdef OMP 
			"       num_threads                       Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)\n"
#endif
			"\n"
            "mF                                       Initial mean Fisher vector (d x 1). When mF is given in input, F output represents variance of Fisher vectors\n" 
            "sF                                       Initial variance of Fisher vector (d x 1). When both mF & sF are given, F output represents centered/standardized Fisher vector\n"
			"\n"
			"\n"
			"Output\n"
			"------\n"
			"\n"
            "F                                        Fisher vector (d x 1) in single/double format where d = K*(options.fv_flags_w + options.fv_flags_M*d + options.fv_flags_S*d)\n" 
			"\n"
			"\n"
			);
		return;
	}

	if(mxIsSingle(prhs[0]))
	{
		sX       = (float *)mxGetData(prhs[0]);
		issingle = 1;
	}
	else
	{
		dX       = (double *)mxGetData(prhs[0]);
	}

	d            = mxGetM(prhs[0]);
	N            = mxGetN(prhs[0]);

	if((mxGetM(prhs[1]) == d))
	{
		if( issingle ) 
		{
			sM       = (float *)mxGetData(prhs[1]);
		}
		else
		{
			dM       = (double *)mxGetData(prhs[1]);
		}
		K           = mxGetN(prhs[1]);
	}
	else
	{
        mexErrMsgTxt("M must be at least(d x K)");
	}

	if((mxGetM(prhs[2]) == d) && (mxGetN(prhs[2]) == K))
	{
		if( issingle ) 
		{
			sS       = (float *)mxGetData(prhs[2]);
		}
		else
		{
			dS       = (double *)mxGetData(prhs[2]);
		}
	}
	else
	{
		mexErrMsgTxt("S must be at least(d x K)");
	}

	if((mxGetM(prhs[3]) == 1) && (mxGetN(prhs[3]) == K))
	{
		if( issingle ) 
		{
			sw       = (float *)mxGetData(prhs[3]);
		}
		else
		{
			dw       = (double *)mxGetData(prhs[3]);
		}
	}
	else
	{
		mexErrMsgTxt("w must be at least(1 x K)");
	}

	if((mxGetM(prhs[4]) == K) && (mxGetN(prhs[4]) == N))
	{
		if( issingle )
		{
			sp       = (float *)mxGetData(prhs[4]);
		}
		else
		{
			dp       = (double *)mxGetData(prhs[4]);
		}
	}
	else
	{
		mexErrMsgTxt("p must be at least(K x N)");
	}

	if ((nrhs > 5) && !mxIsEmpty(prhs[5]))
	{

		mxtemp                            = mxGetField(prhs[5] , 0 , "fv_flags_w");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0) || (tempint > 1))
			{
				mexPrintf("fv_flags_w must be ={0,1}, force to 0\n");	
				options.fv_flags_w       = 0;
			}
			else
			{
				options.fv_flags_w       = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[5] , 0 , "fv_flags_M");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0) || (tempint > 1))
			{
				mexPrintf("fv_flags_M must be ={0,1}, force to 1\n");	
				options.fv_flags_M       = 1;
			}
			else
			{
				options.fv_flags_M       = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[5] , 0 , "fv_flags_S");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0) || (tempint > 1))
			{
				mexPrintf("fv_flags_S must be ={0,1}, force to 1\n");	
				options.fv_flags_S       = 1;
			}
			else
			{
				options.fv_flags_S       = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[5] , 0 , "gmm_flags_no_norm");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0) || (tempint > 1))
			{
				mexPrintf("gmm_flags_no_norm must be ={0,1}, force to 0\n");	
				options.gmm_flags_no_norm  = 0;
			}
			else
			{
				options.gmm_flags_no_norm  = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[5] , 0 , "palpha");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			temp                          = tmp[0];

			if( (temp < 0.0) || (temp > 1.0) )
			{
				mexPrintf("palpha > 0, force to 0.5\n");	
				options.palpha            = 0.5;
			}
			else
			{
				options.palpha            = temp;
			}
		}

		mxtemp                            = mxGetField(prhs[5] , 0 , "norm_l2");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0) || (tempint > 1))
			{
				mexPrintf("norm_l2 must be ={0,1}, force to 1\n");	
				options.norm_l2           = 1;
			}
			else
			{
				options.norm_l2           = tempint;
			}
		}

#ifdef OMP
		mxtemp                            = mxGetField(prhs[4] , 0 , "num_threads");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < -1))
			{
				mexPrintf("num_threads must be >= -1, force to -1\n");	
				options.num_threads       = -1;
			}
			else
			{
				options.num_threads       = tempint;
			}
		}
#endif
	}

	f  = 0;
	if(options.fv_flags_w)
	{
		f +=1;
	}
	if(options.fv_flags_M)
	{
		f +=d;
	}
	if(options.fv_flags_S)
	{
		f +=d;
	}

	f        *=K;
	options.f = f;

	if ((nrhs > 6) && !mxIsEmpty(prhs[6]))
	{
		if((mxGetM(prhs[6]) == f) && (mxGetN(prhs[6]) == 1))
		{
			if( issingle ) 
			{
				smF       = (float *)mxGetData(prhs[6]);
			}
			else
			{
				dmF       = (double *)mxGetData(prhs[6]);
			}
		}
	}

	if ((nrhs > 7) && !mxIsEmpty(prhs[7]))
	{
		if((mxGetM(prhs[7]) == f) && (mxGetN(prhs[7]) == 1))
		{
			if( issingle ) 
			{
				ssF       = (float *)mxGetData(prhs[7]);
			}
			else
			{
				dsF       = (double *)mxGetData(prhs[7]);
			}
		}
	}

	/*------------------------ Main Call ----------------------------*/

	if(issingle)
	{
		plhs[0]    = mxCreateNumericMatrix (f , 1 , mxSINGLE_CLASS, mxREAL);  
		sF         = (float*)mxGetPr(plhs[0]);

		sfisher_gmm(sX , sM , sS , sw , sp , d , N , K , options , smF , ssF , sF );
	}
	else
	{
		plhs[0]    = mxCreateNumericMatrix (f , 1 , mxDOUBLE_CLASS, mxREAL);  
		dF         = (double*)mxGetPr(plhs[0]);

		dfisher_gmm(dX , dM , dS , dw , dp, d , N , K , options , dmF , dsF , dF );
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void sfisher_gmm(float *X , float *M , float *S , float *w, float *p, int d , int N , int K , struct opts options, float *mF , float *sF , float *F)
{
	int i , j;
	int ii = 0 , dK = d*K , dN = d*N , KN = K*N, jd , ind;
	int fv_flags_w = options.fv_flags_w, fv_flags_M = options.fv_flags_M, fv_flags_S = options.fv_flags_S;
	int norm_l2 = options.norm_l2 ;
	float alpha = 1.0f, beta = 0.0f;
	float cteN = 1.0f/N;
	float temp , tempM , tempM2 , tempM3, tempM4 , tempS , tempS2, tempw, sum , sum2 , pj , p2j , nfisher;
	float palpha = (float) options.palpha , sign;
	float *sump=NULL ,*sump2=NULL , *Xpt=NULL , *X2pt=NULL , *X3pt=NULL, *X4pt=NULL , *X2=NULL , *X3=NULL, *X4=NULL;
	float *p2=NULL;
	int ismF=1 , issF=1;

	if(mF == NULL)
	{
		ismF = 0;
	}

	if(sF == NULL)
	{
		issF = 0;
	}

	if ( ((!ismF) && (!issF)) || ((ismF) && (issF)) ) 
	{
		if(fv_flags_w)
		{
			sump = (float *) malloc(K*sizeof(float));
			for(j = 0 ; j < K ; j++)
			{
				sum = 0.0f;
				for(i = 0 ; i < N ; i++)
				{
					sum += p[j + i*K];
				}
				sump[j] = sum;
				F[ii++] = (sum  - N*w[j])*cteN;
			}
		}

		if(fv_flags_M) 
		{
			if(sump==NULL)
			{
				sump = (float *) malloc(K*sizeof(float));
				for(j = 0 ; j < K ; j++)
				{
					sum = 0.0f;
					for(i = 0 ; i < N ; i++)
					{
						sum += p[j + i*K];
					}
					sump[j] = sum;
				}
			}

			Xpt             = (float *) malloc(dK*sizeof(float));
			BLASCALL(sgemm)("Not transposed", "Transposed", &d , &K, &N , &alpha, X , &d , p , &K, &beta, Xpt , &d);

			for (j = 0 ; j < K ; j++)
			{
				jd   = j*d;
				pj   = sump[j];
				for(i = 0 ; i < d ; i++)
				{
					ind     = i + jd;
					tempS   = 1.0f/S[ind]; 
					F[ii++] = (Xpt[ind]*tempS - pj*tempS*M[ind])*cteN;
				}
			}
		}

		if(fv_flags_S) 
		{ 
			if(sump==NULL)
			{
				sump = (float *) malloc(K*sizeof(float));
				for(j = 0 ; j < K ; j++)
				{
					sum = 0.0f;
					for(i = 0 ; i < N ; i++)
					{
						sum += p[j + i*K];
					}
					sump[j] = sum;
				}
			}
			if(Xpt==NULL)
			{
				Xpt             = (float *) malloc(dK*sizeof(float));
				BLASCALL(sgemm)("Not transposed", "Transposed", &d , &K, &N , &alpha, X, &d , p , &K, &beta, Xpt , &d);
			}

			X2         = (float *) malloc(dN*sizeof(float));
			for(i = 0 ; i < dN ; i++)
			{
				X2[i]  = X[i]*X[i];
			}
			X2pt       = (float *) malloc(dK*sizeof(float));
			BLASCALL(sgemm)("Not transposed", "Transposed", &d , &K , &N , &alpha, X2, &d , p , &K , &beta, X2pt , &d);
			free(X2);

			for (j = 0 ; j < K ; j++)
			{
				jd   = j*d;
				pj   = sump[j];
				for(i = 0 ; i < d ; i++)
				{
					ind     = i + jd;
					tempM   = M[ind];
					F[ii++] = 0.5f*( 2.0f*tempM*Xpt[ind] - X2pt[ind] + pj*(S[ind] - tempM*tempM))*cteN;
				}
			}
		}
		if((ismF) && (issF))
		{
			for( i = 0 ; i < options.f ; i++)
			{
				F[i] = (F[i] - mF[i])*sF[i];
			}
		}
	}
	else
	{
		if(fv_flags_w)
		{
			p2 = (float *) malloc(KN*sizeof(float));
			for( i = 0 ; i < KN ; i++)
			{
				p2[i] = p[i]*p[i];
			}

			sump  = (float *) malloc(K*sizeof(float));
			sump2 = (float *) malloc(K*sizeof(float));

			for(j = 0 ; j < K ; j++)
			{
				sum  = 0.0f;
				sum2 = 0.0f;
				for(i = 0 ; i < N ; i++)
				{
					ind   = j + i*K;
					sum  += p[ind];
					sum2 += p2[ind];
				}
				sump[j]  = sum;
				sump2[j] = sum2;
				tempw    = w[j];
				F[ii++]  = (N*tempw*tempw + sum2 - 2.0f*tempw*sum)*cteN;
			}
		}

		if(fv_flags_M) 
		{
			if(p2 == NULL)
			{
				p2 = (float *) malloc(KN*sizeof(float));
				for( i = 0 ; i < KN ; i++)
				{
					p2[i] = p[i]*p[i];
				}
			}
			if(sump == NULL)
			{
				sump  = (float *) malloc(K*sizeof(float));
				for(j = 0 ; j < K ; j++)
				{
					sum  = 0.0f;
					for(i = 0 ; i < N ; i++)
					{
						ind   = j + i*K;
						sum  += p[ind];
					}
					sump[j]  = sum;
				}
			}
			if(sump2 == NULL)
			{
				sump2  = (float *) malloc(K*sizeof(float));
				for(j = 0 ; j < K ; j++)
				{
					sum2  = 0.0f;
					for(i = 0 ; i < N ; i++)
					{
						ind   = j + i*K;
						sum2  += p2[ind];
					}
					sump2[j]  = sum2;
				}
			}

			Xpt        = (float *) malloc(dK*sizeof(float));
			BLASCALL(sgemm)("Not transposed", "Transposed", &d , &K , &N , &alpha, X , &d , p2 , &K , &beta, Xpt , &d);

			X2         = (float *) malloc(dN*sizeof(float));
			for(i = 0 ; i < dN ; i++)
			{
				X2[i]  = X[i]*X[i];
			}
			X2pt       = (float *) malloc(dK*sizeof(float));
			BLASCALL(sgemm)("Not transposed", "Transposed", &d , &K , &N , &alpha, X2 , &d , p2 , &K , &beta , X2pt , &d);
			if(!fv_flags_S)
			{
				free(X2);
			}

			for (j = 0 ; j < K ; j++)
			{
				jd   = j*d;
				pj   = sump[j];
				p2j  = sump2[j];
				for(i = 0 ; i < d ; i++)
				{
					ind     = i + jd;
					tempS   = S[ind]; 
					tempS2  = 1.0f/(tempS*tempS);
					tempM   = M[ind];
					tempM2  = tempM*tempM;
					F[ii++] = (tempS2*X2pt[ind] + p2j*tempS2*tempM2 - 2.0f*tempS2*tempM*Xpt[ind])*cteN;
				}
			}
		}

		if(fv_flags_S) 
		{
			if(sump == NULL)
			{
				sump  = (float *) malloc(K*sizeof(float));
				for(j = 0 ; j < K ; j++)
				{
					sum  = 0.0f;
					for(i = 0 ; i < N ; i++)
					{
						ind   = j + i*K;
						sum  += p[ind];
					}
					sump[j]  = sum;
				}
			}

			if(p2 == NULL)
			{
				p2 = (float *) malloc(KN*sizeof(float));
				for( i = 0 ; i < KN ; i++)
				{
					p2[i] = p[i]*p[i];
				}
			}
			if(sump2 == NULL)
			{
				sump2  = (float *) malloc(K*sizeof(float));
				for(j = 0 ; j < K ; j++)
				{
					sum2  = 0.0f;
					for(i = 0 ; i < N ; i++)
					{
						ind   = j + i*K;
						sum2  += p2[ind];
					}
					sump2[j]  = sum2;
				}
			}
			if(Xpt == NULL)
			{
				Xpt        = (float *) malloc(dK*sizeof(float));
				BLASCALL(sgemm)("Not transposed", "Transposed", &d , &K, &N , &alpha, X , &d , p2 , &K, &beta, Xpt , &d);
			}
			if(X2pt == NULL)
			{

				X2         = (float *) malloc(dN*sizeof(float));
				for(i = 0 ; i < dN ; i++)
				{
					X2[i]  = X[i]*X[i];
				}

				X2pt       = (float *) malloc(dK*sizeof(float));
				BLASCALL(sgemm)("Not transposed", "Transposed", &d , &K , &N , &alpha, X2 , &d , p2 , &K , &beta, X2pt , &d);
			}

			X3         = (float *) malloc(dN*sizeof(float));
			for(i = 0 ; i < dN ; i++)
			{
				X3[i]  = X2[i]*X[i];
			}
			free(X2);

			X3pt        = (float *) malloc(dK*sizeof(float));
			BLASCALL(sgemm)("Not transposed", "Transposed", &d , &K, &N , &alpha, X3 , &d , p2 , &K, &beta, X3pt , &d);

			X4         = (float *) malloc(dN*sizeof(float));
			for(i = 0 ; i < dN ; i++)
			{
				X4[i]  = X3[i]*X[i];
			}
			free(X3);

			X4pt        = (float *) malloc(dK*sizeof(float));
			BLASCALL(sgemm)("Not transposed", "Transposed", &d , &K, &N , &alpha, X4 , &d , p2 , &K, &beta, X4pt , &d);
			free(X4);

			for (j = 0 ; j < K ; j++)
			{
				jd   = j*d;
				p2j  = sump2[j];
				for(i = 0 ; i < d ; i++)
				{
					ind     = i + jd;
					tempS   = S[ind]; 
					tempS2  = tempS*tempS;
					tempM   = M[ind];
					tempM2  = tempM*tempM;
					tempM3  = tempM2*tempM;
					tempM4  = tempM3*tempM;
					F[ii++] = 0.25f*(p2j*(tempS2 - 2.0f*tempS*tempM2 + tempM4) + Xpt[ind]*(4.0f*tempS*tempM - 4.0f*tempM3) + X2pt[ind]*(6.0f*tempM2 - 2.0f*tempS) + X3pt[ind]*(4.0f*tempM) + X4pt[ind])*cteN;
				}
			}
		}

		for (i = 0 ; i < options.f ; i++)
		{
			temp     = mF[i];
			tempS    = F[i] - (temp*temp);
			if(tempS > 10e-20)
			{
				F[i] = 1.0f/sqrt(tempS);
			}
			else
			{
				F[i] = 0.0f;
			}
		}
	}
	if(palpha > 0.0f)
	{
		for (i = 0 ; i < options.f ; i++)
		{
			temp      = F[i];
			if(temp != 0.0f)
			{
				sign  = 2.0f*(temp > 0.0f) - 1.0f;
				F[i]  = (powf(temp*sign , palpha))*sign;
			}
		}
	}
	if(norm_l2)
	{
		nfisher       = svec_normalize(F , options.f, 2.0f);
	}

	if(X4pt != NULL)
	{
		free(X4pt);
	}
	if(X3pt != NULL)
	{
		free(X3pt);
	}
	if(X2pt != NULL)
	{
		free(X2pt);
	}
	if(Xpt != NULL)
	{
		free(Xpt);
	}
	if(sump !=NULL)
	{
		free(sump);
	}
	if(sump2 !=NULL)
	{
		free(sump2);
	}
	if(p2 != NULL)
	{
		free(p2);
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dfisher_gmm(double *X , double *M , double *S , double *w, double *p, int d , int N , int K , struct opts options, double *mF , double *sF , double *F)
{
	int i , j;
	int ii = 0 , dK = d*K , dN = d*N , KN = K*N, jd , ind;
	int fv_flags_w = options.fv_flags_w, fv_flags_M = options.fv_flags_M, fv_flags_S = options.fv_flags_S;
	int norm_l2 = options.norm_l2 ;
	double alpha = 1.0, beta = 0.0;
	double cteN = 1.0/N;
	double temp , tempM , tempM2 , tempM3, tempM4 , tempS , tempS2, tempw, sum , sum2 , pj , p2j , nfisher;
	double palpha = options.palpha , sign;
	double *sump=NULL ,*sump2=NULL , *Xpt=NULL , *X2pt=NULL , *X3pt=NULL, *X4pt=NULL , *X2=NULL , *X3=NULL, *X4=NULL;
	double *p2=NULL;
	int ismF=1 , issF=1;

	if(mF == NULL)
	{
		ismF = 0;
	}

	if(sF == NULL)
	{
		issF = 0;
	}

	if ( ((!ismF) && (!issF)) || ((ismF) && (issF)) ) 
	{
		if(fv_flags_w)
		{
			sump = (double *) malloc(K*sizeof(double));
			for(j = 0 ; j < K ; j++)
			{
				sum = 0.0;
				for(i = 0 ; i < N ; i++)
				{
					sum += p[j + i*K];
				}
				sump[j] = sum;
				F[ii++] = (sum  - N*w[j])*cteN;
			}
		}

		if(fv_flags_M) 
		{
			if(sump==NULL)
			{
				sump = (double *) malloc(K*sizeof(double));
				for(j = 0 ; j < K ; j++)
				{
					sum = 0.0;
					for(i = 0 ; i < N ; i++)
					{
						sum += p[j + i*K];
					}
					sump[j] = sum;
				}
			}
			Xpt             = (double *) malloc(dK*sizeof(double));
			BLASCALL(dgemm)("Not transposed", "Transposed", &d , &K, &N , &alpha, X , &d , p , &K, &beta, Xpt , &d);

			for (j = 0 ; j < K ; j++)
			{
				jd   = j*d;
				pj   = sump[j];
				for(i = 0 ; i < d ; i++)
				{
					ind     = i + jd;
					tempS   = 1.0/S[ind]; 
					F[ii++] = (Xpt[ind]*tempS - pj*tempS*M[ind])*cteN;
				}
			}
		}
		if(fv_flags_S) 
		{ 
			if(sump==NULL)
			{
				sump = (double *) malloc(K*sizeof(double));
				for(j = 0 ; j < K ; j++)
				{
					sum = 0.0;
					for(i = 0 ; i < N ; i++)
					{
						sum += p[j + i*K];
					}
					sump[j] = sum;
				}
			}
			if(Xpt==NULL)
			{
				Xpt             = (double *) malloc(dK*sizeof(double));
				BLASCALL(dgemm)("Not transposed", "Transposed", &d , &K, &N , &alpha, X, &d , p , &K, &beta, Xpt , &d);
			}

			X2         = (double *) malloc(dN*sizeof(double));
			for(i = 0 ; i < dN ; i++)
			{
				X2[i]  = X[i]*X[i];
			}
			X2pt       = (double *) malloc(dK*sizeof(double));
			BLASCALL(dgemm)("Not transposed", "Transposed", &d , &K , &N , &alpha, X2, &d , p , &K , &beta, X2pt , &d);
			free(X2);

			for (j = 0 ; j < K ; j++)
			{
				jd   = j*d;
				pj   = sump[j];
				for(i = 0 ; i < d ; i++)
				{
					ind     = i + jd;
					tempM   = M[ind];
					F[ii++] = 0.5*( 2.0*tempM*Xpt[ind] - X2pt[ind] + pj*(S[ind] - tempM*tempM))*cteN;
				}
			}
		}
		if((ismF) && (issF))
		{
			for( i = 0 ; i < options.f ; i++)
			{
				F[i] = (F[i] - mF[i])*sF[i];
			}
		}
	}
	else
	{
		if(fv_flags_w)
		{
			p2 = (double *) malloc(KN*sizeof(double));
			for( i = 0 ; i < KN ; i++)
			{
				p2[i] = p[i]*p[i];
			}

			sump  = (double *) malloc(K*sizeof(double));
			sump2 = (double *) malloc(K*sizeof(double));

			for(j = 0 ; j < K ; j++)
			{
				sum  = 0.0;
				sum2 = 0.0;
				for(i = 0 ; i < N ; i++)
				{
					ind   = j + i*K;
					sum  += p[ind];
					sum2 += p2[ind];
				}
				sump[j]  = sum;
				sump2[j] = sum2;
				tempw    = w[j];

				F[ii++]  = (N*tempw*tempw + sum2 - 2.0*tempw*sum)*cteN;
			}
		}

		if(fv_flags_M) 
		{
			if(p2 == NULL)
			{
				p2 = (double *) malloc(KN*sizeof(double));
				for( i = 0 ; i < KN ; i++)
				{
					p2[i] = p[i]*p[i];
				}
			}
			if(sump == NULL)
			{
				sump  = (double *) malloc(K*sizeof(double));
				for(j = 0 ; j < K ; j++)
				{
					sum  = 0.0;
					for(i = 0 ; i < N ; i++)
					{
						ind   = j + i*K;
						sum  += p[ind];
					}
					sump[j]  = sum;
				}
			}
			if(sump2 == NULL)
			{
				sump2  = (double *) malloc(K*sizeof(double));
				for(j = 0 ; j < K ; j++)
				{
					sum2  = 0.0;
					for(i = 0 ; i < N ; i++)
					{
						ind   = j + i*K;
						sum2  += p2[ind];
					}
					sump2[j]  = sum2;
				}
			}

			Xpt        = (double *) malloc(dK*sizeof(double));
			BLASCALL(dgemm)("Not transposed", "Transposed", &d , &K, &N , &alpha, X , &d , p2 , &K, &beta, Xpt , &d);

			X2         = (double *) malloc(dN*sizeof(double));
			for(i = 0 ; i < dN ; i++)
			{
				X2[i]  = X[i]*X[i];
			}
			X2pt       = (double *) malloc(dK*sizeof(double));
			BLASCALL(dgemm)("Not transposed", "Transposed", &d , &K , &N , &alpha, X2 , &d , p2 , &K , &beta, X2pt , &d);
			if(!fv_flags_S)
			{
				free(X2);
			}


			for (j = 0 ; j < K ; j++)
			{
				jd   = j*d;
				pj   = sump[j];
				p2j  = sump2[j];
				for(i = 0 ; i < d ; i++)
				{
					ind     = i + jd;
					tempS   = S[ind]; 
					tempS2  = 1.0/(tempS*tempS);
					tempM   = M[ind];
					tempM2  = tempM*tempM;
					F[ii++] = (tempS2*X2pt[ind] + p2j*tempS2*tempM2 - 2.0*tempS2*tempM*Xpt[ind])*cteN;
				}
			}
		}

		if(fv_flags_S) 
		{
			if(p2 == NULL)
			{
				p2 = (double *) malloc(KN*sizeof(double));
				for( i = 0 ; i < KN ; i++)
				{
					p2[i] = p[i]*p[i];
				}
			}
			if(sump == NULL)
			{
				sump  = (double *) malloc(K*sizeof(double));
				for(j = 0 ; j < K ; j++)
				{
					sum  = 0.0;
					for(i = 0 ; i < N ; i++)
					{
						ind   = j + i*K;
						sum  += p[ind];
					}
					sump[j]  = sum;
				}
			}
			if(sump2 == NULL)
			{
				sump2  = (double *) malloc(K*sizeof(double));
				for(j = 0 ; j < K ; j++)
				{
					sum2  = 0.0;
					for(i = 0 ; i < N ; i++)
					{
						ind   = j + i*K;
						sum2 += p2[ind];
					}
					sump2[j]  = sum2;
				}
			}
			if(Xpt == NULL)
			{
				Xpt        = (double *) malloc(dK*sizeof(double));
				BLASCALL(dgemm)("Not transposed", "Transposed", &d , &K, &N , &alpha, X , &d , p2 , &K, &beta, Xpt , &d);
			}
			if(X2pt == NULL)
			{
				X2         = (double *) malloc(dN*sizeof(double));
				for(i = 0 ; i < dN ; i++)
				{
					X2[i]  = X[i]*X[i];
				}

				X2pt       = (double *) malloc(dK*sizeof(double));
				BLASCALL(dgemm)("Not transposed", "Transposed", &d , &K , &N , &alpha, X2 , &d , p2 , &K , &beta, X2pt , &d);
			}

			X3          = (double *) malloc(dN*sizeof(double));
			for(i = 0 ; i < dN ; i++)
			{
				X3[i]   = X2[i]*X[i];
			}
			free(X2);

			X3pt        = (double *) malloc(dK*sizeof(double));
			BLASCALL(dgemm)("Not transposed", "Transposed", &d , &K, &N , &alpha, X3 , &d , p2 , &K, &beta, X3pt , &d);

			X4          = (double *) malloc(dN*sizeof(double));
			for(i = 0 ; i < dN ; i++)
			{
				X4[i]   = X3[i]*X[i];
			}
			free(X3);

			X4pt        = (double *) malloc(dK*sizeof(double));
			BLASCALL(dgemm)("Not transposed", "Transposed", &d , &K, &N , &alpha, X4 , &d , p2 , &K, &beta, X4pt , &d);
			free(X4);

			for (j = 0 ; j < K ; j++)
			{
				jd   = j*d;
				p2j  = sump2[j];
				for(i = 0 ; i < d ; i++)
				{
					ind     = i + jd;
					tempS   = S[ind]; 
					tempS2  = tempS*tempS;
					tempM   = M[ind];
					tempM2  = tempM*tempM;
					tempM3  = tempM2*tempM;
					tempM4  = tempM3*tempM;
					F[ii++] = 0.25*(p2j*(tempS2 - 2.0*tempS*tempM2 + tempM4) + Xpt[ind]*(4.0*tempS*tempM - 4.0*tempM3) + X2pt[ind]*(6.0*tempM2 - 2.0*tempS) + X3pt[ind]*(4.0*tempM) + X4pt[ind])*cteN;
				}
			}
		}

		for (i = 0 ; i < options.f ; i++)
		{
			temp     = mF[i];
			tempS    = F[i] - (temp*temp);
			if(tempS > 10e-20)
			{
				F[i] = 1.0/sqrt(tempS);
			}
			else
			{
				F[i] = 0.0;
			}
		}
	}
	if(palpha > 0.0)
	{
		for (i = 0 ; i < options.f ; i++)
		{
			temp      = F[i];
			if(temp != 0.0)
			{
				sign  = 2.0*(temp > 0.0) - 1.0;
				F[i]  = (pow(temp*sign , palpha))*sign;
			}
		}
	}

	if(norm_l2)
	{
		nfisher       = dvec_normalize(F , options.f, 2.0);
	}

	if(X4pt != NULL)
	{
		free(X4pt);
	}

	if(X3pt != NULL)
	{
		free(X3pt);
	}
	if(X2pt != NULL)
	{
		free(X2pt);
	}
	if(Xpt != NULL)
	{
		free(Xpt);
	}
	if(sump !=NULL)
	{
		free(sump);
	}
	if(sump2 !=NULL)
	{
		free(sump2);
	}
	if(p2 != NULL)
	{
		free(p2);
	}
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
float svec_normalize(float *X , int n, float norm)
{
	int i;
	float nr = 0.0f;

	for (i = 0 ; i < n ; i++)
	{
		nr += (X[i]*X[i]);
	}
	nr = sqrt(nr);

	if(nr==0.0f)
	{
		return 0.0f;
	}
	else
	{
		nr = 1.0f/nr;
		for (i = 0 ; i < n ; i++)
		{

			X[i] *= nr;
		}
	}
	/*  if(nr!=0)*/
	return nr;
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
double dvec_normalize (double *X , int n, double norm)
{
	int i;
	double nr = 0.0;

	for (i = 0 ; i < n ; i++)
	{
		nr += (X[i]*X[i]);
	}
	nr = sqrt(nr);

	if(nr==0.0)
	{
		return 0.0;
	}
	else
	{
		nr = 1.0/nr;
		for (i = 0 ; i < n ; i++)
		{

			X[i] *= nr;
		}
	}
	/*  if(nr!=0)*/
	return nr;
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */

