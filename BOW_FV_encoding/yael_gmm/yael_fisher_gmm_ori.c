/*

Fast mex Fisher vectors of data X, GMM parameters and GMM probabilities p
(mex-interface modified from the original yael package https://gforge.inria.fr/projects/yael)

- Accept single/double precision input
- Support of BLAS/OpenMP for multi-core computation

Usage
-----

F                                       =    yael_fisher_gmm_ori(X , M , S , w , p , [options]);

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
       fv_flagS_1S                       Include a single value for the sigma diagonal in Fisher vector (default gmm_flag_1S = 0)
	   palpha                            alpha-power normalization if alpha > 0.0 and alpha < 1.0 (default palpha = 0.5) 
	   norm_l2                           L2-normalization of fisher vector (default norm_l2 = 0)

If compiled with the "OMP" compilation flag

       num_threads                       Number of threads   (default num_threads = max number of core)

Outputs
-------

F                                        Fisher vector (d x 1) in single/double format where d = K*(options.fv_flags_w + options.fv_flags_M*d + options.fv_flags_S*d  + fv_flags_1S*1)


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

options.seed                         = 1234543;
options.num_threads                  = 2;
options.verbose                      = 1;

options.gmm_1sigma                   = 0;
options.gmm_flags_norm_l2            = 0;
options.gmm_flags_w                  = 1;


[M , S , w]                          = yael_gmm(X , options);
p                                    = yael_proba_gmm(X , M , S , w , options);

options.fv_flags_w                   = 1;
options.fv_flags_M                   = 1;
options.fv_flags_S                   = 1;
options.fv_flags_1S                  = 0;
options.palpha                       = 0.5;
options.norm_l2                      = 1;


F                                    = yael_fisher_gmm_ori(X , M , S , w , p , options);



To compile
----------

mex  -g   yael_fisher_gmm_ori.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex  -g -DBLAS  yael_fisher_gmm_ori.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


mex  -g -DOMP -DBLAS  yael_fisher_gmm_ori.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex -v -g -DBLAS -DOMP yael_fisher_gmm_ori.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


mex  -f mexopts_intel10.bat -DBLAS -DOMP  yael_fisher_gmm_ori.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex  -f mexopts_intel10.bat -DBLAS yael_fisher_gmm_ori.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

If compiled with OMP option, OMP support

mex -v -DOMP  yael_fisher_gmm_ori.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex -v -DOMP -f mexopts_intel10.bat yael_fisher_gmm_ori.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

If compiled with BLAS & OMP options

mex -v -DBLAS -DOMP  yael_fisher_gmm_ori.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex -v -DBLAS -DOMP -f mexopts_intel10.bat yael_fisher_gmm_ori.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


mex -v -DBLAS -f mexopts_intel10.bat yael_fisher_gmm_ori.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"



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
	int    fv_flags_1S;
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

void  sfisher_gmm(float *, float *, float * , float * , float *, int , int , int , struct opts , float *);
float *svec_new (int );
float svec_normalize(float * , int , float);

void   dfisher_gmm(double *, double *, double * , double * , double *, int , int , int , struct opts , double *);
double *dvec_new (int );
double dvec_normalize(double * , int , double);

/* ------------------------------------------------------------------------------------------------------------------------------------ */
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
	int d , N , K , issingle = 0;
	int f;
	float  *sX , *sM , *sS , *sw , *sp , *sF;
	double *dX , *dM , *dS , *dw , *dp , *dF;

#ifdef OMP 
	struct opts options = {0 , 1 , 1 , 0 , 0 , 0.5 , 0 , 4112 , -1};
#else
	struct opts options = {0 , 1 , 1 , 0 , 0 , 0.5 , 0 , 4112};
#endif
	mxArray *mxtemp;
	double *tmp;
	double temp;
	int tempint;

	if ((nrhs < 5) || (nrhs > 6)) 
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
			"F                                        = yael_fisher_gmm_ori(X , M , S , w , p , [options]);\n"
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
            "       gmm_flag_1S                       Include a single value for the sigma diagonal in Fisher vector (default gmm_flag_1S = 0)\n"
			"       palpha                            alpha-power normalization if alpha > 0.0 and alpha < 1.0 (default palpha = 0.5)\n"
			"       norm_l2                           Don't normalize Fisher vector (default norm_l2 = 0)\n"
#ifdef OMP 
			"       num_threads                       Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)\n"
#endif
			"\n"
			"\n"
			"Outputs\n"
			"-------\n"
			"\n"
            "F                                        Fisher vector (d x 1) in single/double format where d = K*(options.fv_flags_w + options.fv_flags_M*d + options.fv_flags_S*d  + fv_flags_1S*1)\n" 
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
		if( mxIsSingle(prhs[1]) ) 
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
		if( mxIsSingle(prhs[2]) ) 
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
		if( mxIsSingle(prhs[3]) ) 
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
		if( mxIsSingle(prhs[4]) )
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

		mxtemp                            = mxGetField(prhs[5] , 0 , "fv_flags_1S");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0) || (tempint > 1))
			{
				mexPrintf("fv_flags_1S must be ={0,1}, force to 0\n");	
				options.fv_flags_1S      = 0;
			}
			else
			{
				options.fv_flags_1S      = tempint;
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
	else if(options.fv_flags_1S)
	{
		f +=1;
	}

	f        *=K;
	options.f = f;

	/*------------------------ Main Call ----------------------------*/

	if(issingle)
	{
		plhs[0]    = mxCreateNumericMatrix (f , 1 , mxSINGLE_CLASS, mxREAL);  
		sF         = (float*)mxGetPr(plhs[0]);

		sfisher_gmm(sX , sM , sS , sw , sp, d , N , K , options , sF );
	}
	else
	{
		plhs[0]    = mxCreateNumericMatrix (f , 1 , mxDOUBLE_CLASS, mxREAL);  
		dF         = (double*)mxGetPr(plhs[0]);

		dfisher_gmm(dX , dM , dS , dw , dp, d , N , K , options , dF );
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void sfisher_gmm(float *X , float *M , float *S , float *w, float *p, int d , int N , int K , struct opts options, float *dp_dlambda)
{
	int i , j , l;
	int iK , ii = 0 , dK = d*K , dN = d*N , jd , ind;
	int fv_flags_w = options.fv_flags_w, fv_flags_M = options.fv_flags_M, fv_flags_S = options.fv_flags_S, fv_flags_1S = options.fv_flags_1S , gmm_flags_no_norm = options.gmm_flags_no_norm;
	int norm_l2 = options.norm_l2 ;
	float accu , accu2 , f, ctew0 , sum , nf , temp , Stemp , Stemp2 , Mtemp , val , nfisher;
	float alpha = 1.0f, beta = 0.0f;
	float *Xp , *X2 , *X2p , *sum_pj;
	float *dp_dmu , *dp_dsigma;
	float palpha = (float) options.palpha , sign;

	if(fv_flags_w)
	{
		ctew0 = 1.0f/w[0];

#ifdef OOMP 
#pragma omp parallel for default(none)  firstprivate(j,val,f) lastprivate(i,accu,iK) shared(p,w,ctew0,K,N,dp_dlambda,ii)
#endif
		for(j = 1 ; j < K ; j++) 
		{
			accu = 0.0f;
			val  = 1.0f/w[j];
			for(i = 0 ; i < N ; i++)
			{
				iK    = i*K;
				accu += (p[j + iK]*val - p[iK]*ctew0);
			}
			/* normalization */
			f                  = N*(val + ctew0);
			dp_dlambda[ii++]   = accu/sqrt(f);
		}
	}

	if(fv_flags_M) 
	{
		dp_dmu = dp_dlambda + ii;
		Xp     = svec_new(dK);
		BLASCALL(sgemm)("Not transposed", "Transposed", &d , &K, &N , &alpha, X, &d , p , &K, &beta, Xp , &d);
		sum_pj = svec_new(K);

#ifdef OMP 
#pragma omp parallel for default(none)  firstprivate(j) lastprivate(i,sum) shared(p,K,N,sum_pj)
#endif
		for(j = 0 ; j < K ; j++) 
		{        
			sum = 0.0f;        
			for(i = 0 ; i < N ; i++) 
			{
				sum += p[j + i*K];   
			}
			sum_pj[j] = sum;
		}

#ifdef OMP 
#pragma omp parallel for default(none)  firstprivate(j) lastprivate(i,jd,ind) shared(Xp,M,S,dp_dmu,sum_pj,K,d)
#endif
		for(j = 0 ; j < K ; j++) 
		{
			jd = j*d;
			for(i = 0 ; i < d ; i++)
			{
				ind         = i + jd;
				dp_dmu[ind] = (Xp[ind] - M[ind]*sum_pj[j])/S[ind];
			}
		}

		if(!gmm_flags_no_norm) 
		{
			for(j = 0 ; j < K ; j++) 
			{
				jd   = j*d;
				temp = N*w[j]; 
				for(i = 0 ; i < d ; i++) 
				{
					ind  = i + jd;
					nf   = sqrt(temp/S[ind]);
					if(nf > 0.0f) 
					{
						dp_dmu[ind] /= nf;
					}
				} 
			}
		}
		ii += dK;
	}

	if((fv_flags_S) || (fv_flags_1S))
	{
		if(fv_flags_1S) 
		{ /* fast not implemented for 1 sigma */

			for(j = 0 ; j < K ; j++) 
			{
				jd    = j*d;
				accu2 = 0.0f;
				val   = 2.0f*N*w[j];
				for(l = 0 ; l < d ; l++) 
				{
					accu    = 0.0f;
					Stemp   = S[l + jd];
					Stemp2  = 1.0f/sqrt(Stemp);
					Mtemp   = M[l + jd];

					for(i = 0 ; i < N ; i++) 
					{
						temp  =  (X[l + i*d] - M[l + jd]);
						accu +=  p[j + i*K] * ((temp*temp)/Stemp - 1.0f)*Stemp2;
					}

					if(fv_flags_S) 
					{
						f                = gmm_flags_no_norm ? 1.0f : val/S[l + jd];
						dp_dlambda[ii++] = accu/sqrt(f);
					} 
					accu2 += accu;        
				}

				if(fv_flags_1S) 
				{
					f                = gmm_flags_no_norm ? 1.0f : d*val/S[jd];
					dp_dlambda[ii++] = accu2/sqrt(f);        
				}
			}  
		} 
		else 
		{ 
			dp_dsigma = dp_dlambda + ii;
			if(!Xp) 
			{
				Xp     = svec_new(dK);
				BLASCALL(sgemm)("Not transposed", "Transposed", &d , &K, &N , &alpha, X , &d , p , &K, &beta , Xp , &d);
			}

			if(!sum_pj) 
			{
				sum_pj = svec_new(K);
#ifdef OMP 
#pragma omp parallel for default(none)  firstprivate(j) lastprivate(i,sum) shared(p,K,N,sum_pj)
#endif
				for(j = 0 ; j < K ; j++) 
				{        
					sum = 0.0f;        
					for(i = 0 ; i < N ; i++) 
					{
						sum += p[j + i*K];  
					}
					sum_pj[j] = sum;
				}
			}

			X2 = svec_new(dN);
			for(i = dN - 1 ; i >= 0 ; i--) 
			{
				X2[i] = (X[i]*X[i]);
			}

			X2p = svec_new(dK);
			BLASCALL(sgemm)("Not transposed" , "Transposed", &d , &K , &N , &alpha, X2 , &d , p , &K , &beta , X2p , &d);

			free(X2);

#ifdef OMP 
#pragma omp parallel for default(none)  firstprivate(j,jd,val) lastprivate(l,ind,accu,Mtemp,Stemp,f) shared(K,d,N,w,Xp,X2p,M,S,sum_pj,gmm_flags_no_norm,dp_dsigma)
#endif
			for(j = 0 ; j < K ; j++) 
			{
				jd  = j*d;
				val = 2.0f*N*w[j]; 
				for(l = 0 ; l < d ; l++) 
				{
					ind   = l + jd;
					accu  = X2p[ind];
					Mtemp = M[ind];
					Stemp = S[ind];
					accu += Xp[ind]*(-2.0f*Mtemp);
					accu += sum_pj[j]*(Mtemp*Mtemp  - Stemp);

					/* normalization */

					if(gmm_flags_no_norm) 
					{
						f = pow(Stemp , -1.5);
					} 
					else 
					{
						f = 1.0f/(Stemp*sqrt(val));
					}
					dp_dsigma[l + jd] = accu * f;
				}
			}  
			free(X2p);
			ii += dK;
		}
	}

	if(palpha > 0.0f)
	{
		for (i = 0 ; i < options.f ; i++)
		{
			temp          = dp_dlambda[i];
			if(temp != 0.0f)
			{
				sign          = 2.0f*(temp > 0.0f) - 1.0f;
				dp_dlambda[i] = (powf(temp*sign , palpha))*sign;
			}
		}
	}

	if(norm_l2)
	{
		nfisher = svec_normalize(dp_dlambda , options.f, 2.0f);
	}
	if(!sum_pj)
	{
		free(sum_pj);
	}
	if(!Xp)
	{
		free(Xp);
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dfisher_gmm(double *X , double *M , double *S , double *w, double *p, int d , int N , int K , struct opts options, double *dp_dlambda)
{
	int i , j , l;
	int iK , ii = 0 , dK = d*K , dN = d*N , jd , ind;
	int fv_flags_w = options.fv_flags_w, fv_flags_M = options.fv_flags_M, fv_flags_S = options.fv_flags_S, fv_flags_1S = options.fv_flags_1S , gmm_flags_no_norm = options.gmm_flags_no_norm;
	int norm_l2 = options.norm_l2 ;
	double accu , accu2 , f, ctew0 , sum , nf , temp , Stemp , Stemp2 , Mtemp , val , nfisher;
	double alpha = 1.0f, beta = 0.0f;
	double *Xp , *X2 , *X2p , *sum_pj;
	double *dp_dmu , *dp_dsigma;
	float palpha = options.palpha , sign;

	if(fv_flags_w)
	{
		ctew0 = 1.0/w[0];
		for(j = 1 ; j < K ; j++) 
		{
			accu = 0.0;
			val  = 1.0/w[j];
			for(i = 0 ; i < N ; i++)
			{
				iK    = i*K;
				accu += (p[j + iK]*val - p[iK]*ctew0);
			}
			/* normalization */
			f                  = N*(val + ctew0);
			dp_dlambda[ii++]   = accu/sqrt(f);
		}
	}

	if(fv_flags_M) 
	{
		dp_dmu = dp_dlambda + ii;
		Xp     = dvec_new(dK);
		BLASCALL(dgemm)("Not transposed", "Transposed", &d , &K, &N , &alpha, X, &d , p , &K, &beta, Xp , &d);
		sum_pj = dvec_new(K);
		for(j = 0 ; j < K ; j++) 
		{        
			sum = 0.0;  
#ifdef OMP 
#pragma omp parallel for default(none)  firstprivate(j) lastprivate(i,sum) shared(p,K,N,sum_pj)
#endif
			for(i = 0 ; i < N ; i++) 
			{
				sum += p[j + i*K];   
			}
			sum_pj[j] = sum;
		}

#ifdef OMP 
#pragma omp parallel for default(none)  firstprivate(j) lastprivate(i,jd,ind) shared(Xp,M,S,dp_dmu,sum_pj,K,d)
#endif
		for(j = 0 ; j < K ; j++) 
		{
			jd = j*d;
			for(i = 0 ; i < d ; i++)
			{
				ind         = i + jd;
				dp_dmu[ind] = (Xp[ind] - M[ind]*sum_pj[j]) / S[ind];
			}
		}

		if(!gmm_flags_no_norm) 
		{
			for(j = 0 ; j < K ; j++) 
			{
				jd   = j*d;
				temp = N*w[j]; 
				for(i = 0 ; i < d ; i++) 
				{
					ind  = i + jd;
					nf   = sqrt(temp/S[ind]);
					if(nf > 0.0) 
					{
						dp_dmu[ind] /= nf;
					}
				} 
			}
		}
		ii += dK;
	}

	if((fv_flags_S) || (fv_flags_1S))
	{

		if(fv_flags_1S) 
		{ /* fast not implemented for 1 sigma */

			for(j = 0 ; j < K ; j++) 
			{
				jd    = j*d;
				accu2 = 0.0;
				val   = 2.0*N*w[j];
				for(l = 0 ; l < d ; l++) 
				{
					accu    = 0.0;
					Stemp   = S[l + jd];
					Stemp2  = 1.0/sqrt(Stemp);
					Mtemp   = M[l + jd];

					for(i = 0 ; i < N ; i++) 
					{
						temp  =  (X[l + i*d] - M[l + jd]);
						accu +=  p[j + i*K] * ((temp*temp)/Stemp - 1.0)*Stemp2;
					}

					if(fv_flags_S) 
					{
						f                = gmm_flags_no_norm ? 1.0 : val/S[l + jd];
						dp_dlambda[ii++] = accu/sqrt(f);
					} 
					accu2 += accu;        
				}

				if(fv_flags_1S) 
				{
					f                = gmm_flags_no_norm ? 1.0 : d*val/S[jd];
					dp_dlambda[ii++] = accu2/sqrt(f);        
				}
			}  
		} 
		else 
		{ 
			dp_dsigma = dp_dlambda + ii;
			if(!Xp) 
			{
				Xp     = dvec_new(dK);
				BLASCALL(dgemm)("Not transposed", "Transposed", &d , &K, &N , &alpha, X , &d , p , &K, &beta , Xp , &d);
			}

			if(!sum_pj) 
			{
				sum_pj = dvec_new(K);

#ifdef OMP 
#pragma omp parallel for default(none)  firstprivate(j) lastprivate(i,sum) shared(p,K,N,sum_pj)
#endif
				for(j = 0 ; j < K ; j++) 
				{        
					sum = 0.0;        
					for(i = 0 ; i < N ; i++) 
					{
						sum += p[j + i*K];  
					}
					sum_pj[j] = sum;
				}
			}

			X2 = dvec_new(dN);
			for(i = dN - 1 ; i >= 0 ; i--) 
			{
				X2[i] = (X[i]*X[i]);
			}

			X2p = dvec_new(dK);
			BLASCALL(dgemm)("Not transposed" , "Transposed", &d , &K , &N , &alpha, X2 , &d , p , &K , &beta , X2p , &d);

			free(X2);

#ifdef OMP 
#pragma omp parallel for default(none)  firstprivate(j,jd,val) lastprivate(l,ind,accu,Mtemp,Stemp,f) shared(K,d,N,w,Xp,X2p,M,S,sum_pj,gmm_flags_no_norm,dp_dsigma)
#endif
			for(j = 0 ; j < K ; j++) 
			{
				jd  = j*d;
				val = 2.0*N*w[j]; 
				for(l = 0 ; l < d ; l++) 
				{
					ind   = l + jd;
					accu  = X2p[ind];
					Mtemp = M[ind];
					Stemp = S[ind];
					accu += Xp[ind]*(-2.0*Mtemp);
					accu += sum_pj[j]*(Mtemp*Mtemp  - Stemp);

					/* normalization */

					if(gmm_flags_no_norm) 
					{
						f = pow(Stemp , -1.5);
					} 
					else 
					{
						f = 1.0/(Stemp*sqrt(val));
					}
					dp_dsigma[l + jd] = accu * f;
				}
			}  
			free(X2p);
			ii += dK;
		}
	}
	if(palpha > 0.0)
	{
		for (i = 0 ; i < options.f ; i++)
		{
			temp          = dp_dlambda[i];
			if(temp != 0.0)
			{
				sign          = 2.0*(temp > 0.0) - 1.0;
				dp_dlambda[i] = (powf(temp*sign , palpha))*sign;
			}
		}
	}

	if(norm_l2)
	{
		nfisher = dvec_normalize(dp_dlambda , options.f, 2.0f);
	}
	if(!sum_pj)
	{
		free(sum_pj);
	}
	if(!Xp)
	{
		free(Xp);
	}
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
float *svec_new(int n)
{
  float *ret = (float *) malloc (n*sizeof(float));
  if (!ret) 
  {
    fprintf (stderr, "svec_new %ld : out of memory\n", n);
    abort();
  }
  return ret;
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
double *dvec_new(int n)
{
  double *ret = (double *) malloc (n*sizeof(double));
  if (!ret) 
  {
    fprintf(stderr, "dvec_new %ld : out of memory\n", n);
    abort();
  }
  return ret;
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
