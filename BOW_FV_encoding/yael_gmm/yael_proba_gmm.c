/*

Fast mex GMM probabilities given input data X
(mex-interface modified from the original yael package https://gforge.inria.fr/projects/yael)

- Accept single/double precision input
- Support of BLAS/OpenMP for multi-core computation

Usage
-----

p                                  =    yael_proba_gmm(X , M , S , w , [options]);

Inputs
------

X                                        Input data matrix (d x N) in single/double format 
M                                        Means matrix of GMM (d x K) in single/double format 
S                                        Variance matrix of GMM (d x K) in single/double format
w                                        Weights vector of GMM (1 x K) in single/double format


options
	   gmm_1sigma                        Compute a single value for the sigma diagonal (default gmm_1sigma = 0)
	   gmm_flags_no_norm                 No normalization in the GMM update (default gmm_flags_no_norm = 0) 
	   gmm_flags_w                       Take weighs into account during probabilities update (default gmm_flags_w = 1) 

If compiled with the "OMP" compilation flag

       num_threads                       Number of threads   (default num_threads = max number of core)

Outputs
-------

p                                        GMM probabilities matrix (K x N) in single/double format for data X 


Example 1
---------

clear

d                                    = 128;                 % dimensionality of the vectors
N                                    = 1000;                % number of vectors

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
options.gmm_flags_no_norm            = 0;
options.gmm_flags_w                  = 1;

options.seed                         = 1234543;
options.num_threads                  = 2;
options.verbose                      = 1;


[M , S , w]                          = yael_gmm(X , options);
p                                    = yael_proba_gmm(X , M , S , w , options);


To compile
----------

mex  -g   yael_proba_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex  -v -g -DOMP -DBLAS  yael_proba_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex -v -g -DBLAS -DOMP yael_proba_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


mex  -f mexopts_intel10.bat -DBLAS -DOMP  yael_proba_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

If compiled with OMP option, OMP support

mex -v -DOMP  yael_proba_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex -v -DOMP -f mexopts_intel10.bat yael_proba_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

If compiled with BLAS & OMP options

mex -v -DBLAS -DOMP  yael_proba_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


mex -v -DBLAS -DOMP -f mexopts_intel10.bat yael_proba_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"



References  [1] Hervé Jégou, Matthijs Douze and Cordelia Schmid, 
----------      "Product quantization for nearest neighbor search"
                IEEE Transactions on Pattern Analysis and Machine Intelligence

            [2] Florent Perronnin, Jorge Sánchez, Thomas Mensink,
		        "Improving the Fisher Kernel for Large-Scale Image Classification", ECCV' 10


Author : Sébastien PARIS : sebastien.paris@lsis.org
-------  

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

#define M_PI 3.14159265358979323846

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
	int    K;
	int    gmm_1sigma;
	int    gmm_flags_no_norm;
	int    gmm_flags_w;
#ifdef OMP 
    int    num_threads;
#endif
};

#ifdef OMP 

typedef struct 
{
  int n,K,d;
  float *X;
  float *M;
  float *S;
  float *w;
  float *p;

  int gmm_flags_no_norm;
  int gmm_flags_w;
  int num_threads; 

} scompute_p_params_t;

typedef struct 
{
  int n,K,d;
  double *X;
  double *M;
  double *S;
  double *w;
  double *p;

  int gmm_flags_no_norm;
  int gmm_flags_w;
  int num_threads; 

} dcompute_p_params_t;
#endif

/* ------------------------------------------------------------------------------------------------------------------------------------ */
extern void   BLASCALL(sgemm)(char *, char *, int*, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *);
extern void   BLASCALL(dgemm)(char *, char *, int*, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);


#ifdef OMP 
void compute_tasks(int , int , void (*task_fun) (void *, int , int ) , void *);
#endif

void  sproba_gmm(float *, float *, float * , float * , int , int , int , struct opts , float *);
void  sgmm_compute_params(float *, int , int , struct opts , float * , float * , float * , float * , float * , float * , float *);
void  sgmm_compute_p(float *, int , int , int , float * , float * , float * , float * , int , int );
void  scompute_mahalanobis_sqr(float * , int , int , int , float * , float * , float *);


#ifdef OMP
void  sgmm_compute_p_thread(float * , int , int , int , struct opts , float * , float *, float * , float *);
void  scompute_p_task_fun(void * , int , int ); 
#endif

float *svec_new(int );
float *svec_new_cpy(float *, int); 
void  svec_sqr (float * , int );
float slog_sum (float , float);

void   dproba_gmm(double *, double *, double * , double * , int , int , int , struct opts , double *);
void   dgmm_compute_params(double *, int , int , struct opts , double * , double * , double * , double * , double * , double * , double *);
void   dgmm_compute_p(double *, int , int , int , double * , double * , double * , double * , int , int );
void   dcompute_mahalanobis_sqr(double * , int , int , int , double * , double * , double *);

#ifdef OMP
void   dgmm_compute_p_thread(double * , int , int , int , struct opts , double * , double *, double * , double *);
void   dcompute_p_task_fun(void * , int , int ); 
#endif

double *dvec_new(int );
double *dvec_new_cpy(double *, int); 
void   dvec_sqr (double * , int );
double dlog_sum (double , double);

/* ------------------------------------------------------------------------------------------------------------------------------------ */
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
	int d,N,K,issingle = 0;
	float  *sX , *sM , *sS , *sw , *sp;
	double *dX , *dM , *dS , *dw , *dp;

#ifdef OMP 
	struct opts options = {16 , 0 , 0 , 1 , -1};
#else
	struct opts options = {16 , 0 , 0 , 1};
#endif
	mxArray *mxtemp;
	double *tmp;
	int tempint;

	if ((nrhs < 4) || (nrhs > 5)) 
	{
		mexPrintf(
			"\n"
			"\n"
			"Fast mex GMM probabilities given input data X\n"
			"(mex-interface modified from the original yael package https://gforge.inria.fr/projects/yael)\n"
			"\n"
			"- Accept single/double precision input\n"
			"- Support of BLAS/OpenMP for multi-core computation\n"
			"\n"
			"Usage\n"
			"-----\n"
			"\n"
			"p                                        = yael_proba_gmm(X , M , S , w , [options]);\n"
			"\n"
			"\n"
			"Inputs\n"
			"------\n"
			"\n"
			"X                                        Input data matrix (d x N) in single/double format\n"
            "M                                        Means matrix of GMM (d x K) in single/double format\n" 
            "S                                        Variance matrix of GMM (d x K) in single/double format\n"
            "w                                        Weights vector of GMM (1 x K) in single/double format\n"
			"\n"
			"options\n"
	        "       gmm_1sigma                        Compute a single value for the sigma diagonal (default gmm_1sigma = 0)\n"
	        "       gmm_flags_no_norm                 No normalization in the GMM update (default gmm_flags_no_norm = 0)\n" 
	        "       gmm_flags_w                       Take weighs into account during probabilities update (default gmm_flags_w = 0)\n" 
#ifdef OMP 
			"       num_threads                       Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)\n"
#endif
			"\n"
			"\n"
			"Outputs\n"
			"-------\n"
			"\n"
            "p                                        GMM probabilities matrix (K x N) in single/double format for data X \n" 
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
		options.K   = K;
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

	if ((nrhs > 4) && !mxIsEmpty(prhs[4]))
	{

		mxtemp                            = mxGetField(prhs[4] , 0 , "gmm_1sigma");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0))
			{
				mexPrintf("gmm_1sigma must {0,1}, force to 0\n");	
				options.gmm_1sigma        = 0;
			}
			else
			{
				options.gmm_1sigma        = tempint;
			}
		}


		mxtemp                            = mxGetField(prhs[4] , 0 , "gmm_flags_no_norm");
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

		mxtemp                            = mxGetField(prhs[4] , 0 , "gmm_flags_w");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0) || (tempint > 1))
			{
				mexPrintf("gmm_flags_w must be ={0,1}, force to 0\n");	
				options.gmm_flags_w       = 0;
			}
			else
			{
				options.gmm_flags_w       = tempint;
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

	/*------------------------ Main Call ----------------------------*/

	if(issingle)
	{
		plhs[0]    = mxCreateNumericMatrix (K , N , mxSINGLE_CLASS, mxREAL);  
		sp         = (float*)mxGetPr(plhs[0]);

		sproba_gmm(sX , sM , sS , sw , d , N , K , options , sp );
	}
	else
	{
		plhs[0]    = mxCreateNumericMatrix (K , N , mxDOUBLE_CLASS, mxREAL);  
		dp         = (double*)mxGetPr(plhs[0]);

		dproba_gmm(dX , dM , dS , dw , d , N , K , options , dp );
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void sproba_gmm(float *X , float *M , float *S , float *w, int d , int N , int K , struct opts options, float *p)
{
	int num_threads = options.num_threads;
	num_threads     = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;

#ifdef OMP 
	sgmm_compute_p_thread(X , d , N , num_threads, options , M , S , w , p);
#else
	sgmm_compute_p(X , d , N , K , M , S , w , p , options.gmm_flags_no_norm , options.gmm_flags_w);
#endif
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dproba_gmm(double *X , double *M , double *S , double *w, int d , int N , int K , struct opts options, double *p)
{
    int num_threads = options.num_threads;
    num_threads     = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;

#ifdef OMP 
	dgmm_compute_p_thread(X , d , N , num_threads , options , M , S , w , p);
#else
	dgmm_compute_p(X , d , N , K , M , S , w , p , options.gmm_flags_no_norm , options.gmm_flags_w);
#endif
}

int svec_arg_max(float *X , int N) 
{
	float m = X[0];
	int i,i0 = 0;
	for (i = 1 ; i < N ; i++) 
	{
		if (X[i] > m)
		{
			m  = X[i]; 
			i0 = i; 
		}
	}
	return i0;
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
int dvec_arg_max(double *X , int N) 
{
	double m = X[0];
	int i,i0 = 0;
	for (i = 1 ; i < N ; i++) 
	{
		if (X[i] > m)
		{
			m  = X[i]; 
			i0 = i; 
		}
	}
	return i0;
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void svec_sqr(float *X, int n)
{
	int i;
	for (i = 0 ; i < n ; i++)
	{
		X[i] *= X[i];
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dvec_sqr(double *X, int n)
{
	int i;
	for (i = 0 ; i < n ; i++)
	{
		X[i] *= X[i];
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
float *svec_new_cpy (float *X, int n) 
{
  float *ret = svec_new(n);
  memcpy(ret, X , n*sizeof (*ret));
  return ret;  
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
double *dvec_new_cpy (double *X, int n) 
{
  double *ret = dvec_new(n);
  memcpy (ret, X , n*sizeof (*ret));
  return ret;  
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
#ifdef OMP
void compute_tasks(int n , int num_threads , void (*task_fun) (void *arg, int tid , int i) , void *task_arg)
{
	int i;
	omp_set_num_threads(num_threads); 

#pragma omp parallel for schedule(dynamic) 
	for(i = 0 ; i < n ; i++) 
	{
		(*task_fun)(task_arg , omp_get_thread_num(), i);
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void sgmm_compute_p_thread(float *X , int d , int N , int num_threads , struct opts options , float *M , float *S, float *w , float *p)
{
	scompute_p_params_t t = {N , options.K , d, X , M , S , w , p , options.gmm_flags_no_norm , options.gmm_flags_w , num_threads};
	compute_tasks(num_threads , num_threads , &scompute_p_task_fun , &t);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dgmm_compute_p_thread(double *X , int d , int N , int num_threads , struct opts options , double *M , double *S , double *w , double *p)
{
	dcompute_p_params_t t = {N , options.K , d, X , M , S , w , p , options.gmm_flags_no_norm , options.gmm_flags_w , num_threads};
	compute_tasks(num_threads , num_threads , &dcompute_p_task_fun , &t);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void scompute_p_task_fun (void *arg, int tid, int i) 
{
  scompute_p_params_t *t=arg;
  int n0 = i*t->n/t->num_threads;
  int n1 = (i+1)*t->n/t->num_threads;

  sgmm_compute_p(t->X + t->d*n0, t->d, n1-n0 , t->K, t->M , t->S , t->w , t->p + t->K*n0, t->gmm_flags_no_norm , t->gmm_flags_w);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dcompute_p_task_fun (void *arg, int tid, int i) 
{
  dcompute_p_params_t *t=arg;
  int n0 = i*t->n/t->num_threads;
  int n1 = (i+1)*t->n/t->num_threads;

  dgmm_compute_p(t->X + t->d*n0, t->d, n1-n0 , t->K, t->M , t->S , t->w , t->p + t->K*n0, t->gmm_flags_no_norm , t->gmm_flags_w);
}
#endif
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void sgmm_compute_p(float *X, int d , int n , int K, float *M, float *S, float *w , float *p , int gmm_flags_no_norm , int gmm_flags_w)
{
  int i, j;
  int jd , iK;
  float tmp;
  float *logdetnr;
  float log_p , log_norm;

  if(n==0) 
  {
	  return; /* sgemm doesn't like empty matrices */
  }

  logdetnr = svec_new(K);

  for (j = 0 ; j < K ; j++) 
  {
	  jd  = j*d;
	  logdetnr[j] = -d / 2.0f * log(2.0f*M_PI);
	  for (i = 0 ; i < d ; i++)
	  {
		  logdetnr[j] -= 0.5f * log(S[i + jd]);
	  }
  }

  /* compute all probabilities in log domain */

  /* compute squared Mahalanobis distances (result in p) */
  /* complicated & fast */

  scompute_mahalanobis_sqr(X , d , n , K , M , S , p);

  /* convert distances to probabilities, staying in the log domain 
     until the very end */
  for (i = 0 ; i < n ; i++) 
  {
	  iK = i*K;
	  for (j = 0 ; j < K ; j++) 
	  {
		  p[j + iK] = logdetnr[j] - 0.5f*p[j + iK];
	  }

	  /* at this point, we have p(x|ci) -> we want p(ci|x) */

	  if(gmm_flags_no_norm) 
	  {    
		  /* compute the normalization factor */
		  tmp = 0.0f;
	  } 
	  else 
	  {
		  tmp = p[0 + iK]; 
		  if(gmm_flags_w) 
		  {
			  tmp +=log(w[0]);
		  }

		  for (j = 1 ; j < K ; j++) 
		  {
			  log_p = p[j + iK];
			  if(gmm_flags_w)
			  {
				  log_p += log(w[j]);
			  }
			  tmp = slog_sum(tmp, log_p);
		  }

		  /* now dtmp contains the log of sums */
	  } 

	  for (j = 0 ; j < K ; j++) 
	  {
		  log_norm = 0.0f;
		  if(gmm_flags_w)
		  {
			  log_norm = log(w[j]) - tmp;
		  }
		  else
		  {
			  log_norm = -tmp;
		  }
		  p[j + iK] = exp(p[j + iK] + log_norm);
	  }
  }
  free(logdetnr);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dgmm_compute_p(double *X, int d , int n , int K , double *M , double *S , double *w , double *p , int gmm_flags_no_norm , int gmm_flags_w)
{
  int i, j;
  int jd , iK;
  double tmp;
  double *logdetnr;
  double log_p , log_norm;

  if(n==0) 
  {
	  return; /* sgemm doesn't like empty matrices */
  }

  logdetnr = dvec_new(K);

  for (j = 0 ; j < K ; j++) 
  {
	  jd  = j*d;
	  logdetnr[j] = -d / 2.0 * log(2.0*M_PI);
	  for (i = 0 ; i < d ; i++)
	  {
		  logdetnr[j] -= 0.5*log(S[i + jd]);
	  }
  }

  /* compute all probabilities in log domain */

  /* compute squared Mahalanobis distances (result in p) */

  dcompute_mahalanobis_sqr(X , d , n , K , M , S , p);

  /* convert distances to probabilities, staying in the log domain  until the very end */
  for (i = 0 ; i < n ; i++) 
  {
	  iK = i*K;
	  for (j = 0 ; j < K ; j++) 
	  {
		  p[j + iK] = logdetnr[j] - 0.5*p[j + iK];
	  }

	  /* at this point, we have p(x|ci) -> we want p(ci|x) */

	  if(gmm_flags_no_norm) 
	  {    
		  /* compute the normalization factor */
		  tmp = 0.0;
	  } 
	  else 
	  {
		  tmp = p[0 + iK]; 
		  if(gmm_flags_w) 
		  {
			  tmp +=log(w[0]);
		  }

		  for (j = 1 ; j < K ; j++) 
		  {
			  log_p = p[j + iK];
			  if(gmm_flags_w)
			  {
				  log_p += log(w[j]);
			  }
			  tmp = dlog_sum(tmp , log_p);
		  }

		  /* now dtmp contains the log of sums */
	  } 

	  for (j = 0 ; j < K ; j++) 
	  {
		  log_norm = 0.0;
		  if(gmm_flags_w)
		  {
			  log_norm = log(w[j]) - tmp;
		  }
		  else
		  {
			  log_norm = -tmp;
		  }
		  p[j + iK] = exp(p[j + iK] + log_norm);
	  }
  }
  free(logdetnr);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
float slog_sum (float log_a , float log_b)
{
  float log_s;
  if (log_a < log_b)
  {
    log_s = log_b + log (1.0f + exp (log_a - log_b));
  }
  else
  {
    log_s = log_a + log (1.0f + exp (log_b - log_a));
  }
  return log_s;
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
double dlog_sum (double log_a , double log_b)
{
  double log_s;
  if (log_a < log_b)
  {
    log_s = log_b + log (1.0 + exp (log_a - log_b));
  }
  else
  {
    log_s = log_a + log (1.0 + exp (log_b - log_a));
  }
  return log_s;
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void scompute_mahalanobis_sqr(float *X , int  d , int n , int K , float *M , float *S , float *p) 
{
  float tmp, sum;
  int i, j, l, iK , jd, dn = d*n, dK = d*K;
  float one = 1.0f , minus_two = -2.0f;
  float *X2 , *inv_S , *M_S;  
  float *M2_sums = svec_new(K);
  
  for (j = 0 ; j < K ; j++) 
  {
	  jd  = j*d;
	  sum = 0.0f;
	  for (l = 0 ; l < d ; l++) 
	  {
		  tmp  = M[l + jd];
		  sum += (tmp*tmp) / S[l + jd];      
	  }
	  M2_sums[j] = sum;
  }
  
  for (i = 0 ; i < n ; i++) 
  {
	  iK = i*K;
	  for (j = 0 ; j < K ; j++) 
	  {
		  p[j + iK] = M2_sums[j];
	  }
  }
  free(M2_sums);
  
  X2 = svec_new(dn);
  for (i = 0 ; i < dn ; i++) 
  {
	  X2[i] = X[i]*X[i];
  }
  
  inv_S = svec_new(dK);
  for (i = 0 ; i < dK ; i++) 
  {
	  inv_S[i] = 1.0f/S[i];
  }
    
  BLASCALL(sgemm)("Transposed","Not transposed",&K , &n , &d , &one , inv_S , &d , X2 , &d , &one , p , &K);
  
  free(X2);
  
  M_S = inv_S;
  for (i = 0 ; i < dK ; i++)
  {
    M_S[i] = M[i]/S[i];
  }
  
  BLASCALL(sgemm)("Transposed","Not transposed" , &K , &n , &d  , &minus_two , M_S , &d , X , &d , &one , p , &K);  
  
  free(M_S);      
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dcompute_mahalanobis_sqr(double *X , int  d , int n , int K , double *M , double *S , double *p) 
{
  double tmp, sum;
  int i, j , l, iK , jd , dn = d*n , dK = d*K;
  double one = 1.0 , minus_two = -2.0;
  double *X2 , *inv_S , *M_S;
  double *M2_sums = dvec_new(K);
  
  for (j = 0 ; j < K ; j++) 
  {
	  jd  = j*d;
	  sum = 0.0;
	  for (l = 0 ; l < d ; l++) 
	  {
		  tmp  = M[l + jd];
		  sum += (tmp*tmp)/S[l + jd];      
	  }
	  M2_sums[j] = sum;
  }
  
  for (i = 0 ; i < n ; i++) 
  {
	  iK = i*K;
	  for (j = 0 ; j < K ; j++) 
	  {
		  p[j + iK] = M2_sums[j];
	  }
  }
  free(M2_sums);
  
  X2 = dvec_new(dn);
  for (i = 0 ; i < dn ; i++) 
  {
	  X2[i] = X[i]*X[i];
  }
  
  inv_S = dvec_new(dK);
  for (i = 0 ; i < dK ; i++) 
  {
	  inv_S[i] = 1.0/S[i];
  }
    
  BLASCALL(dgemm)("Transposed","Not transposed" ,&K , &n , &d , &one , inv_S , &d , X2 , &d , &one , p , &K);
  
  free(X2);
  
  M_S = inv_S;
  for (i = 0 ; i < dK ; i++)
  {
    M_S[i] = M[i]/S[i];
  }
  
  BLASCALL(dgemm)("Transposed","Not transposed" , &K , &n , &d  , &minus_two , M_S , &d , X , &d , &one , p , &K);  
  
  free(M_S);      
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */