/*

Fast mex GMM training algorithm with Kmeans initialization
(mex-interface modified from the original yael package https://gforge.inria.fr/projects/yael)

- Accept single/double precision input
- Support of BLAS/OpenMP for multi-core computation

Usage
------

[M , S , w]                              =    yael_gmm(X , [options]);


Inputs
-------

X                                        Input data matrix (d x N) in single/double format 

options
       K                                 Number of centroid  (default K = 10)
       max_ite_kmeans                    Number of iteration for the initial Kmeans (default max_ite_kmeans = 10)
       max_ite_gmm                       Number of iteration for GMM learning (default max_ite_gmm = 10)
	   gmm_1sigma                        Compute a single value for the sigma diagonal (default gmm_1sigma = 0)
	   gmm_flags_no_norm                 No normalization in the GMM update (default gmm_flags_no_norm = 0) 
	   gmm_flags_w                       Take weighs into account during probabilities update (default gmm_flags_w = 1) 
       redo                              Number of time to restart K-means (default redo = 1)
       verbose                           Verbose level = {0,1} (default verbose = 0)
       init_random_mode                  0 <=> Kmeans++ initialization, 1<=> random selection ...
       normalize_sophisticated_mode      0/1 (No/Yes)
       BLOCK_N1                          Cache size block (default BLOCK_N1 = 1024)
       BLOCK_N2                          Cache size block (default BLOCK_N2 = 1024)
       seed                              Seed number for internal random generator (default random seed according to time)

If compiled with the "OMP" compilation flag

       num_threads                       Number of threads   (default num_threads = max number of core)

Outputs
-------

M                                        Means matrix of GMM (d x K) in single/double format 
S                                        Variance matrix of GMM (d x K) in single/double format
w                                        Weights vector of GMM (1 x K) in single/double format


Example 1
---------

clear

d                                    = 128;                   % dimensionality of the vectors
N                                    = 1000;                % number of vectors

X                                    = randn(d, N , 'single'); % random set of vectors 

options.K                            = 100;
options.max_ite_kmeans               = 10;
options.max_ite_gmm                  = 1;
options.init_random_mode             = 0;
options.redo                         = 1;
options.normalize_sophisticated_mode = 0;
options.BLOCK_N1                     = 1024;
options.BLOCK_N2                     = 1024;
options.gmm_1sigma                   = 0;
options.gmm_flags_no_norm            = 0;
options.gmm_flags_w                  = 0;

options.seed                         = 1234543;
options.num_threads                  = -1;
options.verbose                      = 1;


tic,[M , S , w]                      = yael_gmm(X , options);,toc


Example 2
---------

clear

d                                    = 128;                   % dimensionality of the vectors
N                                    = 100000;                % number of vectors

X                                    = randn(d, N); % random set of vectors 

options.K                            = 100;
options.max_ite_kmeans               = 10;
options.max_ite_gmm                  = 10;
options.init_random_mode             = 0;
options.normalize_sophisticated_mode = 0;
options.BLOCK_N1                     = 1024;
options.BLOCK_N2                     = 1024;
options.gmm_1sigma                   = 0;
options.gmm_flags_no_norm            = 0;
options.gmm_flags_w                  = 0;
options.seed                         = 1234543;
options.num_threads                  = 1;


tic,[M , S , w]                      = yael_gmm(X , options);,toc



Example 3
---------

clear

d                                    = 128;                   % dimensionality of the vectors
N                                    = 100000;                % number of vectors

X                                    = randn(d, N , 'single'); % random set of vectors 

options.K                            = 100;
options.max_ite_kmeans               = 10;
options.max_ite_gmm                  = 0;
options.init_random_mode             = 0;
options.normalize_sophisticated_mode = 0;
options.BLOCK_N1                     = 1024;
options.BLOCK_N2                     = 1024;
options.gmm_1sigma                   = 0;
options.gmm_flags_no_norm            = 0;
options.gmm_flags_w                  = 0;

options.seed                         = 1234543;
options.num_threads                  = -1;
options.verbose                      = 1;


tic,[M , S , w]                      = yael_gmm(X , options);,toc



Example 4
---------

clear

d                                    = 2;                   % dimensionality of the vectors
N                                    = 1000;                % number of vectors

X                                    = [(repmat([-3 ; 10] , 1 , N/2)+0.2*randn(d , N/2 , 'single')) , (repmat([2 ; 3], 1 , N/2) + 1*randn(d , N/2 , 'single'))]; % random set of vectors 

options.K                            = 5;
options.max_ite_kmeans               = 10;
options.max_ite_gmm                  = 10;
options.init_random_mode             = 0;
options.redo                         = 1;
options.normalize_sophisticated_mode = 0;
options.BLOCK_N1                     = 1024;
options.BLOCK_N2                     = 1024;
options.gmm_1sigma                   = 0;
options.gmm_flags_no_norm            = 0;
options.gmm_flags_w                  = 0;

options.seed                         = 1234543;
options.num_threads                  = 2;
options.verbose                      = 1;


tic,[M , S , w]                      = yael_gmm(X , options);,toc


To compile
----------

mex  -g   yael_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex  -g -DOMP -DBLAS  yael_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex -v -g -DBLAS -DOMP yael_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex  -v -f mexopts_intel10.bat -DBLAS -DOMP  yael_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

If compiled with OMP option, OMP support

mex -v -DOMP  yael_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

mex -v -DOMP -f mexopts_intel10.bat yael_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"

If compiled with BLAS & OMP options

mex -v -DBLAS -DOMP  yael_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"


mex -v -DBLAS -DOMP -f mexopts_intel10.bat -output yael_gmm.dll yael_gmm.c "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_core.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_c.lib" "C:\Program Files\Intel\Compiler\11.1\065\mkl\ia32\lib\mkl_intel_thread.lib" "C:\Program Files\Intel\Compiler\11.1\065\lib\ia32\libiomp5md.lib"



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

#include <time.h>
#include <math.h>
#include <malloc.h>
#include <mex.h>

#ifdef OMP
 #include <omp.h>
#endif

#ifndef max
    #define max(a,b) (a >= b ? a : b)
    #define min(a,b) (a <= b ? a : b)
#endif

#define NMAX_KMEANSPP 8192
#define rationNK_KMEANSPP 8
#define min_sigma 1e-10
#define M_PI 3.14159265358979323846


#ifndef MAX_THREADS
#define MAX_THREADS 64
#endif

#if defined(__OS2__)  || defined(__WINDOWS__) || defined(WIN32) || defined(WIN64) || defined(_MSC_VER)
#define BLASCALL(f) f
#else
#define BLASCALL(f) f ##_
#endif

#define SHR3   ( jsr ^= (jsr<<17), jsr ^= (jsr>>13), jsr ^= (jsr<<5) )
#define randint SHR3
#define rand() (0.5 + (signed)randint*2.328306e-10)

#ifdef __x86_64__
    typedef int UL;
#else
    typedef unsigned long UL;
#endif

static UL jsrseed = 31340134 , jsr;

struct opts
{
	int    K;
	int    max_ite_kmeans;
	int    max_ite_gmm;
	int    gmm_1sigma;
	int    redo;
	int    verbose;
	int    init_random_mode;
	int    normalize_sophisticated_mode;
	int    BLOCK_N1;
	int    BLOCK_N2;
	int    gmm_flags_no_norm;
	int    gmm_flags_w;
	UL     seed;
#ifdef OMP 
    int    num_threads;
#endif
};

#ifdef OMP 
typedef struct 
{
	int n,K,d;
	float *X , *M_old , *p;
	float *M ,*S , *w;
	int num_threads;
} scompute_sum_dcov_t;

typedef struct 
{
	int n,K,d;
	double *X , *M_old , *p;
	double *M ,*S , *w;
	int num_threads;
} dcompute_sum_dcov_t;
#endif

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

/* ------------------------------------------------------------------------------------------------------------------------------------ */
extern void   BLASCALL(sgemm)(char *, char *, int*, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *);
extern void   BLASCALL(sgemv)(char *, int *, int *, float *, float *, int *, float *, int *, float *, float *, int *);
extern void   BLASCALL(saxpy)(int *, float *, float *, int *, float *, int *);
extern void   BLASCALL(sscal)(int *, float *, float  *, int *);
extern float  BLASCALL(snrm2)(int * , float * , int *);

extern void   BLASCALL(dgemm)(char *, char *, int*, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
extern void   BLASCALL(dgemv)(char *, int *, int *, double *, double *, int *, double *, int *, double *, double *, int *);
extern void   BLASCALL(daxpy)(int *, double *, double *, int *, double *, int *);
extern void   BLASCALL(dscal)(int *, double *, double  *, int *);
extern double BLASCALL(dnrm2)(int * , double * , int *);


#ifdef OMP 
void compute_tasks(int , int , void (*task_fun) (void *, int , int ) , void *);
#endif

void  sgmm(float *, int , int , struct opts , float *, float *, float *);
void  skmeans(float *, int , int , struct opts , float *, float *, int *, int *,float *);
void  skmeanspp_init(float * , int , int , int , int * , float *, float *);
void  scompute_distances_1(float *, float * , int , int , float *);
void  scompute_cross_distances_nonpacked(float *, float *, int , int , int , float *, float *);
void  snn_single_full(float * , float * , int , int , int , int , int , float * , float * , float *, int *);
int   svec_arg_max(float *, int );
int   svec_count_occurrences(float *, int , float);
void  sgmm_handle_empty(float *, int , int , struct opts , float * , float * , float * , float *); 
void  sgmm_compute_params (float *, int , int , struct opts , float * , float * , float * , float * , float * , float * , float *);
void  scompute_sum_dcov(int , int , int , float * , float * , float * , float * , float * , float *);
void  sgmm_compute_p(float *, int , int , int , float * , float * , float * , float * , int , int );
void  scompute_mahalanobis_sqr(float * , int , int , int , float * , float * , float *);


#ifdef OMP
void  sgmm_compute_p_thread(float * , int , int , int , struct opts , float * , float *, float * , float *);
void  scompute_sum_dcov_thread(int , int , int , float * , float * , float * , float * , float * , float * , int); 
void  scompute_sum_dcov_task_fun(void * , int , int );
void  scompute_p_task_fun(void * , int , int ); 
#endif

float *svec_new (int );
float *svec_new_cpy (float *, int); 
void  svec_sqr (float * , int );
void  svec_cpy (float *, float *, int );
void  svec_add (float * , float * , int);
float slog_sum (float , float);



void   dgmm(double *, int , int , struct opts , double *, double *, double *);
void   dkmeans(double *, int , int , struct opts , double *, double *, int *, int *,double *);
void   dkmeanspp_init(double * , int , int , int , int * , double *, double *);
void   dcompute_distances_1(double *, double * , int , int , double *);
void   dcompute_cross_distances_nonpacked(double *, double *, int , int , int , double *, double *);
void   dnn_single_full(double * , double * , int , int , int , int , int , double * , double * , double *, int *);
int    dvec_arg_max(double *, int ); 
int    dvec_count_occurrences(double *, int , double);
void   dgmm_handle_empty(double *, int , int , struct opts , double * , double * , double * , double *); 
void   dgmm_compute_params(double *, int , int , struct opts , double * , double * , double * , double * , double * , double * , double *);
void   dcompute_sum_dcov(int , int , int , double * , double * , double * , double * , double * , double *);
void   dgmm_compute_p(double *, int , int , int , double * , double * , double * , double * , int , int );
void   dcompute_mahalanobis_sqr(double * , int , int , int , double * , double * , double *);

#ifdef OMP
void   dgmm_compute_p_thread(double * , int , int , int , struct opts , double * , double *, double * , double *);
void   dcompute_sum_dcov_thread(int , int , int , double * , double * , double * , double * , double * , double * , int); 
void   dcompute_sum_dcov_task_fun(void * , int , int );
void   dcompute_p_task_fun(void * , int , int ); 
#endif


double *dvec_new (int );
double *dvec_new_cpy (double *, int); 
void   dvec_sqr (double * , int );
void   dvec_cpy (double *, double *, int );
void   dvec_add (double * , double * , int);
double dlog_sum (double , double);


void randperm(int * , int );
void randini(UL);

/* ------------------------------------------------------------------------------------------------------------------------------------ */
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray*prhs[])
{
	int d,N, issingle = 0;
	float  *sX , *sM , *sS , *sw;
	double *dX , *dM , *dS , *dw;

#ifdef OMP 
	struct opts options = {10 , 10 , 10 , 0, 1 , 0 , 0 , 0 , 1024 , 1024 ,0 , 1 , (UL)NULL , -1};
#else
	struct opts options = {10 , 10 , 10 , 0, 1 , 0 , 0 , 0 , 1024 , 1024 , 0 , 1 , (UL)NULL};
#endif
	mxArray *mxtemp;
	double *tmp;
	int tempint;
	UL templint;

	if ((nrhs < 1) || (nrhs > 2)) 
	{
		mexPrintf(
			"\n"
			"\n"
			"Fast mex GMM training algorithm with Kmeans initialization\n"
			"(mex-interface modified from the original yael package https://gforge.inria.fr/projects/yael)\n"
			"\n"
			"- Accept single/double precision input\n"
			"- Support of BLAS/OpenMP for multi-core computation\n"
			"\n"
			"Usage\n"
			"-----\n"
			"\n"
			"[M, S , w]                              = yael_gmm(X , [options]);\n"
			"\n"
			"\n"
			"Inputs\n"
			"------\n"
			"\n"
			"X                                        Input data matrix (d x N) in single/double format\n"
			"\n"
			"options\n"
			"       K                                 Number of centroid  (default K = 10)\n"
            "       max_ite_kmeans                    Number of iteration for the initial Kmeans (default max_ite_kmeans = 10)\n"
            "       max_ite_gmm                       Number of iteration for GMM learning (default max_ite_gmm = 10)\n"
	        "       gmm_1sigma                        Compute a single value for the sigma diagonal (default gmm_1sigma = 0)\n"
	        "       gmm_flags_no_norm                 No normalization in the GMM update (default gmm_flags_no_norm = 0)\n" 
	        "       gmm_flags_w                       Take weighs into account during probabilities update (default gmm_flags_w = 1)\n" 
			"       redo                              Number of time to restart K-means (default redo = 1)\n"
			"       verbose                           Verbose level = {0,1} (default verbose = 0)\n"
			"       init_random_mode                  0 <=> Kmeans++ initialization, 1<=> random selection ...\n"
			"       normalize_sophisticated_mode      0/1 (No/Yes)\n"
			"       BLOCK_N1                          Cache size block (default BLOCK_N1 = 1024)\n"
			"       BLOCK_N2                          Cache size block (default BLOCK_N2 = 1024)\n"
			"       seed                              Seed number for internal random generator (default random seed according to time)\n"
#ifdef OMP 
			"       num_threads                       Number of threads. If num_threads = -1, num_threads = number of core  (default num_threads = -1)\n"
#endif
			"\n"
			"\n"
			"Outputs\n"
			"-------\n"
			"\n"
            "M                                        Means matrix of GMM (d x K) in single/double format\n" 
            "S                                        Variance matrix of GMM (d x K) in single/double format\n"
            "w                                        Weights vector of GMM (1 x K) in single/double format\n"
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

	d           = mxGetM(prhs[0]);
	N           = mxGetN(prhs[0]);

	if ((nrhs > 1) && !mxIsEmpty(prhs[1]))
	{
		mxtemp                            = mxGetField(prhs[1] , 0 , "K");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if( (tempint < 0))
			{
				mexPrintf("K must be > 0, force to 10\n");	
				options.K                 = 0;
			}
			else
			{
				options.K                 = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "max_ite_kmeans");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if( (tempint < 0))
			{
				mexPrintf("max_ite_kmeans must be > 0, force to 10\n");	
				options.max_ite_kmeans    = 10;
			}
			else
			{
				options.max_ite_kmeans    = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "max_ite_gmm");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];
			if( (tempint < 0))
			{
				mexPrintf("max_ite_gmm must be > 0, force to 10\n");	
				options.max_ite_gmm       = 10;
			}
			else
			{
				options.max_ite_gmm       = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "redo");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0))
			{
				mexPrintf("redo must be >= 0, force to 1\n");	
				options.redo              = 1;
			}
			else
			{
				options.redo              = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "gmm_1sigma");
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

		mxtemp                            = mxGetField(prhs[1] , 0 , "verbose");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0) || (tempint > 1))
			{
				mexPrintf("verbose must be ={0,1}, force to 0\n");	
				options.verbose           = 0;
			}
			else
			{
				options.verbose           = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "init_random_mode");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0) || (tempint > 1))
			{
				mexPrintf("init_random_mode must be ={0,1}, force to 0\n");	
				options.init_random_mode  = 0;
			}
			else
			{
				options.init_random_mode  = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "normalize_sophisticated_mode");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0) || (tempint > 1))
			{
				mexPrintf("normalize_sophisticated_mode must be ={0,1}, force to 0\n");	
				options.normalize_sophisticated_mode  = 0;
			}
			else
			{
				options.normalize_sophisticated_mode  = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "BLOCK_N1");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0))
			{
				mexPrintf("BLOCK_N1 must be >0 a power of 2, force to 1024\n");	
				options.BLOCK_N1          = 1024;
			}
			else
			{
				options.BLOCK_N1          = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "BLOCK_N2");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			tempint                       = (int) tmp[0];

			if( (tempint < 0))
			{
				mexPrintf("BLOCK_N2 must be >0 a power of 2, force to 1024\n");	
				options.BLOCK_N2         = 1024;
			}
			else
			{
				options.BLOCK_N2         = tempint;
			}
		}

		mxtemp                            = mxGetField(prhs[1] , 0 , "gmm_flags_no_norm");
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

		mxtemp                            = mxGetField(prhs[1] , 0 , "gmm_flags_w");
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

		mxtemp                            = mxGetField(prhs[1] , 0 , "seed");
		if(mxtemp != NULL)
		{
			tmp                           = mxGetPr(mxtemp);
			templint                      = (UL) tmp[0];
			if( (templint < 1) )
			{
				mexPrintf("seed >= 1 , force to NULL (random seed)\n");	
				options.seed             = (UL)NULL;
			}
			else
			{
				options.seed             = templint;
			}
		}

#ifdef OMP
		mxtemp                            = mxGetField(prhs[1] , 0 , "num_threads");
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

	if(N < options.K) 
	{
		mexErrMsgTxt("fewer points than centroids");    
	}

	/*------------------------ Main Call ----------------------------*/

	randini(options.seed);

	if(issingle)
	{
		plhs[0]    = mxCreateNumericMatrix (d , options.K , mxSINGLE_CLASS, mxREAL);  
		sM         = (float*)mxGetPr(plhs[0]);

		plhs[1]    = mxCreateNumericMatrix (d, options.K , mxSINGLE_CLASS, mxREAL);
		sS         = (float*) mxGetPr (plhs[1]);

		plhs[2]    = mxCreateNumericMatrix (1 , options.K , mxSINGLE_CLASS, mxREAL);
		sw         = (float*) mxGetPr (plhs[2]);

		sgmm(sX , d , N , options , sM , sS , sw);
	}
	else
	{
		plhs[0]    = mxCreateNumericMatrix (d , options.K , mxDOUBLE_CLASS, mxREAL);  
		dM         = (double*)mxGetPr(plhs[0]);

		plhs[1]    = mxCreateNumericMatrix (d, options.K , mxDOUBLE_CLASS, mxREAL);
		dS         = (double*) mxGetPr (plhs[1]);

		plhs[2]    = mxCreateNumericMatrix (1, options.K , mxDOUBLE_CLASS, mxREAL);
		dw         = (double*) mxGetPr (plhs[2]);

		dgmm(dX , d , N , options , dM , dS , dw);
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void sgmm(float *X , int d , int N , struct opts options , float *M , float *S , float *w)
{
	int i , n , k , iter , iter_tot = 0;
	int K = options.K , verbose = options.verbose;
	int max_ite_kmeans = options.max_ite_kmeans , max_ite_gmm = options.max_ite_gmm;
	int dK = d*K , KN = K*N;
	float *p , *dis , *qerr, *wtemp;
	float *xtemp, *Mtemp;
	float cteK = 1.0f/K , sum , sig;
	float old_key, key = 666;
	int *assign, *nassign;
	
#ifdef OMP 
    int num_threads = options.num_threads;
    num_threads     = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#else
	int gmm_flags_no_norm = options.gmm_flags_no_norm , gmm_flags_w = options.gmm_flags_w;
#endif


	dis           = (float *)malloc(N*sizeof(float));
	assign        = (int *)  malloc(N*sizeof(int)); 
	nassign       = (int *)  malloc(K*sizeof(int)); 
	qerr          = (float *)malloc(max_ite_kmeans*sizeof(float));


	skmeans(X , d , N , options , M , dis , assign , nassign , qerr);

	sum = 0.0f;
	for (n = 0 ; n < N ; n++)
	{
		sum  += dis[n];
	}
	sig  = sum/N;

	free(dis);
	free(assign);
	free(nassign);
	free(qerr);


	for (k = 0 ; k < K ; k++)
	{
		w[k] = cteK;
	}
	
	for(i = 0 ; i < dK ; i++)
	{
		S[i] = sig;
	}

	p             = (float *) malloc(KN*sizeof(float));
	Mtemp         = (float *) malloc(dK*sizeof(float));
	wtemp         = (float *) malloc(K*sizeof(float));
	xtemp         = (float *) malloc(d*sizeof(float));

	for (iter = 1 ; iter <= max_ite_gmm ; iter++) 
	{ 
		
#ifdef OMP 
		sgmm_compute_p_thread(X , d , N , num_threads , options , M , S , w , p);
#else
		sgmm_compute_p(X , d , N , K , M , S , w , p , gmm_flags_no_norm , gmm_flags_w);
#endif

		sgmm_handle_empty(X , d , N , options , M , S , p , wtemp);

		sgmm_compute_params(X , d , N , options , M , S , w , p , xtemp , Mtemp , wtemp);


		iter_tot++;

		/* convergence reached -> leave */
		old_key = key;

		key     = 0.0f;
		for(i = 0 ; i < dK ; i++)
		{
			key  += M[i];
		}

		if(verbose)
		{
			mexPrintf("keys %5d: %.6f -> %.6f\n", iter, old_key, key);
			mexEvalString("drawnow;");
		}

		if (key == old_key)
		{
			break;
		}
	}

	free(p);
	free(Mtemp);
	free(xtemp);
	free(wtemp);
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dgmm(double *X , int d , int N , struct opts options , double *M , double *S , double *w)
{
	int i , n , k , iter , iter_tot = 0;
	int K = options.K , verbose = options.verbose;
	int max_ite_kmeans = options.max_ite_kmeans , max_ite_gmm = options.max_ite_gmm;
	int dK = d*K , KN = K*N;
	double *p , *dis , *qerr, *wtemp;
	double *xtemp, *Mtemp;
	double cteK = 1.0/K , sum , sig;
	double old_key, key = 666;
	int *assign, *nassign;
	
#ifdef OMP 
    int num_threads = options.num_threads;
    num_threads     = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#else
	int gmm_flags_no_norm = options.gmm_flags_no_norm , gmm_flags_w = options.gmm_flags_w;
#endif


	dis           = (double *)malloc(N*sizeof(double));
	assign        = (int *)  malloc(N*sizeof(int)); 
	nassign       = (int *)  malloc(K*sizeof(int)); 
	qerr          = (double *)malloc(max_ite_kmeans*sizeof(double));


	dkmeans(X , d , N , options , M , dis , assign , nassign , qerr);

	sum = 0.0;
	for (n = 0 ; n < N ; n++)
	{
		sum  += dis[n];
	}
	sig  = sum/N;

	free(dis);
	free(assign);
	free(nassign);
	free(qerr);


	for (k = 0 ; k < K ; k++)
	{
		w[k] = cteK;
	}
	
	for(i = 0 ; i < dK ; i++)
	{
		S[i] = sig;
	}

	p             = (double *) malloc(KN*sizeof(double));
	Mtemp         = (double *) malloc(dK*sizeof(double));
	wtemp         = (double *) malloc(K*sizeof(double));
	xtemp         = (double *) malloc(d*sizeof(double));

	for (iter = 1 ; iter <= max_ite_gmm ; iter++) 
	{ 
		
#ifdef OMP 
		dgmm_compute_p_thread(X , d , N , num_threads , options , M , S , w , p);
#else
		dgmm_compute_p(X, d , N , K , M , S , w , p , gmm_flags_no_norm , gmm_flags_w);
#endif

		dgmm_handle_empty(X , d , N , options , M , S , p , wtemp);

		dgmm_compute_params(X , d , N , options , M , S , w , p , xtemp , Mtemp , wtemp);


		iter_tot++;

		/* convergence reached -> leave */
		old_key = key;

		key     = 0.0;
		for(i = 0 ; i < dK ; i++)
		{
			key  += M[i];
		}

		if(verbose)
		{
			mexPrintf("keys %5d: %.6f -> %.6f\n", iter, old_key, key);
			mexEvalString("drawnow;");
		}

		if (key == old_key)
		{
			break;
		}
	}

	free(p);
	free(Mtemp);
	free(xtemp);
	free(wtemp);
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void sgmm_handle_empty(float *X, int d, int N, struct opts options, float *M , float *S , float *p , float *w) 
{
	int i,j,k;
	int K = options.K , iK , id , j2d;
	int j2,split_dim,nt,nnz;
	int nz;
	int bigprime=1000003;

	nz  = svec_count_occurrences(p , K*N , 0.0f);
	for(k = 0 ; k < K ; k++)
	{
		w[k] = 0.0f;
	}
	for (i = 0 ; i < N ; i++) 
	{
		iK = i*K;
		for (j = 0 ; j < K ; j++) 
		{
			w[j] +=p [j + iK];
		}
	}

	for (j = 0 ; j < K ; j++)
	{
		if(w[j]==0) 
		{
			j2 = j;
			for(i=0 ; i<K ;i++) 
			{
				j2 = (j2 + bigprime)%K; 
				if(w[j2] > 0.0f) 
				{
					break;
				}
			}

			/* dimension to split: that with highest variance */

			split_dim = svec_arg_max(S + d*j2 , d);

			/* transfer half(?) of the points from j2 -> j */
			nt  = 0;
			nnz = 0;
			j2d = j2*d;
			for(i=0 ; i < N ; i++) 
			{
				iK = i*K;
				id = i*d;
				if(p[j2 + iK] > 0.0f) 
				{ 
					nnz++;
					if(X[split_dim + id] < M[split_dim + j2d]) 
					{
						p[j + iK]  = p[j2 + iK];
						p[j2 + iK] = 0.0f;
						nt++;
					}
				}
			}
			w[j2] = -1.0f; /* avoid further splits */
		}  
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dgmm_handle_empty(double *X, int d, int N , struct opts options , double *M , double *S , double *p , double *w) 
{
	int i,j,k;
	int K = options.K , iK , id , j2d;
	int j2,split_dim,nt,nnz;
	int nz;
	int bigprime=1000003;

	nz  = dvec_count_occurrences(p , K*N , 0.0);
	
	for(k = 0 ; k < K ; k++)
	{
		w[k] = 0.0;
	}
	for (i = 0 ; i < N ; i++) 
	{
		iK = i*K;
		for (j = 0 ; j < K ; j++) 
		{
			w[j] +=p [j + iK];
		}
	}

	for (j = 0 ; j < K ; j++)
	{
		if(w[j]==0) 
		{
			j2 = j;
			for(i=0 ; i<K ;i++) 
			{
				j2 = (j2 + bigprime)%K; 
				if(w[j2] > 0.0) 
				{
					break;
				}
			}

			/* dimension to split: that with highest variance */

			split_dim = dvec_arg_max(S + d*j2 , d);

			/* transfer half(?) of the points from j2 -> j */
			nt  = 0;
			nnz = 0;
			j2d = j2*d;
			for(i=0 ; i < N ; i++) 
			{
				iK = i*K;
				id = i*d;
				if(p[j2 + iK] > 0.0) 
				{ 
					nnz++;
					if(X[split_dim + id] < M[split_dim + j2d]) 
					{
						p[j + iK]  = p[j2 + iK];
						p[j2 + iK] = 0.0;
						nt++;
					}
				}
			}
			w[j2] = -1.0; /* avoid further splits */
		}  
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
static void sgmm_compute_params(float *X, int d, int N , struct opts options, float *M, float *S , float *w , float *p, float *xtemp, float *Mtemp, float *wtemp)
{
	int i, j  , jd ;
	int K = options.K , dK = d*K;
	float sum;
	int nz;

#ifdef OMP 
	int num_threads = options.num_threads;
	num_threads     = (num_threads == -1) ? min(MAX_THREADS , omp_get_num_procs()) : num_threads;
	omp_set_num_threads(num_threads);
#endif


  for(i = 0 ; i < dK ; i++)
  {
	  Mtemp[i] = M[i];
	  S[i]     = 0.0f;
	  M[i]     = 0.0f;
  }
  for(i = 0 ; i < K; i++)
  {
	  wtemp[i] = w[i];
	  w[i]     = 0.0f;
  }

#ifdef OMP 
  if(num_threads <= 1)
  {
	  scompute_sum_dcov(N , K , d , X , Mtemp , p , M , S , w);
  }
  else
  {
	  scompute_sum_dcov_thread(N , K , d , X , Mtemp , p , M , S , w , num_threads);
  }
#else
  scompute_sum_dcov(N , K , d , X , Mtemp , p , M , S , w);
#endif

 
  if(options.gmm_1sigma) 
  {
	  for (j = 0 ; j < K  ; j++) 
	  {
		  jd  = j*d;
		  sum = 0.0f;
		  for(i = 0 ; i < d ; i++)
		  {
			  sum += S[i + jd];
		  }
		  sum  = sum/d;
		  for(i = 0 ; i < d ; i++)
		  {
			  S[i + jd] = sum;
		  }
	  }
  }

  nz       = 0;
  for(i = 0 ; i < dK ; i++) 
  {
	  if(S[i] < min_sigma) 
	  {
		  S[i] = min_sigma;
		  nz++;
	  }
  }

  if(nz & options.verbose) 
  {
	  mexPrintf("WARN %ld sigma diagonals are too small (set to %g)\n" , nz , min_sigma);
	  mexEvalString("drawnow;");
  }

  for (j = 0 ; j < K ; j++) 
  {
	  jd  = j*d;
	  sum = 1.0f/w[j]; 
	  for(i = 0 ; i < d ; i++)
	  {
		  M[i + jd] *= sum;
		  S[i + jd] *= sum;
	  }
  }

  sum = 0.0f;
  for (j = 0 ; j < K ; j++) 
  {
	  sum +=w[j];
  }
  sum = 1.0f/sum;
  for (j = 0 ; j < K ; j++) 
  {
	  w[j] *=sum;
  }
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
static void dgmm_compute_params(double *X , int d , int N , struct opts options, double *M, double *S , double *w , double *p, double *xtemp, double *Mtemp, double *wtemp)
{
	int i, j  , jd ;
	int K = options.K , dK = d*K;
	double sum;
	int nz;

#ifdef OMP 
	int num_threads = options.num_threads;
	num_threads     = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
	omp_set_num_threads(num_threads);
#endif


  for(i = 0 ; i < dK ; i++)
  {
	  Mtemp[i] = M[i];
	  S[i]     = 0.0;
	  M[i]     = 0.0;
  }
  for(i = 0 ; i < K; i++)
  {
	  wtemp[i] = w[i];
	  w[i]     = 0.0;
  }

#ifdef OMP 
  if(num_threads <= 1)
  {
	  dcompute_sum_dcov(N , K , d , X , Mtemp , p , M , S , w);
  }
  else
  {
	  dcompute_sum_dcov_thread(N , K , d , X , Mtemp , p , M , S , w , num_threads);
  }
#else
  dcompute_sum_dcov(N , K , d , X , Mtemp , p , M , S , w);
#endif

 
  if(options.gmm_1sigma) 
  {
	  for (j = 0 ; j < K  ; j++) 
	  {
		  jd  = j*d;
		  sum = 0.0;
		  for(i = 0 ; i < d ; i++)
		  {
			  sum += S[i + jd];
		  }
		  sum  = sum/d;
		  for(i = 0 ; i < d ; i++)
		  {
			  S[i + jd] = sum;
		  }
	  }
  }

  nz       = 0;
  for(i = 0 ; i < dK ; i++) 
  {
	  if(S[i] < min_sigma) 
	  {
		  S[i] = min_sigma;
		  nz++;
	  }
  }

  if(nz & options.verbose) 
  {
	  mexPrintf("WARN %ld sigma diagonals are too small (set to %g)\n" , nz , min_sigma);
	  mexEvalString("drawnow;");
  }


  for (j = 0 ; j < K ; j++) 
  {
	  jd  = j*d;
	  sum = 1.0/w[j];
	  for(i = 0 ; i < d ; i++)
	  {
		  M[i + jd] *= sum;
		  S[i + jd] *= sum;
	  }
  }

  sum = 0.0;
  for (j = 0 ; j < K ; j++) 
  {
	  sum +=w[j];
  }
  sum = 1.0/sum;
  for (j = 0 ; j < K ; j++) 
  {
	  w[j] *=sum;
  }
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */

void skmeans(float *X , int d , int N , struct opts options , float *centroids_out , float *dis_out, int *assign_out, int *nassign_out , float *qerr_out)
{
	float *dists , *sum_c , *norms , *disbest , *distmp;
	float *centroids, *dis , *quanterr;
	int *assign, *nassign;
	int *selected;
	int K = options.K, Kd = K*d, redo = options.redo , verbose = options.verbose , max_ite = options.max_ite_kmeans;
	int normalize_sophisticated_mode = options.normalize_sophisticated_mode , init_random_mode = options.init_random_mode;
	int step1 = min(N, options.BLOCK_N1), step2 = min(K, options.BLOCK_N2);
	int run , Nsubset;
	int i , j , id , jd , index , iter , iter_tot = 0;
	float temp , sum;
	double qerr, qerr_old , qerr_best = HUGE_VAL ;

#ifdef BLAS
	float one = 1.0f;
	int inc   = 1;
#endif
#ifdef OMP 
    int num_threads = options.num_threads;
    num_threads     = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);
#endif


	centroids     = (float *)malloc(Kd*sizeof(float));
	dis           = (float *)malloc(N*sizeof(float));
	quanterr      = (float *)malloc(max_ite*sizeof(float));

	dists         = (float *)malloc(step1*step2*sizeof(float));
	sum_c         = (float *)malloc(step2*sizeof(float));
	selected      = (int *)  malloc(N*sizeof(int)); /* Only the first K<N will be used */

	assign        = (int *)  malloc(N*sizeof(int)); 
	nassign       = (int *)  malloc(K*sizeof(int)); 


	if(normalize_sophisticated_mode)
	{			
		norms     = (float *)malloc(K*sizeof(float));
	}

	if(!init_random_mode) 
	{
		Nsubset = N;
		if((N>(K*rationNK_KMEANSPP)) && (N>NMAX_KMEANSPP)) 
		{
			Nsubset = K*rationNK_KMEANSPP;
			if(verbose)
			{
				mexPrintf("Restricting k-means++ initialization to %d points\n" , Nsubset);
				mexEvalString("drawnow;");
			}
		}
		disbest       = (float *) malloc(Nsubset*sizeof(float));
		distmp        = (float *) malloc(Nsubset*sizeof(float));
	}

	for (run = 0 ; run < redo ; run++) 
	{
do_redo: 

		if(verbose)
		{
			mexPrintf("<><><><> kmeans / run %d <><><><><>\n", run+1);
			mexEvalString("drawnow;");
		}

		if(init_random_mode) 
		{
			randperm(selected , N);
		} 
		else 
		{
			skmeanspp_init(X , d , Nsubset , K , selected , disbest , distmp);
		}

		for(j = 0 ; j < K ; j++)
		{
			index = selected[j]*d;
			jd    = j*d;
#ifdef BLAS
			memcpy(centroids + jd , X + index , d*sizeof(float));
#else
			for(i = 0 ; i < d ; i++)
			{
				centroids[i + jd] = X[i + index];
			}
#endif
		}

		/* the quantization error */
		qerr = HUGE_VAL;

		for (iter = 1 ; iter <= max_ite ; iter++) 
		{

#ifdef BLAS
			memset(nassign , 0 , K*sizeof(int));
#else
			for(i = 0 ; i < K ; i++)
			{
				nassign[i] = 0;
			}
#endif

			iter_tot++;

			/* Assign point to cluster and count the cluster size */

			snn_single_full(X , centroids , d , N , K , step1 , step2 , dis , dists , sum_c , assign );

			for (i = 0 ; i < N ; i++) 
			{
				nassign[assign[i]]++;
			}

			for (i = 0 ; i < K ; i++) 
			{
				if(nassign[i]==0) 
				{
					if(verbose)
					{
						mexPrintf("WARN nassign %d is 0, redoing!\n",(int)i);
						mexEvalString("drawnow;");

					}
					goto do_redo;
				}
			}

			if(normalize_sophisticated_mode)
			{
#ifdef BLAS
				memset(centroids , 0 , Kd*sizeof(float));
				memset(norms , 0 , K*sizeof(float));
#else
				for( i = 0 ; i < Kd ; i++)
				{
					centroids[i] = 0.0f;
				}
				for( i = 0 ; i < K ; i++)
				{
					norms[i]    = 0.0f;
				}
#endif
				for(i = 0 ; i < N ; i++)
				{
					index = assign[i]*d;
					id    = i*d;
#ifdef BLAS
					BLASCALL(saxpy)(&d , &one , X + id , &inc , centroids + index , &inc);
					sum  = BLASCALL(snrm2)(&d , X + id , &inc);
					sum *= sum;
#else
					sum   = 0.0f;
					for(j = 0  ; j < d ; j++)
					{
						temp                  = X[j + id];
						centroids[j + index] += temp;
						sum                  += (temp*temp);
					}
#endif
					norms[assign[i]]         += (float)sqrt(sum);

				}
				for (i = 0 ; i < K ; i++) 
				{
					id  = i*d;
#ifdef BLAS
					sum = BLASCALL(snrm2)(&d , centroids + id , &inc);
					sum *= sum;
#else
					sum = 0.0f;
					for(j = 0  ; j < d ; j++)
					{
						temp  = centroids[j + id];
						sum  += (temp*temp);
					}
#endif
					sum  = (sum != 0.0f) ? (1.0f/sqrt(sum)) : 1.0f;
					temp = sum*(norms[i]/nassign[i]);
#ifdef BLAS
					BLASCALL(sscal)(&d , &temp , centroids + id , &inc);
#else
					for(j = 0  ; j < d ; j++)
					{
						centroids[j + id] *= temp;
					}
#endif
				}
			} 
			else 
			{
#ifdef BLAS		                
				memset(centroids , 0 , Kd*sizeof(float));
#else
				for( i = 0 ; i < Kd ; i++)
				{
					centroids[i] = 0.0f;
				}
#endif

/*
#ifdef OMP
#ifdef BLAS
#pragma omp parallel for default(none) private(i,j,index,id) shared(X,N,d,assign,centroids,inc,one)
#else
#pragma omp parallel for default(none) private(i,j,index,id) shared(X,N,d,assign,centroids)
#endif
#endif
*/
				for(i = 0 ; i < N ; i++)
				{
					index = assign[i]*d;
					id    = i*d;
#ifdef BLAS
					BLASCALL(saxpy)(&d , &one , X + id , &inc , centroids + index , &inc);
#else
					for(j = 0  ; j < d ; j++)
					{
						centroids[j + index] += X[j + id];
					}
#endif
				}

#ifdef OMP
#ifdef BLAS
#pragma omp parallel for default(none) private(i,j,temp,id) shared(K,d,nassign,centroids,inc)
#else
#pragma omp parallel for default(none) private(i,j,temp,id) shared(K,d,nassign,centroids)
#endif
#endif
				for (i = 0 ; i < K ; i++) 
				{
					id   = i*d;
					temp = 1.0f/nassign[i];
#ifdef BLAS
					BLASCALL(sscal)(&d , &temp , centroids + id , &inc);
#else
					for(j = 0  ; j < d ; j++)
					{
						centroids[j + id] *= temp;
					}
#endif
				}
			}
			qerr_old  = qerr;
			qerr      = 0.0f;
			for(i = 0 ; i < N ; i++)
			{
				qerr += dis[i];
			}
			quanterr[iter-1] = qerr;
			if(verbose)
			{
				mexPrintf("Kmean ite = (%d/%d), qerr = %10.10lf\n",iter,max_ite,qerr);
				mexEvalString("drawnow;");
			}
			if (qerr_old == qerr)
			{
				break;
			}
		}

		if (qerr < qerr_best) 
		{
			qerr_best = qerr;
			memcpy(centroids_out , centroids , Kd*sizeof(float));
			memcpy(dis_out, dis , N*sizeof(float));
			memcpy(qerr_out, quanterr , max_ite*sizeof(float));
			memcpy(assign_out , assign , N*sizeof(int));
			memcpy(nassign_out , nassign , K*sizeof (int));
		}
	}

/*
	if(verbose)
	{
		mexPrintf ("Total number of iterations: %d\n", (int)iter_tot);
		mexEvalString("drawnow;");
	}
*/
	free(centroids);
	free(dis);
	free(quanterr);
	free(assign);
	free(nassign);
	free(selected);
	free(sum_c);
	if(normalize_sophisticated_mode)
	{			
		free(norms);
	}
	if(!init_random_mode) 
	{
		free(disbest);      
		free(distmp);  
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dkmeans(double *X , int d , int N , struct opts options  ,  double *centroids_out , double *dis_out, int *assign_out, int *nassign_out , double *qerr_out)
{
	double *dists , *sum_c , *norms , *disbest , *distmp;
	double *centroids, *dis , *quanterr;
	int *assign, *nassign;
	int *selected;
	int K = options.K, Kd = K*d, redo = options.redo , verbose = options.verbose , max_ite = options.max_ite_kmeans;
	int normalize_sophisticated_mode = options.normalize_sophisticated_mode , init_random_mode = options.init_random_mode;
	int step1 = min(N, options.BLOCK_N1), step2 = min(K, options.BLOCK_N2);
	int run , Nsubset;
	int i , j , id , jd , index , iter , iter_tot = 0;
	double temp , sum;
	double qerr, qerr_old , qerr_best = HUGE_VAL ;

#ifdef BLAS
	double one = 1.0;
	int inc = 1;
#endif

#ifdef OMP 
    int num_threads = options.num_threads;
    num_threads     = (num_threads == -1) ? min(MAX_THREADS,omp_get_num_procs()) : num_threads;
    omp_set_num_threads(num_threads);

#endif

	centroids     = (double *)malloc(Kd*sizeof(double));
	dis           = (double *)malloc(N*sizeof(double));
	quanterr      = (double *)malloc(max_ite*sizeof(double));

	dists         = (double *)malloc(step1*step2*sizeof(double));
	sum_c         = (double *)malloc(step2*sizeof(double));
	selected      = (int *)  malloc(N*sizeof(int)); /* Only the first K<N will be used */

	assign        = (int *)  malloc(N*sizeof(int)); 
	nassign       = (int *)  malloc(K*sizeof(int)); 

	if(normalize_sophisticated_mode)
	{			
		norms     = (double *)malloc(K*sizeof(double));
	}

	if(!init_random_mode) 
	{
		Nsubset = N;
		if((N>(K*rationNK_KMEANSPP)) && (N>NMAX_KMEANSPP)) 
		{
			Nsubset = K*rationNK_KMEANSPP;
			if(verbose)
			{
				mexPrintf("Restricting k-means++ initialization to %d points\n" , Nsubset);
				mexEvalString("drawnow;");
			}
		}
		disbest       = (double *) malloc(Nsubset*sizeof(double));
		distmp        = (double *) malloc(Nsubset*sizeof(double));
	}

	for (run = 0 ; run < redo ; run++) 
	{
do_redo: 

		if(verbose)
		{
			mexPrintf("<><><><> kmeans / run %d <><><><><>\n", run+1);
			mexEvalString("drawnow;");
		}

		if(init_random_mode) 
		{
			randperm(selected , N);
		} 
		else 
		{
			dkmeanspp_init(X , d , Nsubset , K , selected , disbest , distmp);
		}

		for(j = 0 ; j < K ; j++)
		{
			index = selected[j]*d;
			jd    = j*d;
#ifdef BLAS
			memcpy(centroids + jd , X + index , d*sizeof(double));
#else
			for(i = 0 ; i < d ; i++)
			{
				centroids[i + jd] = X[i + index];
			}
#endif
		}

		/* the quantization error */
		qerr = HUGE_VAL;

		for (iter = 1 ; iter <= max_ite ; iter++) 
		{

#ifdef BLAS
			memset(nassign , 0 , K*sizeof(int));
#else
			for(i = 0 ; i < K ; i++)
			{
				nassign[i] = 0;
			}
#endif

			iter_tot++;

			/* Assign point to cluster and count the cluster size */

			dnn_single_full(X , centroids , d , N , K , step1 , step2 , dis , dists , sum_c , assign );

			for (i = 0 ; i < N ; i++) 
			{
				nassign[assign[i]]++;
			}

			for (i = 0 ; i < K ; i++) 
			{
				if(nassign[i]==0) 
				{
					if(verbose)
					{
						mexPrintf("WARN nassign %d is 0, redoing!\n",(int)i);
						mexEvalString("drawnow;");
					}
					goto do_redo;
				}
			}

			if(normalize_sophisticated_mode)
			{
#ifdef BLAS
				memset(centroids , 0 , Kd*sizeof(double));
				memset(norms , 0 , K*sizeof(double));
#else
				for( i = 0 ; i < Kd ; i++)
				{
					centroids[i] = 0.0;
				}
				for( i = 0 ; i < K ; i++)
				{
					norms[i]     = 0.0;
				}
#endif
				for(i = 0 ; i < N ; i++)
				{
					index = assign[i]*d;
					id    = i*d;
#ifdef BLAS
					BLASCALL(daxpy)(&d , &one , X + id , &inc , centroids + index , &inc);
					sum  = BLASCALL(dnrm2)(&d , X + id , &inc);
					sum *= sum;
#else
					sum   = 0.0;
					for(j = 0  ; j < d ; j++)
					{
						temp                  = X[j + id];
						centroids[j + index] += temp;
						sum                  += (temp*temp);
					}
#endif
					norms[assign[i]] += sqrt(sum);

				}
				for (i = 0 ; i < K ; i++) 
				{
					id  = i*d;
#ifdef BLAS
					sum = BLASCALL(dnrm2)(&d , centroids + id , &inc);
					sum *= sum;
#else
					sum = 0.0;
					for(j = 0  ; j < d ; j++)
					{
						temp  = centroids[j + id];
						sum  += (temp*temp);
					}
#endif
					sum  = (sum != 0.0) ? (1.0/sqrt(sum)) : 1.0;
					temp = sum*(norms[i]/nassign[i]);
#ifdef BLAS
					BLASCALL(dscal)(&d , &temp , centroids + id , &inc);
#else
					for(j = 0  ; j < d ; j++)
					{
						centroids[j + id] *= temp;
					}
#endif
				}
			} 
			else 
			{
#ifdef BLAS		                
				memset(centroids , 0 , Kd*sizeof(double));
#else
				for( i = 0 ; i < Kd ; i++)
				{
					centroids[i] = 0.0;
				}
#endif

/*
#ifdef OMP
#ifdef BLAS
#pragma omp parallel for default(none) private(i,j,index,id) shared(X,N,d,assign,centroids,inc,one)
#else
#pragma omp parallel for default(none) private(i,j,index,id) shared(X,N,d,assign,centroids)
#endif
#endif
*/
				for(i = 0 ; i < N ; i++)
				{
					index = assign[i]*d;
					id    = i*d;
#ifdef BLAS
					BLASCALL(daxpy)(&d , &one , X + id , &inc , centroids + index , &inc);
#else
					for(j = 0  ; j < d ; j++)
					{
						centroids[j + index] += X[j + id];
					}
#endif
				}

#ifdef OMP
#ifdef BLAS
#pragma omp parallel for default(none) private(i,j,temp,id) shared(K,d,nassign,centroids,inc)
#else
#pragma omp parallel for default(none) private(i,j,temp,id) shared(K,d,nassign,centroids)
#endif
#endif
				for (i = 0 ; i < K ; i++) 
				{
					id   = i*d;
					temp = 1.0/nassign[i];
#ifdef BLAS
					BLASCALL(dscal)(&d , &temp , centroids + id , &inc);
#else
					for(j = 0  ; j < d ; j++)
					{
						centroids[j + id] *= temp;
					}
#endif
				}
			}
			qerr_old  = qerr;
			qerr      = 0.0;
			for(i = 0 ; i < N ; i++)
			{
				qerr += dis[i];
			}
			quanterr[iter-1] = qerr;
			if(verbose)
			{
				mexPrintf("Kmean ite = (%d/%d), qerr = %10.10lf\n",iter,max_ite,qerr);
				mexEvalString("drawnow;");
			}
			if (qerr_old == qerr)
			{
				break;
			}
		}

		if (qerr < qerr_best) 
		{
			qerr_best = qerr;
			memcpy(centroids_out , centroids , Kd*sizeof(double));
			memcpy(dis_out, dis , N*sizeof(double));
			memcpy(qerr_out, quanterr , max_ite*sizeof(double));
			memcpy(assign_out , assign , N*sizeof(int));
			memcpy(nassign_out , nassign , K*sizeof (int));
		}
	}
/*
	if(verbose)
	{
		mexPrintf ("Total number of iterations: %d\n", (int)iter_tot);
		mexEvalString("drawnow;");
	}
*/
	
    free(centroids);
	free(dis);
	free(quanterr);
	free(assign);
	free(nassign);
	free(selected);
	free(sum_c);
	if(normalize_sophisticated_mode)
	{			
		free(norms);
	}
	if(!init_random_mode) 
	{
		free(disbest);      
		free(distmp);  
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void snn_single_full(float *X , float *centroids, int d , int N, int K , int step1 , int step2 , float *vwdis , float *dists , float *sum_c , int *vw )
{
	int m1 , m2 , imin, i1,i2,j1,j2 , i1d;
	float dmin;
	float *dline;

	for (i1 = 0; i1 < N ; i1 += step1) 
	{  
		m1  = min(step1 , N - i1);
		i1d = i1*d;

		/* clear mins */

		for (j1 = 0; j1 < m1; j1++) 
		{
			vw[j1 + i1]      = -1;
			vwdis[j1 + i1]   = 1e30f;
		}
		for (i2 = 0; i2 < K ; i2 += step2) 
		{     
			m2 = min(step2 , K - i2);

			scompute_cross_distances_nonpacked(centroids + i2*d , X + i1d, d , m2 , m1 , dists , sum_c);

			/* update mins */

#ifdef OMP
#pragma omp parallel for default(none) private(j1,j2) shared(m1,m2,i1,i2,vw,vwdis,dists,dline,dmin,imin)
#endif
			for(j1 = 0 ; j1 < m1 ; j1++) 
			{
				dline   = dists + j1*m2;
				imin    = vw[i1 + j1];
				dmin    = vwdis[i1 + j1];
				for(j2 = 0 ; j2 < m2 ; j2++) 
				{
					if(dline[j2] < dmin) 
					{
						imin = j2 + i2;
						dmin = dline[j2];
					}
				}
				vw[i1 + j1]    = imin;
				vwdis[i1 + j1] = dmin;
			}      
		}  
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dnn_single_full(double *X , double *centroids, int d , int N, int K , int step1 , int step2 , double *vwdis , double *dists , double *sum_c , int *vw )
{
	int m1 , m2 , imin, i1,i2,j1,j2 , i1d;
	double dmin;
	double *dline;

	for (i1 = 0; i1 < N ; i1 += step1) 
	{  
		m1  = min(step1 , N - i1);
		i1d = i1*d;

		/* clear mins */

		for (j1 = 0; j1 < m1; j1++) 
		{
			vw[j1 + i1]      = -1;
			vwdis[j1 + i1]   = 1e200;
		}
		for (i2 = 0; i2 < K ; i2 += step2) 
		{     
			m2 = min(step2 , K - i2);

			dcompute_cross_distances_nonpacked(centroids + i2*d , X + i1d, d , m2 , m1 , dists , sum_c);

			/* update mins */

#ifdef OMP
#pragma omp parallel for default(none) private(j1,j2) shared(m1,m2,i1,i2,vw,vwdis,dists,dline,dmin,imin)
#endif
			for(j1 = 0 ; j1 < m1 ; j1++) 
			{
				dline   = dists + j1*m2;
				imin    = vw[i1 + j1];
				dmin    = vwdis[i1 + j1];
				for(j2 = 0 ; j2 < m2 ; j2++) 
				{
					if(dline[j2] < dmin) 
					{
						imin = j2 + i2;
						dmin = dline[j2];
					}
				}
				vw[i1 + j1]    = imin;
				vwdis[i1 + j1] = dmin;
			}      
		}  
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
/* the kmeans++ initialization */
void skmeanspp_init (float *X , int d , int N , int K , int *selected , float *disbest, float *distmp)
{
	int newsel;
	int i, j;
	double rd;
	float norm;

	for (i = 0 ; i < N ; i++)
	{
		disbest[i] = (float)HUGE_VAL;
	}

	/* select the first centroid and set the others unitialized*/

	selected[0] = (int)floor(rand()* K);

	for (i = 1 ; i < K ; i++) 
	{
		newsel = selected[i - 1];			
		scompute_distances_1(X + newsel*d , X , d , N , distmp);

		for(j = 0 ; j < N ; j++) 
		{
			if(distmp[j] < disbest[j]) 
			{
				disbest[j] = distmp[j];
			}
		}
		norm = 0.0f;
		for(j = 0 ; j < N ; j++)
		{
			distmp[j] = disbest[j];
			norm     += (float)fabs(distmp[j]);
		}
		norm = (norm != 0.0f) ? (1.0f/norm) : 1.0f;
		for(j = 0 ; j < N ; j++)
		{
			distmp[j] *= norm;
		}
		rd = rand();
		for (j = 0 ; j < N - 1 ; j++) 
		{
			rd -= distmp[j];
			if (rd < 0.0)
			{
				break;
			}
		}
		selected[i] = j;
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
/* the kmeans++ initialization */
void dkmeanspp_init (double *X , int d , int N , int K , int *selected , double *disbest, double *distmp)
{
	int newsel;
	int i, j;
	double rd;
	double norm;

	for (i = 0 ; i < N ; i++)
	{
		disbest[i] = HUGE_VAL;
	}

	/* select the first centroid and set the others unitialized*/

	selected[0] = (int)floor(rand()* K);

	for (i = 1 ; i < K ; i++) 
	{
		newsel = selected[i - 1];			
		dcompute_distances_1(X + newsel*d , X , d , N , distmp);

		for(j = 0 ; j < N ; j++) 
		{
			if(distmp[j] < disbest[j]) 
			{
				disbest[j] = distmp[j];
			}
		}
		norm = 0.0;
		for(j = 0 ; j < N ; j++)
		{
			distmp[j] = disbest[j];
			norm     +=  fabs(distmp[j]);
		}
		norm = (norm != 0.0) ? (1.0/norm) : 1.0;
		for(j = 0 ; j < N ; j++)
		{
			distmp[j] *= norm;
		}
		rd = rand();
		for (j = 0 ; j < N - 1 ; j++) 
		{
			rd -= distmp[j];
			if (rd < 0.0)
			{
				break;
			}
		}
		selected[i] = j;
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void randperm(int *selected , int N)
{
	int i,j,tmp;

	for(i = 0 ; i < N ; i++)
	{
		selected[i]      = i;				
	}
	for (i = N - 1 ; i >= 0; i--) 
	{
		j                = (int)floor((i + 1) * rand());  /* j is uniformly distributed on {0, 1, ..., i} */	
		tmp              = selected[j];
		selected[j]      = selected[i];
		selected[i]      = tmp;
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void randini(UL seed)
{
	/* SHR3 Seed initialization */

	if(seed == (UL)NULL)
	{
		jsrseed  = (UL) time( NULL );
		jsr     ^= jsrseed;
	}
	else
	{
		jsr     = (UL)NULL;
		jsrseed = seed;
		jsr    ^= jsrseed;
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void scompute_distances_1(float *a , float *b , int d , int N , float *dist2)
{
	int i, ione = 1;
	float minus_two = -2.0f, one = 1.0f;
	double sum_d2, sum_c2 = 0.0;
	float *dl;
#ifdef BLAS
	sum_c2  = BLASCALL(snrm2)(&d , a , &ione);
	sum_c2 *= sum_c2;
#else
	int j;
	for (j = 0 ; j < d ; j++)
	{
		sum_c2 += (a[j]*a[j]);
	}
#endif
	for (i = 0; i < N ; i++) 
	{
		dl     = b + i*d;
#ifdef BLAS
		sum_d2 = BLASCALL(snrm2)(&d , dl , &ione);
		sum_d2 *= sum_d2;
#else
		sum_d2 = 0.0;
		for (j = 0; j < d ; j++)
		{
			sum_d2 += (dl[j]*dl[j]);
		}
#endif
		dist2[i] = (float)(sum_d2 + sum_c2);
	}
	BLASCALL(sgemv)("Transposed", &d , &N , &minus_two , b , &d , a, &ione, &one , dist2, &ione);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dcompute_distances_1(double *a , double *b , int d , int N , double *dist2)
{
	int i, ione = 1;
	double minus_two = -2.0, one = 1.0;
	double sum_d2, sum_c2 = 0.0;
	double *dl;
#ifdef BLAS
	sum_c2  = BLASCALL(dnrm2)(&d , a , &ione);
	sum_c2 *= sum_c2;
#else
	int j;
	for (j = 0 ; j < d ; j++)
	{
		sum_c2 += (a[j]*a[j]);
	}
#endif
	for (i = 0; i < N ; i++) 
	{
		dl     = b + i*d;
#ifdef BLAS
		sum_d2 = BLASCALL(dnrm2)(&d , dl , &ione);
		sum_d2 *= sum_d2;
#else
		sum_d2 = 0.0;
		for (j = 0; j < d ; j++)
		{
			sum_d2 += (dl[j]*dl[j]);
		}
#endif
		dist2[i] = (sum_d2 + sum_c2);
	}
	BLASCALL(dgemv)("Transposed", &d , &N , &minus_two , b , &d , a, &ione, &one , dist2, &ione);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void scompute_cross_distances_nonpacked (float *a, float *b, int d, int na, int nb, float *dist2, float *sum_c2)
{
	int i, j;
	float *cl , *dl;
	float *d2l;
	float minus_two = -2.0f, one = 1.0f;
	double s_c2, sum_d2;
#ifdef BLAS
	int ione = 1;
#endif

	for (i = 0; i < na; i++) 
	{
		cl = a + d*i;
#ifdef BLAS
		s_c2  = BLASCALL(snrm2)(&d , cl , &ione);
		s_c2 *= s_c2;
#else
		s_c2  = 0.0;
		for (j = 0 ; j < d; j++)
		{
			s_c2 += (cl[j]*cl[j]);
		}
#endif
		sum_c2[i] = (float)s_c2;
	}
	for (i = 0 ; i < nb ; i++) 
	{
		dl     = b + d*i;
#ifdef BLAS
		sum_d2  = BLASCALL(snrm2)(&d , dl , &ione);
		sum_d2 *= sum_d2;
#else
		sum_d2 = 0.0;
		for (j = 0 ; j < d ; j++)
		{
			sum_d2 += (dl[j]*dl[j]);
		}
#endif
		d2l = dist2 + i*na;
		for (j = 0 ; j < na; j++)
		{
			d2l[j] = sum_d2 + sum_c2[j];
		}
	}
	BLASCALL(sgemm)("Transposed", "Not trans", &na , &nb , &d , &minus_two , a , &d , b , &d , &one , dist2 , &na);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dcompute_cross_distances_nonpacked (double *a, double *b, int d, int na, int nb, double *dist2, double *sum_c2)
{
	int i, j;
	double *cl , *dl;
	double *d2l;
	double minus_two = -2.0, one = 1.0;
	double s_c2, sum_d2;
#ifdef BLAS
	int ione = 1;
#endif

	for (i = 0; i < na; i++) 
	{
		cl = a + d*i;
#ifdef BLAS
		s_c2  = BLASCALL(dnrm2)(&d , cl , &ione);
		s_c2 *= s_c2;
#else
		s_c2  = 0.0;
		for (j = 0 ; j < d; j++)
		{
			s_c2 += (cl[j]*cl[j]);
		}
#endif
		sum_c2[i] = (double)s_c2;
	}
	for (i = 0 ; i < nb ; i++) 
	{
		dl     = b + d*i;
#ifdef BLAS
		sum_d2  = BLASCALL(dnrm2)(&d , dl , &ione);
		sum_d2 *= sum_d2;
#else
		sum_d2 = 0.0;
		for (j = 0 ; j < d ; j++)
		{
			sum_d2 += (dl[j]*dl[j]);
		}
#endif
		d2l = dist2 + i*na;
		for (j = 0 ; j < na; j++)
		{
			d2l[j] = sum_d2 + sum_c2[j];
		}
	}
	BLASCALL(dgemm)("Transposed", "Not trans", &na , &nb , &d , &minus_two , a , &d , b , &d , &one , dist2 , &na);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
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
int svec_count_occurrences(float *X , int N, float val) 
{
	int count=0;
	while(N--) 
	{
		if(X[N]==val) 
		{
			count++;
		}
	}
	return count;
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
int dvec_count_occurrences(double *X , int N, double val) 
{
	int count=0;
	while(N--) 
	{
		if(X[N]==val) 
		{
			count++;
		}
	}
	return count;
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void scompute_sum_dcov(int n , int K , int d , float *X , float *M_old , float *p , float *M , float *S , float *w) 
{
	int i,j,l , nd = n*d;
	float zero=0.0f , one=1.0f;
	float *X2 , *S_j , *M_old_j , *M_j;
	float tmp;

	for (j = 0 ; j < K ; j++) 
	{
		tmp = 0.0f;
		for (i = 0 ; i < n ; i++) 
		{
			tmp += p[j + i*K];
		}
		w[j] = tmp;
	}

	BLASCALL(sgemm)("Not transposed","Transposed",&d , &K , &n , &one , X , &d , p , &K , &zero , M ,&d);

	X2  = (float *)malloc(nd*sizeof(float));
	memcpy(X2, X, nd*sizeof(float));
	svec_sqr(X2 , nd);

	BLASCALL(sgemm)("Not transposed","Transposed",&d , &K , &n , &one , X2 , &d , p , &K , &zero , S , &d);
	
	free(X2);

	for (j = 0 ; j < K ; j++) 
	{
		S_j       = S     + j*d;
		M_old_j   = M_old + j*d;
		M_j       = M     + j*d;
		for(l = 0 ; l < d ; l++) 
		{
			S_j[l] += M_old_j[l]*(M_old_j[l]*w[j] - 2.0f*M_j[l]);
		}
	}    
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dcompute_sum_dcov(int n , int K , int d , double *X , double *M_old , double *p , double *M , double *S , double *w) 
{
	int i,j,l , nd = n*d;
	double zero = 0.0 , one=1.0;
	double *X2 , *S_j , *M_old_j , *M_j;
	double tmp;

	for (j = 0 ; j < K ; j++) 
	{
		tmp = 0.0;
		for (i = 0 ; i < n ; i++) 
		{
			tmp += p[j + i*K];
		}
		w[j] = tmp;
	}

	BLASCALL(dgemm)("Not transposed","Transposed",&d , &K , &n , &one , X , &d , p , &K , &zero , M ,&d);

	X2  = (double *)malloc(nd*sizeof(double));
	memcpy(X2, X, nd*sizeof(double));
	dvec_sqr(X2 , nd);

	BLASCALL(dgemm)("Not transposed","Transposed",&d , &K , &n , &one , X2 , &d , p , &K , &zero , S , &d);
	
	free(X2);

	for (j = 0 ; j < K ; j++) 
	{
		S_j       = S     + j*d;
		M_old_j   = M_old + j*d;
		M_j       = M     + j*d;
		for(l = 0 ; l < d ; l++) 
		{
			S_j[l] += M_old_j[l]*(M_old_j[l]*w[j] - 2.0*M_j[l]);
		}
	}    
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
#ifdef OMP
void scompute_sum_dcov_thread(int n ,int K , int d , float *X , float *M_old , float *p , float *M , float *S , float *w , int num_threads) 
{
	int i , dK = d*K;

	scompute_sum_dcov_t t={
		n,K,d,
		X,M_old,p,
		svec_new(num_threads*dK), /* M */
		svec_new(num_threads*dK), /* S */
		svec_new(num_threads*K),  /* w */
		num_threads
	};

	compute_tasks(num_threads , num_threads , &scompute_sum_dcov_task_fun , &t);

	/* accumulate over n's */

	svec_cpy(M , t.M , dK);
	svec_cpy(S , t.S , dK);
	svec_cpy(w , t.w , K);

	for(i = 1 ; i < num_threads ; i++) 
	{
		svec_add(M , t.M + i*dK , dK);
		svec_add(S , t.S + i*dK , dK);    
		svec_add(w , t.w + i*K  , K);    
	}
	free(t.M);
	free(t.S);
	free(t.w);
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dcompute_sum_dcov_thread(int n ,int K , int d , double *X , double *M_old , double *p , double *M , double *S , double *w , int num_threads) 
{
	int i , dK = d*K;

	dcompute_sum_dcov_t t={
		n,K,d,
		X,M_old,p,
		dvec_new(num_threads*dK), /* M */
		dvec_new(num_threads*dK), /* S */
		dvec_new(num_threads*K),  /* w */
		num_threads
	};

	compute_tasks(num_threads , num_threads , &dcompute_sum_dcov_task_fun , &t);

	/* accumulate over n's */

	dvec_cpy(M , t.M , dK);
	dvec_cpy(S , t.S , dK);
	dvec_cpy(w , t.w , K);

	for(i = 1 ; i < num_threads ; i++) 
	{
		dvec_add(M , t.M + i*dK , dK);
		dvec_add(S , t.S + i*dK , dK);    
		dvec_add(w , t.w + i*K  , K);    
	}
	free(t.M);
	free(t.S);
	free(t.w);
}
#endif

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
void svec_cpy (float *Xdest, float *Xsource , int n)
{
  memcpy(Xdest, Xsource, n * sizeof (*Xdest));
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dvec_cpy (double *Xdest , double *Xsource, int n)
{
  memcpy(Xdest, Xsource, n * sizeof (*Xdest));
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void svec_add(float *X1 , float *X2 , int n)
{
	int i;
	for (i = 0 ; i < n ; i++)
	{
		X1[i] += X2[i];
	}
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dvec_add(double *X1 , double *X2 , int n)
{
	int i;
	for (i = 0 ; i < n ; i++)
	{
		X1[i] += X2[i];
	}
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
void scompute_sum_dcov_task_fun (void *arg , int tid, int i) 
{
	scompute_sum_dcov_t *t = arg;
	int n0 = i*t->n/t->num_threads;
	int n1 = (i+1)*t->n/t->num_threads;

	scompute_sum_dcov(n1-n0 , t->K , t->d , t->X + t->d*n0 , t->M_old , t->p + n0*t->K , t->M + i*t->d*t->K , t->S + i*t->d*t->K , t->w + t->K*i);
}
/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dcompute_sum_dcov_task_fun (void *arg , int tid, int i) 
{
	dcompute_sum_dcov_t *t = arg;
	int n0 = i*t->n/t->num_threads;
	int n1 = (i+1)*t->n/t->num_threads;

	dcompute_sum_dcov(n1-n0 , t->K , t->d , t->X + t->d*n0 , t->M_old , t->p + n0*t->K , t->M + i*t->d*t->K , t->S + i*t->d*t->K , t->w + t->K*i);
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
    
  BLASCALL(sgemm)("Transposed" , "Not transposed" , &K , &n , &d , &one , inv_S , &d , X2 , &d , &one , p , &K);
  
  free(X2);
  
  M_S = inv_S;
  for (i = 0 ; i < dK ; i++)
  {
    M_S[i] = M[i]/S[i];
  }
  
  BLASCALL(sgemm)("Transposed" , "Not transposed" , &K , &n , &d  , &minus_two , M_S , &d , X , &d , &one , p , &K);  
  
  free(M_S);      
}

/*----------------------------------------------------------------------------------------------------------------------------------------- */
void dcompute_mahalanobis_sqr(double *X , int  d , int n , int K , double *M , double *S , double *p) 
{
  double tmp, sum;
  int i, j, l, iK , jd , dn = d*n, dK = d*K;
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