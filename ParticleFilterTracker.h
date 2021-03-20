#include <opencv2/opencv.hpp>
#include <math.h>
#include <time.h>
#include <iostream>
using namespace std;
using namespace cv;
#define IA 16807
#define IM 2147483647
#define AM (1.0/IM)
#define IQ 127773
#define IR 2836
#define MASK 123459876

class ParticleFilterTracker
{
private:
	class SpaceState
	{  
	public:
			int xt;               
			int yt;               
			float v_xt;           
			float v_yt;  	        
			int Hxt;              
			int Hyt;              
			float at_dot;         
	} ;
	bool isFirst;
	SpaceState *states;
	float *weights;
	float *ModelHist;
	int NParticle;//number of particles
	int R_BIN,G_BIN,B_BIN;
	int nbin;//bin of hist
	long ran_seed;
	float DELTA_T ;    /*30，25，15，10 */
	int POSITION_DISTURB ;      
	float VELOCITY_DISTURB ;  
	float SCALE_DISTURB ;      
	float SCALE_CHANGE_D;   
	float Pi_Thres; 
	float Weight_Thres ;
private:
	/*some function for generate random*/
	long set_seed( long setvalue )
	{
		if ( setvalue != 0 )
			ran_seed = setvalue;
		else                 
		{
			ran_seed = time(NULL);
		}
		return( ran_seed );
	}

	float ran0(long *idum)
	{
		long k;
		float ans;

		/* *idum ^= MASK;*/      /* XORing with MASK allows use of zero and other */
		k=(*idum)/IQ;            /* simple bit patterns for idum.                 */
		*idum=IA*(*idum-k*IQ)-IR*k;  /* Compute idum=(IA*idum) % IM without over- */
		if (*idum < 0) *idum += IM;  /* flows by Schrage’s method.               */
		ans=AM*(*idum);          /* Convert idum to a floating result.*/
		//  *idum ^= MASK;      /* Unmask before return.*/
		return ans;
	}
		
	float rand0_1()
	{
		return( ran0( &ran_seed ) );
	}
	/*
	N(u,sigma)Gaussian
	*/
	float randGaussian( float u, float sigma )
	{
		float x1, x2, v1, v2;
		float s = 100.0;
		float y;

		
		while ( s > 1.0 )
		{
			x1 = rand0_1();
			x2 = rand0_1();
			v1 = 2 * x1 - 1;
			v2 = 2 * x2 - 1;
			s = v1*v1 + v2*v2;
		}
		y = (float)(sqrt( -2.0 * log(s)/s ) * v1);
		
		return( sigma * y + u );	
	}

	void CalcuColorHistogram(Rect toTrack, Mat img, float *ColorHist, int bins);
	float CalcuBhattacharyya(float *p, float *q, int bins);
	float CalcuWeightedPi(float rho);
	void NormalizeCumulatedWeight(float *weight, float *cumulateWeight, int N);
	int BinarySearch(float v, float *NCumuWeight, int N);

  private:
	void ImportanceSampling(float *weight, int* ResampleIndex, int N);
	void ReSelect(SpaceState *state, float *weight, int N);
	void Propagate(SpaceState *state, int N);
	void Observe(SpaceState *state, float *weight, int N, Mat img, float *ObjectHist, int hbins);
	void Estimation(SpaceState *state, float *weight, int N, SpaceState &EstState);
	int ModelUpdate(SpaceState EstState, float *TargetHist, int bins, float PiT, Mat img);

  public:
	ParticleFilterTracker();
	~ParticleFilterTracker();

	int Initialize(Mat img, Rect toTrack);
	int ColorParticleTracking(Mat img, Rect &toTrack, float &maxWeight);
};