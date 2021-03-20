#include "particleFilterTracker.h"


ParticleFilterTracker::ParticleFilterTracker()
{
    isFirst = true;

    NParticle=150;//number of particles
    R_BIN = G_BIN = B_BIN = 8;
    nbin=R_BIN*G_BIN*B_BIN;//bin of hist
    ran_seed=802163120;
    DELTA_T=0.05 ;    

    POSITION_DISTURB=15 ;      
    VELOCITY_DISTURB=40 ;  
    SCALE_DISTURB =0.0;      
    SCALE_CHANGE_D=0.001; 

    Pi_Thres=90;
    Weight_Thres =0.0001;
}
ParticleFilterTracker::~ParticleFilterTracker()
{
}
int ParticleFilterTracker::Initialize( Mat img, Rect toTrack )
{
    int i, j;
    float rn[7];

    set_seed( 0 ); 

    states = new SpaceState[NParticle]; 
    if ( states == NULL ) return -2;
    weights = new float [NParticle];     
    if ( weights == NULL ) return -3;	
    nbin = R_BIN * G_BIN * B_BIN; 
    ModelHist = new float [nbin];
    if ( ModelHist == NULL ) return -1;

    
    CalcuColorHistogram( toTrack,img, ModelHist, nbin );

    
    states[0].xt = toTrack.x+toTrack.width/2;
    states[0].yt = toTrack.y+toTrack.height/2;
    states[0].v_xt = (float)0.0; 
    states[0].v_yt = (float)0.0; 
    states[0].Hxt = toTrack.width/2;
    states[0].Hyt = toTrack.height/2;
    states[0].at_dot = (float)0.0;
    weights[0] = (float)(1.0/NParticle);

    for ( i = 1; i < NParticle; i++ )
    {
        for ( j = 0; j < 7; j++ ) 
        {
            rn[j] = randGaussian( 0, (float)0.6 ); 
        }
        states[i].xt = (int)( states[0].xt + rn[0] * toTrack.width/2 );
        states[i].yt = (int)( states[0].yt + rn[1] * toTrack.height/2);
        states[i].v_xt = (float)(states[0].v_xt + rn[2] * VELOCITY_DISTURB );
        states[i].v_yt = (float)( states[0].v_yt + rn[3] * VELOCITY_DISTURB );
        states[i].Hxt = (int)( states[0].Hxt + rn[4] * SCALE_DISTURB );
        states[i].Hyt = (int)( states[0].Hyt + rn[5] * SCALE_DISTURB );
        states[i].at_dot = (float)( states[0].at_dot + rn[6] * SCALE_CHANGE_D );
        
        weights[i] = (float)(1.0/NParticle);
        circle(img,Point(states[i].xt,states[i].yt),fabs(states[i].v_xt),Scalar(100,100,100));
    }

    return 1;
}
int ParticleFilterTracker::ColorParticleTracking( Mat img, Rect &toTrack,float & max_weight)
{
    SpaceState EState;

    ReSelect(states, weights, NParticle);
    
    Propagate( states, NParticle);
    
    Observe( states, weights, NParticle, img, ModelHist, nbin );
    
    Estimation( states, weights, NParticle, EState );

    int xc = EState.xt;
    int yc = EState.yt;
    int Wx_h = EState.Hxt;
    int Hy_h = EState.Hyt;
    toTrack=Rect(xc-Wx_h,yc-Hy_h,2*Wx_h,2*Hy_h);
    
    ModelUpdate( EState, ModelHist, nbin, Pi_Thres,	img);

    
    max_weight = weights[0];
    for (int i = 1; i < NParticle; i++ )
    {
        max_weight = max_weight < weights[i] ? weights[i] : max_weight;
    }
    // add CNN to re-calculate the weights of particles

    if ( xc < 0 || yc < 0 || xc >= img.cols || yc >= img.rows ||
        Wx_h <= 0 || Hy_h <= 0 ) 
        return -1;
    else 
        return 1;		
}

/*calculate color histogram of a region*/
void ParticleFilterTracker::CalcuColorHistogram( Rect toTrack, Mat img, float * ColorHist, int bins )
{
    int x_begin, y_begin;  
    int y_end, x_end;
    int  index;
    int r, g, b;
    float k, r2, f;
    int a2; // normalizing constant

    for (int i = 0; i < bins; i++ )    
        ColorHist[i] = 0.0;
    
    Rect whole(0,0,img.cols,img.rows);
    toTrack &=whole;
    x_begin = toTrack.x;               
    y_begin = toTrack.y;
    x_end = x_begin + toTrack.width;
    y_end = y_begin + toTrack.height;
    //the center point of the rect
    int x0=(x_begin+x_end)/2;
    int y0=(y_begin+y_end)/2;

    a2 = (toTrack.width/2)*(toTrack.width/2)+(toTrack.height/2)*(toTrack.height/2);                

    f = 0.0;   
    uchar* image = img.data; 
    int R_SHIFT= log(256/R_BIN)/log(2);
    int G_SHIFT= log(256/G_BIN)/log(2);   
    int B_SHIFT= log(256/B_BIN)/log(2);                    
    for (int y = y_begin; y < y_end; y++ )
    {
        for (int x = x_begin; x < x_end; x++ )
        {
            r = (int)(image[(y*img.cols+x)*3+2]) >> R_SHIFT;   
            
            g = (int)(image[(y*img.cols+x)*3+1])>> G_SHIFT; 
            b = (int)(image[(y*img.cols+x)*3]) >> B_SHIFT;
        
            index = r * G_BIN * B_BIN + g * B_BIN + b;
            
            r2 = (float)(((y-y0)*(y-y0)+(x-x0)*(x-x0))*1.0/a2); 
            
            k = 1 - r2;   /*k(r) = 1-r^2, |r| < 1;k(r) = 0 */
            
            f = f + k;
            ColorHist[index] = ColorHist[index] + k; 
        }
    }
    // normalize distribution by a factor f
    for (int i = 0; i < bins; i++ )     
        ColorHist[i] = ColorHist[i]/f;
    return;
}

float ParticleFilterTracker::CalcuBhattacharyya( float * p, float * q, int bins )
{
    float rho = 0.0f;

    for (int i = 0; i < bins; ++i )
    {
        rho = (float)(rho + sqrt( p[i]*q[i] ));
        
    }
    
    return rho;
}



/*# define RECIP_SIGMA  3.98942280401  / * 1/(sqrt(2*pi)*sigma), sigma = 0.1 * /*/
# define SIGMA2       0.02           /* 2*sigma^2, sigma = 0.1 */

float ParticleFilterTracker::CalcuWeightedPi( float rho )
{
    float pi_n, d2;

    d2 = 1 - rho;
    //pi_n = (float)(RECIP_SIGMA * exp( - d2/SIGMA2 ));
    pi_n = (float)(exp( - d2/SIGMA2 ));

    return( pi_n );
}


void ParticleFilterTracker::NormalizeCumulatedWeight( float * weight, float * cumulateWeight, int N )
{
    int i;

    for ( i = 0; i < N+1; i++ ) 
        cumulateWeight[i] = 0;
    for ( i = 0; i < N; i++ )
        cumulateWeight[i+1] = cumulateWeight[i] + weight[i];
    for ( i = 0; i < N+1; i++ )
        cumulateWeight[i] = cumulateWeight[i]/ cumulateWeight[N];

    return;
}

int ParticleFilterTracker::BinarySearch( float v, float * NCumuWeight, int N )
{
    int l, r, m;

    l = 0; 	r = N-1;   /* extreme left and extreme right components' indexes */
    while ( r >= l)
    {
        m = (l+r)/2;
        if ( v >= NCumuWeight[m] && v < NCumuWeight[m+1] ) return( m );
        if ( v < NCumuWeight[m] ) r = m - 1;
        else l = m + 1;
    }
    return 0;
}


void ParticleFilterTracker::ImportanceSampling( float * weight, int * ResampleIndex, int N)
{
    float rnum, * cumulateWeight;
    int i, j;

    cumulateWeight = new float [N+1];
    //check if resampling or not
    float sum_weight = 0.0f;
    for (i = 0; i < N;i++)
    {
        sum_weight += weight[i];
    }
    float thresh = 0.02;
    cout << sum_weight;
    if(isFirst == false)
    {
        if(1/sum_weight < thresh)
        {
            return;
        }
    }
    if(isFirst == true)
    {
        isFirst = false;
    }
    NormalizeCumulatedWeight(weight, cumulateWeight, N); 
    for ( i = 0; i < N; i++ )
    {
        do
        {
            rnum = rand0_1();
        } while (rnum < 0.5);


        j = BinarySearch( rnum, cumulateWeight, N+1 );
        if ( j == N ) j--;
        ResampleIndex[i] = j;
    }

    delete cumulateWeight;

    return;	
}
/*
SPACESTATE * state：     
float * weight：         
int N：                  
SPACESTATE * state：     
*/
void ParticleFilterTracker::ReSelect( SpaceState * state, float * weight, int N )
{
    SpaceState * tmpState;
    int i, * rsIdx;

    tmpState = new SpaceState[N];
    rsIdx = new int[N];

    ImportanceSampling( weight, rsIdx, N);
    for (i = 0; i < N; i++)
        tmpState[i] = state[rsIdx[i]];
    for ( i = 0; i < N; i++ )
        state[i] = tmpState[i];

    delete[] tmpState;
    delete[] rsIdx;

    return;
}

/*
    S(t) = A S(t-1) + W(t-1)


    SPACESTATE * state：      
    int N：                   

    SPACESTATE * state
*/
void ParticleFilterTracker::Propagate( SpaceState * state, int N)
{
    int i;
    int j;
    float rn[7];


    for ( i = 0; i < N; i++ )  
    {
        for ( j = 0; j < 7; j++ ) rn[j] = randGaussian( 0, (float)0.6 ); 
        state[i].xt = (int)(state[i].xt + state[i].v_xt * DELTA_T + rn[0] * state[i].Hxt + 0.5);
        state[i].yt = (int)(state[i].yt + state[i].v_yt * DELTA_T + rn[1] * state[i].Hyt + 0.5);
        state[i].v_xt = (float)(state[i].v_xt + rn[2] * VELOCITY_DISTURB);
        state[i].v_yt = (float)(state[i].v_yt + rn[3] * VELOCITY_DISTURB);
        state[i].Hxt = (int)(state[i].Hxt+state[i].Hxt*state[i].at_dot + rn[4] * SCALE_DISTURB + 0.5);
        state[i].Hyt = (int)(state[i].Hyt+state[i].Hyt*state[i].at_dot + rn[5] * SCALE_DISTURB + 0.5);
        state[i].at_dot = (float)(state[i].at_dot + rn[6] * SCALE_CHANGE_D);
    }
    return;
}

void ParticleFilterTracker::Observe( SpaceState * state, float * weight, int N,
                Mat img,float * ObjectHist, int hbins )
{
    int i;
    float * ColorHist;
    float rho;

    ColorHist = new float[hbins];

    for ( i = 0; i < N; i++ )
    {
        CalcuColorHistogram(Rect(state[i].xt-state[i].Hxt, state[i].yt-state[i].Hyt,2*state[i].Hxt, 2*state[i].Hyt),
            img,ColorHist, hbins );
        
        rho = CalcuBhattacharyya( ColorHist, ObjectHist, hbins );
    
        weight[i] = CalcuWeightedPi( rho );
    }

    delete ColorHist;

    return;	
}

void ParticleFilterTracker::Estimation( SpaceState * state, float * weight, int N, 
                SpaceState & EstState )
{
    int i;
    float at_dot, Hxt, Hyt, v_xt, v_yt, xt, yt;
    float weight_sum;

    at_dot = 0;
    Hxt = 0; 	Hyt = 0;
    v_xt = 0;	v_yt = 0;
    xt = 0;  	yt = 0;
    weight_sum = 0;
    for ( i = 0; i < N; i++ ) 
    {
        at_dot += state[i].at_dot * weight[i];
        Hxt += state[i].Hxt * weight[i];
        Hyt += state[i].Hyt * weight[i];
        v_xt += state[i].v_xt * weight[i];
        v_yt += state[i].v_yt * weight[i];
        xt += state[i].xt * weight[i];
        yt += state[i].yt * weight[i];
        weight_sum += weight[i];
    }
    
    if ( weight_sum <= 0 ) weight_sum = 1; 
    EstState.at_dot = at_dot/weight_sum;
    EstState.Hxt = (int)(Hxt/weight_sum + 0.5 );
    EstState.Hyt = (int)(Hyt/weight_sum + 0.5 );
    EstState.v_xt = v_xt/weight_sum;
    EstState.v_yt = v_yt/weight_sum;
    EstState.xt = (int)(xt/weight_sum + 0.5 );
    EstState.yt = (int)(yt/weight_sum + 0.5 );

    return;
}

    

int ParticleFilterTracker::ModelUpdate( SpaceState EstState, float * TargetHist, int bins, float PiT,Mat img)
{
    float * EstHist, Bha, Pi_E;
    int i, rvalue = -1;

    EstHist = new float [bins];

    CalcuColorHistogram( Rect(EstState.xt, EstState.yt, EstState.Hxt,EstState.Hyt), img,EstHist, bins );
    
    Bha  = CalcuBhattacharyya( EstHist, TargetHist, bins );
    

    Pi_E = CalcuWeightedPi( Bha );
    float ALPHA_COEFFICIENT = 0.2;
    if ( Pi_E > PiT ) 
    {
        for ( i = 0; i < bins; i++ )
        {
            TargetHist[i] = (float)((1.0 - ALPHA_COEFFICIENT) * TargetHist[i]
            + ALPHA_COEFFICIENT * EstHist[i]);
        }
        rvalue = 1;
    }

    delete EstHist;

    return rvalue;
}