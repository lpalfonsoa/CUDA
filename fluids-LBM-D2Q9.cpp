#include <iostream>
#include <fstream>
#include <cmath>
using namespace std;

#define Lx 256
#define Ly 64
#define N 32 //Threads per Block
const int M=(Lx*Ly+N-1)/N; //Blocks per Grid

#define Q 9
const int ArraySize=Lx*Ly*Q;

const float tau=0.55;
const float Utau=1.0/tau;
const float UmUtau=1-Utau;

//const int ixc=Lx/8, iyc=Ly/2, R=Ly/5;
const int ix_test=Lx/2, iy_test=Ly/2, i_test=1;

//------------ PROGRAMMING ON THE DEVICE ----------------
//---------------Constants (Symbols)----------------
__constant__ float d_w[Q];
__constant__ int d_Vx[Q];
__constant__ int d_Vy[Q];
__constant__ float d_tau[3]; // d_tau[0]=tau,  d_tau[1]=Utau,  d_tau[2]=UmUtau,
__constant__ int d_Cylinder[4]; // d_Cylinder[0]=ixc, d_Cylinder[1]=iyc, d_Cylinder[2]=R, d_Cylinder[3]=R*R,
//----------Functions called by the device itself
//Data index
__device__ int d_n(int ix,int iy,int i){
  return (iy+ix*Ly)*Q+i;  
}
//Macroscopic Fields
__device__ float d_rho(int ix,int iy,float *d_f){
  float sum=0; int i,n0;
  for(i=0;i<Q;i++){
    n0=d_n(ix,iy,i); sum+=d_f[n0];
  }
  return sum;
}
__device__ float d_Jx(int ix,int iy,float *d_f){
  float sum=0; int i,n0;
  for(i=0;i<Q;i++){
    n0=d_n(ix,iy,i); sum+=d_Vx[i]*d_f[n0];
  }
  return sum;
}
__device__ float d_Jy(int ix,int iy,float *d_f){
  float sum=0; int i,n0;
  for(i=0;i<Q;i++){
    n0=d_n(ix,iy,i); sum+=d_Vy[i]*d_f[n0];
  }
  return sum;
}
//Equilibrium Functions
__device__ float d_feq(float rho0,float Ux0,float Uy0,int i){
  float UdotVi=Ux0*d_Vx[i]+Uy0*d_Vy[i], U2=Ux0*Ux0+Uy0*Uy0;
  return rho0*d_w[i]*(1+3*UdotVi+4.5*UdotVi*UdotVi-1.5*U2);
}
//---------------------KERNELS----------------------------
__global__ void d_Collision(float *d_f,float *d_fnew){
  //Define internal registers
  int icell,ix,iy,i,n0;  float rho0,Ux0,Uy0;
  //Find which thread an which cell should I work
  icell=blockIdx.x*blockDim.x+threadIdx.x;
  ix=icell/Ly; iy=icell%Ly;
  //Compute the macroscopic fields
  rho0=d_rho(ix,iy,d_f);    //rho
  Ux0=d_Jx(ix,iy,d_f)/rho0; //Ux0
  Uy0=d_Jy(ix,iy,d_f)/rho0; //Uy0
  //Collide and compute fnew
  for(i=0;i<Q;i++){ //on each direction
    n0=d_n(ix,iy,i); d_fnew[n0]=d_tau[2]*d_f[n0]+d_tau[1]*d_feq(rho0,Ux0,Uy0,i);
  }
}
__global__ void d_ImposeFields(float *d_f,float *d_fnew,float Ufan){
  //Define internal registers
  int icell,i,ix,iy,n0; float rho0;
  //Find which thread an which cell should I work
  icell=blockIdx.x*blockDim.x+threadIdx.x;
  ix=icell/Ly; iy=icell%Ly;
   //Compute the macroscopic field
  rho0=d_rho(ix,iy,d_f); //rho
  //fan
  if(ix==0)
    for(i=0;i<Q;i++){n0=d_n(ix,iy,i); d_fnew[n0]=d_feq(rho0,Ufan,0,i);}
   // /*
  //obstacle
  else if((ix-d_Cylinder[0])*(ix-d_Cylinder[0])+(iy-d_Cylinder[1])*(iy-d_Cylinder[1])<=d_Cylinder[3])
    for(i=0;i<Q;i++) {n0=d_n(ix,iy,i); d_fnew[n0]=d_feq(rho0,0,0,i);}
  //An extra point at one side to break the symmetry
  else if(ix==d_Cylinder[0] && iy==d_Cylinder[1]+d_Cylinder[2]+1)
    for(i=0;i<Q;i++){n0=d_n(ix,iy,i); d_fnew[n0]=d_feq(rho0,0,0,i);}	
   // */
}
__global__ void d_Advection(float *d_f,float *d_fnew){
  //Define internal registers
  int icell,ix,iy,i,ixnext,iynext,n0,n0next;
  //Find which thread an which cell should I work
  icell=blockIdx.x*blockDim.x+threadIdx.x;
  ix=icell/Ly; iy=icell%Ly;
  //Move the contents to the neighboring cells
  for(i=0;i<Q;i++){ //on each direction
    ixnext=(ix+d_Vx[i]+Lx)%Lx; iynext=(iy+d_Vy[i]+Ly)%Ly;//periodic boundaries
    n0=d_n(ix,iy,i); n0next=d_n(ixnext,iynext,i);
    d_f[n0next]=d_fnew[n0]; 
  }
}
//------------ PROGRAMMING ON THE HOST ----------------
//-------------LatticeBoltzmann class------------
class LatticeBoltzmann{
private:
  int h_Cylinder[4];// d_Cylinder[0]=ixc, d_Cylinder[1]=iyc, d_Cylinder[2]=R, d_Cylinder[3]=R*R, 
  float h_tau[3]; // h_tau[0]=tau,  h_tau[1]=Utau,  h_tau[2]=UmUtau, 
  float h_w[Q]; // w[i]
  int h_Vx[Q],h_Vy[Q]; // Vx[i],Vy[i]
  float *h_f,*h_fnew;  float *d_f,*d_fnew;// f[ix][iy][i]
  float *h_Test,*d_Test; //Just for tests
public:
  LatticeBoltzmann(void);
  ~LatticeBoltzmann(void);
  int n(int ix,int iy,int i){return (ix*Ly+iy)*Q+i;};
  float h_rho(int ix,int iy);
  float h_Jx(int ix,int iy);
  float h_Jy(int ix,int iy);
  float h_feq(float rho0,float Ux0,float Uy0,int i);
  void Start(float rho0,float Ux0,float Uy0);
  void Collision(void);
  void ImposeFields(float Ufan);
  void Advection(void);
  void Print(const char * NameFile,double Ufan);
  void ShowTest(void);
};

LatticeBoltzmann::LatticeBoltzmann(void){
  //CONSTANTS(d_Symbols)
  //---Charge constantes on the Host-----------------
  //running constants
  h_Cylinder[0]=Lx/8;  h_Cylinder[1]=Ly/2;  h_Cylinder[2]=Ly/5;  h_Cylinder[3]=h_Cylinder[2]*h_Cylinder[2];
  h_tau[0]=tau;  h_tau[1]=Utau;  h_tau[2]=UmUtau;
  //weights
  h_w[0]=4.0/9; h_w[1]=h_w[2]=h_w[3]=h_w[4]=1.0/9; h_w[5]=h_w[6]=h_w[7]=h_w[8]=1.0/36;
  //velocity vectors
  h_Vx[0]=0;  h_Vx[1]=1;  h_Vx[2]=0;  h_Vx[3]=-1; h_Vx[4]=0;
  h_Vy[0]=0;  h_Vy[1]=0;  h_Vy[2]=1;  h_Vy[3]=0;  h_Vy[4]=-1;

            h_Vx[5]=1;  h_Vx[6]=-1; h_Vx[7]=-1; h_Vx[8]=1;
            h_Vy[5]=1;  h_Vy[6]=1;  h_Vy[7]=-1; h_Vy[8]=-1;
  //------Send to the Device-----------------
  cudaMemcpyToSymbol(d_w,h_w,Q*sizeof(float),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_Vx,h_Vx,Q*sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_Vy,h_Vy,Q*sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_Cylinder,h_Cylinder,4*sizeof(int),0,cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(d_tau,h_tau,3*sizeof(float),0,cudaMemcpyHostToDevice);
  //DISTRIBUTION FUNCTIONS
  //Build the dynamic matrices on the host
  h_f=new float [ArraySize];  h_fnew=new float [ArraySize];
  //Build the dynamic matrices on the device
  cudaMalloc((void**) &d_f,ArraySize*sizeof(float));
  cudaMalloc((void**) &d_fnew,ArraySize*sizeof(float));
  //Test variables
   h_Test=new float [1]; cudaMalloc((void**) &d_Test,sizeof(float));
}
LatticeBoltzmann::~LatticeBoltzmann(void){
  delete[] h_f;  delete[] h_fnew;
  cudaFree(d_f);  cudaFree(d_fnew);
  //Test variables
  delete[] h_Test; cudaFree(d_Test);
}
float LatticeBoltzmann::h_rho(int ix,int iy){
  //Note: Please import data from device before running
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=n(ix,iy,i); sum+=h_fnew[n0];
  }
  return sum;
}
float LatticeBoltzmann::h_Jx(int ix,int iy){
  //Note: Please import data from device before running
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=n(ix,iy,i); sum+=h_Vx[i]*h_fnew[n0];
  }
  return sum;
}
float LatticeBoltzmann::h_Jy(int ix,int iy){
  //Note: Please import data from device before running
  float sum; int i,n0;
  for(sum=0,i=0;i<Q;i++){
    n0=n(ix,iy,i); sum+=h_Vy[i]*h_fnew[n0];
  }
  return sum;
}
float LatticeBoltzmann::h_feq(float rho0,float Ux0,float Uy0,int i){
  float UdotVi=Ux0*h_Vx[i]+Uy0*h_Vy[i], U2=Ux0*Ux0+Uy0*Uy0;
 // cout << "Valorea en fequilibrio:  " <<rho0<<" "<<h_w[i]<< " " << 1+3*UdotVi+4.5*UdotVi*UdotVi-1.5*U2 << endl;
  return rho0*h_w[i]*(1+3*UdotVi+4.5*UdotVi*UdotVi-1.5*U2);
}
void LatticeBoltzmann::Start(float rho0,float Ux0,float Uy0){
  int ix,iy,i,n0;
  //Charge on the Host
  for(ix=0;ix<Lx;ix++) //for each cell
    for(iy=0;iy<Ly;iy++)
      for(i=0;i<Q;i++){ //on each direction
	n0=n(ix,iy,i); h_f[n0]=h_feq(rho0,Ux0,Uy0,i);
  cout <<ix<<" "<<iy<< h_f[n0] << endl;
      }
  //Send to the Device
  cudaMemcpy(d_f,h_f,ArraySize*sizeof(float),cudaMemcpyHostToDevice);

}  
void LatticeBoltzmann::Collision(void){
  //Do everything on the Device
  dim3 ThreadsPerBlock(N,1,1);
  dim3 BlocksPerGrid(M,1,1);
  d_Collision<<<BlocksPerGrid,ThreadsPerBlock>>>(d_f,d_fnew); //OJO, quitar test
  //  cudaMemcpy(h_Test,d_Test,sizeof(float),cudaMemcpyDeviceToHost); //OJO
  //  cout<<"Test="<<h_Test[0]<<endl; //OJO
}
void LatticeBoltzmann::ImposeFields(float Ufan){
  //A thread for each cell
  dim3 ThreadsPerBlock(N,1,1);
  dim3 BlocksPerGrid(M,1,1);
  d_ImposeFields<<<BlocksPerGrid,ThreadsPerBlock>>>(d_f,d_fnew,Ufan);
}
void LatticeBoltzmann::Advection(void){
  //Do everything on the Device
  dim3 ThreadsPerBlock(N,1,1);
  dim3 BlocksPerGrid(M,1,1);
  d_Advection<<<BlocksPerGrid,ThreadsPerBlock>>>(d_f,d_fnew);
}
void LatticeBoltzmann::Print(const char * NameFile,double Ufan){
  ofstream MyFile(NameFile); double rho0,Ux0,Uy0; int ix,iy;
  //Bring back the data from Device to Host
  cudaMemcpy(h_fnew,d_fnew,ArraySize*sizeof(float),cudaMemcpyDeviceToHost);
  //Print for gnuplot plot vec
  for(ix=0;ix<Lx;ix+=4){
    for(iy=0;iy<Ly;iy+=4){
      rho0=h_rho(ix,iy); Ux0=h_Jx(ix,iy)/rho0; Uy0=h_Jy(ix,iy)/rho0;
      MyFile<<ix<<" "<<iy<<" "<<Ux0/Ufan*4<<" "<<Uy0/Ufan*4<<endl;
    }
    MyFile<<endl;
  }
  MyFile.close();
}
void LatticeBoltzmann::ShowTest(void){
  //Bring back test data from Device to Host
  cudaMemcpy(h_Test,d_Test,sizeof(float),cudaMemcpyDeviceToHost);
  cout<<"Test="<<h_Test[0]<<endl;
}

//--------------- GLOBAL FUNCTIONS ------------

int main(void){
  LatticeBoltzmann Tunnel;
  int t,tmax=5000;//=10000;
  float rho0=1.0,Ufan0=0.1;

  //Start
  Tunnel.Start(rho0,Ufan0*0.0,0);
  //Run
  for(t=0;t<tmax;t++){
    Tunnel.Collision();
    Tunnel.ImposeFields(Ufan0);
    Tunnel.Advection();
  }
  //Print Results
  Tunnel.Print("TunelDeViento-SRT.dat",Ufan0);
 
  return 0;
}  
