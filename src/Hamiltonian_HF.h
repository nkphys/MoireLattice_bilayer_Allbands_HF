#include <algorithm>
#include <functional>
#include <math.h>
#include "random"
#include "tensor_type.h"
#include "ParametersEngine.h"
#include "Coordinates_ContinuumModel.h"
#include "Hamiltonian_ContinuumModel.h"
#include "Coordinates.h"

#define PI acos(-1.0)

#ifndef Hamiltonian_class
#define Hamiltonian_class

extern "C" void   zheev_(char *,char *,int *,std::complex<double> *, int *, double *,
                         std::complex<double> *,int *, double *, int *);
//zheev_(&jobz,&uplo,&n,&(Ham_(0,0)),&lda,&(eigs_[0]),&(work[0]),&lwork,&(rwork[0]),&info);



class Hamiltonian {
public:

    Hamiltonian(Parameters& Parameters__, Coordinates&  Coordinates__, Hamiltonian_ContinuumModel&  HamiltonianCont__, mt19937_64& Generator__  )
        :Parameters_(Parameters__),Coordinates_(Coordinates__),HamiltonianCont_(HamiltonianCont__), Generator_(Generator__)

    {
        Initialize();
    }


    void Initialize();    //::DONE
    void Calculate_FormFactors();
    void Copy_BlochSpectrum(Mat_4_Complex_doub &BlochStates_old, Mat_3_doub &eigvals);
    void Folding_to_BrillouinZone(int k1, int k2, int &k1_new, int &k2_new, int &G1_ind, int &G2_ind);
    complex<double> FormFactor(int spin, int band1, int band2, int k1_vec_ind1,int k1_vec_ind2, int k2_vec_ind1, int k2_vec_ind2);
    void PrintFormFactors(int band1, int band2, int spin);
    void PrintBlochStates();
    complex<double> Interaction_value(int spin, int spin_p, int band1, int band2, int band3, int band4, int k1_ind, int k2_ind, int q_ind);
    void Create_Amat_and_Bmat();
    void Save_InteractionVal();
    double V_int(double q_val);
    void Print_Interaction_value();
    void Print_Interaction_value2(int k1_ind, int k2_ind);
    void Print_Vint();
    void Create_k_sublattices();
    void Print_k_sublattices();
    bool AreSitesRelated(int k1,int k2);
    void RunSelfConsistency();
    complex<double> Hartree_coefficient(int k2_ind, int k3_ind, int band2, int band3, int spin_p);
    complex<double> Hartree_coefficient_new(int k2_ind, int k3_ind, int band2, int band3, int spin_p);
    complex<double> Fock_coefficient(int k2_ind, int k3_ind, int band2, int band4, int spin, int spin_p);
    complex<double> Fock_coefficient_new(int k2_ind, int k3_ind, int band2, int band4, int spin, int spin_p);
   // complex<double> Xmat_val(kSL_ind, k1_ind, k2_ind, spin, spin_p, layer1, layer2, band1, band2, );
    void Create_Hamiltonian(int kset_ind);
    void Diagonalize(char option);
    void AppendEigenspectrum(int kset_ind);
    void Calculate_OParams_and_diff(double &diff_);    
    double FermiFunction(double Eval);
    void Update_OParams();
    double chemicalpotential(double muin,double Particles);
    void Get_max_and_min_eigvals();
    double Myrandom();
    void Initialize_OParams();
    void Print_Spectrum(int kset_ind, string filename);
    void Update_Hartree_Coefficients();
    void Update_Fock_Coefficients();
    void Calculate_Total_Spin();
    double DispersionTriangularLattice(int k_ind);
    double Lorentzian(double eta, double x);
    void Print_SPDOS(string filename);
    void Calculate_RealSpace_OParams(string filename, string filename2);
    void Calculate_RealSpace_OParams_new(string filename, string filename2);
    void Print_HF_Bands();
    Mat_1_intpair Get_k_path(int path_no);
    void Get_layer_overlaps(double &overlap_top, double &overlap_bottom, int band, int spin, int q_ind1, int q_ind2);
    void Saving_BlochState_Overlaps();
    void Calculate_ChernNumbers_HFBands();
    void Calculate_layer_resolved_densities();
    //---------------------------------------
   


    Parameters &Parameters_;
    Coordinates &Coordinates_;
    Hamiltonian_ContinuumModel &HamiltonianCont_;
    Matrix<complex<double>> Ham_;
    Mat_2_doub EigValues;
    vector<Matrix<complex<double>>> EigVectors;
    Mat_1_doub eigs_;

    Mat_1_intpair Inverse_kSublattice_mapping;
    int Nbands, l1_k, l2_k, MUC_L1, MUC_L2, space_k;
    int ns_, l1_, l2_;
    int N_Blocks_l1, N_Blocks_l2;
    int G_grid_L1, G_grid_L2;

    Matrix<int> NMat_MUC;
    int NMat_det;

    Mat_2_int k_sublattices;

    double mu_, KB_, Temperature, beta_;

    double nu_holes_target;
    double nu_holes, nu_holes_new;
    double d_gate; //in Angstorm
    double Area;

    double EigVal_max, EigVal_min;

    int HF_max_iterations;
    double HF_convergence_error;
    double alpha_mixing;
    string Convergence_technique;

    vector<Matrix<complex<double>>> OParams; //Single particle density matrix
    vector<Matrix<complex<double>>> OParams_new;

    Mat_6_Complex_doub HartreeCoefficients;
    Mat_7_Complex_doub FockCoefficients;
    //(int k2_ind, int k3_ind, int band2, int band3, int spin_p);
    //Form factors
    Mat_5_Complex_doub Lambda_;
    Mat_9_Complex_doub Bmat, Amat;
    Mat_9_Complex_doub Xmat;
    Mat_7_Complex_doub Omat;

    Mat_9_Complex_doub Interaction_val;
    //(int spin, int spin_p, int band1, int band2, int band3, int band4, int k1_ind, int k2_ind, int q_ind);


    Mat_4_Complex_doub BlochStates, BlochStates_old_; //[valley(spin)][band][k][G,l]
    Mat_3_doub BlochEigvals; //[valley(spin)][band][k]
    Mat_6_Complex_doub BO; //[band][spin][k][band'][spin'][k']


    Mat_2_Complex_doub N_layer_tau;
//------------------
    
    double kx_, ky_;
    double k_plusx, k_minusx, k_plusy, k_minusy;
    double k_plusx_p, k_minusx_p, k_plusy_p, k_minusy_p;
    double kx_offset, ky_offset;
    int valley;

    Matrix<complex<double>> HTB_;
    Matrix<double> Tx,Ty,Tpxpy,Tpxmy;
    

    double Overlap_bottom, Overlap_top;

    //real space  effective H params
    int L1_eff, L2_eff;
    Mat_2_Complex_doub Tij;
    Mat_2_Complex_doub Uij;

    mt19937_64 &Generator_; //for random 
    uniform_real_distribution<double> dis_;//for random fields



    //Observables
    double Total_n_up, Total_n_dn; 
    double Total_Sz, Total_Sx, Total_Sy;

};


void Hamiltonian::Calculate_RealSpace_OParams_new(string filename, string filename2){

int M1, M2; //no. of slices of one moire unit cell
M1=10;M2=10;
int m1_max, m1_min, m2_max, m2_min; //range for full 2d lattice
m1_max = M1*l1_;m1_min=0;
m2_max = M2*l1_;m2_min=0;


double rx,ry;

int comp1, comp2, G1_ind1, G1_ind2;
int G2_ind1, G2_ind2;
int k1_ind_val, k2_ind_val;
int col_val, row_val;

int spin_up=0;
int spin_dn=1;
int h1_1, h1_2, h2_1, h2_2;
int l1, l2;

int col_val_up, row_val_up;
int col_val_dn, row_val_dn;
string filename_new, filename2_new;
for(int layer=0;layer<2;layer++){
filename_new = "layer_"+to_string(layer)+"_"+filename;
filename2_new = "layer_"+to_string(layer)+"_"+filename2;
ofstream fileout(filename_new.c_str());
ofstream fileout2(filename2_new.c_str());
fileout<<"# m1  m2  rx  ry   density   sz  sx  sy"<<endl;
fileout2<<"#cell1  cell2   m1  m2  rx  ry   density   sz  sx  sy"<<endl;

l1=layer;
l2=layer;
complex<double> density_, Sz_, Sx_, Sy_;
for(int m1=m1_min;m1<=m1_max;m1++){
for(int m2=m2_min;m2<=m2_max;m2++){

rx = (  ((m1*1.0)/(M1*1.0)) + ((m2*1.0)/(M2*1.0)) )*sqrt(3.0)*0.5;
ry = (  -((m1*1.0)/(M1*1.0)) + ((m2*1.0)/(M2*1.0)) )*0.5;

density_ =0.0;
Sz_=0.0;Sx_=0.0;Sy_=0.0;
for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
for(int k1_ind=0;k1_ind<k_sublattices[kSL_ind].size();k1_ind++){
for(int k2_ind=0;k2_ind<k_sublattices[kSL_ind].size();k2_ind++){
for(int G1_ind=0;G1_ind<G_grid_L1*G_grid_L2;G1_ind++){
for(int G2_ind=0;G2_ind<G_grid_L1*G_grid_L2;G2_ind++){
for(int band1=0;band1<Nbands;band1++){
for(int band2=0;band2<Nbands;band2++){

G1_ind1 = HamiltonianCont_.Coordinates_.indx_cellwise(G1_ind);
G1_ind2 = HamiltonianCont_.Coordinates_.indy_cellwise(G1_ind);
G2_ind1 = HamiltonianCont_.Coordinates_.indx_cellwise(G2_ind);
G2_ind2 = HamiltonianCont_.Coordinates_.indy_cellwise(G2_ind);


k1_ind_val = k_sublattices[kSL_ind][k1_ind];
k2_ind_val = k_sublattices[kSL_ind][k2_ind];
h1_1 = Coordinates_.indx_cellwise(k1_ind_val);
h1_2 = Coordinates_.indy_cellwise(k1_ind_val);
h2_1 = Coordinates_.indx_cellwise(k2_ind_val);
h2_2 = Coordinates_.indy_cellwise(k2_ind_val);


comp1 = HamiltonianCont_.Coordinates_.Nbasis(G1_ind1, G1_ind2, layer);
comp2 = HamiltonianCont_.Coordinates_.Nbasis(G2_ind1, G2_ind2, layer);

row_val_up = k1_ind + 
          k_sublattices[kSL_ind].size()*band1 +
          k_sublattices[kSL_ind].size()*Nbands*spin_up;
col_val_up = k2_ind + 
          k_sublattices[kSL_ind].size()*band2 +
          k_sublattices[kSL_ind].size()*Nbands*spin_up;

row_val_dn = k1_ind + 
          k_sublattices[kSL_ind].size()*band1 +
          k_sublattices[kSL_ind].size()*Nbands*spin_dn;
col_val_dn = k2_ind + 
          k_sublattices[kSL_ind].size()*band2 +
          k_sublattices[kSL_ind].size()*Nbands*spin_dn;

density_ += (1.0/Area)*( 
            exp(iota_complex*2.0*PI*(  (m1*(h1_1-h2_1)*1.0)/(M1*l1_*1.0)   +  (m2*(h1_2-h2_2)*1.0)/(M2*l2_*1.0)  ))
            *
            exp(iota_complex*2.0*PI*(  (m1*(G1_ind1-G2_ind1)*1.0)/(M1*1.0)   +  (m2*(G1_ind2-G2_ind2)*1.0)/(M2*1.0)  ))
            *
            (
            (conj(BlochStates[spin_up][band1][k1_ind_val][comp1])*BlochStates[spin_up][band2][k2_ind_val][comp2]*OParams[kSL_ind](row_val_up,col_val_up))  
            +
            (conj(BlochStates[spin_dn][band1][k1_ind_val][comp1])*BlochStates[spin_dn][band2][k2_ind_val][comp2]*OParams[kSL_ind](row_val_dn,col_val_dn)) 
            )
            );
Sz_ += (0.5/Area)*( 
            exp(iota_complex*2.0*PI*(  (m1*(h1_1-h2_1)*1.0)/(M1*l1_*1.0)   +  (m2*(h1_2-h2_2)*1.0)/(M2*l2_*1.0)  ))
            *
            exp(iota_complex*2.0*PI*(  (m1*(G1_ind1-G2_ind1)*1.0)/(M1*1.0)   +  (m2*(G1_ind2-G2_ind2)*1.0)/(M2*1.0)  ))
            *
            (
            (conj(BlochStates[spin_up][band1][k1_ind_val][comp1])*BlochStates[spin_up][band2][k2_ind_val][comp2]*OParams[kSL_ind](row_val_up,col_val_up))  
            -
            (conj(BlochStates[spin_dn][band1][k1_ind_val][comp1])*BlochStates[spin_dn][band2][k2_ind_val][comp2]*OParams[kSL_ind](row_val_dn,col_val_dn)) 
            )
            );

Sx_ += (0.5/Area)*( 
            exp(iota_complex*2.0*PI*(  (m1*(h1_1-h2_1)*1.0)/(M1*l1_*1.0)   +  (m2*(h1_2-h2_2)*1.0)/(M2*l2_*1.0)  ))
            *
            exp(iota_complex*2.0*PI*(  (m1*(G1_ind1-G2_ind1)*1.0)/(M1*1.0)   +  (m2*(G1_ind2-G2_ind2)*1.0)/(M2*1.0)  ))
            *
            (
            (conj(BlochStates[spin_up][band1][k1_ind_val][comp1])*BlochStates[spin_dn][band2][k2_ind_val][comp2]*OParams[kSL_ind](row_val_up,col_val_dn))  
            +
            (conj(BlochStates[spin_dn][band1][k1_ind_val][comp1])*BlochStates[spin_up][band2][k2_ind_val][comp2]*OParams[kSL_ind](row_val_dn,col_val_up)) 
            )
            );

Sy_ += ((-0.5*iota_complex)/Area)*( 
            exp(iota_complex*2.0*PI*(  (m1*(h1_1-h2_1)*1.0)/(M1*l1_*1.0)   +  (m2*(h1_2-h2_2)*1.0)/(M2*l2_*1.0)  ))
            *
            exp(iota_complex*2.0*PI*(  (m1*(G1_ind1-G2_ind1)*1.0)/(M1*1.0)   +  (m2*(G1_ind2-G2_ind2)*1.0)/(M2*1.0)  ))
            *
            (
            (conj(BlochStates[spin_up][band1][k1_ind_val][comp1])*BlochStates[spin_dn][band2][k2_ind_val][comp2]*OParams[kSL_ind](row_val_up,col_val_dn))  
            -
            (conj(BlochStates[spin_dn][band1][k1_ind_val][comp1])*BlochStates[spin_up][band2][k2_ind_val][comp2]*OParams[kSL_ind](row_val_dn,col_val_up)) 
            )
            );


}}
}}
}}}


fileout<<m1<<"  "<<m2<<"  "<<rx<<"  "<<ry<<"  "<<density_.real()<<"  "<<Sz_.real()<<"  "<<Sx_.real()<<"  "<<Sy_.real()<<
"  "<<density_.imag()<<"  "<<Sz_.imag()<<"  "<<Sx_.imag()<<"  "<<Sy_.imag()<<endl;

if(m1%M1==0  && m2%M2==0){
fileout2<<m1/M1<<"  "<<m2/M2<<"  "<<m1<<"  "<<m2<<"  "<<rx<<"  "<<ry<<"  "<<Area*density_.real()<<"  "<<Area*Sz_.real()<<"  "<<Area*Sx_.real()<<"  "<<Area*Sy_.real()<<
"  "<<Area*density_.imag()<<"  "<<Area*Sz_.imag()<<"  "<<Area*Sx_.imag()<<"  "<<Area*Sy_.imag()<<endl;
}
}

fileout<<endl;
if( (m1%M1==0)){
    fileout2<<endl;
}

}

}


}


void Hamiltonian::Calculate_RealSpace_OParams(string filename, string filename2){

int M1, M2; //no. of slices of one moire unit cell
M1=10;M2=10;
int m1_max, m1_min, m2_max, m2_min; //range for full 2d lattice
m1_max = M1*l1_;m1_min=0;
m2_max = M2*l1_;m2_min=0;


double rx,ry;

Xmat.resize(k_sublattices.size());
Omat.resize(k_sublattices.size());
for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
Xmat[kSL_ind].resize(k_sublattices[kSL_ind].size());
Omat[kSL_ind].resize(k_sublattices[kSL_ind].size()); 
for(int k1_ind=0;k1_ind<k_sublattices[kSL_ind].size();k1_ind++){
Xmat[kSL_ind][k1_ind].resize(k_sublattices[kSL_ind].size());
Omat[kSL_ind][k1_ind].resize(k_sublattices[kSL_ind].size());
for(int k2_ind=0;k2_ind<k_sublattices[kSL_ind].size();k2_ind++){
Xmat[kSL_ind][k1_ind][k2_ind].resize(2);
Omat[kSL_ind][k1_ind][k2_ind].resize(2);
for(int spin=0;spin<2;spin++){
Xmat[kSL_ind][k1_ind][k2_ind][spin].resize(2);
Omat[kSL_ind][k1_ind][k2_ind][spin].resize(2);
for(int spin_p=0;spin_p<2;spin_p++){
Xmat[kSL_ind][k1_ind][k2_ind][spin][spin_p].resize(2);
Omat[kSL_ind][k1_ind][k2_ind][spin][spin_p].resize(2);
for(int layer1=0;layer1<2;layer1++){
Xmat[kSL_ind][k1_ind][k2_ind][spin][spin_p][layer1].resize(2);
Omat[kSL_ind][k1_ind][k2_ind][spin][spin_p][layer1].resize(2);
for(int layer2=0;layer2<2;layer2++){
Xmat[kSL_ind][k1_ind][k2_ind][spin][spin_p][layer1][layer2].resize(Nbands);
for(int band1=0;band1<Nbands;band1++){
Xmat[kSL_ind][k1_ind][k2_ind][spin][spin_p][layer1][layer2][band1].resize(Nbands);
}}}}}}}}





int comp1, comp2, G_ind1, G_ind2;
int G2_ind1, G2_ind2;
int k1_ind_val, k2_ind_val;
for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
for(int k1_ind=0;k1_ind<k_sublattices[kSL_ind].size();k1_ind++){
k1_ind_val = k_sublattices[kSL_ind][k1_ind];
for(int k2_ind=0;k2_ind<k_sublattices[kSL_ind].size();k2_ind++){
k2_ind_val = k_sublattices[kSL_ind][k2_ind];
for(int spin=0;spin<2;spin++){
for(int spin_p=0;spin_p<2;spin_p++){
for(int layer1=0;layer1<2;layer1++){
for(int layer2=0;layer2<2;layer2++){
for(int band1=0;band1<Nbands;band1++){
for(int band2=0;band2<Nbands;band2++){
Xmat[kSL_ind][k1_ind][k2_ind][spin][spin_p][layer1][layer2][band1][band2]=0.0;
for(int G_ind=0;G_ind<G_grid_L1*G_grid_L2;G_ind++){
    for(int G2_ind=0;G2_ind<G_grid_L1*G_grid_L2;G2_ind++){
    G_ind1 = HamiltonianCont_.Coordinates_.indx_cellwise(G_ind);
    G_ind2 = HamiltonianCont_.Coordinates_.indy_cellwise(G_ind);
    G2_ind1 = HamiltonianCont_.Coordinates_.indx_cellwise(G2_ind);
    G2_ind2 = HamiltonianCont_.Coordinates_.indy_cellwise(G2_ind);
    comp1 = HamiltonianCont_.Coordinates_.Nbasis(G_ind1, G_ind2, layer1);
    comp2 = HamiltonianCont_.Coordinates_.Nbasis(G2_ind1, G2_ind2, layer2);
    Xmat[kSL_ind][k1_ind][k2_ind][spin][spin_p][layer1][layer2][band1][band2] +=
    conj(BlochStates[spin][band1][k1_ind_val][comp1])*
    BlochStates[spin_p][band2][k2_ind_val][comp2];
}}

//Xmat[kSL_ind][k1_ind][k2_ind][spin][spin_p][layer1][layer2][band1][band2]=1.0;
}}}}}}}}}

int col_val, row_val;
for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
for(int k1_ind=0;k1_ind<k_sublattices[kSL_ind].size();k1_ind++){
for(int k2_ind=0;k2_ind<k_sublattices[kSL_ind].size();k2_ind++){
for(int spin=0;spin<2;spin++){
for(int spin_p=0;spin_p<2;spin_p++){
for(int layer1=0;layer1<2;layer1++){
for(int layer2=0;layer2<2;layer2++){
Omat[kSL_ind][k1_ind][k2_ind][spin][spin_p][layer1][layer2]=0.0;
for(int band1=0;band1<Nbands;band1++){
for(int band2=0;band2<Nbands;band2++){

row_val = k1_ind + 
          k_sublattices[kSL_ind].size()*band1 +
          k_sublattices[kSL_ind].size()*Nbands*spin;

col_val = k2_ind + 
          k_sublattices[kSL_ind].size()*band2 +
          k_sublattices[kSL_ind].size()*Nbands*spin_p;

Omat[kSL_ind][k1_ind][k2_ind][spin][spin_p][layer1][layer2] +=
Xmat[kSL_ind][k1_ind][k2_ind][spin][spin_p][layer1][layer2][band1][band2]*OParams[kSL_ind](row_val,col_val) ;
}}}}}}}}}




int spin_up=0;
int spin_dn=1;
int h1_1, h1_2, h2_1, h2_2;
int l1, l2;

string filename_new, filename2_new;
for(int layer=0;layer<2;layer++){
filename_new = "layer_"+to_string(layer)+"_"+filename;
filename2_new = "layer_"+to_string(layer)+"_"+filename2;
ofstream fileout(filename_new.c_str());
ofstream fileout2(filename2_new.c_str());
fileout<<"# m1  m2  rx  ry   density   sz  sx  sy"<<endl;
fileout2<<"#cell1  cell2   m1  m2  rx  ry   density   sz  sx  sy"<<endl;

l1=layer;
l2=layer;
complex<double> density_, Sz_, Sx_, Sy_;
for(int m1=m1_min;m1<=m1_max;m1++){
for(int m2=m2_min;m2<=m2_max;m2++){

rx = (  ((m1*1.0)/(M1*1.0)) + ((m2*1.0)/(M2*1.0)) )*sqrt(3.0)*0.5;
ry = (  -((m1*1.0)/(M1*1.0)) + ((m2*1.0)/(M2*1.0)) )*0.5;

density_ =0.0;
Sz_=0.0;Sx_=0.0;Sy_=0.0;
for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
for(int k1_ind=0;k1_ind<k_sublattices[kSL_ind].size();k1_ind++){
for(int k2_ind=0;k2_ind<k_sublattices[kSL_ind].size();k2_ind++){
k1_ind_val = k_sublattices[kSL_ind][k1_ind];
k2_ind_val = k_sublattices[kSL_ind][k2_ind];
h1_1 = Coordinates_.indx_cellwise(k1_ind_val);
h1_2 = Coordinates_.indy_cellwise(k1_ind_val);
h2_1 = Coordinates_.indx_cellwise(k2_ind_val);
h2_2 = Coordinates_.indy_cellwise(k2_ind_val);

density_ += (1.0/Area)*( 
            exp(iota_complex*2.0*PI*(  (m1*(h1_1-h2_1)*1.0)/(M1*l1_*1.0)   +  (m2*(h1_2-h2_2)*1.0)/(M2*l2_*1.0)  ))
            *(
            Omat[kSL_ind][k1_ind][k2_ind][spin_up][spin_up][l1][l2] +
            Omat[kSL_ind][k1_ind][k2_ind][spin_dn][spin_dn][l1][l2]
            )  
            );
Sz_ += (0.5/Area)*( 
            exp(iota_complex*2.0*PI*(  (m1*(h1_1-h2_1)*1.0)/(M1*l1_*1.0)   +  (m2*(h1_2-h2_2)*1.0)/(M2*l2_*1.0)  ))*
            (
            Omat[kSL_ind][k1_ind][k2_ind][spin_up][spin_up][l1][l2] -
            Omat[kSL_ind][k1_ind][k2_ind][spin_dn][spin_dn][l1][l2]
            )  
            );

Sx_ += (0.5/Area)*( 
            exp(iota_complex*2.0*PI*(  (m1*(h1_1-h2_1)*1.0)/(M1*l1_*1.0)   +  (m2*(h1_2-h2_2)*1.0)/(M2*l2_*1.0)  ))*
            (
            Omat[kSL_ind][k1_ind][k2_ind][spin_up][spin_dn][l1][l2] +
            Omat[kSL_ind][k1_ind][k2_ind][spin_dn][spin_up][l1][l2]
            )  
            );

Sy_ += ((-0.5*iota_complex)/Area)*( 
            exp(iota_complex*2.0*PI*(  (m1*(h1_1-h2_1)*1.0)/(M1*l1_*1.0)   +  (m2*(h1_2-h2_2)*1.0)/(M2*l2_*1.0)  ))*
            (
            Omat[kSL_ind][k1_ind][k2_ind][spin_up][spin_dn][l1][l2] -
            Omat[kSL_ind][k1_ind][k2_ind][spin_dn][spin_up][l1][l2]
            )  
            );

}}}


fileout<<m1<<"  "<<m2<<"  "<<rx<<"  "<<ry<<"  "<<density_.real()<<"  "<<Sz_.real()<<"  "<<Sx_.real()<<"  "<<Sy_.real()<<
"  "<<density_.imag()<<"  "<<Sz_.imag()<<"  "<<Sx_.imag()<<"  "<<Sy_.imag()<<endl;

if(m1%M1==0  && m2%M2==0){
fileout2<<m1/M1<<"  "<<m2/M2<<"  "<<m1<<"  "<<m2<<"  "<<rx<<"  "<<ry<<"  "<<Area*density_.real()<<"  "<<Area*Sz_.real()<<"  "<<Area*Sx_.real()<<"  "<<Area*Sy_.real()<<
"  "<<Area*density_.imag()<<"  "<<Area*Sz_.imag()<<"  "<<Area*Sx_.imag()<<"  "<<Area*Sy_.imag()<<endl;
}
}

fileout<<endl;
if( (m1%M1==0)){
    fileout2<<endl;
}

}

}


}

double Hamiltonian::Lorentzian(double eta, double x){
double val;
    val = (1.0/PI)*( (eta/1.0) / (  (x*x) + ((eta*eta)/1.0)   ) );
    return val;

}

void Hamiltonian::Print_SPDOS(string filename){

double dw=0.01;
double eta=0.01;
double w_min = EigVal_min-1.0;
double w_max = EigVal_max+1.0;
double dos_;

ofstream file_out(filename.c_str());
file_out<<"#omega  DOS(omega)"<<endl;

double w_val=w_min;
while(w_val<=w_max){
dos_=0.0;
for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
    for(int i=0;i<EigValues[kSL_ind].size();i++){
    dos_ += Lorentzian(eta, w_val - EigValues[kSL_ind][i]);
    }
}

file_out<<w_val<< "   "<<dos_<<endl;
w_val +=dw;
}

}


double Hamiltonian::DispersionTriangularLattice(int k_ind){


double val;
int i1_, i2_;
i1_=Coordinates_.indx_cellwise(k_ind);
i2_=Coordinates_.indy_cellwise(k_ind);
val = -2.0*(
           cos(2*PI* (  ((1.0*i2_)/(1.0*l2_))-((1.0*i1_)/(1.0*l1_)) ) ) +
           cos(2*PI* ( ((1.0*i1_)/(1.0*l1_)) ) ) +
           cos(2*PI* ( ((1.0*i2_)/(1.0*l2_)) ) )
           );

// cout<<"Ek : "<<k_ind<<"  "<<i1_<<"  "<<i2_<<"  "<<val<<endl;

//cout<<"Ek : "<<k_ind<<"  "<<Coordinates_.indx_cellwise(k_ind)<<"  "<<Coordinates_.indy_cellwise(k_ind)<<"  "<<val<<endl;
return val;
}

void Hamiltonian::Calculate_Total_Spin(){

complex<double> S_plus_total;
int spin_up=0;
int spin_dn=1;
int row_val, col_val;
Total_Sz = ns_*0.5*(Total_n_up - Total_n_dn);

S_plus_total=0.0;
for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
for(int k_ind=0;k_ind<k_sublattices[kSL_ind].size();k_ind++){
for(int band1=0;band1<Nbands;band1++){
row_val = k_ind + 
          k_sublattices[kSL_ind].size()*band1 +
          k_sublattices[kSL_ind].size()*Nbands*spin_up;
col_val = k_ind + 
          k_sublattices[kSL_ind].size()*band1 +
          k_sublattices[kSL_ind].size()*Nbands*spin_dn;

S_plus_total += OParams_new[kSL_ind](row_val,col_val);

}}
}

Total_Sx = 0.5*(S_plus_total + conj(S_plus_total)).real();
Total_Sy = -0.5*(iota_complex*(S_plus_total - conj(S_plus_total)) ).real();

cout<<"Total_Sz = "<<Total_Sz<<endl;
cout<<"Total_Sx = "<<Total_Sx<<endl;
cout<<"Total_Sy = "<<Total_Sy<<endl;
cout<<"|Total_S_vec| = "<<sqrt(Total_Sz*Total_Sz + Total_Sx*Total_Sx + Total_Sy*Total_Sy)<<endl;

}

void Hamiltonian::Update_Hartree_Coefficients(){

//Hartree_Coefficients;
    //(int k2_ind, int k3_ind, int band2, int band3, int spin_p)

    
    for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
    for(int k2_ind_=0;k2_ind_<k_sublattices[kSL_ind].size();k2_ind_++){
    for(int k3_ind_=0;k3_ind_<k_sublattices[kSL_ind].size();k3_ind_++){
    for(int band2=0;band2<Nbands;band2++){
    for(int band3=0;band3<Nbands;band3++){
        for(int spin_p=0;spin_p<2;spin_p++){
        HartreeCoefficients[kSL_ind][k2_ind_][k3_ind_][band2][band3][spin_p]=Hartree_coefficient_new(k_sublattices[kSL_ind][k2_ind_], k_sublattices[kSL_ind][k3_ind_], band2, band3, spin_p);
        }
    }
    }
    }
    }
    }



}


void Hamiltonian::Update_Fock_Coefficients(){

//Hartree_Coefficients;
    //(int k2_ind, int k3_ind, int band2, int band3, int spin_p)

    
    for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
    for(int k2_ind_=0;k2_ind_<k_sublattices[kSL_ind].size();k2_ind_++){
    for(int k3_ind_=0;k3_ind_<k_sublattices[kSL_ind].size();k3_ind_++){
    for(int band2=0;band2<Nbands;band2++){
    for(int band4=0;band4<Nbands;band4++){
        for(int spin=0;spin<2;spin++){
        for(int spin_p=0;spin_p<2;spin_p++){
        FockCoefficients[kSL_ind][k2_ind_][k3_ind_][band2][band4][spin][spin_p]=Fock_coefficient_new(k_sublattices[kSL_ind][k2_ind_], k_sublattices[kSL_ind][k3_ind_], band2, band4, spin, spin_p);
                
        }
        }
    }
    }
    }
    }
    }



}


double Hamiltonian::Myrandom(){

    return dis_(Generator_);

}


void Hamiltonian::Get_max_and_min_eigvals(){

    double val_max=-10000;
    double val_min=10000;
    for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
        for(int i=0;i<EigValues[kSL_ind].size();i++){
            if(EigValues[kSL_ind][i]>val_max){
                    val_max=EigValues[kSL_ind][i];
            }
            if(EigValues[kSL_ind][i]<val_min){
                    val_min=EigValues[kSL_ind][i];
            }

        }
    }

    EigVal_max=val_max;
    EigVal_min=val_min;


}

double Hamiltonian::chemicalpotential(double muin,double Particles){


    double mu_out;
    Get_max_and_min_eigvals();

    if(!Parameters_.FixingMu)
    {
        double n1,N;
        double dMubydN;
        int nstate = eigs_.size();
        dMubydN = 0.0005*(EigVal_max - EigVal_min)/nstate;
        N=Particles;
        //temp=Parameters_.temp;
        mu_out = muin;
        bool converged=false;
        int final_i;


        if(1==2){
            for(int i=0;i<50000;i++){
                n1=0.0;
                for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
        for(int j=0;j<EigValues[kSL_ind].size();j++){
                    n1+=double(1.0/( exp( (EigValues[kSL_ind][j]-mu_out)*beta_) + 1.0));
                }}
                //cout <<"i  "<< i << "  n1  " << n1 << "  mu  " << mu_out<< endl;
                if(abs(N-n1)<double(0.000001)){
                    //cout<<abs(N-n1)<<endl;
                    converged=true;
                    final_i=i;
                    break;
                }
                else {
                    mu_out += (N-n1)*dMubydN;
                    //cout<<i<<"    "<<n1<<"    "<<N-n1<<endl;

                }
            }

            if(!converged){
                cout<<"mu_not_converged, N = "<<n1<<endl;
            }
            else{
                //cout<<"mu converged, N = "<<n1<<" in "<<final_i<<" iters"<<endl;
            }

        }


        double mu1, mu2;
        double mu_temp = muin;
        //cout<<"mu_input = "<<mu_temp<<endl;
        if(1==1){
            mu1=EigVal_min - (5.0/beta_);
            mu2=EigVal_max + (5.0/beta_);
            for(int i=0;i<40000;i++){
                n1=0.0;
               for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
        for(int j=0;j<EigValues[kSL_ind].size();j++){
                    n1+=double(1.0/( exp( (EigValues[kSL_ind][j]-mu_temp)*beta_) + 1.0));
                }}
                //cout <<"i  "<< i << "  n1  " << n1 << "  mu  " << mu_out<< endl;
                if(abs(N-n1)<double(0.00001)){
                    //cout<<abs(N-n1)<<endl;
                    converged=true;
                    break;
                }
                else {
                    if(n1 >N){
                        mu2=mu_temp;
                        mu_temp=0.5*(mu1 + mu_temp);
                    }
                    else{
                        mu1=mu_temp;
                        mu_temp=0.5*(mu2 + mu_temp);
                    }

                }
                //cout<<"mu_temp = "<<mu_temp<<"   "<<mu1<<"    "<<mu2<<"   "<<eigs_[nstate-1]<<"  "<<eigs_[0]<<"  "<<n1<<endl;
            }

            if(!converged){
                cout<<"mu_not_converged, N = "<<n1<<endl;
            }
            else{
                //cout<<"mu converged, N = "<<n1<<endl;
            }

            mu_out = mu_temp;
        }

    }

    else{
        mu_out = Parameters_.MuValueFixed;
    }

    return mu_out;
} // ----------





double Hamiltonian::V_int(double q_val){

    double val;
    if(q_val>0.0001){
    val = (2.0*PI*10000000.0*tanh(q_val*d_gate))/(4.0*PI*Parameters_.eps_DE*55.263494*q_val);
    }
    else{
    val = (2.0*PI*10000000.0*d_gate)/(4.0*PI*Parameters_.eps_DE*55.263494);
    }

    return val; //this is in meV X (Angstorm)^2
}


void Hamiltonian::Print_Vint(){

string file_Int="V_int.txt";
 ofstream fl_Int_out(file_Int.c_str());

 double q_val=0;
 while(q_val<50){
   fl_Int_out<<q_val<<"  "<<V_int(q_val)<<endl;
   q_val+=0.00001;
 }

}

void Hamiltonian::Print_Interaction_value(){

for(int spin1=0;spin1<2;spin1++){
for(int spin2=0;spin2<2;spin2++){
    
    for(int band1=0;band1<Nbands;band1++){
    for(int band2=0;band2<Nbands;band2++){
    for(int band3=0;band3<Nbands;band3++){
    for(int band4=0;band4<Nbands;band4++){

        for(int q_ind=0;q_ind<1;q_ind++){ //ns_

string file_Int="Interation_spins"+to_string(spin1)+"_"+to_string(spin2)+"_bands_"+to_string(band1)+"_"+
                to_string(band2)+"_"+to_string(band3)+"_"+to_string(band4)+"_qind_"+to_string(q_ind)+
                ".txt";
 ofstream fl_Int_out(file_Int.c_str());
 fl_Int_out<<"#k1(=kx+ky*l1_)   k2    Interaction.real()  Interaction.imag()"<<endl;


        for(int k1_ind=0;k1_ind<ns_;k1_ind++){
        for(int k2_ind=0;k2_ind<ns_;k2_ind++){
        
            fl_Int_out<<k1_ind<<"  "<<k2_ind<<"  "<<Interaction_value(spin1, spin2, band1, band2, band3, band4,  k1_ind, k2_ind, q_ind).real()<<"  "<<Interaction_value(spin1, spin2, band1, band2, band3, band4,  k1_ind, k2_ind, q_ind).imag()<<endl;


        }
        fl_Int_out<<endl;
        }
        
        }
    }
    }
    }
    }


}   
}


}

void Hamiltonian::Print_Interaction_value2(int k1_ind, int k2_ind){

int q_ind;
for(int spin1=0;spin1<2;spin1++){
for(int spin2=0;spin2<2;spin2++){
    
    for(int band1=0;band1<Nbands;band1++){
    for(int band2=0;band2<Nbands;band2++){
    for(int band3=0;band3<Nbands;band3++){
    for(int band4=0;band4<Nbands;band4++){

       

string file_Int="Interation_spins"+to_string(spin1)+"_"+to_string(spin2)+"_bands_"+to_string(band1)+"_"+
                to_string(band2)+"_"+to_string(band3)+"_"+to_string(band4)+"_k1_ind_"+to_string(k1_ind)+
                "_k2_ind_"+to_string(k2_ind)+
                ".txt";
 ofstream fl_Int_out(file_Int.c_str());
 fl_Int_out<<"#q1(q=q1+q2*l1_)   q2    Interaction.real()  Interaction.imag()"<<endl;


        for(int q1_ind=0;q1_ind<l1_;q1_ind++){
        for(int q2_ind=0;q2_ind<l2_;q2_ind++){
        q_ind = q1_ind + l1_*q2_ind;
            fl_Int_out<<q1_ind<<"  "<<q2_ind<<"  "<<Interaction_value(spin1, spin2, band1, band2, band3, band4,  k1_ind, k2_ind, q_ind).real()<<"  "<<Interaction_value(spin1, spin2, band1, band2, band3, band4,  k1_ind, k2_ind, q_ind).imag()<<endl;
        }
        fl_Int_out<<endl;
        }
        
        
    }
    }
    }
    }


}   
}


}


void Hamiltonian::Save_InteractionVal(){

    Interaction_val.resize(2);
    for(int spin=0;spin<2;spin++){
    Interaction_val[spin].resize(2);
    for(int spin_p=0;spin_p<2;spin_p++){
    Interaction_val[spin][spin_p].resize(Nbands);
    for(int band1=0;band1<Nbands;band1++){
    Interaction_val[spin][spin_p][band1].resize(Nbands);
    for(int band2=0;band2<Nbands;band2++){
    Interaction_val[spin][spin_p][band1][band2].resize(Nbands);
    for(int band3=0;band3<Nbands;band3++){
    Interaction_val[spin][spin_p][band1][band2][band3].resize(Nbands);
    for(int band4=0;band4<Nbands;band4++){
    Interaction_val[spin][spin_p][band1][band2][band3][band4].resize(ns_);
    for(int k1=0;k1<ns_;k1++){
    Interaction_val[spin][spin_p][band1][band2][band3][band4][k1].resize(ns_);
    for(int k2=0;k2<ns_;k2++){
    Interaction_val[spin][spin_p][band1][band2][band3][band4][k1][k2].resize(ns_);
    for(int q=0;q<ns_;q++){
    Interaction_val[spin][spin_p][band1][band2][band3][band4][k1][k2][q] = 
    Interaction_value(spin,spin_p,band1,band2,band3,band4,k1,k2,q);
    }}}
    }}}}
    }}

}

complex<double> Hamiltonian::Interaction_value(int spin, int spin_p, int band1, int band2, int band3, int band4, int k1_ind, int k2_ind, int q_ind){

int q_ind1, q_ind2, k1_ind1, k1_ind2, k2_ind1, k2_ind2;
q_ind1=Coordinates_.indx_cellwise(q_ind);q_ind2=Coordinates_.indy_cellwise(q_ind);
k1_ind1=Coordinates_.indx_cellwise(k1_ind);k1_ind2=Coordinates_.indy_cellwise(k1_ind);
k2_ind1=Coordinates_.indx_cellwise(k2_ind);k2_ind2=Coordinates_.indy_cellwise(k2_ind);
double qpGx_temp, qpGy_temp;
complex<double> val;
val=0.0;
for(int g_ind1=-G_grid_L1/2;g_ind1<=G_grid_L1/2;g_ind1++){
for(int g_ind2=-G_grid_L2/2;g_ind2<=G_grid_L2/2;g_ind2++){

//if(g_ind1!=0 && g_ind2!=0){
//  for(int g_ind1=0;g_ind1<=0;g_ind1++){
//  for(int g_ind2=0;g_ind2<=0;g_ind2++){

//kx_=(2.0*PI/Parameters_.a_moire)*((n1)*(1.0/(sqrt(3)*L1_))  +  (n2)*(1.0/(sqrt(3)*L2_)));
//ky_=(2.0*PI/Parameters_.a_moire)*((n1)*(-1.0/(L1_))  +  (n2)*(1.0/(L2_)));
qpGx_temp = (2.0*PI/Parameters_.a_moire)*(
            (q_ind1)*(1.0/(sqrt(3)*l1_))  +  (q_ind2)*(1.0/(sqrt(3)*l2_)) //q
            + g_ind1*(1.0/sqrt(3)) + g_ind2*(1.0/sqrt(3)) //G
            );

qpGy_temp = (2.0*PI/Parameters_.a_moire)*(
            (q_ind1)*(-1.0/l1_)  +  (q_ind2)*(1.0/l2_) //q
            + g_ind1*(-1.0) + g_ind2*(1.0) //G
            );



val += V_int(sqrt( (qpGx_temp*qpGx_temp) + (qpGy_temp*qpGy_temp) ) ) *
      FormFactor(spin,band1,band4, k1_ind1-q_ind1, k1_ind2-q_ind2, k1_ind1 + (g_ind1*l1_), k1_ind2 + (g_ind2*l2_))*
      conj(FormFactor(spin_p,band3,band2, k2_ind1, k2_ind2, k2_ind1+q_ind1+(g_ind1*l1_), k2_ind2+q_ind2+(g_ind2*l2_)) );

//val += V_int(sqrt( (qpGx_temp*qpGx_temp) + (qpGy_temp*qpGy_temp) ) );


//FormFactor(int spin, int band1, int band2, int k1_vec_ind1, int k1_vec_ind2, int k2_vec_ind1, int k2_vec_ind2)


//}

 //}}

}}



//Overwriting val, remove later
//n_up X n_dn ; Only Hubbard term
// if(spin==spin_p){
//     val=0;
// }
// else{
//     val=1.0/Parameters_.eps_DE;
// }

//val=10000;
return val;
}



bool Hamiltonian::AreSitesRelated(int k2,int k1){

    bool temp_check;
    int k1_1=Coordinates_.indx_cellwise(k1);
    int k1_2=Coordinates_.indy_cellwise(k1);

    int k2_1=Coordinates_.indx_cellwise(k2);
    int k2_2=Coordinates_.indy_cellwise(k2);

    double n2_, n1_;
    n2_ = (((k2_1 - k1_1)*1.0*NMat_MUC(1,0))/l1_) + (((k2_2 - k1_2)*1.0*NMat_MUC(1,1))/l2_) ;
    n1_ = (((k2_1 - k1_1)*1.0*NMat_MUC(0,0))/l1_) + (((k2_2 - k1_2)*1.0*NMat_MUC(0,1))/l2_) ;

    if( ( abs((floor(n2_+0.5)) - n2_)<0.00001 ) &&
        ( abs((floor(n1_+0.5)) - n1_)<0.00001 ) 
         ){
            temp_check=true;
         }
         else{
            temp_check=false;
         }


    //  if(k1==8 || k1==7){
    //      cout<<"check : "<<k1<<"("<<k1_1<<","<<k1_2<<")  "<<k2<<"("<<k2_1<<","<<k2_2<<") : "<<n1_<<"  "<<n2_<<endl;
    //  }

    return temp_check;

}

void Hamiltonian::Print_k_sublattices(){

    cout<<"XXXXXXXXXXXXXX k sublattices XXXXXXXXXXXXXXXX"<<endl;
    for(int i=0;i<k_sublattices.size();i++){
        cout <<i<<" : ";
        for(int j=0;j<k_sublattices[i].size();j++){
            cout<<k_sublattices[i][j]<<" ";
        }
        cout<<endl;
    }
    cout<<"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"<<endl;

}

void Hamiltonian::Create_k_sublattices(){

    Mat_1_int temp_list;
    
    bool listed_;

    temp_list.push_back(0);
    k_sublattices.push_back(temp_list);
    for(int i=1;i<ns_;i++){
        
            listed_=false;
            for(int list_no=0;list_no<k_sublattices.size();list_no++){
                if(AreSitesRelated(i,k_sublattices[list_no][0])){
                    k_sublattices[list_no].push_back(i);
                    listed_=true;
                    break;
                }
            }

            if(!listed_){
                temp_list.clear();
                temp_list.push_back(i);
                k_sublattices.push_back(temp_list);
            }
        
    }


    Inverse_kSublattice_mapping.resize(ns_);
    for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
        for(int temp_ind=0;temp_ind<k_sublattices[kSL_ind].size();temp_ind++){
          Inverse_kSublattice_mapping[k_sublattices[kSL_ind][temp_ind]].first=kSL_ind;  
          Inverse_kSublattice_mapping[k_sublattices[kSL_ind][temp_ind]].second=temp_ind;
        }
    }

}


void Hamiltonian::Initialize(){

    //lowest Nbands no of bands used for Hartree-Fock in hole language
    Nbands = Parameters_.N_bands_HF;

    //number of k points moire brillioun zone
    // which is = number of moire unit cells in real space along direction a1 and a2
    l1_= Parameters_.moire_BZ_L1;
    l2_= Parameters_.moire_BZ_L2;
    ns_ = l1_*l2_; // total number of moire unit cells

    Area = l1_*l2_*sqrt(3.0)*0.5*Parameters_.a_moire*Parameters_.a_moire; //in Angstorm^2
    d_gate = Parameters_.d_gate;
 

    //From input file later on
    mu_=0.0;
    Temperature=Parameters_.Temperature; //in Kelvin
    nu_holes_target=Parameters_.holesdensity_per_moire_unit_cell;  //per moire unit cell
    HF_max_iterations=Parameters_.Max_HFIterations;
    HF_convergence_error=Parameters_.HF_convergence_error;
    alpha_mixing=Parameters_.SimpleMixing_alpha;
    Convergence_technique="SimpleMixing";



    KB_=0.08617332; //in meV/K
    beta_=1.0/(KB_*Temperature); //in meV^-1

    NMat_MUC = Parameters_.NMat_MUC;
    NMat_det = Parameters_.NMat_det;


    Create_k_sublattices();
    Print_k_sublattices();
    //assert(false);

    //eigs_.resize(k_sublattices.size());
    EigValues.resize(k_sublattices.size());
    EigVectors.resize(k_sublattices.size());

    OParams.resize(k_sublattices.size());
    OParams_new.resize(k_sublattices.size());

    
    // OParams.resize(ns_*2*Nbands);
    // OParams_new.resize(ns_*2*Nbands);
    // for(int i=0;i<OParams.size();i++){
    //     Oparams[i].resize(ns_2*Nbands);
    //     Oparams_new[i].resize(ns_2*Nbands);
    //     for(int j=0;j<OParams[i].size();j++){
    //         OParams[i][j] = 0.0;
    //         OParams_new[i][j] = 0.0;
    //     }
    // }



    Lambda_.resize(2);
    for(int spin=0;spin<2;spin++){
        Lambda_[spin].resize(Nbands);
        for(int n1=0;n1<Nbands;n1++){
            Lambda_[spin][n1].resize(Nbands);
            for(int n2=0;n2<Nbands;n2++){
                Lambda_[spin][n1][n2].resize(ns_);
              for(int k1=0;k1<ns_;k1++){
                  Lambda_[spin][n1][n2][k1].resize(ns_);  
              }  
            }
        }
    }

    
    G_grid_L1=Parameters_.Grid_moireRL_L1;
    G_grid_L2=Parameters_.Grid_moireRL_L2;
     BlochStates.resize(2);
        for(int spin=0;spin<2;spin++){
          BlochStates[spin].resize(Nbands);
          for(int n=0;n<Nbands;n++){
            BlochStates[spin][n].resize(ns_);
            for(int i=0;i<ns_;i++){  //k_ind
               BlochStates[spin][n][i].resize(G_grid_L1*G_grid_L2*2); //G_ind*layer
            }
          }
        }

    BlochEigvals.resize(2);
        for(int spin=0;spin<2;spin++){
        BlochEigvals[spin].resize(Nbands);
            for(int n=0;n<Nbands;n++){
                BlochEigvals[spin][n].resize(ns_);
            }
        }

    

   assert(HamiltonianCont_.mbz_factor==1);
   int comp_norm;
   comp_norm=HamiltonianCont_.Coordinates_.Nbasis((G_grid_L1/2), (G_grid_L2/2), 0);
   //comp_norm=HamiltonianCont_.Coordinates_.Nbasis(0, 0, 0);
   //comp_norm=1.0; 
    complex<double> phase_;
    for(int spin=0;spin<2;spin++){
          for(int n=0;n<Nbands;n++){
            for(int i1=0;i1<l1_;i1++){  //k_ind
            for(int i2=0;i2<l2_;i2++){
                
                phase_= conj(HamiltonianCont_.BlochStates[spin][n][i1+i2*(l1_+1)][comp_norm])/
                        (abs(HamiltonianCont_.BlochStates[spin][n][i1+i2*(l1_+1)][comp_norm]));
              for(int comp=0;comp<G_grid_L1*G_grid_L2*2;comp++){
                BlochStates[spin][n][i1+i2*l1_][comp]=phase_*HamiltonianCont_.BlochStates[spin][n][i1+i2*(l1_+1)][comp];
              }
            }
            }
          }
        }
    BlochStates_old_ = HamiltonianCont_.BlochStates;

    for(int spin=0;spin<2;spin++){
        for(int n=0;n<Nbands;n++){
        for(int i1=0;i1<l1_;i1++){  //k_ind
        for(int i2=0;i2<l2_;i2++){
            BlochEigvals[spin][n][i1+i2*l1_]=HamiltonianCont_.eigvals[spin][n][i1+i2*(l1_+1)];
        }}
        }
    }


    
//cout<<"Here -22"<<endl;
    HartreeCoefficients.resize(k_sublattices.size());
    for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
    HartreeCoefficients[kSL_ind].resize(k_sublattices[kSL_ind].size());
for(int k2_ind=0;k2_ind<k_sublattices[kSL_ind].size();k2_ind++){
    HartreeCoefficients[kSL_ind][k2_ind].resize(k_sublattices[kSL_ind].size());
    for(int k3_ind=0;k3_ind<k_sublattices[kSL_ind].size();k3_ind++){
    HartreeCoefficients[kSL_ind][k2_ind][k3_ind].resize(Nbands);
    for(int band2=0;band2<Nbands;band2++){
    HartreeCoefficients[kSL_ind][k2_ind][k3_ind][band2].resize(Nbands);
    for(int band3=0;band3<Nbands;band3++){
    HartreeCoefficients[kSL_ind][k2_ind][k3_ind][band2][band3].resize(2);
    }
    }
    }
    }
    }
//cout<<"Here -21"<<endl;


    FockCoefficients.resize(k_sublattices.size());
    for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
    FockCoefficients[kSL_ind].resize(k_sublattices[kSL_ind].size());
for(int k2_ind=0;k2_ind<k_sublattices[kSL_ind].size();k2_ind++){
    FockCoefficients[kSL_ind][k2_ind].resize(k_sublattices[kSL_ind].size());
    for(int k3_ind=0;k3_ind<k_sublattices[kSL_ind].size();k3_ind++){
    FockCoefficients[kSL_ind][k2_ind][k3_ind].resize(Nbands);
    for(int band2=0;band2<Nbands;band2++){
    FockCoefficients[kSL_ind][k2_ind][k3_ind][band2].resize(Nbands);
    for(int band3=0;band3<Nbands;band3++){
    FockCoefficients[kSL_ind][k2_ind][k3_ind][band2][band3].resize(2);
    for(int spin=0;spin<2;spin++){
    FockCoefficients[kSL_ind][k2_ind][k3_ind][band2][band3][spin].resize(2);    
    }
    }
    }
    }
    }
    }

    //cout<<"Here -20"<<endl;

    //assert(k_sublattices[0].size() == NMat_det);
    Amat.resize(ns_);
    Bmat.resize(ns_);
    for(int k1=0;k1<ns_;k1++){
        Amat[k1].resize(ns_);
        Bmat[k1].resize(ns_);
        for(int k2=0;k2<ns_;k2++){
            Amat[k1][k2].resize(k_sublattices[0].size());
            Bmat[k1][k2].resize(k_sublattices[0].size());
            for(int q_ind=0;q_ind<k_sublattices[0].size();q_ind++){
                Amat[k1][k2][q_ind].resize(Nbands);
                Bmat[k1][k2][q_ind].resize(Nbands);
                for(int n1=0;n1<Nbands;n1++){
                Amat[k1][k2][q_ind][n1].resize(Nbands);
                Bmat[k1][k2][q_ind][n1].resize(Nbands);
                for(int n2=0;n2<Nbands;n2++){
                Amat[k1][k2][q_ind][n1][n2].resize(Nbands);
                Bmat[k1][k2][q_ind][n1][n2].resize(Nbands);
                for(int n3=0;n3<Nbands;n3++){
                Amat[k1][k2][q_ind][n1][n2][n3].resize(Nbands);
                Bmat[k1][k2][q_ind][n1][n2][n3].resize(Nbands);
                for(int n4=0;n4<Nbands;n4++){
                Amat[k1][k2][q_ind][n1][n2][n3][n4].resize(2);
                Bmat[k1][k2][q_ind][n1][n2][n3][n4].resize(2);
                for(int spin=0;spin<2;spin++){
                Amat[k1][k2][q_ind][n1][n2][n3][n4][spin].resize(2);
                Bmat[k1][k2][q_ind][n1][n2][n3][n4][spin].resize(2);
                }
                }}}}
                }
            }
    }



    
    //(int spin, int spin_p, int band1, int band2, int band3, int band4, int k1_ind, int k2_ind, int q_ind)


    cout<<"Saving Interaction val"<<endl;
    Save_InteractionVal();
    cout<<"Interaction val completed"<<endl;

    cout<<"Started:  Creating Amat and Bmat"<<endl;
    Create_Amat_and_Bmat();
    cout<<"Completed: Creating Amat and Bmat"<<endl;

    //-----------
  

    N_layer_tau.resize(2);
    for(int i=0;i<2;i++){
        N_layer_tau[i].resize(2);
    }


    //----------

} // ----------


void Hamiltonian::Create_Amat_and_Bmat(){

int q_ind1, q_ind2, q_ind_val;
int minus_q_ind1, minus_q_ind2, minus_q_ind_val;
int G1_ind_temp, G2_ind_temp;

int k1_1, k1_2, k2_1, k2_2;

int q_ind1_temp, q_ind2_temp, k2_mk1_mq, k1_mk2_pq;
int q_ind1_new, q_ind2_new;
for(int k1=0;k1<ns_;k1++){
    k1_1=Coordinates_.indx_cellwise(k1);
    k1_2=Coordinates_.indy_cellwise(k1);
                
        for(int k2=0;k2<ns_;k2++){
            k2_1=Coordinates_.indx_cellwise(k2);
            k2_2=Coordinates_.indy_cellwise(k2);

            for(int q_ind=0;q_ind<k_sublattices[0].size();q_ind++){
                q_ind_val = k_sublattices[0][q_ind];
                q_ind1=Coordinates_.indx_cellwise(q_ind_val);
                q_ind2=Coordinates_.indy_cellwise(q_ind_val);
                
                Folding_to_BrillouinZone(-1*q_ind1, -1*q_ind2, minus_q_ind1, minus_q_ind2, G1_ind_temp, G2_ind_temp);
                minus_q_ind_val=minus_q_ind1 + minus_q_ind2*l1_;

                q_ind1_temp = k2_1 - k1_1 - q_ind1;
                q_ind2_temp = k2_2 - k1_2 - q_ind2;
                Folding_to_BrillouinZone(q_ind1_temp, q_ind2_temp, q_ind1_new, q_ind2_new, G1_ind_temp, G2_ind_temp);
                k2_mk1_mq = q_ind1_new + q_ind2_new*l1_;

                Folding_to_BrillouinZone(-1*q_ind1_temp, -1*q_ind2_temp, q_ind1_new, q_ind2_new, G1_ind_temp, G2_ind_temp);
                k1_mk2_pq = q_ind1_new + q_ind2_new*l1_;


                for(int n1=0;n1<Nbands;n1++){
                for(int n2=0;n2<Nbands;n2++){
                for(int n3=0;n3<Nbands;n3++){
                for(int n4=0;n4<Nbands;n4++){
                for(int spin=0;spin<2;spin++){
                for(int spin_p=0;spin_p<2;spin_p++){
                
                Amat[k1][k2][q_ind][n1][n2][n3][n4][spin][spin_p] = 
                Interaction_val[spin][spin_p][n1][n2][n3][n4][k1][k2][q_ind_val] + 
                Interaction_val[spin_p][spin][n2][n1][n4][n3][k2][k1][minus_q_ind_val];
                //Interaction_value(spin, spin_p, n1, n2, n3, n4, k1, k2, q_ind_val) + 
                //Interaction_value(spin_p, spin, n2, n1, n4, n3, k2, k1, minus_q_ind_val);
                Bmat[k1][k2][q_ind][n1][n2][n3][n4][spin][spin_p] = 
                Interaction_val[spin][spin_p][n1][n2][n3][n4][k2][k1][k2_mk1_mq] + 
                Interaction_val[spin_p][spin][n2][n1][n4][n3][k1][k2][k1_mk2_pq];
                //Interaction_value(spin, spin_p, n1, n2, n3, n4, k2, k1, k2_mk1_mq) + 
                //Interaction_value(spin_p, spin, n2, n1, n4, n3, k1, k2, k1_mk2_pq);

                }}
                }}}}
                }
            }
    }


}

complex<double> Hamiltonian::Hartree_coefficient(int k2_ind, int k3_ind, int band2, int band3, int spin_p){

complex<double> value;
int q_ind1, q_ind2, q_ind1_new, q_ind2_new;
int G_ind1_temp, G_ind2_temp;
int k3_ind1, k3_ind2, k2_ind1, k2_ind2;
int k1_ind, k1_ind1, k1_ind2;
int q1_ind, q2_ind, q3_ind;
int q3_kSL_internal_ind, q3_kSL_ind;

int OP_row, OP_col;
k3_ind1 = Coordinates_.indx_cellwise(k3_ind);
k3_ind2 = Coordinates_.indy_cellwise(k3_ind);
k2_ind1 = Coordinates_.indx_cellwise(k2_ind);
k2_ind2 = Coordinates_.indy_cellwise(k2_ind);

    q_ind1 = k3_ind1 - k2_ind1; 
    q_ind2 = k3_ind2 - k2_ind2;
    Folding_to_BrillouinZone(q_ind1, q_ind2, q_ind1_new, q_ind2_new, G_ind1_temp, G_ind2_temp);
    q1_ind = q_ind1_new + l1_*q_ind2_new;

    q_ind1 = -k3_ind1 + k2_ind1; 
    q_ind2 = -k3_ind2 + k2_ind2;
    Folding_to_BrillouinZone(q_ind1, q_ind2, q_ind1_new, q_ind2_new, G_ind1_temp, G_ind2_temp);
    q2_ind = q_ind1_new + l1_*q_ind2_new;

    
    

    value=0.0;
    for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
        for(int temp_ind=0;temp_ind<k_sublattices[kSL_ind].size();temp_ind++){
        k1_ind = k_sublattices[kSL_ind][temp_ind];
        k1_ind1 = Coordinates_.indx_cellwise(k1_ind);
        k1_ind2 = Coordinates_.indy_cellwise(k1_ind);
        q_ind1 = -k3_ind1 + k2_ind1 + k1_ind1; 
        q_ind2 = -k3_ind2 + k2_ind2 + k1_ind2;
        Folding_to_BrillouinZone(q_ind1, q_ind2, q_ind1_new, q_ind2_new, G_ind1_temp, G_ind2_temp);
        q3_ind = q_ind1_new + l1_*q_ind2_new; //k1-k3+k2 for OPs

        //q3_ind ---> sublattice, ind
        q3_kSL_ind = Inverse_kSublattice_mapping[q3_ind].first;
        q3_kSL_internal_ind = Inverse_kSublattice_mapping[q3_ind].second;

        assert(q3_kSL_ind == kSL_ind);

        for(int band1=0;band1<Nbands;band1++){
        for(int band4=0;band4<Nbands;band4++){
            for(int spin=0;spin<2;spin++){

                OP_row =  q3_kSL_internal_ind + 
                          k_sublattices[kSL_ind].size()*band1 +
                          k_sublattices[kSL_ind].size()*Nbands*spin;

                OP_col =  temp_ind + 
                          k_sublattices[kSL_ind].size()*band4 +
                          k_sublattices[kSL_ind].size()*Nbands*spin;

                value += (1.0/(2.0*Area)) * ( 
                        Interaction_value(spin, spin_p, band1, band2, band3, band4, k1_ind, k2_ind, q1_ind) +
                        Interaction_value(spin_p, spin, band2, band1, band4, band3, k2_ind, k1_ind, q2_ind)
                        )*
                        OParams[kSL_ind](OP_row,OP_col);

            }
        }
        }
    }
    }

//cout<<"Hartree coefficient : "<<value<<endl;
return value;

}


complex<double> Hamiltonian::Hartree_coefficient_new(int k2_ind, int k3_ind, int band2, int band3, int spin_p){


complex<double> value;
int q_ind1, q_ind2, q_ind1_new, q_ind2_new;
int G_ind1_temp, G_ind2_temp;
int k3_ind1, k3_ind2, k2_ind1, k2_ind2;
int k1_ind, k1_ind1, k1_ind2;
int q1_ind, q2_ind, q3_ind;
int q3_kSL_internal_ind, q3_kSL_ind;
int q1_kSL_internal_ind, q1_kSL_ind;

int OP_row, OP_col;
k3_ind1 = Coordinates_.indx_cellwise(k3_ind);
k3_ind2 = Coordinates_.indy_cellwise(k3_ind);
k2_ind1 = Coordinates_.indx_cellwise(k2_ind);
k2_ind2 = Coordinates_.indy_cellwise(k2_ind);

    q_ind1 = k3_ind1 - k2_ind1; 
    q_ind2 = k3_ind2 - k2_ind2;
    Folding_to_BrillouinZone(q_ind1, q_ind2, q_ind1_new, q_ind2_new, G_ind1_temp, G_ind2_temp);
    q1_ind = q_ind1_new + l1_*q_ind2_new;

    q1_kSL_ind = Inverse_kSublattice_mapping[q1_ind].first;
    q1_kSL_internal_ind = Inverse_kSublattice_mapping[q1_ind].second;
    assert(q1_kSL_ind==0);

    q_ind1 = -k3_ind1 + k2_ind1; 
    q_ind2 = -k3_ind2 + k2_ind2;
    Folding_to_BrillouinZone(q_ind1, q_ind2, q_ind1_new, q_ind2_new, G_ind1_temp, G_ind2_temp);
    q2_ind = q_ind1_new + l1_*q_ind2_new;

    

    value=0.0;
    for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
        for(int temp_ind=0;temp_ind<k_sublattices[kSL_ind].size();temp_ind++){
        k1_ind = k_sublattices[kSL_ind][temp_ind];
        k1_ind1 = Coordinates_.indx_cellwise(k1_ind);
        k1_ind2 = Coordinates_.indy_cellwise(k1_ind);
        q_ind1 = -k3_ind1 + k2_ind1 + k1_ind1; 
        q_ind2 = -k3_ind2 + k2_ind2 + k1_ind2;
        Folding_to_BrillouinZone(q_ind1, q_ind2, q_ind1_new, q_ind2_new, G_ind1_temp, G_ind2_temp);
        q3_ind = q_ind1_new + l1_*q_ind2_new; //k1-k3+k2 for OPs

        //q3_ind ---> sublattice, ind
        q3_kSL_ind = Inverse_kSublattice_mapping[q3_ind].first;
        q3_kSL_internal_ind = Inverse_kSublattice_mapping[q3_ind].second;

        assert(q3_kSL_ind == kSL_ind);

        for(int band1=0;band1<Nbands;band1++){
        for(int band4=0;band4<Nbands;band4++){
            for(int spin=0;spin<2;spin++){

                OP_row =  q3_kSL_internal_ind + 
                          k_sublattices[kSL_ind].size()*band1 +
                          k_sublattices[kSL_ind].size()*Nbands*spin;

                OP_col =  temp_ind + 
                          k_sublattices[kSL_ind].size()*band4 +
                          k_sublattices[kSL_ind].size()*Nbands*spin;


                //cout<<spin<<" "<<spin_p<<"  "<<k1_ind<<"  "<<k2_ind<<"  "<<q1_ind<<endl;

                //cout<<"here 0.1"<<endl;
                value += (1.0/(2.0*Area)) * ( 
                        Amat[k1_ind][k2_ind][q1_kSL_internal_ind][band1][band2][band3][band4][spin][spin_p]
                        )*
                        OParams[kSL_ind](OP_row,OP_col);
                //cout<<"here 0.2"<<endl;

            
            }
        }
        }
    }
    }

//cout<<"Hartree coefficient : "<<value<<endl;
return value;

}


complex<double> Hamiltonian::Fock_coefficient(int k2_ind, int k3_ind, int band2, int band4, int spin, int spin_p){

complex<double> value;
int q_ind1, q_ind2, q_ind1_new, q_ind2_new;
int G_ind1_temp, G_ind2_temp;
int k3_ind1, k3_ind2, k2_ind1, k2_ind2;
int k1_ind, k1_ind1, k1_ind2;
int q1_ind, q2_ind, q3_ind;
int q3_kSL_internal_ind, q3_kSL_ind;

int OP_row, OP_col;
k3_ind1 = Coordinates_.indx_cellwise(k3_ind);
k3_ind2 = Coordinates_.indy_cellwise(k3_ind);
k2_ind1 = Coordinates_.indx_cellwise(k2_ind);
k2_ind2 = Coordinates_.indy_cellwise(k2_ind);

 

    

    value=0.0;
    for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
        for(int temp_ind=0;temp_ind<k_sublattices[kSL_ind].size();temp_ind++){
        k1_ind = k_sublattices[kSL_ind][temp_ind];
        k1_ind1 = Coordinates_.indx_cellwise(k1_ind);
        k1_ind2 = Coordinates_.indy_cellwise(k1_ind);

    q_ind1 = k3_ind1 - k1_ind1; 
    q_ind2 = k3_ind2 - k1_ind2;
    Folding_to_BrillouinZone(q_ind1, q_ind2, q_ind1_new, q_ind2_new, G_ind1_temp, G_ind2_temp);
    q1_ind = q_ind1_new + l1_*q_ind2_new;

    q_ind1 = -k3_ind1 + k1_ind1; 
    q_ind2 = -k3_ind2 + k1_ind2;
    Folding_to_BrillouinZone(q_ind1, q_ind2, q_ind1_new, q_ind2_new, G_ind1_temp, G_ind2_temp);
    q2_ind = q_ind1_new + l1_*q_ind2_new;



        q_ind1 = -k3_ind1 + k2_ind1 + k1_ind1; 
        q_ind2 = -k3_ind2 + k2_ind2 + k1_ind2;
        Folding_to_BrillouinZone(q_ind1, q_ind2, q_ind1_new, q_ind2_new, G_ind1_temp, G_ind2_temp);
        q3_ind = q_ind1_new + l1_*q_ind2_new; //k1-k3+k2 for OPs

        //q3_ind ---> sublattice, ind
        q3_kSL_ind = Inverse_kSublattice_mapping[q3_ind].first;
        q3_kSL_internal_ind = Inverse_kSublattice_mapping[q3_ind].second;

        assert(q3_kSL_ind == kSL_ind);

        for(int band1=0;band1<Nbands;band1++){
        for(int band3=0;band3<Nbands;band3++){
          
                OP_row =  q3_kSL_internal_ind + 
                          k_sublattices[kSL_ind].size()*band1 +
                          k_sublattices[kSL_ind].size()*Nbands*spin;

                OP_col =  temp_ind + 
                          k_sublattices[kSL_ind].size()*band3 +
                          k_sublattices[kSL_ind].size()*Nbands*spin_p;

                value += (1.0/(2.0*Area)) * ( //HERE
                        Interaction_value(spin, spin_p, band1, band2, band3, band4, k2_ind, k1_ind, q1_ind) +
                        Interaction_value(spin_p, spin, band2, band1, band4, band3, k1_ind, k2_ind, q2_ind)
                        )*
                        OParams[kSL_ind](OP_row,OP_col);

            
        }
        }
    }
    }

//cout<<"Hartree coefficient : "<<value<<endl;
return value;

}

complex<double> Hamiltonian::Fock_coefficient_new(int k2_ind, int k3_ind, int band2, int band4, int spin, int spin_p){


//HERE
complex<double> value;
int q_ind1, q_ind2, q_ind1_new, q_ind2_new;
int G_ind1_temp, G_ind2_temp;
int k3_ind1, k3_ind2, k2_ind1, k2_ind2;
int k1_ind, k1_ind1, k1_ind2;
int q1_ind, q2_ind, q3_ind;
int k2_mk3;
int q3_kSL_internal_ind, q3_kSL_ind;
int k2_mk3_kSL_internal_ind, k2_mk3_kSL_ind;

int OP_row, OP_col;
k3_ind1 = Coordinates_.indx_cellwise(k3_ind);
k3_ind2 = Coordinates_.indy_cellwise(k3_ind);
k2_ind1 = Coordinates_.indx_cellwise(k2_ind);
k2_ind2 = Coordinates_.indy_cellwise(k2_ind);
q_ind1 = k2_ind1 - k3_ind1; 
q_ind2 = k2_ind2 - k3_ind2;
Folding_to_BrillouinZone(q_ind1, q_ind2, q_ind1_new, q_ind2_new, G_ind1_temp, G_ind2_temp);
k2_mk3 = q_ind1_new + l1_*q_ind2_new;

k2_mk3_kSL_ind = Inverse_kSublattice_mapping[k2_mk3].first;
k2_mk3_kSL_internal_ind = Inverse_kSublattice_mapping[k2_mk3].second; 
assert(k2_mk3_kSL_ind==0);
    

    value=0.0;
    for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
        for(int temp_ind=0;temp_ind<k_sublattices[kSL_ind].size();temp_ind++){
        k1_ind = k_sublattices[kSL_ind][temp_ind];
        k1_ind1 = Coordinates_.indx_cellwise(k1_ind);
        k1_ind2 = Coordinates_.indy_cellwise(k1_ind);

    // q_ind1 = k3_ind1 - k1_ind1; 
    // q_ind2 = k3_ind2 - k1_ind2;
    // Folding_to_BrillouinZone(q_ind1, q_ind2, q_ind1_new, q_ind2_new, G_ind1_temp, G_ind2_temp);
    // q1_ind = q_ind1_new + l1_*q_ind2_new;

    // q_ind1 = -k3_ind1 + k1_ind1; 
    // q_ind2 = -k3_ind2 + k1_ind2;
    // Folding_to_BrillouinZone(q_ind1, q_ind2, q_ind1_new, q_ind2_new, G_ind1_temp, G_ind2_temp);
    // q2_ind = q_ind1_new + l1_*q_ind2_new;



        q_ind1 = -k3_ind1 + k2_ind1 + k1_ind1; 
        q_ind2 = -k3_ind2 + k2_ind2 + k1_ind2;
        Folding_to_BrillouinZone(q_ind1, q_ind2, q_ind1_new, q_ind2_new, G_ind1_temp, G_ind2_temp);
        q3_ind = q_ind1_new + l1_*q_ind2_new; //k1-k3+k2 for OPs

        //q3_ind ---> sublattice, ind
        q3_kSL_ind = Inverse_kSublattice_mapping[q3_ind].first;
        q3_kSL_internal_ind = Inverse_kSublattice_mapping[q3_ind].second;

        assert(q3_kSL_ind == kSL_ind);

        for(int band1=0;band1<Nbands;band1++){
        for(int band3=0;band3<Nbands;band3++){
          
                OP_row =  q3_kSL_internal_ind + 
                          k_sublattices[kSL_ind].size()*band1 +
                          k_sublattices[kSL_ind].size()*Nbands*spin;

                OP_col =  temp_ind + 
                          k_sublattices[kSL_ind].size()*band3 +
                          k_sublattices[kSL_ind].size()*Nbands*spin_p;

                value += (1.0/(2.0*Area)) * ( //HERE
                        Bmat[k1_ind][k2_ind][k2_mk3_kSL_internal_ind][band1][band2][band3][band4][spin][spin_p]
                        )*
                        OParams[kSL_ind](OP_row,OP_col);

            
        }
        }
    }
    }

//cout<<"Hartree coefficient : "<<value<<endl;
return value;

}

void Hamiltonian::Create_Hamiltonian(int kset_ind){


    


    int subspace_;    
    int row_ind, col_ind;
    int q_ind1, q_ind2, k3_ind1, k3_ind2, k1_ind1, k1_ind2, k2_ind1, k2_ind2;
    int G_ind1, G_ind2;

        subspace_= 2*Nbands*k_sublattices[kset_ind].size();
        Ham_.resize(subspace_,subspace_);


        //Kinetic Energy
        for(int spin=0;spin<2;spin++){
            for(int band=0;band<Nbands;band++){
                for(int k_ind=0;k_ind<k_sublattices[kset_ind].size();k_ind++){
                    row_ind = k_ind + 
                              k_sublattices[kset_ind].size()*band +
                              k_sublattices[kset_ind].size()*Nbands*spin;
                    Ham_(row_ind, row_ind) += 1.0*BlochEigvals[spin][band][k_sublattices[kset_ind][k_ind]];
                    //Ham_(row_ind, row_ind) += DispersionTriangularLattice(k_sublattices[kset_ind][k_ind]);
                }
            }
        }


        //Hartree Term
            for(int spin_p=0;spin_p<2;spin_p++){
               
                for(int band2=0;band2<Nbands;band2++){
                for(int band3=0;band3<Nbands;band3++){
                
                
                for(int k2_ind=0;k2_ind<k_sublattices[kset_ind].size();k2_ind++){
                 for(int k3_ind=0;k3_ind<k_sublattices[kset_ind].size();k3_ind++){
                    
                    row_ind = k3_ind + 
                              k_sublattices[kset_ind].size()*band2 +
                              k_sublattices[kset_ind].size()*Nbands*spin_p;
                    col_ind = k2_ind + 
                              k_sublattices[kset_ind].size()*band3 +
                              k_sublattices[kset_ind].size()*Nbands*spin_p;

                    //k_sublattices[kset_ind][k3_ind]);

                   Ham_(row_ind,col_ind) += 1.0*HartreeCoefficients[kset_ind][k2_ind][k3_ind][band2][band3][spin_p];
                   //cout<<"Hartree Coeff :" <<HartreeCoefficients[kset_ind][k2_ind][k3_ind][band2][band3][spin_p]<<endl;
                   //Hartree_coefficient(k_sublattices[kset_ind][k2_ind], k_sublattices[kset_ind][k3_ind], band2, band3, spin_p);
                }   
                }

                
                }
                }
                

            }

            //Fock Term 
            for(int spin=0;spin<2;spin++){
            for(int spin_p=0;spin_p<2;spin_p++){
               
                for(int band2=0;band2<Nbands;band2++){
                for(int band4=0;band4<Nbands;band4++){
                
                
                for(int k2_ind=0;k2_ind<k_sublattices[kset_ind].size();k2_ind++){
                 for(int k3_ind=0;k3_ind<k_sublattices[kset_ind].size();k3_ind++){
                    
                    row_ind = k3_ind + 
                              k_sublattices[kset_ind].size()*band2 +
                              k_sublattices[kset_ind].size()*Nbands*spin_p;
                    col_ind = k2_ind + 
                              k_sublattices[kset_ind].size()*band4 +
                              k_sublattices[kset_ind].size()*Nbands*spin;

                    //k_sublattices[kset_ind][k3_ind]);

                   Ham_(row_ind,col_ind) += -1.0*FockCoefficients[kset_ind][k2_ind][k3_ind][band2][band4][spin][spin_p];
                   //cout<<"Fock Coeff : "<<FockCoefficients[kset_ind][k2_ind][k3_ind][band2][band4][spin][spin_p]<<endl;
                   //Fock_coefficient(k_sublattices[kset_ind][k2_ind], k_sublattices[kset_ind][k3_ind], band2, band4, spin, spin_p);
                }   
                }

                }
                }
                
            }
            }



}


void Hamiltonian::Diagonalize(char option){

    //extern "C" void   zheev_(char *,char *,int *,std::complex<double> *, int *, double *,
    //                       std::complex<double> *,int *, double *, int *);


    char jobz=option;
    char uplo='L'; //WHY ONLY 'L' WORKS?
    int n=Ham_.n_row();
    int lda=Ham_.n_col();
    vector<complex<double>> work(3);
    vector<double> rwork(3*n -2);
    int info;
    int lwork= -1;

    eigs_.clear();
    eigs_.resize(Ham_.n_row());
    fill(eigs_.begin(),eigs_.end(),0);
    // query:
    zheev_(&jobz,&uplo,&n,&(Ham_(0,0)),&lda,&(eigs_[0]),&(work[0]),&lwork,&(rwork[0]),&info);
    //lwork = int(real(work[0]))+1;
    lwork = int((work[0].real()));
    work.resize(lwork);
    // real work:
    zheev_(&jobz,&uplo,&n,&(Ham_(0,0)),&lda,&(eigs_[0]),&(work[0]),&lwork,&(rwork[0]),&info);
    if (info!=0) {
        std::cerr<<"info="<<info<<"\n";
        perror("diag: zheev: failed with info!=0.\n");
    }

    // Ham_.print();

    //  for(int i=0;i<eigs_.size();i++){
    //    cout<<eigs_[i]<<endl;
    //}


}


void Hamiltonian::AppendEigenspectrum(int kset_ind){

EigValues[kset_ind] = eigs_;
EigVectors[kset_ind] = Ham_;

}




void Hamiltonian::Calculate_layer_resolved_densities(){

int spin_up=0;
int spin_dn=1;
int BOTTOM_=0;
int TOP_=1;

int q_ind,comp;

int col_val, row_val;
for(int layer=0;layer<2;layer++){
for(int spin=0;spin<2;spin++){
N_layer_tau[layer][spin]=0.0;


for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
for(int k_ind=0;k_ind<k_sublattices[kSL_ind].size();k_ind++){

q_ind = k_sublattices[kSL_ind][k_ind];

for(int m=0;m<EigVectors[kSL_ind].n_row();m++){

for(int bandn=0;bandn<Nbands;bandn++){
for(int bandnp=0;bandnp<Nbands;bandnp++){

col_val = k_ind + 
          k_sublattices[kSL_ind].size()*bandnp +
          k_sublattices[kSL_ind].size()*Nbands*spin;
row_val = k_ind + 
          k_sublattices[kSL_ind].size()*bandn +
          k_sublattices[kSL_ind].size()*Nbands*spin;

for(int g_ind1=0;g_ind1<G_grid_L1;g_ind1++){
for(int g_ind2=0;g_ind2<G_grid_L2;g_ind2++){
 
comp = HamiltonianCont_.Coordinates_.Nbasis(g_ind1, g_ind2, layer);

N_layer_tau[layer][spin] += FermiFunction(EigValues[kSL_ind][m])*
                            conj(EigVectors[kSL_ind](col_val,m))*
                            EigVectors[kSL_ind](row_val,m)*
                            conj(BlochStates[spin][bandnp][q_ind][comp])*
                            BlochStates[spin][bandn][q_ind][comp];
                            
}}
}}
}}}

cout<<"No. Fermions ";
if(layer==BOTTOM_ && spin==spin_up){
cout<<"BOTTOM SPIN_UP"<<N_layer_tau[layer][spin]<<endl;
}
if(layer==TOP_ && spin==spin_up){
cout<<"TOP SPIN_UP"<<N_layer_tau[layer][spin]<<endl;
}
if(layer==BOTTOM_ && spin==spin_dn){
cout<<"BOTTOM SPIN_DN"<<N_layer_tau[layer][spin]<<endl;
}
if(layer==TOP_ && spin==spin_dn){
cout<<"TOP SPIN_DN"<<N_layer_tau[layer][spin]<<endl;
}


}}


}

void Hamiltonian::Calculate_OParams_and_diff(double &diff_){

// calculate OParams_new
int row_val, col_val;
nu_holes_new=0.0;
Total_n_up=0.0;Total_n_dn=0.0;
double distance_sqr;

distance_sqr=0.0;
for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){

OParams_new[kSL_ind].resize(k_sublattices[kSL_ind].size()*Nbands*2,k_sublattices[kSL_ind].size()*Nbands*2);

for(int spin=0;spin<2;spin++){
for(int k_ind=0;k_ind<k_sublattices[kSL_ind].size();k_ind++){
for(int band1=0;band1<Nbands;band1++){
row_val = k_ind + 
          k_sublattices[kSL_ind].size()*band1 +
          k_sublattices[kSL_ind].size()*Nbands*spin;

for(int spin_p=0;spin_p<2;spin_p++){
for(int k_ind_p=0;k_ind_p<k_sublattices[kSL_ind].size();k_ind_p++){
for(int band2=0;band2<Nbands;band2++){
col_val = k_ind_p + 
          k_sublattices[kSL_ind].size()*band2 +
          k_sublattices[kSL_ind].size()*Nbands*spin_p;


OParams_new[kSL_ind](row_val,col_val) = 0.0;

for(int n=0;n<EigVectors[kSL_ind].n_row();n++){
OParams_new[kSL_ind](row_val,col_val) += FermiFunction(EigValues[kSL_ind][n])*EigVectors[kSL_ind](col_val,n)*conj(EigVectors[kSL_ind](row_val,n));

if(row_val==col_val){
nu_holes_new += (1.0/ns_)*FermiFunction(EigValues[kSL_ind][n])*abs(EigVectors[kSL_ind](row_val,n))*abs(EigVectors[kSL_ind](row_val,n));
if(spin==0){
Total_n_up += (1.0/ns_)*FermiFunction(EigValues[kSL_ind][n])*abs(EigVectors[kSL_ind](row_val,n))*abs(EigVectors[kSL_ind](row_val,n));
}
else{
Total_n_dn += (1.0/ns_)*FermiFunction(EigValues[kSL_ind][n])*abs(EigVectors[kSL_ind](row_val,n))*abs(EigVectors[kSL_ind](row_val,n));  
}
}

}

distance_sqr =  abs(OParams_new[kSL_ind](row_val,col_val) - OParams[kSL_ind](row_val,col_val))*
                abs(OParams_new[kSL_ind](row_val,col_val) - OParams[kSL_ind](row_val,col_val));

}}}
}}}
}

diff_ = sqrt(distance_sqr);
}



void Hamiltonian::Update_OParams(){

if(Convergence_technique=="SimpleMixing"){

for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
for(int row_val=0;row_val<OParams[kSL_ind].n_row();row_val++){
for(int col_val=0;col_val<OParams[kSL_ind].n_col();col_val++){
OParams[kSL_ind](row_val,col_val) = alpha_mixing*OParams_new[kSL_ind](row_val,col_val) + 
                                    (1.0-alpha_mixing)*OParams[kSL_ind](row_val,col_val);

}}}

}


}

void Hamiltonian::Initialize_OParams(){


for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
    OParams[kSL_ind].resize(k_sublattices[kSL_ind].size()*2*Nbands,k_sublattices[kSL_ind].size()*2*Nbands);
for(int row_val=0;row_val<OParams[kSL_ind].n_row();row_val++){
for(int col_val=0;col_val<OParams[kSL_ind].n_col();col_val++){
    if(row_val!=col_val){
OParams[kSL_ind](row_val,col_val) = complex<double>(Myrandom(),Myrandom());
    }
    else{
OParams[kSL_ind](row_val,col_val) = complex<double>(Myrandom(),0.0);
    }
}}}



    string file_OP="Initial_Oparams.txt";
    ofstream file_OP_out(file_OP.c_str());
    file_OP_out<<"#k_sublattice  row_val   col_val  value.real  value.imag"<<endl;
for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
for(int row_val=0;row_val<OParams[kSL_ind].n_row();row_val++){
for(int col_val=0;col_val<OParams[kSL_ind].n_col();col_val++){
    file_OP_out<<kSL_ind<<"  "<<row_val<<"  "<<col_val<<"  "<<OParams[kSL_ind](row_val,col_val).real()<<"  "<<OParams[kSL_ind](row_val,col_val).imag()<<endl;
}}}



}

double Hamiltonian::FermiFunction(double Eval){
    double val;
    val=1.0/(  exp((Eval-mu_)/(KB_*Temperature)) +1.0  );

    return val;
}


void Hamiltonian::Print_Spectrum(int kset_ind, string filename){

//string file_spec="Spectrum_kSL_" + to_string(kset_ind) +".txt";
 string file_spec = filename;
    ofstream file_out(file_spec.c_str());

file_out<<"#----EigenVectors------------"<<endl;
for(int n=0;n<Ham_.n_row();n++){
for(int m=0;m<Ham_.n_col();m++){
file_out<<Ham_(n,m)<<"  ";
}
file_out<<endl;
}

file_out<<endl;
file_out<<"#----------Eigenvalues------"<<endl;
for(int n=0;n<EigValues[kset_ind].size();n++){
file_out<<EigValues[kset_ind][n]<<endl;
}


}


void Hamiltonian::RunSelfConsistency(){


    // cout<<"CHECK Coordinates (in HF Hamiltonian)"<<endl;
    //     for(int i=0;i<l1_*l2_;i++){
    //     cout<<i<<"  "<<Coordinates_.indx_cellwise(i)<<"  "<<Coordinates_.indy_cellwise(i)<<endl;
    //     }



    string file_conv="HF_out.txt";
    ofstream Convergence_out(file_conv.c_str());
    Convergence_out<<"#iter   OP_diff   mu    Total_n_up    Total_n_dn"<<endl;


    int N_Particles;
    N_Particles = nu_holes_target*ns_;
    double diff_=1000;
    int iter=0;
    double mu_old;

    //cout<<"here -1"<<endl;
    Initialize_OParams();
    //cout<<"here 0"<<endl;
    Update_Hartree_Coefficients();
    //cout<<"here 1"<<endl;
    Update_Fock_Coefficients();
    //cout<<"here 2"<<endl;

    while(iter<HF_max_iterations && diff_>HF_convergence_error){ 

    for(int kset_ind=0;kset_ind<k_sublattices.size();kset_ind++){
        
        Create_Hamiltonian(kset_ind);
        char Dflag='V';

        // string filename1 = "Iter_"+to_string(iter)+"_kSL_ind_"+to_string(kset_ind)+"_Hamil.txt";
        // Print_Spectrum(kset_ind, filename1);

        Diagonalize(Dflag);
        AppendEigenspectrum(kset_ind);

        // string filename = "Iter_"+to_string(iter)+"_kSL_ind_"+to_string(kset_ind)+"_Spectrum.txt";
        // Print_Spectrum(kset_ind, filename);

    }

    if(iter==0){
        //near bottom of first band
        mu_old = EigValues[0][0] + 0.5;
        cout<<"initial mu = "<<mu_old<<endl;
    }
    
    
    mu_=chemicalpotential(mu_old,N_Particles);
    Calculate_OParams_and_diff(diff_);
    Update_OParams();
    Update_Hartree_Coefficients();
    Update_Fock_Coefficients();
    mu_old=mu_;

    Convergence_out<<iter<<"   "<<diff_<<"   "<<mu_<<"   "<<Total_n_up<<"  "<<Total_n_dn<<endl;

//-------REMOVE LATER ------------
/* 
string file_OP_temp="Oparams_Iter_" + to_string(iter) +".txt";
    ofstream file_OP_out_temp(file_OP_temp.c_str());
    file_OP_out_temp<<"#k_sublattice  row_val   col_val  value.real  value.imag"<<endl;
for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
for(int row_val=0;row_val<OParams_new[kSL_ind].n_row();row_val++){
for(int col_val=0;col_val<OParams_new[kSL_ind].n_col();col_val++){
    file_OP_out_temp<<kSL_ind<<"  "<<row_val<<"  "<<col_val<<"  "<<OParams_new[kSL_ind](row_val,col_val).real()<<"  "<<OParams_new[kSL_ind](row_val,col_val).imag()<<endl;
}}}
*/
//-----------------------------------



    iter++;
    }




    string file_OP="Final_Oparams.txt";
    ofstream file_OP_out(file_OP.c_str());
    file_OP_out<<"#k_sublattice  row_val   col_val  value.real  value.imag"<<endl;
for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
for(int row_val=0;row_val<OParams[kSL_ind].n_row();row_val++){
for(int col_val=0;col_val<OParams[kSL_ind].n_col();col_val++){
    file_OP_out<<kSL_ind<<"  "<<row_val<<"  "<<col_val<<"  "<<OParams[kSL_ind](row_val,col_val).real()<<"  "<<OParams[kSL_ind](row_val,col_val).imag()<<endl;
}}}


    Calculate_Total_Spin();
    //Print_SPDOS("DOS.txt");
    Calculate_layer_resolved_densities();

    Print_HF_Bands();

    if(abs(Parameters_.NMat_det)==1){
    Calculate_ChernNumbers_HFBands();
    }
    Calculate_RealSpace_OParams_new("RealSpace_OParams.txt", "RealSpace_OParams_moiresites.txt");

}



void Hamiltonian::Saving_BlochState_Overlaps(){

BO.resize(Nbands);
for(int n=0;n<Nbands;n++){
BO[n].resize(2);
for(int spin=0;spin<2;spin++){
BO[n][spin].resize((l1_+1)*(l2_+1));
for(int k_ind=0;k_ind<((l1_+1)*(l2_+1));k_ind++){
BO[n][spin][k_ind].resize(Nbands);
for(int np=0;np<Nbands;np++){
BO[n][spin][k_ind][np].resize(2);
for(int spinp=0;spinp<2;spinp++){
BO[n][spin][k_ind][np][spinp].resize((l1_+1)*(l2_+1));

}}}}}

int k_ind, kp_ind;
for(int n=0;n<Nbands;n++){
for(int spin=0;spin<2;spin++){
for(int k_ind1=0;k_ind1<(l1_+1);k_ind1++){
for(int k_ind2=0;k_ind2<(l2_+1);k_ind2++){
k_ind = k_ind1 + k_ind2*(l1_+1);

for(int np=0;np<Nbands;np++){
for(int spinp=0;spinp<2;spinp++){
for(int kp_ind1=0;kp_ind1<(l1_+1);kp_ind1++){
for(int kp_ind2=0;kp_ind2<(l2_+1);kp_ind2++){
kp_ind = kp_ind1 + kp_ind2*(l1_+1);

BO[n][spin][k_ind][np][spinp][kp_ind] =0.0;

for(int comp=0;comp<G_grid_L1*G_grid_L2*2;comp++){
    //cout<<"Here: "<<n<<" "<<spin<<" "<<k_ind<<" "<<np<<" "<<spinp<<" "<<kp_ind<<" "<<comp<<"  "<<endl;
BO[n][spin][k_ind][np][spinp][kp_ind] += conj(BlochStates_old_[spin][n][k_ind1+k_ind2*(l1_+1)][comp])*
                                         (BlochStates_old_[spinp][np][kp_ind1+kp_ind2*(l1_+1)][comp]);
}

}}}}
}}}}



}


void Hamiltonian::Calculate_ChernNumbers_HFBands(){

    assert(abs(Parameters_.NMat_det)==1);


    int mbz_factor=1;
    Saving_BlochState_Overlaps();
     int L1_,L2_;
     L1_=l1_; //along G1 (b6)
     L2_=l2_; //along G2 (b2)
     int N_bands_Chern = Nbands*2; //2 here for spin


    
    
    cout<<"XXXXXXXXXXXXXXX Chern numbers for HF bands XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"<<endl;

    Matrix<complex<double>> F_mat; //F1, F2, F3, F4, F5;
    F_mat.resize(N_bands_Chern, ((mbz_factor*L1_)+1)*((mbz_factor*L2_)+1));

    complex<double> Ux_k, Uy_k, Ux_kpy, Uy_kpx;
    vector<complex<double>> F_bands;
    F_bands.resize(N_bands_Chern);
    vector<complex<double>> Chern_num;
    Chern_num.resize(N_bands_Chern);

    vector<complex<double>> F_bands_orgnl;
    F_bands_orgnl.resize(N_bands_Chern);
    vector<complex<double>> Chern_num_orgnl;
    Chern_num_orgnl.resize(N_bands_Chern);
    for (int band = 0; band < N_bands_Chern; band++)
    {
        string file_Fk="Fk_HF_band"+to_string(band)+".txt";
        ofstream fl_Fk_out(file_Fk.c_str());
        fl_Fk_out<<"#nx  ny  tilde_F(nx,ny).real()  tilde_F(nx,ny).imag()  ArgofLog.real()  ArgofLog.imag()"<<endl;
        fl_Fk_out<<"#Extra momentum point for pm3d corners2color c1"<<endl;

//        string file_Fk_orgnl="Fk_original_band"+to_string(band)+".txt";
//        ofstream fl_Fk_orgnl_out(file_Fk_orgnl.c_str());
//        fl_Fk_orgnl_out<<"#nx  ny  F(nx,ny).real()*(2pi/Lx)*(2pi/Ly)  F(nx,ny).imag()*(2pi/Lx)*(2pi/Ly)"<<endl;

        F_bands[band] = 0.0;
        F_bands_orgnl[band] = 0.0;
        // for (int nx = L1_/2; nx < 3*L1_/2; nx++)
        // {
        //     for (int ny = L2_/2; ny < 3*L2_/2; ny++)
        //     {

        for (int nx = 0; nx < L1_; nx++)
        {
            for (int ny = 0; ny < L2_; ny++)
            {

                int n = nx + ny*((mbz_factor*L1_)+1);
                int n_left, n_right, nx_left, ny_left, nx_right, ny_right;
                int n_HF, n_left_HF, n_right_HF, nx_left_HF, ny_left_HF, nx_right_HF, ny_right_HF;
                int q_kSL_internal_ind_left, q_kSL_ind_left;
                int q_kSL_internal_ind_right, q_kSL_ind_right;
                int col_val_left, col_val_right;

                n_HF = nx + ny*(L1_);


                //U1_k
                Ux_k = 0;
                n_left = n;
                nx_right = (nx + 1);// % (mbz_factor*L1_);
                ny_right = ny;
                n_right = nx_right + ny_right*((mbz_factor*L1_)+1);
 
                n_left_HF = n_HF;
                nx_right_HF = (nx + 1)%(L1_);
                ny_right_HF = ny;
                n_right_HF = nx_right_HF + ny_right_HF*(L1_);

                q_kSL_ind_left = Inverse_kSublattice_mapping[n_left_HF].first;
                q_kSL_internal_ind_left = Inverse_kSublattice_mapping[n_left_HF].second;
                q_kSL_ind_right = Inverse_kSublattice_mapping[n_right_HF].first;
                q_kSL_internal_ind_right = Inverse_kSublattice_mapping[n_right_HF].second;

                for(int band_n=0;band_n<Nbands;band_n++){
                for(int spin=0;spin<2;spin++){
                for(int band_np=0;band_np<Nbands;band_np++){
                for(int spin_p=0;spin_p<2;spin_p++){

                col_val_left =  q_kSL_internal_ind_left + 
                                k_sublattices[q_kSL_ind_left].size()*band_n +
                                k_sublattices[q_kSL_ind_left].size()*Nbands*spin;

                col_val_right =  q_kSL_internal_ind_right + 
                                k_sublattices[q_kSL_ind_right].size()*band_np +
                                k_sublattices[q_kSL_ind_right].size()*Nbands*spin_p;


                 Ux_k += BO[band_n][spin][n_left][band_np][spin_p][n_right]*
                        conj(EigVectors[q_kSL_ind_left](col_val_left,band))*
                        EigVectors[q_kSL_ind_right](col_val_right,band);
                }}
                }}
                Ux_k = Ux_k * (1.0 / abs(Ux_k));

                //U2_kpx
                Uy_kpx = 0;
                nx_left = (nx + 1);// % (mbz_factor*L1_);
                ny_left = ny;
                n_left = nx_left + ny_left*((mbz_factor*L1_)+1);
                nx_right = nx_left;
                ny_right = (ny_left + 1);// % (mbz_factor*L2_);
                n_right = nx_right + ny_right*((mbz_factor*L1_)+1);

                nx_left_HF = (nx + 1)%(L1_);
                ny_left_HF = ny;
                n_left_HF = nx_left_HF + ny_left_HF*((L1_));
                nx_right_HF = nx_left_HF;
                ny_right_HF = (ny_left_HF + 1)%(L2_);
                n_right_HF = nx_right_HF + ny_right_HF*((L1_));
                

                q_kSL_ind_left = Inverse_kSublattice_mapping[n_left_HF].first;
                q_kSL_internal_ind_left = Inverse_kSublattice_mapping[n_left_HF].second;
                q_kSL_ind_right = Inverse_kSublattice_mapping[n_right_HF].first;
                q_kSL_internal_ind_right = Inverse_kSublattice_mapping[n_right_HF].second;

                for(int band_n=0;band_n<Nbands;band_n++){
                for(int spin=0;spin<2;spin++){
                for(int band_np=0;band_np<Nbands;band_np++){
                for(int spin_p=0;spin_p<2;spin_p++){

                col_val_left =  q_kSL_internal_ind_left + 
                                k_sublattices[q_kSL_ind_left].size()*band_n +
                                k_sublattices[q_kSL_ind_left].size()*Nbands*spin;

                col_val_right =  q_kSL_internal_ind_right + 
                                k_sublattices[q_kSL_ind_right].size()*band_np +
                                k_sublattices[q_kSL_ind_right].size()*Nbands*spin_p;


                 Uy_kpx += BO[band_n][spin][n_left][band_np][spin_p][n_right]*
                        conj(EigVectors[q_kSL_ind_left](col_val_left,band))*
                        EigVectors[q_kSL_ind_right](col_val_right,band);
                }}
                }}
                Uy_kpx = Uy_kpx * (1.0 / abs(Uy_kpx));

                //U1_kpy
                Ux_kpy = 0;
                nx_left = nx;
                ny_left = (ny + 1);// % (mbz_factor*L2_);
                n_left = nx_left + ny_left*((mbz_factor*L1_)+1);
                nx_right = (nx_left + 1);// % (mbz_factor*L1_);
                ny_right = ny_left;
                n_right = nx_right + ny_right*((mbz_factor*L1_)+1);

                nx_left_HF = nx;
                ny_left_HF = (ny + 1)%(mbz_factor*L2_);
                n_left_HF = nx_left_HF + ny_left_HF*((mbz_factor*L1_));
                nx_right_HF = (nx_left_HF + 1)% (mbz_factor*L1_);
                ny_right_HF = ny_left_HF;
                n_right_HF = nx_right_HF + ny_right_HF*((mbz_factor*L1_));


                q_kSL_ind_left = Inverse_kSublattice_mapping[n_left_HF].first;
                q_kSL_internal_ind_left = Inverse_kSublattice_mapping[n_left_HF].second;
                q_kSL_ind_right = Inverse_kSublattice_mapping[n_right_HF].first;
                q_kSL_internal_ind_right = Inverse_kSublattice_mapping[n_right_HF].second;

                for(int band_n=0;band_n<Nbands;band_n++){
                for(int spin=0;spin<2;spin++){
                for(int band_np=0;band_np<Nbands;band_np++){
                for(int spin_p=0;spin_p<2;spin_p++){

                col_val_left =  q_kSL_internal_ind_left + 
                                k_sublattices[q_kSL_ind_left].size()*band_n +
                                k_sublattices[q_kSL_ind_left].size()*Nbands*spin;

                col_val_right =  q_kSL_internal_ind_right + 
                                k_sublattices[q_kSL_ind_right].size()*band_np +
                                k_sublattices[q_kSL_ind_right].size()*Nbands*spin_p;


                 Ux_kpy += BO[band_n][spin][n_left][band_np][spin_p][n_right]*
                        conj(EigVectors[q_kSL_ind_left](col_val_left,band))*
                        EigVectors[q_kSL_ind_right](col_val_right,band);
                }}
                }}
                Ux_kpy = Ux_kpy * (1.0 / abs(Ux_kpy));

                //U2_k
                Uy_k = 0;
                nx_left = nx;
                ny_left = ny;
                n_left = nx_left + ny_left*((mbz_factor*L1_)+1);
                nx_right = nx_left;
                ny_right = (ny_left + 1);// % (mbz_factor*L2_);
                n_right = nx_right + ny_right*((mbz_factor*L1_)+1);

                nx_left_HF = nx;
                ny_left_HF = ny;
                n_left_HF = nx_left_HF + ny_left_HF*((mbz_factor*L1_));
                nx_right_HF = nx_left_HF;
                ny_right_HF = (ny_left_HF + 1)%(mbz_factor*L2_);
                n_right_HF = nx_right_HF + ny_right_HF*((mbz_factor*L1_));

                q_kSL_ind_left = Inverse_kSublattice_mapping[n_left_HF].first;
                q_kSL_internal_ind_left = Inverse_kSublattice_mapping[n_left_HF].second;
                q_kSL_ind_right = Inverse_kSublattice_mapping[n_right_HF].first;
                q_kSL_internal_ind_right = Inverse_kSublattice_mapping[n_right_HF].second;

                for(int band_n=0;band_n<Nbands;band_n++){
                for(int spin=0;spin<2;spin++){
                for(int band_np=0;band_np<Nbands;band_np++){
                for(int spin_p=0;spin_p<2;spin_p++){

                col_val_left =  q_kSL_internal_ind_left + 
                                k_sublattices[q_kSL_ind_left].size()*band_n +
                                k_sublattices[q_kSL_ind_left].size()*Nbands*spin;

                col_val_right =  q_kSL_internal_ind_right + 
                                k_sublattices[q_kSL_ind_right].size()*band_np +
                                k_sublattices[q_kSL_ind_right].size()*Nbands*spin_p;


                Uy_k += BO[band_n][spin][n_left][band_np][spin_p][n_right]*
                        conj(EigVectors[q_kSL_ind_left](col_val_left,band))*
                        EigVectors[q_kSL_ind_right](col_val_right,band);
                }}
                }}
                Uy_k = Uy_k * (1.0 / abs(Uy_k));



                // Calculating tilde F12
                F_mat(band, n) = log(Ux_k *
                                     Uy_kpx *
                                     conj(Ux_kpy) * conj(Uy_k));

                F_bands[band] += F_mat(band, n);


                fl_Fk_out.precision(10);

                fl_Fk_out<<nx<<"  "<<ny<<"  "<<F_mat(band, n).real()<<"  "<<F_mat(band, n).imag()<<
                           "  "<<(Ux_k*Uy_kpx*conj(Ux_kpy)*conj(Uy_k)).real()<<
                           "  "<<(Ux_k*Uy_kpx*conj(Ux_kpy)*conj(Uy_k)).imag()<<endl;




                if(abs((abs(F_mat(band, n).imag()) - PI))<0.0000001){
                    cout<<ny<<"  "<<nx<<"  gives Pi for band"<< band <<endl;
                    // assert (abs((abs(F_mat(band, n).imag()) - M_PI))>0.0000001);
                }


                if(ny==(L2_-1)){//For pm3d corners2color c1
                    fl_Fk_out<<nx<<"  "<<ny+1<<"  "<<F_mat(band, n).real()<<"  "<<F_mat(band, n).imag()<<
                               "  "<<(Ux_k*Uy_kpx*conj(Ux_kpy)*conj(Uy_k)).real()<<
                               "  "<<(Ux_k*Uy_kpx*conj(Ux_kpy)*conj(Uy_k)).imag()<<endl;
                }
            }
            if(nx==(L1_)-1){//For pm3d corners2color c1
                fl_Fk_out<<endl;
                for(int ny_=0;ny_<L2_;ny_++){
                    int n_ = nx + mbz_factor*L1_*ny_;
                    fl_Fk_out<<nx+1<<"  "<<ny_<<"  "<<F_mat(band, n_).real()<<"  "<<F_mat(band, n_).imag()<<
                               "  "<<(Ux_k*Uy_kpx*conj(Ux_kpy)*conj(Uy_k)).real()<<
                               "  "<<(Ux_k*Uy_kpx*conj(Ux_kpy)*conj(Uy_k)).imag()<<endl;

                   if(ny_==(L2_)-1){//For pm3d corners2color c1
                    fl_Fk_out<<nx+1<<"  "<<ny_+1<<"  "<<F_mat(band, n_).real()<<"  "<<F_mat(band, n_).imag()<<
                               "  "<<(Ux_k*Uy_kpx*conj(Ux_kpy)*conj(Uy_k)).real()<<
                               "  "<<(Ux_k*Uy_kpx*conj(Ux_kpy)*conj(Uy_k)).imag()<<endl;
                }
                }
            }
            fl_Fk_out<<endl;
        }


        Chern_num[band] = (-1.0 * iota_complex / (2 * PI)) * F_bands[band];
        // Chern_num_orgnl[band] = (-1.0 * iota_complex / (2 * PI)) * F_bands_orgnl[band];
        fl_Fk_out<<"#Chern no*2pi*Iota= "<<F_bands[band].real()<<"  "<<F_bands[band].imag()<<endl;
        cout << "tilde Chern number [" << band << "] = " << Chern_num[band].real() << "        " << Chern_num[band].imag() << endl;
        //  cout << "Chern number [" << band << "] = " << Chern_num_orgnl[band].real() << " " << Chern_num_orgnl[band].imag() << endl;

    }

cout<<"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"<<endl;


}

void Hamiltonian::Print_HF_Bands(){

string file_bands="Bands_HF.txt";
ofstream file_bands_out(file_bands.c_str());

double overlap_top, overlap_bottom;
int q_ind1, q_ind2, q_ind;
int q1_kSL_ind, q1_kSL_internal_ind;
Mat_1_intpair k_path_;
double kx_val, ky_val;
k_path_ = Get_k_path(1);

int q_ind1_new, q_ind2_new, G1_ind_temp, G2_ind_temp;

for(int index=0;index<k_path_.size();index++){
        q_ind1=k_path_[index].first;
        q_ind2=k_path_[index].second;
        

Folding_to_BrillouinZone(q_ind1, q_ind2, q_ind1_new, q_ind2_new, G1_ind_temp, G2_ind_temp);
q_ind = q_ind1_new + l1_*q_ind2_new;

//cout<< q_ind1<< "  "<<  q_ind2 << "  "<<q_ind1_new<< "  "<< q_ind2_new<<endl;
q1_kSL_ind = Inverse_kSublattice_mapping[q_ind].first;
q1_kSL_internal_ind = Inverse_kSublattice_mapping[q_ind].second;


kx_val=(2.0*PI/Parameters_.a_moire)*(q_ind1*(1.0/(sqrt(3)*l1_))  +  q_ind2*(1.0/(sqrt(3)*l2_)));
ky_val=(2.0*PI/Parameters_.a_moire)*(q_ind1*(-1.0/(l1_))  +  q_ind2*(1.0/(l2_)));


file_bands_out<<index<<"  "<<kx_val<<"  "<<ky_val<<"  ";
for(int m=0;m<EigValues[q1_kSL_ind].size();m++){
file_bands_out<<EigValues[q1_kSL_ind][m]<<"  ";
for(int spin=0;spin<2;spin++){
Get_layer_overlaps(overlap_top, overlap_bottom, m, spin, q_ind1_new, q_ind2_new);
file_bands_out<<"  "<<overlap_top<<"  "<<overlap_bottom<<"  ";
}
}
file_bands_out<<endl;
}

}

void Hamiltonian::Get_layer_overlaps(double &overlap_top, double &overlap_bottom, int band_m, int spin, int q_ind1, int q_ind2){

int BOTTOM_=0;
int TOP_=1;
int comp;
int q_ind = q_ind1 + l1_*q_ind2;
int q_kSL_ind, q_kSL_internal_ind;
q_kSL_ind = Inverse_kSublattice_mapping[q_ind].first;
q_kSL_internal_ind = Inverse_kSublattice_mapping[q_ind].second;
int col_val;

Mat_1_doub overlaps_;
overlaps_.resize(2);

complex<double> temp_overlap;

for(int layer=0;layer<2;layer++){
overlaps_[layer]=0.0;

for(int g_ind1=0;g_ind1<G_grid_L1;g_ind1++){
for(int g_ind2=0;g_ind2<G_grid_L2;g_ind2++){

comp = HamiltonianCont_.Coordinates_.Nbasis(g_ind1, g_ind2, layer);

temp_overlap=0.0;
for(int n=0;n<BlochStates[spin].size();n++){

col_val = q_kSL_internal_ind + 
          k_sublattices[q_kSL_ind].size()*n +
          k_sublattices[q_kSL_ind].size()*Nbands*spin;

temp_overlap += BlochStates[spin][n][q_ind][comp]*EigVectors[q_kSL_ind](col_val,band_m) *
                conj(BlochStates[spin][n][q_ind][comp]*EigVectors[q_kSL_ind](col_val,band_m)); 

}

overlaps_[layer] += temp_overlap.real();

}}


}


overlap_top=overlaps_[TOP_];
overlap_bottom=overlaps_[BOTTOM_];

}

Mat_1_intpair Hamiltonian::Get_k_path(int path_no){

int n1, n2;
        Mat_1_intpair k_path;
        k_path.clear();

        pair_int temp_pair;
        int L1_,L2_;
        L1_=Parameters_.moire_BZ_L1; //along G1 (b6)
        L2_=Parameters_.moire_BZ_L2; //along G2 (b2)



        if(path_no==1){
        //K+' to K-
        n1=int(2*L1_/3);
        n2=int(L2_/3);
        while(n2>=int(-L2_/3)){
            temp_pair.first = n1;
            temp_pair.second = n2;
            k_path.push_back(temp_pair);
            n2--;
            n1=int(2*L1_/L2_)*n2;
        }

        //K- to K+
        n1=int(-2*L1_/3);
        n2=int(-L2_/3);
        n2--;n1++;
        while(n1<=int(-L1_/3)){
            temp_pair.first = n1;
            temp_pair.second = n2;
            k_path.push_back(temp_pair);
            n2--;
            n1++;
        }

        //K+ to K+'
        n1=int(-L1_/3);
        n2=int(-2*L2_/3);
        n2=n2+2; //in principle it should be n2=n2+1, n1=n1+1
        n1=n1+2;
        while(n1<=int(2*L1_/3)){
            temp_pair.first = n1;
            temp_pair.second = n2;
            k_path.push_back(temp_pair);
            n2=n2+2;  //in principle it should be n2=n2+1, n1=n1+1
            n1=n1+2;
        }

        } //path no. = 1

return k_path;

}

void Hamiltonian::Copy_BlochSpectrum(Mat_4_Complex_doub &BlochStates_old, Mat_3_doub &eigvals){

     for(int spin=0;spin<2;spin++){
          for(int n=0;n<Nbands;n++){
            for(int i1=0;i1<l1_;i1++){  //k_ind
            for(int i2=0;i2<l2_;i2++){
              for(int comp=0;comp<G_grid_L1*G_grid_L2*2;comp++){
                BlochStates[spin][n][i1+i2*l1_][comp]=BlochStates_old[spin][n][i1+i2*(l1_+1)][comp];
              }
            }
            }
          }
        }



    BlochStates_old_ = BlochStates_old;
    
    for(int spin=0;spin<2;spin++){
        for(int n=0;n<Nbands;n++){
        for(int i1=0;i1<l1_;i1++){  //k_ind
        for(int i2=0;i2<l2_;i2++){
            BlochEigvals[spin][n][i1+i2*l1_]=eigvals[spin][n][i1+i2*(l1_+1)];
        }}
        }
    }

}


void Hamiltonian::Calculate_FormFactors(){


cout<<"Calculating FormFactors"<<endl;
for(int spin=0;spin<2;spin++){
for(int n1=0;n1<Nbands;n1++){
for(int n2=0;n2<Nbands;n2++){
 
 string file_FF="Lambda_k_band_spin"+to_string(spin)+"_bands"+to_string(n1)+"_"+to_string(n2)+"Version1.txt";
 ofstream fl_FF_out(file_FF.c_str());
 fl_FF_out<<"#k1(=kx+ky*l1_)   k2    Lambda[spin][n1][n2][k1][k2].real()   Lambda[spin][n1][n2][k1][k2].imag()"<<endl;

for(int k1_ind=0;k1_ind<ns_;k1_ind++){
for(int k2_ind=0;k2_ind<ns_;k2_ind++){

cout<<spin<<"(2) "<<n1<<"("<<Nbands<<") "<<n2<<"("<<Nbands<<") "<<k1_ind<<"("<<ns_<<") "<<k2_ind<<"("<<ns_<<")"<<endl;

Lambda_[spin][n1][n2][k1_ind][k2_ind]=0.0;
for(int comp=0;comp<G_grid_L1*G_grid_L2*2;comp++){
    Lambda_[spin][n1][n2][k1_ind][k2_ind] += conj(BlochStates[spin][n1][k1_ind][comp])*
                                             BlochStates[spin][n2][k2_ind][comp];
}

fl_FF_out<<k1_ind<<"  "<<k2_ind<<"  "<<Lambda_[spin][n1][n2][k1_ind][k2_ind].real()<<"  "<<Lambda_[spin][n1][n2][k1_ind][k2_ind].imag()<<endl;
}
fl_FF_out<<endl;
}
}
}
}



}


void Hamiltonian::PrintFormFactors(int band1, int band2, int spin){

string file_FF="Lambda_k_band_spin"+to_string(spin)+"_bands"+to_string(band1)+"_"+to_string(band2)+".txt";
 ofstream fl_FF_out(file_FF.c_str());
 fl_FF_out<<"#k1(=kx+ky*l1_)   k2    Lambda[spin][k1][k2].real()   Lambda[spin][k1][k2].imag()"<<endl;

int k1_ind, k2_ind;

//for(int k1_ind2=-l2_;k1_ind2<2*l2_;k1_ind2++){
//for(int k1_ind1=-l1_;k1_ind1<2*l1_;k1_ind1++){

for(int k1_ind2=0;k1_ind2<=0;k1_ind2++){
for(int k1_ind1=0;k1_ind1<=0;k1_ind1++){
// for(int k1_ind2=-6*l2_;k1_ind2<6*l2_;k1_ind2++){
// for(int k1_ind1=-6*l1_;k1_ind1<6*l1_;k1_ind1++){
k1_ind = k1_ind1 + k1_ind2*l1_*12;


for(int k2_ind2=-6*l2_;k2_ind2<6*l2_;k2_ind2++){
for(int k2_ind1=-6*l1_;k2_ind1<6*l1_;k2_ind1++){
k2_ind = k2_ind1 + k2_ind2*l1_*12;


fl_FF_out<<k1_ind<<"  "<<k1_ind1<<"  "<<k1_ind2<<"  "<<k2_ind<<"   "<<k2_ind1<<"  "<<k2_ind2<<"  "<<FormFactor(spin, band1, band2, k1_ind1, k1_ind2, k2_ind1, k2_ind2).real()<<"  "<<FormFactor(spin, band1, band2, k1_ind1, k1_ind2, k2_ind1, k2_ind2).imag()<<endl;

}
fl_FF_out<<endl;
}
fl_FF_out<<endl;
}
}

}

complex<double> Hamiltonian::FormFactor(int spin, int band1, int band2, int k1_vec_ind1,int k1_vec_ind2, int k2_vec_ind1, int k2_vec_ind2){


int G1_off1, G1_off2, G2_off1, G2_off2; 
int k1_vec_ind1_new, k1_vec_ind2_new, k2_vec_ind1_new, k2_vec_ind2_new;
int k1_ind, k2_ind;

Folding_to_BrillouinZone(k1_vec_ind1, k1_vec_ind2, k1_vec_ind1_new, k1_vec_ind2_new, G1_off1, G1_off2);
Folding_to_BrillouinZone(k2_vec_ind1, k2_vec_ind2, k2_vec_ind1_new, k2_vec_ind2_new, G2_off1, G2_off2);

k1_ind=k1_vec_ind1_new + k1_vec_ind2_new*l1_;
k2_ind=k2_vec_ind1_new + k2_vec_ind2_new*l1_;


//cout<<k1_vec_ind1<<"  "<<k1_vec_ind2<<"  "<<k1_vec_ind1_new<<"  "<<k1_vec_ind2_new<<endl;
assert(k1_ind>=0 && k1_ind<ns_);
assert(k2_ind>=0 && k2_ind<ns_);

complex<double> lambda_temp=0.0;
int comp1, comp2;
int g1_ind1, g1_ind2, g2_ind1, g2_ind2;
for(int g_ind1=0;g_ind1<G_grid_L1;g_ind1++){
for(int g_ind2=0;g_ind2<G_grid_L2;g_ind2++){
for(int layer=0;layer<2;layer++){
    
    g1_ind1= g_ind1 + G1_off1;
    g1_ind2= g_ind2 + G1_off2;
    g2_ind1= g_ind1 + G2_off1;
    g2_ind2= g_ind2 + G2_off2;

    //assert(G1_off1==0);
    //assert(G1_off2==0);
    //assert(G2_off1==0);
    //assert(G2_off2==0);


    if( (g1_ind1>=0 && g1_ind1<G_grid_L1) &&
        (g1_ind2>=0 && g1_ind2<G_grid_L2) &&
        (g2_ind1>=0 && g2_ind1<G_grid_L1) &&
        (g2_ind2>=0 && g2_ind2<G_grid_L2)
       ){
    comp1 = HamiltonianCont_.Coordinates_.Nbasis(g1_ind1, g1_ind2, layer);
    comp2 = HamiltonianCont_.Coordinates_.Nbasis(g2_ind1, g2_ind2, layer);
    lambda_temp += conj(BlochStates[spin][band1][k1_ind][comp1])*
                   BlochStates[spin][band2][k2_ind][comp2];
    }
}
}
}

return lambda_temp;
}


void Hamiltonian::PrintBlochStates(){

int comp, k_ind;
for(int spin=0;spin<2;spin++){
    for(int layer=0;layer<2;layer++){
        for(int band=0;band<Nbands;band++){
string file_Bloch="BlochState_spin"+to_string(spin)+ "_layer"+ to_string(layer) +"_band"+to_string(band)+".txt";
ofstream fl_bloch_out(file_Bloch.c_str());
fl_bloch_out<<"#k,G_1    k1   G1  k,G_2  k2   G2   u.real    u.imag"<<endl;

        for(int G_ind2=0;G_ind2<G_grid_L2;G_ind2++){
        for(int k_ind2=0;k_ind2<l2_;k_ind2++){
        

        for(int G_ind1=0;G_ind1<G_grid_L1;G_ind1++){
        for(int k_ind1=0;k_ind1<l1_;k_ind1++){
        
        
        k_ind = k_ind1 + k_ind2*l1_;
        comp = HamiltonianCont_.Coordinates_.Nbasis(G_ind1, G_ind2, layer);

        fl_bloch_out<< k_ind1 + G_ind1*l1_<<"   "<<k_ind1<<"  "<<G_ind1<<"  "<<k_ind2 + G_ind2*l2_<<"   "<<k_ind2<<"  "<<G_ind2<<"  "<<BlochStates[spin][band][k_ind][comp].real()<<"  "<<BlochStates[spin][band][k_ind][comp].imag()<<endl;
    
        }
        }
        fl_bloch_out<<endl;
        }
    }


        }
    }
}


} 

void Hamiltonian::Folding_to_BrillouinZone(int k1, int k2, int &k1_new, int &k2_new, int &G1_ind, int &G2_ind){

//k1
if(k1<0){
G1_ind = -( int((-k1-1)/l1_) + 1);
k1_new = k1 - (G1_ind*l1_); 
}
else{
G1_ind = int(k1/l1_);
k1_new = k1%l1_;
}

//k1
if(k2<0){
G2_ind = -( int((-k2-1)/l2_) + 1);
k2_new = k2 - (G2_ind*l2_); 
}
else{
G2_ind = int(k2/l2_);
k2_new = k2%l2_;
}


}





#endif
