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


extern "C" void dgesdd_ (char *, int *, int *, double *, int *, double *, double *, int *, double *, int*,
                         double *, int *, int *, int *);


extern "C" void zgesdd_ (char *, int *, int *, std::complex<double> *, int *, double *, std::complex<double> *, int *, std::complex<double> *, int*,
                         std::complex<double> *, int *, double * , int *, int *);


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
    void PrintFormFactors2(int band1, int band2, int spin);
    void PrintBlochStates_old();
    complex<double> Interaction_value(int spin, int spin_p, int band1, int band2, int band3, int band4, int k1_ind, int k2_ind, int q_ind);
    complex<double> Interaction_value_new(int spin, int spin_p, int band1, int band2, int band3, int band4, int k1_ind, int k2_ind, int q_ind);
    void Create_Amat_and_Bmat();
    void Create_Hbar_and_Fbar();
    void Save_InteractionVal();
    double V_int(double q_val);
    void Print_Interaction_value();
    void Print_Interaction_value2(int k1_ind, int k2_ind);
    void Print_Interaction_value3();
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
    void Update_OrderParameters_AndersonMixing(int iter);
    void Perform_SVD_complex(Matrix<complex<double>> & A_, Matrix<complex<double>> & VT_, Matrix<complex<double>> & U_, vector<double> & Sigma_);
    void Perform_SVD(Matrix<double> & A_, Matrix<double> & VT_, Matrix<double> & U_, vector<double> & Sigma_);
    void Update_OParams_SimpleMixing();
    double chemicalpotential(double muin,double Particles);
    void Get_max_and_min_eigvals();
    double Myrandom();
    void Initialize_OParams();
    void Kick_OParams(double kick);
    void Print_Spectrum(int kset_ind, string filename);
    void Update_Hartree_Coefficients();
    void Update_Fock_Coefficients();
    void Calculate_Total_Spin();
    void Calculate_Total_Energy();
    double DispersionTriangularLattice(int k_ind);
    double Lorentzian(double eta, double x);
    void Print_SPDOS(string filename);
    void Calculate_RealSpace_OParams(string filename, string filename2);
    void Calculate_RealSpace_OParams_new(string filename);
    void Calculate_RealSpace_OParams_new2(string filename);
    void Calculate_RealSpace_OParams_new3(string filename);
    void Calculate_RealSpace_OParams_important_positions(string filename2);
    void Calculate_RealSpace_OParams_important_positions_new(string filename);
    void Calculate_RealSpace_OParams_important_positions_new3(string filename);
    void Print_HF_Bands();
    Mat_1_intpair Get_k_path(int path_no);
    void Get_layer_overlaps(double &overlap_top, double &overlap_bottom, int band, int spin, int q_ind1, int q_ind2);
    void Saving_BlochState_Overlaps();
    void Saving_PBZ_BlochState_Overlaps();
    void Calculate_ChernNumbers_HFBands();
    void Calculate_layer_resolved_densities();
    void Write_ordered_spectrum(string filename);
    void Imposing_ZeroSz();
    void Create_PMat();
    void PrintBlochStates();
    void PrintBlochStatesPBZ();
    void Print_HF_Band_Projected_Interaction();
    void Create_Lambda_PBZ();
    bool Present(string type, pair_int k_temp);
    void PrintFormFactors_PBZ(int band1, int band2, int spin);
    int Get_maximum_comp(Mat_1_Complex_doub Vec_);
    void Print_HF_Band_Projected_Interaction_TR_and_Inversion_imposed();
    void Calculate_NonAbelianChernNumbers_HFBands();
    complex<double> Get_U_mat_NonAbelian(int mx_left, int my_left, int mx_right, int my_right, Mat_1_int Bands_);
    complex<double> Determinant(Matrix<complex<double>> & A_);
    void Calculate_Band_Projector(Mat_1_int Bands, int nx_, int ny_, Mat_2_Complex_doub & Projector_full);
    void Calculate_QuantumGeometry_using_Projectors();
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
    Mat_9_Complex_doub LambdaNew_;
    Mat_9_Complex_doub LambdaPBZ_k1_m_q, LambdaPBZ_k2_p_q;
    int Lambda_G_grid_L1_min, Lambda_G_grid_L1_max, Lambda_G1_grid_size;
    int Lambda_G_grid_L2_min, Lambda_G_grid_L2_max, Lambda_G2_grid_size;
    Mat_9_Complex_doub Bmat, Amat;
    Mat_9_Complex_doub Xmat;
    Mat_7_Complex_doub Omat;

    Mat_8_Complex_doub Pmat;

    Mat_4_Complex_doub Hbar, Fbar;

    Mat_9_Complex_doub Interaction_val;
    //(int spin, int spin_p, int band1, int band2, int band3, int band4, int k1_ind, int k2_ind, int q_ind);


    Mat_4_Complex_doub BlochStates, BlochStates_old_; //[valley(spin)][band][k][G,l]
    Mat_3_doub BlochEigvals; //[valley(spin)][band][k]
    Mat_6_Complex_doub BO, BO_PBZ; //[band][spin][k][band'][spin'][k']


    Mat_2_Complex_doub N_layer_tau;

    complex<double> Total_QuantEnergy, Total_ClassEnergy;

    Mat_1_doub Eigenvalues_ordered;

    Mat_1_intpair Possible_k1_m_q, Possible_k2_p_q;

    //Mat_3_Complex_doub Projector_band_resolved; //[m][comp][comp]
    //Mat_2_Complex_doub Projector_full; //[comp][comp]

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


    //Declarations for Anderson Mixing
    Mat_1_doub x_km1_, x_k_, Del_x_km1;
    Mat_1_doub f_k_, f_km1_, Del_f_km1;
    Mat_1_doub xbar_k_, fbar_k_, gamma_k_, x_kp1_;
    Matrix<double> X_mat, F_mat;

};





void Hamiltonian::Write_ordered_spectrum(string filename){

    ofstream fileout(filename.c_str());

    double value_;
    Mat_1_Complex_doub Vec_temp;
    Vec_temp.resize(6);

    fileout<<"#index  Eigenvalue"<<endl;
    Eigenvalues_ordered.clear();
    for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
        for(int i=0;i<EigValues[kSL_ind].size();i++){
       Eigenvalues_ordered.push_back(EigValues[kSL_ind][i]);
        }}


   //in increasing order
    for(int i=0;i<Eigenvalues_ordered.size();i++){
        for(int j=i+1;j<Eigenvalues_ordered.size();j++){
            if(Eigenvalues_ordered[j]<Eigenvalues_ordered[i]){
                value_=Eigenvalues_ordered[i];
                Eigenvalues_ordered[i]=Eigenvalues_ordered[j];
                Eigenvalues_ordered[j]=value_;
            }
        }
    }

    for(int i=0;i<Eigenvalues_ordered.size();i++){
        fileout<<i<<"  "<<Eigenvalues_ordered[i]<<endl;
    }


}




void Hamiltonian::Calculate_RealSpace_OParams_important_positions_new3(string filename){



    int M1, M2; //no. of slices of one moire unit cell
    M1=10;M2=10;
    int m1_max, m1_min, m2_max, m2_min; //range for full 2d lattice
    m1_max = M1*l1_;m1_min=0;
    m2_max = M2*l1_;m2_min=0;


    double rx,ry;


    Mat_8_Complex_doub Omat;
    Omat.resize(2);//spin
    for(int spin1=0;spin1<2;spin1++){
    Omat[spin1].resize(2);
    for(int spin2=0;spin2<2;spin2++){
    Omat[spin1][spin2].resize(Parameters_.max_layer_ind);
    for(int layer1=0;layer1<Parameters_.max_layer_ind;layer1++){
    Omat[spin1][spin2][layer1].resize(Parameters_.max_layer_ind);
    for(int layer2=0;layer2<Parameters_.max_layer_ind;layer2++){
    Omat[spin1][spin2][layer1][layer2].resize(2*l1_-1); //-(l1-1)....(l1-1)
    for(int kprime1_ind=0;kprime1_ind<2*l1_-1;kprime1_ind++){
    Omat[spin1][spin2][layer1][layer2][kprime1_ind].resize(2*l2_-1);//-(l2-1)....(l2-1)
    for(int kprime2_ind=0;kprime2_ind<2*l2_-1;kprime2_ind++){
    Omat[spin1][spin2][layer1][layer2][kprime1_ind][kprime2_ind].resize(2*G_grid_L1-1);//-(L1-1)...(L1-1)
    for(int Gprime1_ind=0;Gprime1_ind<2*G_grid_L1-1;Gprime1_ind++){
    Omat[spin1][spin2][layer1][layer2][kprime1_ind][kprime2_ind][Gprime1_ind].resize(2*G_grid_L2-1);//-(L2-1)...(L2-1)
    }}}}}}}



    int kprime1_;
    int kprime2_;
    int Gprime1_;
    int Gprime2_;
    int k1_min, k2_min, k1_size;
    int k1_max, k2_max, k2_size;
    int G1_min, G2_min, G1_size;
    int G1_max, G2_max, G2_size;

    int k_minus_kprime_1;
    int k_minus_kprime_2;
    int G_minus_Gprime_1;
    int G_minus_Gprime_2;


    for(int spin1=0;spin1<2;spin1++){
    for(int spin2=0;spin2<2;spin2++){
    for(int layer1=0;layer1<Parameters_.max_layer_ind;layer1++){
    for(int layer2=0;layer2<Parameters_.max_layer_ind;layer2++){
    for(int kprime1_ind=0;kprime1_ind<2*l1_-1;kprime1_ind++){
    for(int kprime2_ind=0;kprime2_ind<2*l2_-1;kprime2_ind++){
    for(int Gprime1_ind=0;Gprime1_ind<2*G_grid_L1-1;Gprime1_ind++){
    for(int Gprime2_ind=0;Gprime2_ind<2*G_grid_L2-1;Gprime2_ind++){

    Omat[spin1][spin2][layer1][layer2][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind]=0.0;


    kprime1_ = kprime1_ind - (l1_-1);
    kprime2_ = kprime2_ind - (l2_-1);
    Gprime1_ = Gprime1_ind - (G_grid_L1-1);
    Gprime2_ = Gprime2_ind - (G_grid_L2-1);

    if(kprime1_<0){k1_min=0;k1_max=(l1_-1)-abs(kprime1_);}
    else{k1_min=kprime1_;k1_max=(l1_-1);}

    if(kprime2_<0){k2_min=0;k2_max=(l2_-1)-abs(kprime2_);}
    else{k2_min=kprime2_;k2_max=(l2_-1);}

    if(Gprime1_<0){G1_min=0;G1_max=(G_grid_L1-1)-abs(Gprime1_);}
    else{G1_min=Gprime1_;G1_max=(G_grid_L1-1);}

    if(Gprime2_<0){G2_min=0;G2_max=(G_grid_L2-1)-abs(Gprime2_);}
    else{G2_min=Gprime2_;G2_max=(G_grid_L2-1);}

    k1_size = k1_max - k1_min +1;
    k2_size = k2_max - k2_min +1;
    G1_size = G1_max - G1_min +1;
    G2_size = G2_max - G2_min +1;


    for(int k1_val=k1_min;k1_val<=k1_max;k1_val++){
        for(int k2_val=k2_min;k2_val<=k2_max;k2_val++){
        k_minus_kprime_1 = k1_val - kprime1_;
        k_minus_kprime_2 = k2_val - kprime2_;

        assert(k_minus_kprime_1>=0 && k_minus_kprime_1<l1_);
        assert(k_minus_kprime_2>=0 && k_minus_kprime_2<l2_);

        int k_row = Coordinates_.Ncell(k1_val,k2_val);
        int k_col = Coordinates_.Ncell(k_minus_kprime_1, k_minus_kprime_2);

        bool allow=(Inverse_kSublattice_mapping[k_row].first==Inverse_kSublattice_mapping[k_col].first);
        int kSL_ind = Inverse_kSublattice_mapping[k_row].first;
        int k_row_ind = Inverse_kSublattice_mapping[k_row].second;
        int k_col_ind = Inverse_kSublattice_mapping[k_col].second;

        if(allow){
            for(int G1_val=G1_min;G1_val<=G1_max;G1_val++){
                for(int G2_val=G2_min;G2_val<=G2_max;G2_val++){
                    G_minus_Gprime_1 = G1_val - Gprime1_;
                    G_minus_Gprime_2 = G2_val - Gprime2_;

                    assert(G_minus_Gprime_1>=0 && G_minus_Gprime_1<G_grid_L1);
                    assert(G_minus_Gprime_2>=0 && G_minus_Gprime_2<G_grid_L2);

                    int comp1 = HamiltonianCont_.Coordinates_.Nbasis(G1_val, G2_val, layer1);
                    int comp2 = HamiltonianCont_.Coordinates_.Nbasis(G_minus_Gprime_1, G_minus_Gprime_2, layer2);

                    for(int band1=0;band1<Nbands;band1++){
                        for(int band2=0;band2<Nbands;band2++){

                     int row_val = k_row_ind +
                                      k_sublattices[kSL_ind].size()*band1 +
                                      k_sublattices[kSL_ind].size()*Nbands*spin1;
                     int col_val = k_col_ind +
                                      k_sublattices[kSL_ind].size()*band2 +
                                      k_sublattices[kSL_ind].size()*Nbands*spin2;

        Omat[spin1][spin2][layer1][layer2][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind]+=
                    (conj(BlochStates[spin1][band1][k_row][comp1])*BlochStates[spin2][band2][k_col][comp2]*
                    OParams[kSL_ind](row_val,col_val));


                        }
                    }
                }
            }
        }//bool allow
        }
    }


    }}}}}}}}



    complex<double> density_, Sz_, Sx_, Sy_;

    for(int layer=0;layer<Parameters_.max_layer_ind;layer++){
    string filename_new = "layer_"+to_string(layer)+"_"+filename;
    ofstream fileout(filename_new.c_str());
    fileout<<"# m1  m2  rx  ry   density   sz  sx  sy"<<endl;

    for(int m1=m1_min;m1<=m1_max;m1++){
    for(int m2=m2_min;m2<=m2_max;m2++){
    if(m1%M1==0  && m2%M2==0){

    rx = (  ((m1*1.0)/(M1*1.0)) + ((m2*1.0)/(M2*1.0)) )*sqrt(3.0)*0.5;
    ry = (  -((m1*1.0)/(M1*1.0)) + ((m2*1.0)/(M2*1.0)) )*0.5;

    density_ =0.0;
    Sz_ =0.0;
    Sx_=0.0;
    Sy_=0.0;

    for(int kprime1_ind=0;kprime1_ind<2*l1_-1;kprime1_ind++){
    for(int kprime2_ind=0;kprime2_ind<2*l2_-1;kprime2_ind++){
    for(int Gprime1_ind=0;Gprime1_ind<2*G_grid_L1-1;Gprime1_ind++){
    for(int Gprime2_ind=0;Gprime2_ind<2*G_grid_L2-1;Gprime2_ind++){


    kprime1_ = kprime1_ind - (l1_-1);
    kprime2_ = kprime2_ind - (l2_-1);
    Gprime1_ = Gprime1_ind - (G_grid_L1-1);
    Gprime2_ = Gprime2_ind - (G_grid_L2-1);

    density_ += (1.0/Area)*(
                exp(iota_complex*2.0*PI*(  (m1*(kprime1_)*1.0)/(M1*l1_*1.0)   +  (m2*(kprime2_)*1.0)/(M2*l2_*1.0)  ))
                *
                exp(iota_complex*2.0*PI*(  (m1*(Gprime1_)*1.0)/(M1*1.0)   +  (m2*(Gprime2_)*1.0)/(M2*1.0)  ))
                *
                (Omat[0][0][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind] +
                Omat[1][1][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind])
                );

    Sz_ += (0.5/Area)*(
                exp(iota_complex*2.0*PI*(  (m1*(kprime1_)*1.0)/(M1*l1_*1.0)   +  (m2*(kprime2_)*1.0)/(M2*l2_*1.0)  ))
                *
                exp(iota_complex*2.0*PI*(  (m1*(Gprime1_)*1.0)/(M1*1.0)   +  (m2*(Gprime2_)*1.0)/(M2*1.0)  ))
                *
                (Omat[0][0][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind] -
                Omat[1][1][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind])
                );

    Sx_ += (0.5/Area)*(
                exp(iota_complex*2.0*PI*(  (m1*(kprime1_)*1.0)/(M1*l1_*1.0)   +  (m2*(kprime2_)*1.0)/(M2*l2_*1.0)  ))
                *
                exp(iota_complex*2.0*PI*(  (m1*(Gprime1_)*1.0)/(M1*1.0)   +  (m2*(Gprime2_)*1.0)/(M2*1.0)  ))
                *
                (Omat[0][1][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind] +
                Omat[1][0][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind])
                );
    Sy_ += ((-0.5*iota_complex)/Area)*(
                exp(iota_complex*2.0*PI*(  (m1*(kprime1_)*1.0)/(M1*l1_*1.0)   +  (m2*(kprime2_)*1.0)/(M2*l2_*1.0)  ))
                *
                exp(iota_complex*2.0*PI*(  (m1*(Gprime1_)*1.0)/(M1*1.0)   +  (m2*(Gprime2_)*1.0)/(M2*1.0)  ))
                *
                (Omat[0][1][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind] -
                Omat[1][0][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind])
                );

    }}}}

    fileout<<m1<<"  "<<m2<<"  "<<rx<<"  "<<ry<<"  "<<density_.real()<<"  "<<Sz_.real()<<"  "<<Sx_.real()<<"  "<<Sy_.real()<<
    "  "<<density_.imag()<<"  "<<Sz_.imag()<<"  "<<Sx_.imag()<<"  "<<Sy_.imag()<<endl;

    }

    }
    if(m1%M1==0){
    fileout<<endl;}
    }}
}

void Hamiltonian::Calculate_RealSpace_OParams_important_positions_new(string filename){



    int M1, M2; //no. of slices of one moire unit cell
    M1=10;M2=10;
    int m1_max, m1_min, m2_max, m2_min; //range for full 2d lattice
    m1_max = M1*l1_;m1_min=0;
    m2_max = M2*l1_;m2_min=0;


    double rx,ry;


    Mat_8_Complex_doub Omat;
    Omat.resize(2);//spin
    for(int spin1=0;spin1<2;spin1++){
    Omat[spin1].resize(2);
    for(int spin2=0;spin2<2;spin2++){
    Omat[spin1][spin2].resize(Parameters_.max_layer_ind);
    for(int layer1=0;layer1<Parameters_.max_layer_ind;layer1++){
    Omat[spin1][spin2][layer1].resize(Parameters_.max_layer_ind);
    for(int layer2=0;layer2<Parameters_.max_layer_ind;layer2++){
    Omat[spin1][spin2][layer1][layer2].resize(2*l1_-1); //-(l1-1)....(l1-1)
    for(int kprime1_ind=0;kprime1_ind<2*l1_-1;kprime1_ind++){
    Omat[spin1][spin2][layer1][layer2][kprime1_ind].resize(2*l2_-1);//-(l2-1)....(l2-1)
    for(int kprime2_ind=0;kprime2_ind<2*l2_-1;kprime2_ind++){
    Omat[spin1][spin2][layer1][layer2][kprime1_ind][kprime2_ind].resize(2*G_grid_L1-1);//-(L1-1)...(L1-1)
    for(int Gprime1_ind=0;Gprime1_ind<2*G_grid_L1-1;Gprime1_ind++){
    Omat[spin1][spin2][layer1][layer2][kprime1_ind][kprime2_ind][Gprime1_ind].resize(2*G_grid_L2-1);//-(L2-1)...(L2-1)
    }}}}}}}



    int kprime1_;
    int kprime2_;
    int Gprime1_;
    int Gprime2_;
    int k1_min, k2_min, k1_size;
    int k1_max, k2_max, k2_size;
    int G1_min, G2_min, G1_size;
    int G1_max, G2_max, G2_size;

    int k_minus_kprime_1;
    int k_minus_kprime_2;
    int G_minus_Gprime_1;
    int G_minus_Gprime_2;


    for(int spin1=0;spin1<2;spin1++){
    for(int spin2=0;spin2<2;spin2++){
    for(int layer1=0;layer1<Parameters_.max_layer_ind;layer1++){
    for(int layer2=0;layer2<Parameters_.max_layer_ind;layer2++){
    for(int kprime1_ind=0;kprime1_ind<2*l1_-1;kprime1_ind++){
    for(int kprime2_ind=0;kprime2_ind<2*l2_-1;kprime2_ind++){
    for(int Gprime1_ind=0;Gprime1_ind<2*G_grid_L1-1;Gprime1_ind++){
    for(int Gprime2_ind=0;Gprime2_ind<2*G_grid_L2-1;Gprime2_ind++){

    Omat[spin1][spin2][layer1][layer2][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind]=0.0;


    kprime1_ = kprime1_ind - (l1_-1);
    kprime2_ = kprime2_ind - (l2_-1);
    Gprime1_ = Gprime1_ind - (G_grid_L1-1);
    Gprime2_ = Gprime2_ind - (G_grid_L2-1);

    if(kprime1_<0){k1_min=0;k1_max=(l1_-1)-abs(kprime1_);}
    else{k1_min=kprime1_;k1_max=(l1_-1);}

    if(kprime2_<0){k2_min=0;k2_max=(l2_-1)-abs(kprime2_);}
    else{k2_min=kprime2_;k2_max=(l2_-1);}

    if(Gprime1_<0){G1_min=0;G1_max=(G_grid_L1-1)-abs(Gprime1_);}
    else{G1_min=Gprime1_;G1_max=(G_grid_L1-1);}

    if(Gprime2_<0){G2_min=0;G2_max=(G_grid_L2-1)-abs(Gprime2_);}
    else{G2_min=Gprime2_;G2_max=(G_grid_L2-1);}

    k1_size = k1_max - k1_min +1;
    k2_size = k2_max - k2_min +1;
    G1_size = G1_max - G1_min +1;
    G2_size = G2_max - G2_min +1;


    for(int k1_val=k1_min;k1_val<=k1_max;k1_val++){
        for(int k2_val=k2_min;k2_val<=k2_max;k2_val++){
        k_minus_kprime_1 = k1_val - kprime1_;
        k_minus_kprime_2 = k2_val - kprime2_;

        assert(k_minus_kprime_1>=0 && k_minus_kprime_1<l1_);
        assert(k_minus_kprime_2>=0 && k_minus_kprime_2<l2_);

        int k_row = Coordinates_.Ncell(k1_val,k2_val);
        int k_col = Coordinates_.Ncell(k_minus_kprime_1, k_minus_kprime_2);

        bool allow=(Inverse_kSublattice_mapping[k_row].first==Inverse_kSublattice_mapping[k_col].first);
        int kSL_ind = Inverse_kSublattice_mapping[k_row].first;

        if(allow){
            for(int G1_val=G1_min;G1_val<=G1_max;G1_val++){
                for(int G2_val=G2_min;G2_val<=G2_max;G2_val++){
                    G_minus_Gprime_1 = G1_val - Gprime1_;
                    G_minus_Gprime_2 = G2_val - Gprime2_;

                    assert(G_minus_Gprime_1>=0 && G_minus_Gprime_1<G_grid_L1);
                    assert(G_minus_Gprime_2>=0 && G_minus_Gprime_2<G_grid_L2);

                    int comp1 = HamiltonianCont_.Coordinates_.Nbasis(G1_val, G2_val, layer1);
                    int comp2 = HamiltonianCont_.Coordinates_.Nbasis(G_minus_Gprime_1, G_minus_Gprime_2, layer2);

                    for(int band1=0;band1<Nbands;band1++){
                        for(int band2=0;band2<Nbands;band2++){

                     int row_val = k_row +
                                      k_sublattices[kSL_ind].size()*band1 +
                                      k_sublattices[kSL_ind].size()*Nbands*spin1;
                     int col_val = k_col +
                                      k_sublattices[kSL_ind].size()*band2 +
                                      k_sublattices[kSL_ind].size()*Nbands*spin2;

        Omat[spin1][spin2][layer1][layer2][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind]+=
                    (conj(BlochStates[spin1][band1][k_row][comp1])*BlochStates[spin2][band2][k_col][comp2]*
                    OParams[kSL_ind](row_val,col_val));


                        }
                    }
                }
            }
        }//bool allow
        }
    }


    }}}}}}}}



    complex<double> density_, Sz_, Sx_, Sy_;

    for(int layer=0;layer<Parameters_.max_layer_ind;layer++){
    string filename_new = "layer_"+to_string(layer)+"_"+filename;
    ofstream fileout(filename_new.c_str());
    fileout<<"# m1  m2  rx  ry   density   sz  sx  sy"<<endl;

    for(int m1=m1_min;m1<=m1_max;m1++){
    for(int m2=m2_min;m2<=m2_max;m2++){
    if(m1%M1==0  && m2%M2==0){

    rx = (  ((m1*1.0)/(M1*1.0)) + ((m2*1.0)/(M2*1.0)) )*sqrt(3.0)*0.5;
    ry = (  -((m1*1.0)/(M1*1.0)) + ((m2*1.0)/(M2*1.0)) )*0.5;

    density_ =0.0;
    Sz_ =0.0;
    Sx_=0.0;
    Sy_=0.0;

    for(int kprime1_ind=0;kprime1_ind<2*l1_-1;kprime1_ind++){
    for(int kprime2_ind=0;kprime2_ind<2*l2_-1;kprime2_ind++){
    for(int Gprime1_ind=0;Gprime1_ind<2*G_grid_L1-1;Gprime1_ind++){
    for(int Gprime2_ind=0;Gprime2_ind<2*G_grid_L2-1;Gprime2_ind++){


    kprime1_ = kprime1_ind - (l1_-1);
    kprime2_ = kprime2_ind - (l2_-1);
    Gprime1_ = Gprime1_ind - (G_grid_L1-1);
    Gprime2_ = Gprime2_ind - (G_grid_L2-1);

    density_ += (1.0/Area)*(
                exp(iota_complex*2.0*PI*(  (m1*(kprime1_)*1.0)/(M1*l1_*1.0)   +  (m2*(kprime2_)*1.0)/(M2*l2_*1.0)  ))
                *
                exp(iota_complex*2.0*PI*(  (m1*(Gprime1_)*1.0)/(M1*1.0)   +  (m2*(Gprime2_)*1.0)/(M2*1.0)  ))
                *
                (Omat[0][0][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind] +
                Omat[1][1][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind])
                );

    Sz_ += (0.5/Area)*(
                exp(iota_complex*2.0*PI*(  (m1*(kprime1_)*1.0)/(M1*l1_*1.0)   +  (m2*(kprime2_)*1.0)/(M2*l2_*1.0)  ))
                *
                exp(iota_complex*2.0*PI*(  (m1*(Gprime1_)*1.0)/(M1*1.0)   +  (m2*(Gprime2_)*1.0)/(M2*1.0)  ))
                *
                (Omat[0][0][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind] -
                Omat[1][1][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind])
                );

    Sx_ += (0.5/Area)*(
                exp(iota_complex*2.0*PI*(  (m1*(kprime1_)*1.0)/(M1*l1_*1.0)   +  (m2*(kprime2_)*1.0)/(M2*l2_*1.0)  ))
                *
                exp(iota_complex*2.0*PI*(  (m1*(Gprime1_)*1.0)/(M1*1.0)   +  (m2*(Gprime2_)*1.0)/(M2*1.0)  ))
                *
                (Omat[0][1][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind] +
                Omat[1][0][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind])
                );
    Sy_ += ((-0.5*iota_complex)/Area)*(
                exp(iota_complex*2.0*PI*(  (m1*(kprime1_)*1.0)/(M1*l1_*1.0)   +  (m2*(kprime2_)*1.0)/(M2*l2_*1.0)  ))
                *
                exp(iota_complex*2.0*PI*(  (m1*(Gprime1_)*1.0)/(M1*1.0)   +  (m2*(Gprime2_)*1.0)/(M2*1.0)  ))
                *
                (Omat[0][1][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind] -
                Omat[1][0][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind])
                );

    }}}}


    //fileout<<m1/M1<<"  "<<m2/M2<<"  "<<m1<<"  "<<m2<<"  "<<rx<<"  "<<ry<<"  "<<Area*density_.real()<<"  "<<Area*Sz_.real()<<"  "<<Area*Sx_.real()<<"  "<<Area*Sy_.real()<<
   // "  "<<Area*density_.imag()<<"  "<<Area*Sz_.imag()<<"  "<<Area*Sx_.imag()<<"  "<<Area*Sy_.imag()<<endl;

    fileout<<m1/M1<<"  "<<m2/M2<<"  "<<m1<<"  "<<m2<<"  "<<rx<<"  "<<ry<<"  "<<density_.real()<<"  "<<Sz_.real()<<"  "<<Sx_.real()<<"  "<<Sy_.real()<<
    "  "<<density_.imag()<<"  "<<Sz_.imag()<<"  "<<Sx_.imag()<<"  "<<Sy_.imag()<<endl;


    }//if
    }//m2
    if(m1%M1==0){
    fileout<<endl;
    }

    }}
}

void Hamiltonian::Calculate_RealSpace_OParams_important_positions(string filename2){

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
string  filename2_new;
for(int layer=0;layer<Parameters_.max_layer_ind;layer++){
filename2_new = "layer_"+to_string(layer)+"_"+filename2;
ofstream fileout2(filename2_new.c_str());
fileout2<<"#cell1  cell2   m1  m2  rx  ry   density   sz  sx  sy"<<endl;

l1=layer;
l2=layer;
complex<double> density_, Sz_, Sx_, Sy_;
for(int m1=m1_min;m1<=m1_max;m1++){
for(int m2=m2_min;m2<=m2_max;m2++){
if(m1%M1==0  && m2%M2==0){

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


fileout2<<m1/M1<<"  "<<m2/M2<<"  "<<m1<<"  "<<m2<<"  "<<rx<<"  "<<ry<<"  "<<Area*density_.real()<<"  "<<Area*Sz_.real()<<"  "<<Area*Sx_.real()<<"  "<<Area*Sy_.real()<<
"  "<<Area*density_.imag()<<"  "<<Area*Sz_.imag()<<"  "<<Area*Sx_.imag()<<"  "<<Area*Sy_.imag()<<endl;

}

}
if(m1%M1==0){
fileout2<<endl;
}

}

}


}



void Hamiltonian::Calculate_RealSpace_OParams_new(string filename){

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
string filename_new;
for(int layer=0;layer<Parameters_.max_layer_ind;layer++){
filename_new = "layer_"+to_string(layer)+"_"+filename;
ofstream fileout(filename_new.c_str());
fileout<<"# m1  m2  rx  ry   density   sz  sx  sy"<<endl;

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
}

fileout<<endl;

}

}


}


void Hamiltonian::Calculate_RealSpace_OParams_new3(string filename){



    int M1, M2; //no. of slices of one moire unit cell
    M1=12;M2=12;
    int m1_max, m1_min, m2_max, m2_min; //range for full 2d lattice
    m1_max = M1*l1_;m1_min=0;
    m2_max = M2*l1_;m2_min=0;


    double rx,ry;


    Mat_8_Complex_doub Omat;
    Omat.resize(2);//spin
    for(int spin1=0;spin1<2;spin1++){
    Omat[spin1].resize(2);
    for(int spin2=0;spin2<2;spin2++){
    Omat[spin1][spin2].resize(Parameters_.max_layer_ind);
    for(int layer1=0;layer1<Parameters_.max_layer_ind;layer1++){
    Omat[spin1][spin2][layer1].resize(Parameters_.max_layer_ind);
    for(int layer2=0;layer2<Parameters_.max_layer_ind;layer2++){
    Omat[spin1][spin2][layer1][layer2].resize(2*l1_-1); //-(l1-1)....(l1-1)
    for(int kprime1_ind=0;kprime1_ind<2*l1_-1;kprime1_ind++){
    Omat[spin1][spin2][layer1][layer2][kprime1_ind].resize(2*l2_-1);//-(l2-1)....(l2-1)
    for(int kprime2_ind=0;kprime2_ind<2*l2_-1;kprime2_ind++){
    Omat[spin1][spin2][layer1][layer2][kprime1_ind][kprime2_ind].resize(2*G_grid_L1-1);//-(L1-1)...(L1-1)
    for(int Gprime1_ind=0;Gprime1_ind<2*G_grid_L1-1;Gprime1_ind++){
    Omat[spin1][spin2][layer1][layer2][kprime1_ind][kprime2_ind][Gprime1_ind].resize(2*G_grid_L2-1);//-(L2-1)...(L2-1)
    }}}}}}}



    int kprime1_;
    int kprime2_;
    int Gprime1_;
    int Gprime2_;
    int k1_min, k2_min, k1_size;
    int k1_max, k2_max, k2_size;
    int G1_min, G2_min, G1_size;
    int G1_max, G2_max, G2_size;

    int k_minus_kprime_1;
    int k_minus_kprime_2;
    int G_minus_Gprime_1;
    int G_minus_Gprime_2;


    for(int spin1=0;spin1<2;spin1++){
    for(int spin2=0;spin2<2;spin2++){
    for(int layer1=0;layer1<Parameters_.max_layer_ind;layer1++){
    for(int layer2=0;layer2<Parameters_.max_layer_ind;layer2++){
    for(int kprime1_ind=0;kprime1_ind<2*l1_-1;kprime1_ind++){
    for(int kprime2_ind=0;kprime2_ind<2*l2_-1;kprime2_ind++){
    for(int Gprime1_ind=0;Gprime1_ind<2*G_grid_L1-1;Gprime1_ind++){
    for(int Gprime2_ind=0;Gprime2_ind<2*G_grid_L2-1;Gprime2_ind++){

    Omat[spin1][spin2][layer1][layer2][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind]=0.0;


    kprime1_ = kprime1_ind - (l1_-1);
    kprime2_ = kprime2_ind - (l2_-1);
    Gprime1_ = Gprime1_ind - (G_grid_L1-1);
    Gprime2_ = Gprime2_ind - (G_grid_L2-1);

    if(kprime1_<0){k1_min=0;k1_max=(l1_-1)-abs(kprime1_);}
    else{k1_min=kprime1_;k1_max=(l1_-1);}

    if(kprime2_<0){k2_min=0;k2_max=(l2_-1)-abs(kprime2_);}
    else{k2_min=kprime2_;k2_max=(l2_-1);}

    if(Gprime1_<0){G1_min=0;G1_max=(G_grid_L1-1)-abs(Gprime1_);}
    else{G1_min=Gprime1_;G1_max=(G_grid_L1-1);}

    if(Gprime2_<0){G2_min=0;G2_max=(G_grid_L2-1)-abs(Gprime2_);}
    else{G2_min=Gprime2_;G2_max=(G_grid_L2-1);}

    k1_size = k1_max - k1_min +1;
    k2_size = k2_max - k2_min +1;
    G1_size = G1_max - G1_min +1;
    G2_size = G2_max - G2_min +1;


    for(int k1_val=k1_min;k1_val<=k1_max;k1_val++){
        for(int k2_val=k2_min;k2_val<=k2_max;k2_val++){
        k_minus_kprime_1 = k1_val - kprime1_;
        k_minus_kprime_2 = k2_val - kprime2_;

        assert(k_minus_kprime_1>=0 && k_minus_kprime_1<l1_);
        assert(k_minus_kprime_2>=0 && k_minus_kprime_2<l2_);

        int k_row = Coordinates_.Ncell(k1_val,k2_val);
        int k_col = Coordinates_.Ncell(k_minus_kprime_1, k_minus_kprime_2);

        bool allow=(Inverse_kSublattice_mapping[k_row].first==Inverse_kSublattice_mapping[k_col].first);
        int kSL_ind = Inverse_kSublattice_mapping[k_row].first;
        int k_row_ind = Inverse_kSublattice_mapping[k_row].second;
        int k_col_ind = Inverse_kSublattice_mapping[k_col].second;

        if(allow){
            for(int G1_val=G1_min;G1_val<=G1_max;G1_val++){
                for(int G2_val=G2_min;G2_val<=G2_max;G2_val++){
                    G_minus_Gprime_1 = G1_val - Gprime1_;
                    G_minus_Gprime_2 = G2_val - Gprime2_;

                    assert(G_minus_Gprime_1>=0 && G_minus_Gprime_1<G_grid_L1);
                    assert(G_minus_Gprime_2>=0 && G_minus_Gprime_2<G_grid_L2);

                    int comp1 = HamiltonianCont_.Coordinates_.Nbasis(G1_val, G2_val, layer1);
                    int comp2 = HamiltonianCont_.Coordinates_.Nbasis(G_minus_Gprime_1, G_minus_Gprime_2, layer2);

                    for(int band1=0;band1<Nbands;band1++){
                        for(int band2=0;band2<Nbands;band2++){

                     int row_val = k_row_ind +
                                      k_sublattices[kSL_ind].size()*band1 +
                                      k_sublattices[kSL_ind].size()*Nbands*spin1;
                     int col_val = k_col_ind +
                                      k_sublattices[kSL_ind].size()*band2 +
                                      k_sublattices[kSL_ind].size()*Nbands*spin2;

        Omat[spin1][spin2][layer1][layer2][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind]+=
                    (conj(BlochStates[spin1][band1][k_row][comp1])*BlochStates[spin2][band2][k_col][comp2]*
                    OParams[kSL_ind](row_val,col_val));


                        }
                    }
                }
            }
        }//bool allow
        }
    }


    }}}}}}}}


    complex<double> Sum_density, Sum_Sz, Sum_Sx, Sum_Sy;

    complex<double> density_, Sz_, Sx_, Sy_;


    Sum_density=0; Sum_Sz=0; Sum_Sx=0; Sum_Sy=0;
    for(int layer=0;layer<Parameters_.max_layer_ind;layer++){
    string filename_new = "layer_"+to_string(layer)+"_"+filename;
    ofstream fileout(filename_new.c_str());
    fileout<<"# m1  m2  rx  ry   density   sz  sx  sy"<<endl;

    for(int m1=m1_min;m1<=m1_max;m1++){
    for(int m2=m2_min;m2<=m2_max;m2++){

    rx = (  ((m1*1.0)/(M1*1.0)) + ((m2*1.0)/(M2*1.0)) )*sqrt(3.0)*0.5;
    ry = (  -((m1*1.0)/(M1*1.0)) + ((m2*1.0)/(M2*1.0)) )*0.5;

    density_ =0.0;
    Sz_ =0.0;
    Sx_=0.0;
    Sy_=0.0;

    for(int kprime1_ind=0;kprime1_ind<2*l1_-1;kprime1_ind++){
    for(int kprime2_ind=0;kprime2_ind<2*l2_-1;kprime2_ind++){
    for(int Gprime1_ind=0;Gprime1_ind<2*G_grid_L1-1;Gprime1_ind++){
    for(int Gprime2_ind=0;Gprime2_ind<2*G_grid_L2-1;Gprime2_ind++){


    kprime1_ = kprime1_ind - (l1_-1);
    kprime2_ = kprime2_ind - (l2_-1);
    Gprime1_ = Gprime1_ind - (G_grid_L1-1);
    Gprime2_ = Gprime2_ind - (G_grid_L2-1);

    density_ += (1.0/Area)*(
                exp(iota_complex*2.0*PI*(  (m1*(kprime1_)*1.0)/(M1*l1_*1.0)   +  (m2*(kprime2_)*1.0)/(M2*l2_*1.0)  ))
                *
                exp(iota_complex*2.0*PI*(  (m1*(Gprime1_)*1.0)/(M1*1.0)   +  (m2*(Gprime2_)*1.0)/(M2*1.0)  ))
                *
                (Omat[0][0][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind] +
                Omat[1][1][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind])
                );

    Sz_ += (0.5/Area)*(
                exp(iota_complex*2.0*PI*(  (m1*(kprime1_)*1.0)/(M1*l1_*1.0)   +  (m2*(kprime2_)*1.0)/(M2*l2_*1.0)  ))
                *
                exp(iota_complex*2.0*PI*(  (m1*(Gprime1_)*1.0)/(M1*1.0)   +  (m2*(Gprime2_)*1.0)/(M2*1.0)  ))
                *
                (Omat[0][0][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind] -
                Omat[1][1][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind])
                );

    Sx_ += (0.5/Area)*(
                exp(iota_complex*2.0*PI*(  (m1*(kprime1_)*1.0)/(M1*l1_*1.0)   +  (m2*(kprime2_)*1.0)/(M2*l2_*1.0)  ))
                *
                exp(iota_complex*2.0*PI*(  (m1*(Gprime1_)*1.0)/(M1*1.0)   +  (m2*(Gprime2_)*1.0)/(M2*1.0)  ))
                *
                (Omat[0][1][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind] +
                Omat[1][0][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind])
                );
    Sy_ += ((-0.5*iota_complex)/Area)*(
                exp(iota_complex*2.0*PI*(  (m1*(kprime1_)*1.0)/(M1*l1_*1.0)   +  (m2*(kprime2_)*1.0)/(M2*l2_*1.0)  ))
                *
                exp(iota_complex*2.0*PI*(  (m1*(Gprime1_)*1.0)/(M1*1.0)   +  (m2*(Gprime2_)*1.0)/(M2*1.0)  ))
                *
                (Omat[0][1][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind] -
                Omat[1][0][layer][layer][kprime1_ind][kprime2_ind][Gprime1_ind][Gprime2_ind])
                );

    }}}}

    fileout<<m1<<"  "<<m2<<"  "<<rx<<"  "<<ry<<"  "<<density_.real()<<"  "<<Sz_.real()<<"  "<<Sx_.real()<<"  "<<Sy_.real()<<
    "  "<<density_.imag()<<"  "<<Sz_.imag()<<"  "<<Sx_.imag()<<"  "<<Sy_.imag()<<endl;
    Sum_density +=density_;
    Sum_Sz +=Sz_;
    Sum_Sx +=Sx_;
    Sum_Sy +=Sy_;
    }

    fileout<<endl;
    }}


    cout<<"Real space Total density = "<<Sum_density<<endl;
    cout<<"Real space Total Sz = "<<Sum_Sz<<endl;
    cout<<"Real space Total Sx = "<<Sum_Sx<<endl;
    cout<<"Real space Total Sy = "<<Sum_Sy<<endl;
}


void Hamiltonian::Create_PMat(){

 Pmat.resize(2*l1_-1); //k1_
 for(int k1_ind=0;k1_ind<(2*l1_-1);k1_ind++){
     Pmat[k1_ind].resize(2*l2_-1); //k2_
     for(int k2_ind=0;k2_ind<(2*l2_-1);k2_ind++){
     Pmat[k1_ind][k2_ind].resize(2*G_grid_L1-1); //g1_;
     for(int g1_ind=0;g1_ind<(2*G_grid_L1-1);g1_ind++){
     Pmat[k1_ind][k2_ind][g1_ind].resize(2*G_grid_L2-1); //g2;
     for(int g2_ind=0;g2_ind<(2*G_grid_L2-1);g2_ind++){
     Pmat[k1_ind][k2_ind][g1_ind][g2_ind].resize(2);//spin1
     for(int s1=0;s1<2;s1++){
     Pmat[k1_ind][k2_ind][g1_ind][g2_ind][s1].resize(2); //spin2
     for(int s2=0;s2<2;s2++){
     Pmat[k1_ind][k2_ind][g1_ind][g2_ind][s1][s2].resize(Parameters_.max_layer_ind);//layer1
    for(int l1=0;l1<Parameters_.max_layer_ind;l1++){
    Pmat[k1_ind][k2_ind][g1_ind][g2_ind][s1][s2][l1].resize(Parameters_.max_layer_ind);//layer2
    for(int l2=0;l2<Parameters_.max_layer_ind;l2++){
    Pmat[k1_ind][k2_ind][g1_ind][g2_ind][s1][s2][l1][l2]=0.0;
    }}}}}}}}



 ifstream file_local_OP(Parameters_.OP_input_file.c_str());

 string linetemp;
 getline(file_local_OP, linetemp);


 int M1, M2; //no. of slices of one moire unit cell
 int l1_inp, l2_inp;
 M1=Parameters_.M1_inp;M2=Parameters_.M2_inp;
 l1_inp =Parameters_.l1_inp;l2_inp=Parameters_.l2_inp;


 double m1_, m2_, rx_, ry_, den_, sz_, sx_, sy_;
 while(getline(file_local_OP,linetemp)){
 stringstream line_stream(linetemp);
 line_stream >> m1_>>m2_>>rx_>>ry_>>den_>>sz_>>sx_>>sy_;


 int tau1_, tau2_, layer1_, layer2_;
 layer1_=0;
 layer2_=0;
 int k1_ind, k2_ind, g1_l1_ind, g2_l2_ind;
 int k1_t1_ind, k2_t2_ind;

 int k1_val, k2_val, g1_val, g2_val;
 for(int k_1_ind=0;k_1_ind<(2*l1_-1);k_1_ind++){
        //k1_ind =  Coordinates_.Ncell(k1_1,k1_2);
            k1_val = k_1_ind - (l1_-1);
         for(int k_2_ind=0;k_2_ind<(2*l2_-1);k_2_ind++){
           //k2_ind =  Coordinates_.Ncell(k2_1,k2_2);
            k2_val = k_2_ind - (l2_-1);
            for(int g_1_ind=0;g_1_ind<(2*G_grid_L1-1);g_1_ind++){
                   // g1_l1_ind=HamiltonianCont_.Coordinates_.Nbasis(g1_1, g1_2, layer1_);
            g1_val = g_1_ind - (G_grid_L1-1);
                for(int g_2_ind=0;g_2_ind<(2*G_grid_L2-1);g_2_ind++){
                    //g2_l2_ind=HamiltonianCont_.Coordinates_.Nbasis(g2_1, g2_2, layer2_);
            g2_val = g_2_ind - (G_grid_L2-1);


            tau1_=0; tau2_=0;
            Pmat[k_1_ind][k_2_ind][g_1_ind][g_2_ind][tau1_][tau2_][layer1_][layer2_] += (1.0*(1.0/1.0))*
                                    exp(iota_complex*2.0*PI*(((m1_*(k1_val))/(M1*l1_inp*1.0)) + ((m2_*(k2_val))/(M2*l2_inp*1.0)) ))*
                                    exp(iota_complex*2.0*PI*(((m1_*(g1_val))/(M1*1.0)) + ((m2_*(g2_val))/(M2*1.0)) ))*
                                    (sz_ + 0.5*(den_));

            tau1_=1; tau2_=1;
            Pmat[k_1_ind][k_2_ind][g_1_ind][g_2_ind][tau1_][tau2_][layer1_][layer2_] += (1.0*(1.0/1.0))*
                                    exp(iota_complex*2.0*PI*(((m1_*(k1_val))/(M1*l1_inp*1.0)) + ((m2_*(k2_val))/(M2*l2_inp*1.0)) ))*
                                    exp(iota_complex*2.0*PI*(((m1_*(g1_val))/(M1*1.0)) + ((m2_*(g2_val))/(M2*1.0)) ))*
                                    (-1.0*sz_ + 0.5*(den_));


            tau1_=0; tau2_=1;
            Pmat[k_1_ind][k_2_ind][g_1_ind][g_2_ind][tau1_][tau2_][layer1_][layer2_] += (1.0*(1.0/1.0))*
                                    exp(iota_complex*2.0*PI*(((m1_*(k1_val))/(M1*l1_inp*1.0)) + ((m2_*(k2_val))/(M2*l2_inp*1.0)) ))*
                                    exp(iota_complex*2.0*PI*(((m1_*(g1_val))/(M1*1.0)) + ((m2_*(g2_val))/(M2*1.0)) ))*
                                    (sx_ + iota_complex*sy_);

            tau1_=1; tau2_=0;
            Pmat[k_1_ind][k_2_ind][g_1_ind][g_2_ind][tau1_][tau2_][layer1_][layer2_] += (1.0*(1.0/1.0))*
                                    exp(iota_complex*2.0*PI*(((m1_*(k1_val))/(M1*l1_inp*1.0)) + ((m2_*(k2_val))/(M2*l2_inp*1.0)) ))*
                                    exp(iota_complex*2.0*PI*(((m1_*(g1_val))/(M1*1.0)) + ((m2_*(g2_val))/(M2*1.0)) ))*
                                    (sx_ - iota_complex*sy_);



                        }

                }

             }
     }


 }



 cout<<"Pmat for using read OP's created"<<endl;

}


void Hamiltonian::Calculate_RealSpace_OParams_new2(string filename){

    //THIS ROUTINE IS NOT WORKING

    int M1, M2; //no. of slices of one moire unit cell
    M1=10;M2=10;
    int m1_max, m1_min, m2_max, m2_min; //range for full 2d lattice
    m1_max = M1*l1_;m1_min=0;
    m2_max = M2*l1_;m2_min=0;


    double rx,ry;

    int q_1_min, q_1_max, q_1_size;
    int q_2_min, q_2_max, q_2_size;
    q_1_min=-(2*l1_*G_grid_L1);q_1_max=(2*l1_*G_grid_L1);
    q_2_min=-(2*l2_*G_grid_L2);q_2_max=(2*l2_*G_grid_L2);
    q_1_size=q_1_max-q_1_min+1;
    q_2_size=q_2_max-q_2_min+1;


    int k_1_min, k_1_max, k_1_size;
    int k_2_min, k_2_max, k_2_size;
    k_1_min=-(1*l1_*G_grid_L1);k_1_max=(1*l1_*G_grid_L1);
    k_2_min=-(1*l2_*G_grid_L2);k_2_max=(1*l2_*G_grid_L2);
    k_1_size=k_1_max-k_1_min+1;
    k_2_size=k_2_max-k_2_min+1;

    Mat_6_Complex_doub Omat;
    Omat.resize(2);
    for(int spin1=0;spin1<2;spin1++){
        Omat[spin1].resize(2);
    for(int spin2=0;spin2<2;spin2++){
        Omat[spin1][spin2].resize(Parameters_.max_layer_ind);
    for(int layer1=0;layer1<Parameters_.max_layer_ind;layer1++){
    Omat[spin1][spin2][layer1].resize(Parameters_.max_layer_ind);
    for(int layer2=0;layer2<Parameters_.max_layer_ind;layer2++){
    Omat[spin1][spin2][layer1][layer2].resize(q_1_size);
    for(int q_1_ind=0;q_1_ind<q_1_size;q_1_ind++){
    Omat[spin1][spin2][layer1][layer2][q_1_ind].resize(q_2_size);
    }}}}}


    //Folding_to_BrillouinZone(k1_vec_ind1, k1_vec_ind2, k1_vec_ind1_new, k1_vec_ind2_new, G1_off1, G1_off2);


    int k_plus_q_1_new, k_plus_q_2_new, G2_off1, G2_off2;
    int k_1_new, k_2_new, G1_off1, G1_off2;
    int k_plus_q_1, k_plus_q_2;
    for(int spin1=0;spin1<2;spin1++){
    for(int spin2=0;spin2<2;spin2++){
    for(int layer1=0;layer1<Parameters_.max_layer_ind;layer1++){
    for(int layer2=0;layer2<Parameters_.max_layer_ind;layer2++){
    for(int q_1_ind=0;q_1_ind<q_1_size;q_1_ind++){
    for(int q_2_ind=0;q_2_ind<q_2_size;q_2_ind++){
    int q_1_ = q_1_min + q_1_ind;
    int q_2_ = q_2_min + q_2_ind;

    Omat[spin1][spin2][layer1][layer2][q_1_ind][q_2_ind]=0.0;

    for(int k_1_ind=0;k_1_ind<k_1_size;k_1_ind++){
    for(int k_2_ind=0;k_2_ind<k_2_size;k_2_ind++){
    int k_1_ = k_1_min + k_1_ind;
    int k_2_ = k_2_min + k_2_ind;

    k_plus_q_1 = q_1_ + k_1_;
    k_plus_q_2 = q_2_ + k_2_;

    Folding_to_BrillouinZone(k_1_, k_2_, k_1_new, k_2_new, G1_off1, G1_off2);
    Folding_to_BrillouinZone(k_plus_q_1, k_plus_q_2, k_plus_q_1_new, k_plus_q_2_new, G2_off1, G2_off2);

    int k_row = Coordinates_.Ncell(k_1_new,k_2_new);
    int k_col = Coordinates_.Ncell(k_plus_q_1_new,k_plus_q_2_new);

    bool allow=(Inverse_kSublattice_mapping[k_row].first==Inverse_kSublattice_mapping[k_col].first);

    int kSL_ind = Inverse_kSublattice_mapping[k_row].first;
    int k_row_ind = Inverse_kSublattice_mapping[k_row].second;
    int k_col_ind = Inverse_kSublattice_mapping[k_col].second;
    if( allow &&
       (G1_off1>=0 && G1_off1<G_grid_L1) &&
       (G1_off2>=0 && G1_off2<G_grid_L2) &&
       (G2_off1>=0 && G2_off1<G_grid_L1) &&
       (G2_off2>=0 && G2_off2<G_grid_L2)
       ){

        int comp1 = HamiltonianCont_.Coordinates_.Nbasis(G1_off1, G1_off1, layer1);
        int comp2 = HamiltonianCont_.Coordinates_.Nbasis(G2_off1, G2_off2, layer2);



        for(int band1=0;band1<Nbands;band1++){
        for(int band2=0;band2<Nbands;band2++){

            int row_val = k_row_ind +
                      k_sublattices[kSL_ind].size()*band1 +
                      k_sublattices[kSL_ind].size()*Nbands*spin1;
            int col_val = k_col_ind +
                      k_sublattices[kSL_ind].size()*band2 +
                      k_sublattices[kSL_ind].size()*Nbands*spin2;

    Omat[spin1][spin2][layer1][layer2][q_1_ind][q_2_ind] +=
            (conj(BlochStates[spin1][band1][k_row][comp1])*BlochStates[spin2][band2][k_col][comp2]*
             OParams[kSL_ind](row_val,col_val));
        }}

    }


    }}
    }}}}}}





    complex<double> density_, Sz_, Sx_, Sy_;

    for(int layer=0;layer<Parameters_.max_layer_ind;layer++){
    string filename_new = "layer_"+to_string(layer)+"_"+filename;
    ofstream fileout(filename_new.c_str());
    fileout<<"# m1  m2  rx  ry   density   sz  sx  sy"<<endl;

    for(int m1=m1_min;m1<=m1_max;m1++){
    for(int m2=m2_min;m2<=m2_max;m2++){

    rx = (  ((m1*1.0)/(M1*1.0)) + ((m2*1.0)/(M2*1.0)) )*sqrt(3.0)*0.5;
    ry = (  -((m1*1.0)/(M1*1.0)) + ((m2*1.0)/(M2*1.0)) )*0.5;

    density_ =0.0;



    for(int q_1_ind=0;q_1_ind<q_1_size;q_1_ind++){
    for(int q_2_ind=0;q_2_ind<q_2_size;q_2_ind++){
        int q_1_ = q_1_min + q_1_ind;
        int q_2_ = q_2_min + q_2_ind;

    density_ += (1.0/Area)*(
                exp(iota_complex*2.0*PI*(  (m1*(q_1_)*1.0)/(M1*l1_*1.0)   +  (m2*(q_2_)*1.0)/(M2*l2_*1.0)  ))
                *
                (Omat[0][0][layer][layer][q_1_ind][q_2_ind] + Omat[1][1][layer][layer][q_1_ind][q_2_ind])
                );
    }}

    fileout<<m1<<"  "<<m2<<"  "<<rx<<"  "<<ry<<"  "<<density_.real()<<"  "<<Sz_.real()<<"  "<<Sx_.real()<<"  "<<Sy_.real()<<
    "  "<<density_.imag()<<"  "<<Sz_.imag()<<"  "<<Sx_.imag()<<"  "<<Sy_.imag()<<endl;
    }

    fileout<<endl;
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
Xmat[kSL_ind][k1_ind][k2_ind][spin][spin_p].resize(Parameters_.max_layer_ind);
Omat[kSL_ind][k1_ind][k2_ind][spin][spin_p].resize(Parameters_.max_layer_ind);
for(int layer1=0;layer1<Parameters_.max_layer_ind;layer1++){
Xmat[kSL_ind][k1_ind][k2_ind][spin][spin_p][layer1].resize(Parameters_.max_layer_ind);
Omat[kSL_ind][k1_ind][k2_ind][spin][spin_p][layer1].resize(Parameters_.max_layer_ind);
for(int layer2=0;layer2<Parameters_.max_layer_ind;layer2++){
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
for(int layer1=0;layer1<Parameters_.max_layer_ind;layer1++){
for(int layer2=0;layer2<Parameters_.max_layer_ind;layer2++){
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
for(int layer1=0;layer1<Parameters_.max_layer_ind;layer1++){
for(int layer2=0;layer2<Parameters_.max_layer_ind;layer2++){
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
for(int layer=0;layer<Parameters_.max_layer_ind;layer++){
filename_new = "layer_"+to_string(layer)+"_"+filename;
filename2_new = "layer_"+to_string(layer)+"_"+filename2;
ofstream fileout(filename_new.c_str());
ofstream fileout2(filename2_new.c_str());
fileout<<"# m1  m2  rx  ry   density   sz  sx  sy"<<endl;
//fileout2<<"#cell1  cell2   m1  m2  rx  ry   density   sz  sx  sy"<<endl;

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




//density_ += (1.0/Area)*(
//            exp(iota_complex*2.0*PI*(  (m1*(h1_1-h2_1)*1.0)/(M1*l1_*1.0)   +  (m2*(h1_2-h2_2)*1.0)/(M2*l2_*1.0)  ))
//            *
//            exp(iota_complex*2.0*PI*(  (m1*(G1_ind1-G2_ind1)*1.0)/(M1*1.0)   +  (m2*(G1_ind2-G2_ind2)*1.0)/(M2*1.0)  ))
//            *
//            (
//            (conj(BlochStates[spin_up][band1][k1_ind_val][comp1])*BlochStates[spin_up][band2][k2_ind_val][comp2]*OParams[kSL_ind](row_val_up,col_val_up))
//            +
//            (conj(BlochStates[spin_dn][band1][k1_ind_val][comp1])*BlochStates[spin_dn][band2][k2_ind_val][comp2]*OParams[kSL_ind](row_val_dn,col_val_dn))
//            )
//            );



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

//if(m1%M1==0  && m2%M2==0){
//fileout2<<m1/M1<<"  "<<m2/M2<<"  "<<m1<<"  "<<m2<<"  "<<rx<<"  "<<ry<<"  "<<Area*density_.real()<<"  "<<Area*Sz_.real()<<"  "<<Area*Sx_.real()<<"  "<<Area*Sy_.real()<<
//"  "<<Area*density_.imag()<<"  "<<Area*Sz_.imag()<<"  "<<Area*Sx_.imag()<<"  "<<Area*Sy_.imag()<<endl;
//}
}

fileout<<endl;
//if( (m1%M1==0)){
//    fileout2<<endl;
//}

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
double eta=0.2;
double w_min = EigVal_min-5.0;
double w_max = EigVal_max+5.0;
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

file_out<<"#mu = "<<mu_<<endl;
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



void Hamiltonian::Calculate_Total_Energy(){

Total_QuantEnergy=0.0;
Total_ClassEnergy=0.0;


for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
for(int n=0;n<EigVectors[kSL_ind].n_row();n++){
  Total_QuantEnergy += EigValues[kSL_ind][n]*FermiFunction(EigValues[kSL_ind][n]);
}
}


int row_ind, col_ind;
//Classical Energy Terms

//Hartree Term
for(int kset_ind=0;kset_ind<k_sublattices.size();kset_ind++){
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

       Total_ClassEnergy += (-0.5)*HartreeCoefficients[kset_ind][k2_ind][k3_ind][band2][band3][spin_p]*
                                OParams[kset_ind](row_ind,col_ind);
    }
    }
    }
    }
}}


//Fock Term
for(int kset_ind=0;kset_ind<k_sublattices.size();kset_ind++){
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

      Total_ClassEnergy += (0.5)*FockCoefficients[kset_ind][k2_ind][k3_ind][band2][band4][spin][spin_p]*
                            OParams[kset_ind](row_ind,col_ind);
    }
    }

    }
    }

}
}}


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
            //cout<<"Particles : "<<Particles<<" , n1 = "<<n1<<endl;
        }

    }

    else{
        mu_out = Parameters_.MuValueFixed;
    }



    return mu_out;
} // ----------





double Hamiltonian::V_int(double q_val){

    double val;
    if(q_val>0.00001){
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



void Hamiltonian::Print_Interaction_value3(){

for(int spin1=0;spin1<2;spin1++){
for(int spin2=0;spin2<2;spin2++){

    for(int band1=0;band1<Nbands;band1++){
    for(int band2=0;band2<Nbands;band2++){
    for(int band3=0;band3<Nbands;band3++){
    for(int band4=0;band4<Nbands;band4++){

        string file_Int="Interation_spins"+to_string(spin1)+"_"+to_string(spin2)+"_bands_"+to_string(band1)+"_"+
                        to_string(band2)+"_"+to_string(band3)+"_"+to_string(band4)+
                        ".txt";
         ofstream fl_Int_out(file_Int.c_str());
         fl_Int_out<<"#k1(=k1_1+k1_2*l1_)   k2    q  Interaction.real()  Interaction.imag()"<<endl;

        for(int k1_ind=0;k1_ind<ns_;k1_ind++){ //ns_
        for(int k2_ind=0;k2_ind<ns_;k2_ind++){
        for(int q_ind=0;q_ind<ns_;q_ind++){
            fl_Int_out<<k1_ind<<"  "<<k2_ind<<"  "<<q_ind<< "   "<<Interaction_val[spin1][spin2][band1][band2][band3][band4][k1_ind][k2_ind][q_ind].real()<<"  "
                                                <<Interaction_val[spin1][spin2][band1][band2][band3][band4][k1_ind][k2_ind][q_ind].imag()<<endl;
        }
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




void Hamiltonian::Print_HF_Band_Projected_Interaction(){



    complex<double> W_val;
    int k1_1,k1_2, k2_1, k2_2, q_1, q_2;

    int k1_1_minus_q1, k1_2_minus_q2, k1_minus_q;
    int k1_1_minus_q1_temp, k1_2_minus_q2_temp;

    int k2_1_plus_q1, k2_2_plus_q2, k2_plus_q;
    int k2_1_plus_q1_temp, k2_2_plus_q2_temp;

    int G1_ind_temp, G2_ind_temp;
    int k1_minus_q_SL, k1_minus_q_ind;
    int k2_plus_q_SL, k2_plus_q_ind;

    int k2_SL, k2_ind, k1_SL, k1_ind;
    int col_val_1,col_val_2,col_val_3,col_val_4;



    string fileout_str="HF_Band_projected_Interaction_bands_2_3.txt";
    ofstream fileout(fileout_str.c_str());
    fileout<<"#m1   m2   m3  m4 spin  spin_p   k1  k2  q   W(k1,k2k,q).real   W().imag"<<endl;


    //m=2 is spin=0
    for(int m1=2;m1<=3;m1++){
    for(int m2=2;m2<=3;m2++){
        for(int m3=2;m3<=3;m3++){
            for(int m4=2;m4<=3;m4++){

        for(int spin=0;spin<2;spin++){
            for(int spin_p=0;spin_p<2;spin_p++){

                for(int k1=0;k1<ns_;k1++){
                k1_1 = Coordinates_.indx_cellwise(k1);
                k1_2 = Coordinates_.indy_cellwise(k1);
                k1_SL = Inverse_kSublattice_mapping[k1].first;
                k1_ind = Inverse_kSublattice_mapping[k1].second;

                for(int k2=0;k2<ns_;k2++){
                    k2_1 = Coordinates_.indx_cellwise(k2);
                    k2_2 = Coordinates_.indy_cellwise(k2);
                    k2_SL = Inverse_kSublattice_mapping[k2].first;
                    k2_ind = Inverse_kSublattice_mapping[k2].second;

                for(int q=0;q<ns_;q++){
                    fileout<<m1<<"  "<<m2<<"   "<<m3<<"   "<<m4<<"  "<<spin<<"  "<<spin_p<<"  "<<k1<<"  "<<k2<<"  "<<q<<"  ";

                    q_1 = Coordinates_.indx_cellwise(q);
                    q_2 = Coordinates_.indy_cellwise(q);

                    k1_1_minus_q1_temp = k1_1 - q_1;
                    k1_2_minus_q2_temp = k1_2 - q_2;
                    Folding_to_BrillouinZone(k1_1_minus_q1_temp, k1_2_minus_q2_temp, k1_1_minus_q1, k1_2_minus_q2, G1_ind_temp, G2_ind_temp);
                    k1_minus_q = k1_1_minus_q1 + k1_2_minus_q2*l1_;
                    k1_minus_q_SL = Inverse_kSublattice_mapping[k1_minus_q].first;
                    k1_minus_q_ind = Inverse_kSublattice_mapping[k1_minus_q].second;

                    k2_1_plus_q1_temp = k2_1 + q_1;
                    k2_2_plus_q2_temp = k2_2 + q_2;
                    Folding_to_BrillouinZone(k2_1_plus_q1_temp, k2_2_plus_q2_temp, k2_1_plus_q1, k2_2_plus_q2, G1_ind_temp, G2_ind_temp);
                    k2_plus_q = k2_1_plus_q1 + k2_2_plus_q2*l1_;
                    k2_plus_q_SL = Inverse_kSublattice_mapping[k2_plus_q].first;
                    k2_plus_q_ind = Inverse_kSublattice_mapping[k2_plus_q].second;



                W_val=0.0;
                int n_min=0;
                int n_max=Nbands;
                for(int n1=n_min;n1<n_max;n1++){
                    for(int n2=n_min;n2<n_max;n2++){
                        for(int n3=n_min;n3<n_max;n3++){
                            for(int n4=n_min;n4<n_max;n4++){

                 col_val_1 = k1_minus_q_ind +
                 k_sublattices[k1_minus_q_SL].size()*n1 +
                 k_sublattices[k1_minus_q_SL].size()*Nbands*spin;

                 col_val_2 = k2_plus_q_ind +
                 k_sublattices[k2_plus_q_SL].size()*n2 +
                 k_sublattices[k2_plus_q_SL].size()*Nbands*spin_p;

                 col_val_3 = k2_ind +
                 k_sublattices[k2_SL].size()*n3 +
                 k_sublattices[k2_SL].size()*Nbands*spin_p;

                 col_val_4 = k1_ind +
                 k_sublattices[k1_SL].size()*n4 +
                 k_sublattices[k1_SL].size()*Nbands*spin;

                W_val += (1.0/(2.0*Area))*Interaction_val[spin][spin_p][n1][n2][n3][n4][k1][k2][q]*
                        conj(EigVectors[k1_minus_q_SL](col_val_1,m1))*
                        conj(EigVectors[k2_plus_q_SL](col_val_2,m2))*
                        EigVectors[k2_SL](col_val_3,m3)*
                        EigVectors[k1_SL](col_val_4,m4);


                            }
                        }
                    }
                }

                //fileout<<0.0<<"  "<<0.0<<endl;
                fileout<<W_val.real()<<"  "<<W_val.imag()<<endl;


                }}}
            }
        }

    }}}}



    for(int m=2;m<=3;m++){
    string fileout_str="HF_Band_Eigenvalues_band" + to_string(m) + ".txt";
    ofstream fileout(fileout_str.c_str());
    fileout<<"k  E(k)"<<endl;


    for(int k1=0;k1<ns_;k1++){
    k1_1 = Coordinates_.indx_cellwise(k1);
    k1_2 = Coordinates_.indy_cellwise(k1);
    k1_SL = Inverse_kSublattice_mapping[k1].first;
    k1_ind = Inverse_kSublattice_mapping[k1].second;

    fileout<<k1<<"  "<<EigValues[k1_SL][m]<<endl;

    }
    }

}


void Hamiltonian::Print_HF_Band_Projected_Interaction_TR_and_Inversion_imposed(){



    Mat_5_Complex_doub W_val_mat;
    //[spin][spin_p][k1][k2][q]
    W_val_mat.resize(2);
    for(int spin=0;spin<2;spin++){
    W_val_mat[spin].resize(2);
    for(int spin_p=0;spin_p<2;spin_p++){
      W_val_mat[spin][spin_p].resize(ns_);
      for(int k1=0;k1<ns_;k1++){
        W_val_mat[spin][spin_p][k1].resize(ns_);
       for(int k2=0;k2<ns_;k2++){
        W_val_mat[spin][spin_p][k1][k2].resize(ns_);
       }
          }
    }
    }

    Mat_5_Complex_doub W_val_mat2;
    //[spin][spin_p][k1][k2][q]
    W_val_mat2.resize(2);
    for(int spin=0;spin<2;spin++){
    W_val_mat2[spin].resize(2);
    for(int spin_p=0;spin_p<2;spin_p++){
      W_val_mat2[spin][spin_p].resize(ns_);
      for(int k1=0;k1<ns_;k1++){
        W_val_mat2[spin][spin_p][k1].resize(ns_);
       for(int k2=0;k2<ns_;k2++){
        W_val_mat2[spin][spin_p][k1][k2].resize(ns_);
       }
          }
    }
    }


    Mat_5_Complex_doub W_val_mat3;
    //[spin][spin_p][k1][k2][q]
    W_val_mat3.resize(2);
    for(int spin=0;spin<2;spin++){
    W_val_mat3[spin].resize(2);
    for(int spin_p=0;spin_p<2;spin_p++){
      W_val_mat3[spin][spin_p].resize(ns_);
      for(int k1=0;k1<ns_;k1++){
        W_val_mat3[spin][spin_p][k1].resize(ns_);
       for(int k2=0;k2<ns_;k2++){
        W_val_mat3[spin][spin_p][k1][k2].resize(ns_);
       }
          }
    }
    }


    complex<double> W_val;
    int k1_1,k1_2, k2_1, k2_2, q_1, q_2;

    int mk1_1, mk1_2, mk2_1, mk2_2, mq_1, mq_2;
    int mk1, mk2, mq;


    int k1_1_minus_q1, k1_2_minus_q2, k1_minus_q;
    int k1_1_minus_q1_temp, k1_2_minus_q2_temp;

    int k2_1_plus_q1, k2_2_plus_q2, k2_plus_q;
    int k2_1_plus_q1_temp, k2_2_plus_q2_temp;

    int G1_ind_temp, G2_ind_temp;
    int k1_minus_q_SL, k1_minus_q_ind;
    int k2_plus_q_SL, k2_plus_q_ind;

    int k2_SL, k2_ind, k1_SL, k1_ind;
    int col_val_1,col_val_2,col_val_3,col_val_4;

    string fileout_str="HF_Band_projected_Interaction_bands_2_3_TR_and_Inversion_imposed.txt";
    ofstream fileout(fileout_str.c_str());
    fileout<<"#m1   m2   m3  m4 spin  spin_p   k1  k2  q   W(k1,k2k,q).real   W().imag"<<endl;


    //m=2 is spin=0
    int m1, m2, m3 ,m4;

        for(int spin=0;spin<2;spin++){ //m1, m4
            m1 = spin+2;
            m4=m1;
            for(int spin_p=0;spin_p<2;spin_p++){ //m2, m3
            m2 = spin_p +2;
            m3=m2;

                for(int k1=0;k1<ns_;k1++){
                k1_1 = Coordinates_.indx_cellwise(k1);
                k1_2 = Coordinates_.indy_cellwise(k1);
                k1_SL = Inverse_kSublattice_mapping[k1].first;
                k1_ind = Inverse_kSublattice_mapping[k1].second;

//                mk1_1 = (-k1_1 + l1_)%l1_;
//                mk1_2 = (-k1_2 + l2_)%l2_;
//                mk1 =  (mk1_1 + l1_*mk1_2);

                for(int k2=0;k2<ns_;k2++){
                    k2_1 = Coordinates_.indx_cellwise(k2);
                    k2_2 = Coordinates_.indy_cellwise(k2);
                    k2_SL = Inverse_kSublattice_mapping[k2].first;
                    k2_ind = Inverse_kSublattice_mapping[k2].second;

//                    mk2_1 = (-k2_1 + l1_)%l1_;
//                    mk2_2 = (-k2_2 + l2_)%l2_;
//                    mk2 =  (mk2_1 + l1_*mk2_2);

                for(int q=0;q<ns_;q++){
                   // fileout<<m1<<"  "<<m2<<"   "<<m3<<"   "<<m4<<"  "<<spin<<"  "<<spin_p<<"  "<<k1<<"  "<<k2<<"  "<<q<<"  ";

                    q_1 = Coordinates_.indx_cellwise(q);
                    q_2 = Coordinates_.indy_cellwise(q);

//                    mq_1 = (-q_1 + l1_)%l1_;
//                    mq_2 = (-q_2 + l2_)%l2_;
//                    mq =  (mq_1 + l1_*mq_2);

                    k1_1_minus_q1_temp = k1_1 - q_1;
                    k1_2_minus_q2_temp = k1_2 - q_2;
                    Folding_to_BrillouinZone(k1_1_minus_q1_temp, k1_2_minus_q2_temp, k1_1_minus_q1, k1_2_minus_q2, G1_ind_temp, G2_ind_temp);
                    k1_minus_q = k1_1_minus_q1 + k1_2_minus_q2*l1_;
                    k1_minus_q_SL = Inverse_kSublattice_mapping[k1_minus_q].first;
                    k1_minus_q_ind = Inverse_kSublattice_mapping[k1_minus_q].second;

                    k2_1_plus_q1_temp = k2_1 + q_1;
                    k2_2_plus_q2_temp = k2_2 + q_2;
                    Folding_to_BrillouinZone(k2_1_plus_q1_temp, k2_2_plus_q2_temp, k2_1_plus_q1, k2_2_plus_q2, G1_ind_temp, G2_ind_temp);
                    k2_plus_q = k2_1_plus_q1 + k2_2_plus_q2*l1_;
                    k2_plus_q_SL = Inverse_kSublattice_mapping[k2_plus_q].first;
                    k2_plus_q_ind = Inverse_kSublattice_mapping[k2_plus_q].second;



                W_val=0.0;
                int n_min=0;
                int n_max=Nbands;
                for(int n1=n_min;n1<n_max;n1++){
                    for(int n2=n_min;n2<n_max;n2++){
                        for(int n3=n_min;n3<n_max;n3++){
                            for(int n4=n_min;n4<n_max;n4++){

                 col_val_1 = k1_minus_q_ind +
                 k_sublattices[k1_minus_q_SL].size()*n1 +
                 k_sublattices[k1_minus_q_SL].size()*Nbands*spin;

                 col_val_2 = k2_plus_q_ind +
                 k_sublattices[k2_plus_q_SL].size()*n2 +
                 k_sublattices[k2_plus_q_SL].size()*Nbands*spin_p;

                 col_val_3 = k2_ind +
                 k_sublattices[k2_SL].size()*n3 +
                 k_sublattices[k2_SL].size()*Nbands*spin_p;

                 col_val_4 = k1_ind +
                 k_sublattices[k1_SL].size()*n4 +
                 k_sublattices[k1_SL].size()*Nbands*spin;

                W_val += (1.0/(2.0*Area))*Interaction_val[spin][spin_p][n1][n2][n3][n4][k1][k2][q]
                        *conj(EigVectors[k1_minus_q_SL](col_val_1,m1))*
                        conj(EigVectors[k2_plus_q_SL](col_val_2,m2))*
                        EigVectors[k2_SL](col_val_3,m3)*
                        EigVectors[k1_SL](col_val_4,m4);


                            }
                        }
                    }
                }


                //spin<<"  "<<spin_p<<"  "<<k1<<"  "<<k2<<"  "<<q
                W_val_mat[spin][spin_p][k1][k2][q]=W_val;
                //fileout<<0.0<<"  "<<0.0<<endl;
                //fileout<<W_val.real()<<"  "<<W_val.imag()<<endl;

//                if(spin==0 && spin_p==0){
//                fileout<<m1+1<<"  "<<m2+1<<"   "<<m3+1<<"   "<<m4+1<<"  "<<spin+1<<"  "<<spin_p+1<<"  ";
//                int mk1_1 = (-k1_1 + l1_)%l1_;
//                int mk1_2 = (-k1_2 + l2_)%l2_;
//                int mk2_1 = (-k2_1 + l1_)%l1_;
//                int mk2_2 = (-k2_2 + l2_)%l2_;
//                int mk1 = mk1_1 + l1_*mk1_2;
//                int mk2 = mk2_1 + l1_*mk2_2;
//                fileout<<mk2<<"  "<<mk1<<"  "<<q<<"  "<<W_val.real()<<"  "<<-1.0*W_val.imag()<<endl;
//                }

//                if(spin==0 && spin_p==1){ //2 3 3 2 0 1
//                fileout<<3<<"  "<<2<<"   "<<2<<"   "<<3<<"  "<<spin_p<<"  "<<spin<<"  ";
//                int mk1_1 = (-k1_1 + l1_)%l1_;
//                int mk1_2 = (-k1_2 + l2_)%l2_;
//                int mk2_1 = (-k2_1 + l1_)%l1_;
//                int mk2_2 = (-k2_2 + l2_)%l2_;
//                int mq_1 = (-q_1 + l1_)%l1_;
//                int mq_2 = (-q_2 + l2_)%l2_;
//                int mk1 = mk1_1 + l1_*mk1_2;
//                int mk2 = mk2_1 + l1_*mk2_2;
//                int mq = mq_1 + l1_*mq_2;
//                fileout<<mk1<<"  "<<mk2<<"  "<<mq<<"  "<<W_val.real()<<"  "<<-1.0*W_val.imag()<<endl;
//                }

                }}}
            }
        }



        for(int k1=0;k1<ns_;k1++){
            k1_1 = Coordinates_.indx_cellwise(k1);
            k1_2 = Coordinates_.indy_cellwise(k1);
            mk1_1 = (-k1_1 + l1_)%l1_;
            mk1_2 = (-k1_2 + l2_)%l2_;
            mk1 =  (mk1_1 + l1_*mk1_2);
         for(int k2=0;k2<ns_;k2++){
             k2_1 = Coordinates_.indx_cellwise(k2);
             k2_2 = Coordinates_.indy_cellwise(k2);
             mk2_1 = (-k2_1 + l1_)%l1_;
             mk2_2 = (-k2_2 + l2_)%l2_;
             mk2 =  (mk2_1 + l1_*mk2_2);
        for(int q=0;q<ns_;q++){
        W_val_mat2[0][0][k1][k2][q]=W_val_mat[0][0][k1][k2][q];
        W_val_mat2[1][1][mk2][mk1][q]=conj(W_val_mat[0][0][k1][k2][q]);
        W_val_mat2[0][1][mk2][mk1][q]=0.5*(conj(W_val_mat[0][1][k1][k2][q]) +
                                        W_val_mat[0][1][mk2][mk1][q]);
        }}}


        for(int k1=0;k1<ns_;k1++){
            k1_1 = Coordinates_.indx_cellwise(k1);
            k1_2 = Coordinates_.indy_cellwise(k1);
            mk1_1 = (-k1_1 + l1_)%l1_;
            mk1_2 = (-k1_2 + l2_)%l2_;
            mk1 =  (mk1_1 + l1_*mk1_2);

         for(int k2=0;k2<ns_;k2++){
             k2_1 = Coordinates_.indx_cellwise(k2);
             k2_2 = Coordinates_.indy_cellwise(k2);
             mk2_1 = (-k2_1 + l1_)%l1_;
             mk2_2 = (-k2_2 + l2_)%l2_;
             mk2 =  (mk2_1 + l1_*mk2_2);

        for(int q=0;q<ns_;q++){
            q_1 = Coordinates_.indx_cellwise(q);
            q_2 = Coordinates_.indy_cellwise(q);
            mq_1 = (-q_1 + l1_)%l1_;
            mq_2 = (-q_2 + l2_)%l2_;
            mq =  (mq_1 + l1_*mq_2);
        W_val_mat2[1][0][k1][k2][q]=conj(W_val_mat2[0][1][mk1][mk2][mq]);
        }}}



        //Hermiticity
        for(int spin=0;spin<2;spin++){ //m1, m4
            for(int spin_p=0;spin_p<2;spin_p++){
        for(int k1=0;k1<ns_;k1++){
            k1_1 = Coordinates_.indx_cellwise(k1);
            k1_2 = Coordinates_.indy_cellwise(k1);
        for(int k2=0;k2<ns_;k2++){
            k2_1 = Coordinates_.indx_cellwise(k2);
            k2_2 = Coordinates_.indy_cellwise(k2);
        for(int q=0;q<ns_;q++){
            q_1 = Coordinates_.indx_cellwise(q);
            q_2 = Coordinates_.indy_cellwise(q);

            k1_1_minus_q1 = (k1_1 - q_1 + l1_)%l1_;
            k1_2_minus_q2 = (k1_2 - q_2 + l2_)%l2_;
            k1_minus_q = k1_1_minus_q1 + k1_2_minus_q2*l1_;

            k2_1_plus_q1 = (k2_1 + q_1)%l1_;
            k2_2_plus_q2 = (k2_2 + q_2)%l2_;
            k2_plus_q = k2_1_plus_q1 + k2_2_plus_q2*l1_;

            mq_1 = (-q_1 + l1_)%l1_;
            mq_2 = (-q_2 + l2_)%l2_;
            mq =  (mq_1 + l1_*mq_2);

           W_val_mat3[spin][spin_p][k1][k2][q] = 0.5*(W_val_mat2[spin][spin_p][k1][k2][q] +
                                                conj(W_val_mat2[spin][spin_p][k1_minus_q][k2_plus_q][mq]));


        }}}
        }}


        for(int spin=0;spin<2;spin++){ //m1, m4
            for(int spin_p=0;spin_p<2;spin_p++){
                //if(spin!=spin_p){
        for(int k1=0;k1<ns_;k1++){
        for(int k2=0;k2<ns_;k2++){
        for(int q=0;q<ns_;q++){
    fileout<<spin+2<<"  "<<spin_p+2<<"   "<<spin_p+2<<"   "<<spin+2<<"  "<<spin<<"  "<<spin_p<<"  "<<k1<<"  "<<k2<<"  "<<q<<"  "<<W_val_mat3[spin][spin_p][k1][k2][q].real()<<"  "<<W_val_mat3[spin][spin_p][k1][k2][q].imag()<<endl;
       //}
        }}}
        }}








    for(int m=2;m<=3;m++){
    string fileout_str="HF_Band_Eigenvalues_band" + to_string(m) + "_TR_and_Inversion_imposed.txt";
    ofstream fileout(fileout_str.c_str());
    fileout<<"k  E(k)"<<endl;


    for(int k1=0;k1<ns_;k1++){
    k1_1 = Coordinates_.indx_cellwise(k1);
    k1_2 = Coordinates_.indy_cellwise(k1);
    k1_SL = Inverse_kSublattice_mapping[k1].first;
    k1_ind = Inverse_kSublattice_mapping[k1].second;


    //Inversion symmetry is imposed
    int mk1_1 = (-k1_1 + l1_)%l1_;
    int mk1_2 = (-k1_2 + l2_)%l2_;
    int mk1 = mk1_1 + l1_*mk1_2;
    int mk1_SL = Inverse_kSublattice_mapping[mk1].first;
    int mk1_ind = Inverse_kSublattice_mapping[mk1].second;

    fileout<<k1<<"  "<<0.5*(EigValues[k1_SL][m] + EigValues[mk1_SL][m])<<endl;

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
    Interaction_value_new(spin,spin_p,band1,band2,band3,band4,k1,k2,q);
    }}}
    }}}}
    }}

}


complex<double> Hamiltonian::Interaction_value_new(int spin, int spin_p, int band1, int band2, int band3, int band4, int k1_ind, int k2_ind, int q_ind){


    int temp_int;
    int k1_1_ind, k1_2_ind, k2_1_ind, k2_2_ind, g1_ind, g2_ind;
    int q_ind1, q_ind2, k1_ind1, k1_ind2, k2_ind1, k2_ind2;
    int q_ind1_new, q_ind2_new;
    q_ind1=Coordinates_.indx_cellwise(q_ind);q_ind2=Coordinates_.indy_cellwise(q_ind);
    k1_ind1=Coordinates_.indx_cellwise(k1_ind);k1_ind2=Coordinates_.indy_cellwise(k1_ind);
    k2_ind1=Coordinates_.indx_cellwise(k2_ind);k2_ind2=Coordinates_.indy_cellwise(k2_ind);
    double qpGx_temp, qpGy_temp;
    complex<double> val;
    val=0.0;
    for(int g_ind1=Lambda_G_grid_L1_min;g_ind1<=Lambda_G_grid_L1_max;g_ind1++){
    g1_ind = g_ind1 - Lambda_G_grid_L1_min;
    for(int g_ind2=Lambda_G_grid_L2_min;g_ind2<=Lambda_G_grid_L2_max;g_ind2++){
    g2_ind = g_ind2 - Lambda_G_grid_L2_min;

    HamiltonianCont_.Getting_kx_ky_in_Primitive_BZ(qpGx_temp, qpGy_temp, q_ind1, q_ind2, q_ind1_new, q_ind2_new);

    qpGx_temp = qpGx_temp +
                (2.0*PI/Parameters_.a_moire)*(g_ind1*(1.0/sqrt(3)) + g_ind2*(1.0/sqrt(3)));
    qpGy_temp = qpGy_temp +
                (2.0*PI/Parameters_.a_moire)*(g_ind1*(-1.0) + g_ind2*(1.0));


    val += V_int(sqrt( (qpGx_temp*qpGx_temp) + (qpGy_temp*qpGy_temp) ) ) *
            LambdaPBZ_k1_m_q[spin][band1][band4][k1_ind1][k1_ind2][q_ind1][q_ind2][g1_ind][g2_ind]*
            conj(LambdaPBZ_k2_p_q[spin_p][band3][band2][k2_ind1][k2_ind2][q_ind1][q_ind2][g1_ind][g2_ind]);

    }}

    return val;
}

complex<double> Hamiltonian::Interaction_value(int spin, int spin_p, int band1, int band2, int band3, int band4, int k1_ind, int k2_ind, int q_ind){



int temp_int;
int k1_1_ind, k1_2_ind, k2_1_ind, k2_2_ind, g1_ind, g2_ind;
int q_ind1, q_ind2, k1_ind1, k1_ind2, k2_ind1, k2_ind2;
q_ind1=Coordinates_.indx_cellwise(q_ind);q_ind2=Coordinates_.indy_cellwise(q_ind);
k1_ind1=Coordinates_.indx_cellwise(k1_ind);k1_ind2=Coordinates_.indy_cellwise(k1_ind);
k2_ind1=Coordinates_.indx_cellwise(k2_ind);k2_ind2=Coordinates_.indy_cellwise(k2_ind);
double qpGx_temp, qpGy_temp;
complex<double> val;
val=0.0;
for(int g_ind1=Lambda_G_grid_L1_min;g_ind1<=Lambda_G_grid_L1_max;g_ind1++){
g1_ind = g_ind1 - Lambda_G_grid_L1_min;
for(int g_ind2=Lambda_G_grid_L2_min;g_ind2<=Lambda_G_grid_L2_max;g_ind2++){
g2_ind = g_ind2 - Lambda_G_grid_L2_min;

//if(g_ind1!=0 && g_ind2!=0){
//  for(int g_ind1=0;g_ind1<=0;g_ind1++){
//  for(int g_ind2=0;g_ind2<=0;g_ind2++){

//kx_=(2.0*PI/Parameters_.a_moire)*((n1)*(1.0/(sqrt(3)*L1_))  +  (n2)*(1.0/(sqrt(3)*L2_)));
//ky_=(2.0*PI/Parameters_.a_moire)*((n1)*(-1.0/(L1_))  +  (n2)*(1.0/(L2_)));

//HamiltonianCont_.Getting_kx_ky_in_Primitive_BZ(qpGx_temp, qpGy_temp, q_ind1, q_ind2);

//qpGx_temp = qpGx_temp +
//            (2.0*PI/Parameters_.a_moire)*(g_ind1*(1.0/sqrt(3)) + g_ind2*(1.0/sqrt(3)));
//qpGy_temp = qpGy_temp +
//            (2.0*PI/Parameters_.a_moire)*(g_ind1*(-1.0) + g_ind2*(1.0));

qpGx_temp = (2.0*PI/Parameters_.a_moire)*(
            (q_ind1)*(1.0/(sqrt(3)*l1_))  +  (q_ind2)*(1.0/(sqrt(3)*l2_)) //q
            + g_ind1*(1.0/sqrt(3)) + g_ind2*(1.0/sqrt(3)) //G
            );

qpGy_temp = (2.0*PI/Parameters_.a_moire)*(
            (q_ind1)*(-1.0/l1_)  +  (q_ind2)*(1.0/l2_) //q
            + g_ind1*(-1.0) + g_ind2*(1.0) //G
            );


k1_1_ind = k1_ind1-q_ind1+(l1_-1);
k1_2_ind = k1_ind2-q_ind2+(l2_-1);

k2_1_ind = k2_ind1 + q_ind1;
k2_2_ind = k2_ind2 + q_ind2;

val += V_int(sqrt( (qpGx_temp*qpGx_temp) + (qpGy_temp*qpGy_temp) ) ) *
        LambdaNew_[spin][band1][band4][k1_1_ind][k1_2_ind][k1_ind1][k1_ind2][g1_ind][g2_ind]*
        conj(LambdaNew_[spin_p][band3][band2][k2_ind1+(l1_-1)][k2_ind2+(l2_-1)][k2_1_ind][k2_2_ind][g1_ind][g2_ind]);

//cout<<"Comp : "<<LambdaNew_[spin][band1][band4][k1_1_ind][k1_2_ind][k1_ind1][k1_ind2][g1_ind][g2_ind]<<"   "<<
//      FormFactor(spin,band1,band4, k1_ind1-q_ind1, k1_ind2-q_ind2, k1_ind1 + (g_ind1*l1_), k1_ind2 + (g_ind2*l2_))<<"   "<<
 //     conj(LambdaNew_[spin_p][band3][band2][k2_ind1+(l1_-1)][k2_ind2+(l2_-1)][k2_1_ind][k2_2_ind][g1_ind][g2_ind])<<"   "<<
 //     conj(FormFactor(spin_p,band3,band2, k2_ind1, k2_ind2, k2_ind1+q_ind1+(g_ind1*l1_), k2_ind2+q_ind2+(g_ind2*l2_)))<<
 //     endl;
//cin>>temp_int;


//val += V_int(sqrt( (qpGx_temp*qpGx_temp) + (qpGy_temp*qpGy_temp) ) ) *
//      FormFactor(spin,band1,band4, k1_ind1-q_ind1, k1_ind2-q_ind2, k1_ind1 + (g_ind1*l1_), k1_ind2 + (g_ind2*l2_))*
//      conj(FormFactor(spin_p,band3,band2, k2_ind1, k2_ind2, k2_ind1+q_ind1+(g_ind1*l1_), k2_ind2+q_ind2+(g_ind2*l2_)) );

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


int Hamiltonian::Get_maximum_comp(Mat_1_Complex_doub Vec_){

  int max_comp=0;
  for(int i=0;i<Vec_.size();i++){
      if(abs(Vec_[i])>abs(Vec_[max_comp])){
        max_comp=i;
      }
  }

  return max_comp;
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
    Convergence_technique=Parameters_.Convergence_technique;

    assert(Convergence_technique=="SimpleMixing" ||
           Convergence_technique=="AndersonMixing");


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
               BlochStates[spin][n][i].resize(G_grid_L1*G_grid_L2*Parameters_.max_layer_ind); //G_ind*layer
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
   int comp_norm, comp_norm_PBZ;
   comp_norm=HamiltonianCont_.Coordinates_.Nbasis((G_grid_L1/2), (G_grid_L2/2), 0);
   //comp_norm=HamiltonianCont_.Coordinates_.Nbasis(0, 0, 0);
   //comp_norm=1.0; 
    complex<double> phase_, phase_PBZ;
    for(int spin=0;spin<2;spin++){
          for(int n=0;n<Nbands;n++){
            for(int i1=0;i1<l1_;i1++){  //k_ind
            for(int i2=0;i2<l2_;i2++){
                
               //Why this does not work for MoTe2 homobilayer??
              //  comp_norm = Get_maximum_comp(HamiltonianCont_.BlochStates[spin][n][i1+i2*(l1_+1)]);
              //  comp_norm_PBZ = Get_maximum_comp(HamiltonianCont_.BlochStates_PBZ[spin][n][i1+i2*(l1_)]);

                comp_norm_PBZ=comp_norm;

                phase_= conj(HamiltonianCont_.BlochStates[spin][n][i1+i2*(l1_+1)][comp_norm])/
                        (abs(HamiltonianCont_.BlochStates[spin][n][i1+i2*(l1_+1)][comp_norm]));

                phase_PBZ= conj(HamiltonianCont_.BlochStates_PBZ[spin][n][i1+i2*(l1_)][comp_norm_PBZ])/
                        (abs(HamiltonianCont_.BlochStates_PBZ[spin][n][i1+i2*(l1_)][comp_norm_PBZ]));


               // phase_=1.0;
                //phase_PBZ=1.0;
                for(int comp=0;comp<G_grid_L1*G_grid_L2*Parameters_.max_layer_ind;comp++){
                BlochStates[spin][n][i1+i2*l1_][comp]=phase_*HamiltonianCont_.BlochStates[spin][n][i1+i2*(l1_+1)][comp];
                HamiltonianCont_.BlochStates_PBZ[spin][n][i1+i2*(l1_)][comp] = phase_PBZ*HamiltonianCont_.BlochStates_PBZ[spin][n][i1+i2*(l1_)][comp];
              }
            }
            }
          }
        }




    //Imposing T-Reversal symmetry in Bloch states, because otherwise phase is random
    //and in Interaction matrix it is hard to investigate the TR symmetry
    //This should not change the final results.

    //bloch(dn,k,G)=conj(bloch(up,-k,-G))
    /*
    int m_g1_ind, m_g2_ind;
         int m_g_ind1, m_g_ind2;
         int g_ind1, g_ind2;
         int comp_up, comp_dn;
         int i1_new, i2_new;
         int m_i1_temp, m_i2_temp;
         int m_i1, m_i2;
         int g1_temp, g2_temp;
         int g1_temp2, g2_temp2;
         int g_off1, g_off2;
         int i1_val, i2_val;

          for(int n=0;n<Nbands;n++){
            for(int i1=0;i1<l1_;i1++){  //k_ind
                for(int i2=0;i2<l2_;i2++){
                HamiltonianCont_.Getting_n1_n2_in_Primitive_BZ(i1,i2,i1_val, i2_val, g1_temp, g2_temp);

                Folding_to_BrillouinZone(-i1_val, -i2_val,
                                        m_i1_temp, m_i2_temp,
                                         g1_temp, g2_temp);

               HamiltonianCont_.Getting_n1_n2_in_Primitive_BZ(m_i1_temp,m_i2_temp,
                                                               m_i1,m_i2,
                                                               g1_temp2, g2_temp2);

                 g_off1=g1_temp - g1_temp2;
                 g_off2=g2_temp - g2_temp2;



                for(int layer=0;layer<Parameters_.max_layer_ind;layer++){
                    for(int g1_ind=0;g1_ind<G_grid_L1;g1_ind++){
                    for(int g2_ind=0;g2_ind<G_grid_L2;g2_ind++){
                    g_ind1 = g1_ind  - (G_grid_L1/2);
                    g_ind2 = g2_ind  - (G_grid_L2/2);
                    m_g_ind1 = - g_ind1;
                    m_g_ind2 = - g_ind2;

                   m_g1_ind= m_g_ind1 + g_off1 + (G_grid_L1/2);
                   m_g2_ind= m_g_ind2 + g_off2 +(G_grid_L2/2);

                   //if is used, because grid is not symmeric around 0,0 for even sized grid
                   if(m_g1_ind<G_grid_L1 && m_g2_ind<G_grid_L2){
                   comp_dn=HamiltonianCont_.Coordinates_.Nbasis(g1_ind, g2_ind, layer);
                   comp_up=HamiltonianCont_.Coordinates_.Nbasis(m_g1_ind, m_g2_ind, layer);

                HamiltonianCont_.BlochStates_PBZ[0][n][m_i1_temp+m_i2_temp*(l1_)][comp_up] = conj(HamiltonianCont_.BlochStates_PBZ[1][n][i1+i2*(l1_)][comp_dn]);
                        }
                    }
                    }
            }
            }
          }
        }
        */

          //If Interaction_new is used
          BlochStates = HamiltonianCont_.BlochStates_PBZ;


    BlochStates_old_ = HamiltonianCont_.BlochStates;

    //PrintBlochStates();
    //PrintBlochStatesPBZ();

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


    Hbar.resize(ns_);
    Fbar.resize(ns_);
    for(int k_=0;k_<ns_;k_++){
    Hbar[k_].resize(Nbands);
    Fbar[k_].resize(Nbands);
    for(int n1=0;n1<Nbands;n1++){
    Hbar[k_][n1].resize(Nbands);
    Fbar[k_][n1].resize(Nbands);
    for(int n2=0;n2<Nbands;n2++){
    Hbar[k_][n1][n2].resize(2);//For Spin
    Fbar[k_][n1][n2].resize(2);//For Spin
    }
    }
    }


    
    //(int spin, int spin_p, int band1, int band2, int band3, int band4, int k1_ind, int k2_ind, int q_ind)



    Lambda_G_grid_L1_min= int((-1.0*G_grid_L1/2.0)+0.5);
    Lambda_G_grid_L1_max= int((1.0*G_grid_L1/2.0)+0.5);
    Lambda_G_grid_L2_min= int((-1.0*G_grid_L2/2.0)+0.5);
    Lambda_G_grid_L2_max= int((1.0*G_grid_L2/2.0)+0.5);

    Lambda_G1_grid_size = Lambda_G_grid_L1_max - Lambda_G_grid_L1_min + 1;
    Lambda_G2_grid_size = Lambda_G_grid_L2_max - Lambda_G_grid_L2_min + 1;


    LambdaNew_.resize(2);  //[k][q+G]; k is in {-(l-1),(l-1)}; q is in {0,2l-2}; G is in {-G_Grid/2,G_Grid/2}
    for(int spin=0;spin<2;spin++){
        LambdaNew_[spin].resize(Nbands);
        for(int n1=0;n1<Nbands;n1++){
            LambdaNew_[spin][n1].resize(Nbands);
            for(int n2=0;n2<Nbands;n2++){
                LambdaNew_[spin][n1][n2].resize(2*l1_ - 1); //-(l1-1)....to....(l1-1)
              for(int k1=0;k1<(2*l1_-1);k1++){
                  LambdaNew_[spin][n1][n2][k1].resize(2*l2_ -1); //-(l2-1)....to....(l2-1)
                  for(int k2=0;k2<(2*l2_ -1);k2++){
                LambdaNew_[spin][n1][n2][k1][k2].resize(2*l1_ -1); // 0.....to....(2*l1-2)
                for(int q1=0;q1<(2*l1_ -1);q1++){
              LambdaNew_[spin][n1][n2][k1][k2][q1].resize(2*l2_ -1); // 0.....to....(2*l2-2)
                for(int q2=0;q2<(2*l2_ -1);q2++){
                LambdaNew_[spin][n1][n2][k1][k2][q1][q2].resize(Lambda_G1_grid_size); //min...to...max
                for(int g1=0;g1<Lambda_G1_grid_size;g1++){
                LambdaNew_[spin][n1][n2][k1][k2][q1][q2][g1].resize(Lambda_G2_grid_size); //min...to...max
                }
                }
                }
                }
              }
            }
        }
    }



    Create_Lambda_PBZ();
//    PrintFormFactors_PBZ(0, 0, 0);
//    PrintFormFactors_PBZ(0, 0, 1);
//    PrintFormFactors_PBZ(1, 0, 1);
//    PrintFormFactors_PBZ(1, 0, 0);
//    PrintFormFactors_PBZ(0, 1, 1);
//    PrintFormFactors_PBZ(0, 1, 0);

 //   Calculate_FormFactors();
//    PrintFormFactors2(0 ,0, 0);
//    PrintFormFactors2(0 ,0, 1);
//    PrintFormFactors2(1 ,0, 1);
//    PrintFormFactors2(1 ,0, 0);
//    PrintFormFactors2(0 ,1, 1);
//    PrintFormFactors2(0 ,1, 0);

    cout<<"Saving Interaction val"<<endl;
    Save_InteractionVal();
    cout<<"Interaction val completed"<<endl;



    //Print_Interaction_value3();


    cout<<"Started:  Creating Amat and Bmat"<<endl;
    Create_Amat_and_Bmat();
    cout<<"Completed: Creating Amat and Bmat"<<endl;

    cout<<"Started:  Creating Hbar and Fbar"<<endl;
    Create_Hbar_and_Fbar();
    cout<<"Completed: Creating Hbar and Fbar"<<endl;


    //-----------
  

    N_layer_tau.resize(Parameters_.max_layer_ind);
    for(int i=0;i<Parameters_.max_layer_ind;i++){
        N_layer_tau[i].resize(2);
    }



    BO_PBZ.resize(Nbands);
    for(int n=0;n<Nbands;n++){
    BO_PBZ[n].resize(2);
    for(int spin=0;spin<2;spin++){
    BO_PBZ[n][spin].resize((l1_)*(l2_));
    for(int k_ind=0;k_ind<((l1_)*(l2_));k_ind++){
    BO_PBZ[n][spin][k_ind].resize(Nbands);
    for(int np=0;np<Nbands;np++){
    BO_PBZ[n][spin][k_ind][np].resize(2);
    for(int spinp=0;spinp<2;spinp++){
    BO_PBZ[n][spin][k_ind][np][spinp].resize((l1_)*(l2_));

    }}}}}


    //----------

} // ----------

bool Hamiltonian::Present(string type, pair_int k_temp){
    bool present_=false;

    if(type=="k1_m_q"){
        for(int i=0;i<Possible_k1_m_q.size();i++){
            if(Possible_k1_m_q[i].first == k_temp.first
               || Possible_k1_m_q[i].second == k_temp.second ){
            present_=true;
            break;
            }
        }
    }

    if(type=="k2_p_q"){
        for(int i=0;i<Possible_k2_p_q.size();i++){
            if(Possible_k2_p_q[i].first == k_temp.first
               || Possible_k2_p_q[i].second == k_temp.second ){
            present_=true;
            break;
            }
        }
    }

    return present_;

}

void Hamiltonian::Create_Lambda_PBZ(){


int k1_1_val, k1_2_val;
int q_1_val, q_2_val;
/*
Possible_k1_m_q.clear();
Possible_k2_p_q.clear();

    for(int k1_1_ind=0;k1_1_ind<l1_;k1_1_ind++){
        for(int k1_2_ind=0;k1_2_ind<l2_;k1_2_ind++){
            k1_1_val = HamiltonianCont_.PBZ_map_n1[k1_1_ind][k1_2_ind];
            k1_2_val = HamiltonianCont_.PBZ_map_n2[k1_1_ind][k1_2_ind];

            for(int q_1_ind=0;q_1_ind<l1_;q_1_ind++){
                for(int q_2_ind=0;q_2_ind<l2_;q_2_ind++){
            q_1_val = HamiltonianCont_.PBZ_map_n1[q_1_ind][q_2_ind];
            q_2_val = HamiltonianCont_.PBZ_map_n2[q_1_ind][q_2_ind];

            pair_int k1_m_q_temp;
            k1_m_q_temp.first = k1_1_val - q_1_val;
            k1_m_q_temp.second = k1_2_val - q_2_val;
            if(!Present("k1_m_q", k1_m_q_temp)){
            Possible_k1_m_q.push_back(k1_m_q_temp);
            }
                }
            }
        }
    }

    for(int k2_1_ind=0;k2_1_ind<l1_;k2_1_ind++){
        for(int k2_2_ind=0;k2_2_ind<l2_;k2_2_ind++){
            k2_1_val = HamiltonianCont_.PBZ_map_n1[k2_1_ind][k2_2_ind];
            k2_2_val = HamiltonianCont_.PBZ_map_n2[k2_1_ind][k2_2_ind];

            for(int q_1_ind=0;q_1_ind<l1_;q_1_ind++){
                for(int q_2_ind=0;q_2_ind<l2_;q_2_ind++){
            q_1_val = HamiltonianCont_.PBZ_map_n1[q_1_ind][q_2_ind];
            q_2_val = HamiltonianCont_.PBZ_map_n2[q_1_ind][q_2_ind];

            pair_int k2_p_q_temp;
            k2_p_q_temp.first = k2_1_val + q_1_val;
            k2_p_q_temp.second = k2_2_val + q_2_val;
            if(!Present("k2_p_q", k2_p_q_temp )){
            Possible_k2_p_q.push_back(k2_p_q_temp);
            }
        }}
        }}

    */


    LambdaPBZ_k1_m_q.resize(2);  //[k-q][k+G];  G is in {-G_Grid/2,G_Grid/2}
    for(int spin=0;spin<2;spin++){
        LambdaPBZ_k1_m_q[spin].resize(Nbands);
        for(int n1=0;n1<Nbands;n1++){
            LambdaPBZ_k1_m_q[spin][n1].resize(Nbands);
            for(int n2=0;n2<Nbands;n2++){
                LambdaPBZ_k1_m_q[spin][n1][n2].resize(l1_);
              for(int k1_1=0;k1_1<l1_;k1_1++){
                 LambdaPBZ_k1_m_q[spin][n1][n2][k1_1].resize(l2_);
                  for(int k1_2=0;k1_2<l2_;k1_2++){
                LambdaPBZ_k1_m_q[spin][n1][n2][k1_1][k1_2].resize(l1_);
                for(int q_1=0;q_1<l1_;q_1++){
                   LambdaPBZ_k1_m_q[spin][n1][n2][k1_1][k1_2][q_1].resize(l2_);
                   for(int q_2=0;q_2<l2_;q_2++){
                      LambdaPBZ_k1_m_q[spin][n1][n2][k1_1][k1_2][q_1][q_2].resize(Lambda_G1_grid_size); //min...to...max
                      for(int g1=0;g1<Lambda_G1_grid_size;g1++){
                      LambdaPBZ_k1_m_q[spin][n1][n2][k1_1][k1_2][q_1][q_2][g1].resize(Lambda_G2_grid_size); //min...to...max
                      }
                }
              }
            }
        }
    }
}}


    LambdaPBZ_k2_p_q.resize(2);  //[k-q][k+G];  G is in {-G_Grid/2,G_Grid/2}
    for(int spin=0;spin<2;spin++){
        LambdaPBZ_k2_p_q[spin].resize(Nbands);
        for(int n1=0;n1<Nbands;n1++){
            LambdaPBZ_k2_p_q[spin][n1].resize(Nbands);
            for(int n2=0;n2<Nbands;n2++){
                LambdaPBZ_k2_p_q[spin][n1][n2].resize(l1_);
              for(int k1_1=0;k1_1<l1_;k1_1++){
                 LambdaPBZ_k2_p_q[spin][n1][n2][k1_1].resize(l2_);
                  for(int k1_2=0;k1_2<l2_;k1_2++){
                LambdaPBZ_k2_p_q[spin][n1][n2][k1_1][k1_2].resize(l1_);
                for(int q_1=0;q_1<l1_;q_1++){
                   LambdaPBZ_k2_p_q[spin][n1][n2][k1_1][k1_2][q_1].resize(l2_);
                   for(int q_2=0;q_2<l2_;q_2++){
                      LambdaPBZ_k2_p_q[spin][n1][n2][k1_1][k1_2][q_1][q_2].resize(Lambda_G1_grid_size); //min...to...max
                      for(int g1=0;g1<Lambda_G1_grid_size;g1++){
                      LambdaPBZ_k2_p_q[spin][n1][n2][k1_1][k1_2][q_1][q_2][g1].resize(Lambda_G2_grid_size); //min...to...max
                      }
                }
              }
            }
        }
    }
}}






    //LambdaPBZ_k1_m_q[spin][n1][n2][k1_1][k1_2][q_1][q_2][g1][g2]

    for(int spin=0;spin<2;spin++){
        for(int n1=0;n1<Nbands;n1++){
            for(int n2=0;n2<Nbands;n2++){
              for(int k1_1=0;k1_1<l1_;k1_1++){
                  for(int k1_2=0;k1_2<l2_;k1_2++){
         k1_1_val = HamiltonianCont_.PBZ_map_n1[k1_1][k1_2];
         k1_2_val = HamiltonianCont_.PBZ_map_n2[k1_1][k1_2];
                for(int q_1=0;q_1<l1_;q_1++){
                   for(int q_2=0;q_2<l2_;q_2++){
         q_1_val = HamiltonianCont_.PBZ_map_n1[q_1][q_2];
         q_2_val = HamiltonianCont_.PBZ_map_n2[q_1][q_2];

         int k1_1_m_q_1_val = k1_1_val - q_1_val;
         int k1_2_m_q_2_val = k1_2_val - q_2_val;

         int k1_1_m_q_1_val_temp, k1_2_m_q_2_val_temp; // in FBZ (0,2pi)
         int g1_temp, g2_temp;

         int k1_1_m_q_1_val_temp2, k1_2_m_q_2_val_temp2;  //inn PBZ
         int g1_temp2, g2_temp2;

         int g1_left, g2_left;

         Folding_to_BrillouinZone(k1_1_m_q_1_val, k1_2_m_q_2_val,
                                  k1_1_m_q_1_val_temp, k1_2_m_q_2_val_temp,
                                  g1_temp, g2_temp);
         HamiltonianCont_.Getting_n1_n2_in_Primitive_BZ(k1_1_m_q_1_val_temp, k1_2_m_q_2_val_temp,
                                                        k1_1_m_q_1_val_temp2, k1_2_m_q_2_val_temp2,
                                                        g1_temp2, g2_temp2);

          g1_left=g1_temp - g1_temp2;
          g2_left=g2_temp - g2_temp2;


         for(int g3_1=0;g3_1<Lambda_G1_grid_size;g3_1++){
                    for(int g3_2=0;g3_2<Lambda_G2_grid_size;g3_2++){
         int k1_1_p_g3_1_val = k1_1_val + (g3_1 + Lambda_G_grid_L1_min)*l1_;
         int k1_2_p_g3_2_val = k1_2_val + (g3_2 + Lambda_G_grid_L2_min)*l2_;
         int k1_1_p_g3_1_val_temp, k1_2_p_g3_2_val_temp; // in FBZ (0,2pi)
         int g1_right_temp, g2_right_temp;

         int k1_1_p_g3_1_val_temp2, k1_2_p_g3_2_val_temp2; // in PBZ
         int g1_right_temp2, g2_right_temp2;
         int g1_right, g2_right;
         Folding_to_BrillouinZone(k1_1_p_g3_1_val, k1_2_p_g3_2_val,
                                  k1_1_p_g3_1_val_temp, k1_2_p_g3_2_val_temp,
                                  g1_right_temp, g2_right_temp);
         HamiltonianCont_.Getting_n1_n2_in_Primitive_BZ(k1_1_p_g3_1_val_temp, k1_2_p_g3_2_val_temp,
                                                        k1_1_p_g3_1_val_temp2, k1_2_p_g3_2_val_temp2,
                                                        g1_right_temp2, g2_right_temp2);

         g1_right=g1_right_temp - g1_right_temp2;
         g2_right=g2_right_temp - g2_right_temp2;

         LambdaPBZ_k1_m_q[spin][n1][n2][k1_1][k1_2][q_1][q_2][g3_1][g3_2]=0.0;
         for(int g_1=0;g_1<Lambda_G1_grid_size;g_1++){
                    for(int g_2=0;g_2<Lambda_G2_grid_size;g_2++){
           int g1_left_ind = (g_1 + Lambda_G_grid_L1_min + g1_left) - Lambda_G_grid_L1_min;
           int g2_left_ind = (g_2 + Lambda_G_grid_L2_min + g2_left) - Lambda_G_grid_L2_min;

           int g1_right_ind = (g_1 + Lambda_G_grid_L1_min + g1_right) - Lambda_G_grid_L1_min;
           int g2_right_ind = (g_2 + Lambda_G_grid_L2_min + g2_right) - Lambda_G_grid_L2_min;


           if( (g1_left_ind>=0 && g1_left_ind<G_grid_L1) &&
               (g2_left_ind>=0 && g2_left_ind<G_grid_L2) &&
               (g1_right_ind>=0 && g1_right_ind<G_grid_L1) &&
               (g2_right_ind>=0 && g2_right_ind<G_grid_L2)
                   ){

           for(int layer=0;layer<Parameters_.max_layer_ind;layer++){
           int comp_left = HamiltonianCont_.Coordinates_.Nbasis(g1_left_ind, g2_left_ind, layer);
           int comp_right = HamiltonianCont_.Coordinates_.Nbasis(g1_right_ind, g2_right_ind, layer);
           LambdaPBZ_k1_m_q[spin][n1][n2][k1_1][k1_2][q_1][q_2][g3_1][g3_2] +=
           conj(HamiltonianCont_.BlochStates_PBZ[spin][n1][ k1_1_m_q_1_val_temp + l1_*k1_2_m_q_2_val_temp][comp_left])*
           HamiltonianCont_.BlochStates_PBZ[spin][n2][ k1_1_p_g3_1_val_temp + l1_*k1_2_p_g3_2_val_temp][comp_right];
        }
           }

          }}


                          }
                      }


                }
              }
            }
        }
    }
}}







//Lambda_k2_p_q , writing k1 for k2 below
    for(int spin=0;spin<2;spin++){
        for(int n1=0;n1<Nbands;n1++){
            for(int n2=0;n2<Nbands;n2++){
              for(int k1_1=0;k1_1<l1_;k1_1++){;
                  for(int k1_2=0;k1_2<l2_;k1_2++){
         k1_1_val = HamiltonianCont_.PBZ_map_n1[k1_1][k1_2];
         k1_2_val = HamiltonianCont_.PBZ_map_n2[k1_1][k1_2];
                for(int q_1=0;q_1<l1_;q_1++){
                   for(int q_2=0;q_2<l2_;q_2++){
         q_1_val = HamiltonianCont_.PBZ_map_n1[q_1][q_2];
         q_2_val = HamiltonianCont_.PBZ_map_n2[q_1][q_2];


         int k1_1_val_temp, k1_2_val_temp;
         int g1_temp, g2_temp;

         int k1_1_val_temp2, k1_2_val_temp2;
         int g1_temp2, g2_temp2;

         int g1_left, g2_left;
         Folding_to_BrillouinZone(k1_1_val, k1_2_val,
                                  k1_1_val_temp, k1_2_val_temp,
                                  g1_temp, g2_temp);
         HamiltonianCont_.Getting_n1_n2_in_Primitive_BZ(k1_1_val_temp, k1_2_val_temp,
                                                        k1_1_val_temp2, k1_2_val_temp2,
                                                        g1_temp2, g2_temp2);

          g1_left=g1_temp - g1_temp2;
          g2_left=g2_temp - g2_temp2;


         for(int g3_1=0;g3_1<Lambda_G1_grid_size;g3_1++){
                    for(int g3_2=0;g3_2<Lambda_G2_grid_size;g3_2++){
         int k1_1_p_q_1_p_g3_1_val = k1_1_val + (g3_1 + Lambda_G_grid_L1_min)*l1_ + q_1_val;
         int k1_2_p_q_2_p_g3_2_val = k1_2_val + (g3_2 + Lambda_G_grid_L1_min)*l2_ + q_2_val;

         int k1_1_p_q_1_p_g3_1_val_temp, k1_2_p_q_2_p_g3_2_val_temp; // in FBZ (0,2pi)
         int g1_right_temp, g2_right_temp;

         int k1_1_p_q_1_p_g3_1_val_temp2, k1_2_p_q_2_p_g3_2_val_temp2; // in PBZ
         int g1_right_temp2, g2_right_temp2;
         int g1_right, g2_right;

         Folding_to_BrillouinZone(k1_1_p_q_1_p_g3_1_val, k1_2_p_q_2_p_g3_2_val,
                                  k1_1_p_q_1_p_g3_1_val_temp, k1_2_p_q_2_p_g3_2_val_temp,
                                  g1_right_temp, g2_right_temp);
         HamiltonianCont_.Getting_n1_n2_in_Primitive_BZ(k1_1_p_q_1_p_g3_1_val_temp, k1_2_p_q_2_p_g3_2_val_temp,
                                                        k1_1_p_q_1_p_g3_1_val_temp2, k1_2_p_q_2_p_g3_2_val_temp2,
                                                        g1_right_temp2, g2_right_temp2);

         g1_right=g1_right_temp - g1_right_temp2;
         g2_right=g2_right_temp - g2_right_temp2;

         LambdaPBZ_k2_p_q[spin][n1][n2][k1_1][k1_2][q_1][q_2][g3_1][g3_2]=0.0;
         for(int g_1=0;g_1<Lambda_G1_grid_size;g_1++){
                    for(int g_2=0;g_2<Lambda_G2_grid_size;g_2++){
           int g1_left_ind = (g_1 + Lambda_G_grid_L1_min + g1_left) - Lambda_G_grid_L1_min;
           int g2_left_ind = (g_2 + Lambda_G_grid_L2_min + g2_left) - Lambda_G_grid_L2_min;

           int g1_right_ind = (g_1 + Lambda_G_grid_L1_min + g1_right) - Lambda_G_grid_L1_min;
           int g2_right_ind = (g_2 + Lambda_G_grid_L2_min + g2_right) - Lambda_G_grid_L2_min;


           if( (g1_left_ind>=0 && g1_left_ind<G_grid_L1) &&
               (g2_left_ind>=0 && g2_left_ind<G_grid_L2) &&
               (g1_right_ind>=0 && g1_right_ind<G_grid_L1) &&
               (g2_right_ind>=0 && g2_right_ind<G_grid_L2)
                   ){

           for(int layer=0;layer<Parameters_.max_layer_ind;layer++){
           int comp_left = HamiltonianCont_.Coordinates_.Nbasis(g1_left_ind, g2_left_ind, layer);
           int comp_right = HamiltonianCont_.Coordinates_.Nbasis(g1_right_ind, g2_right_ind, layer);
           LambdaPBZ_k2_p_q[spin][n1][n2][k1_1][k1_2][q_1][q_2][g3_1][g3_2] +=
           conj(HamiltonianCont_.BlochStates_PBZ[spin][n1][ k1_1_val_temp + l1_*k1_2_val_temp][comp_left])*
           HamiltonianCont_.BlochStates_PBZ[spin][n2][ k1_1_p_q_1_p_g3_1_val_temp + l1_*k1_2_p_q_2_p_g3_2_val_temp][comp_right];
        }
           }

          }}


                          }
                      }


                }
              }
            }
        }
    }
}}












}




void Hamiltonian::PrintBlochStatesPBZ(){

    for(int spin=0;spin<2;spin++){
          for(int n=0;n<Nbands;n++){

        string filename = "PBZ_BlochState_spin"+to_string(spin)+"_band"+to_string(n)+".txt";
        ofstream fileout(filename.c_str());
        fileout<<"#comp  Psi(k1,k2).real()  Psi(k2,k2).imag  ..  ..  .."<<endl;


               // phase_= conj(HamiltonianCont_.BlochStates[spin][n][i1+i2*(l1_+1)][comp_norm])/
               //         (abs(HamiltonianCont_.BlochStates[spin][n][i1+i2*(l1_+1)][comp_norm]));

                for(int g2=0;g2<G_grid_L2;g2++){
                for(int g1=0;g1<G_grid_L1;g1++){
                for(int layer_=0;layer_<1;layer_++){//Parameters_.max_layer_ind
                    int comp = HamiltonianCont_.Coordinates_.Nbasis(g1,g2,layer_);
                //for(int comp=0;comp<G_grid_L1*G_grid_L2*Parameters_.max_layer_ind;comp++){
                fileout<<comp<<"  "<< g1<<"  "<<g2<<"  "<<layer_;
              for(int i2=0;i2<l2_;i2++){
              for(int i1=0;i1<l1_;i1++){  //k_ind
               fileout<<"  "<<HamiltonianCont_.BlochStates_PBZ[spin][n][i1+i2*l1_][comp].real()<<"  "<<HamiltonianCont_.BlochStates_PBZ[spin][n][i1+i2*l1_][comp].imag();
              }
            }
              fileout<<endl;
            }}
               fileout<<endl;
                }
          }
        }


}

void Hamiltonian::PrintBlochStates(){

    for(int spin=0;spin<2;spin++){
          for(int n=0;n<Nbands;n++){

        string filename = "BlochState_spin"+to_string(spin)+"_band"+to_string(n)+".txt";
        ofstream fileout(filename.c_str());
        fileout<<"#comp  Psi(k1,k2).real()  Psi(k2,k2).imag  ..  ..  .."<<endl;


               // phase_= conj(HamiltonianCont_.BlochStates[spin][n][i1+i2*(l1_+1)][comp_norm])/
               //         (abs(HamiltonianCont_.BlochStates[spin][n][i1+i2*(l1_+1)][comp_norm]));

                for(int g2=0;g2<G_grid_L2;g2++){
                for(int g1=0;g1<G_grid_L1;g1++){
                for(int layer_=0;layer_<1;layer_++){//Parameters_.max_layer_ind
                    int comp = HamiltonianCont_.Coordinates_.Nbasis(g1,g2,layer_);
                //for(int comp=0;comp<G_grid_L1*G_grid_L2*Parameters_.max_layer_ind;comp++){
                fileout<<comp<<"  "<< g1<<"  "<<g2<<"  "<<layer_;
              for(int i2=0;i2<l2_;i2++){
              for(int i1=0;i1<l1_;i1++){  //k_ind
               fileout<<"  "<<BlochStates[spin][n][i1+i2*l1_][comp].real()<<"  "<<BlochStates[spin][n][i1+i2*l1_][comp].imag();
              }
            }
              fileout<<endl;
            }}
               fileout<<endl;
                }
          }
        }


}

void Hamiltonian::Create_Hbar_and_Fbar(){

    string Hbarfilestr="Hbar.txt";
    ofstream Hbarfile(Hbarfilestr.c_str());

    string Fbarfilestr="Fbar.txt";
    ofstream Fbarfile(Fbarfilestr.c_str());


    int k_ind1, k_ind2, q_ind1, q_ind2;
    int kpq_temp1, kpq_temp2;
    int kpq_new;
    int kpq_new1, kpq_new2, G1_ind_temp, G2_ind_temp;

    for(int k_=0;k_<ns_;k_++){
        k_ind1=Coordinates_.indx_cellwise(k_);
        k_ind2=Coordinates_.indy_cellwise(k_);
    for(int n1=0;n1<Nbands;n1++){
    for(int n2=0;n2<Nbands;n2++){
    for(int spin=0;spin<2;spin++){


    Hbar[k_][n1][n2][spin]=0.0;
    Fbar[k_][n1][n2][spin]=0.0;
    for(int q_=0;q_<ns_;q_++){
        q_ind1=Coordinates_.indx_cellwise(q_);
        q_ind2=Coordinates_.indy_cellwise(q_);

        kpq_temp1 = k_ind1 + q_ind1;
        kpq_temp2 = k_ind2 + q_ind2;

        Folding_to_BrillouinZone(kpq_temp1, kpq_temp2, kpq_new1, kpq_new2, G1_ind_temp, G2_ind_temp);
        kpq_new = kpq_new1 + kpq_new2*l1_;

        for(int n3=0;n3<Nbands;n3++){
        Hbar[k_][n1][n2][spin] += (-1.0/(2.0*Area))*Interaction_val[spin][spin][n1][n3][n2][n3][kpq_new][k_][q_];

        for(int spin_p=0;spin_p<2;spin_p++){
            //if( !Parameters_.Imposing_SzZero ||  (spin_p==spin)){
        Fbar[k_][n1][n2][spin] += (1.0/(2.0*Area))*Interaction_val[spin][spin_p][n1][n3][n3][n2][k_][q_][0];
                //}
            }

        }


    }



    }

    if(n1==0 && n2==1){
      Hbarfile<< k_<<"  "<<  Hbar[k_][n1][n2][0].real()<<"  "<<Hbar[k_][n1][n2][0].imag()<<"  "
              <<  Hbar[k_][n1][n2][1].real()<<"  "<<Hbar[k_][n1][n2][1].imag()<<"  "<<endl;
    }

    if(n1==0 && n2==1){
      Fbarfile<< k_<<"  "<<  Fbar[k_][n1][n2][0].real()<<"  "<<Fbar[k_][n1][n2][0].imag()<<"  "
              <<  Fbar[k_][n1][n2][1].real()<<"  "<<Fbar[k_][n1][n2][1].imag()<<"  "<<endl;
    }

    }}}

}

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

                if( !Parameters_.Imposing_SzZero ||  (spin_p==spin)){
                value += (1.0/(2.0*Area)) * ( //HERE
                        Bmat[k1_ind][k2_ind][k2_mk3_kSL_internal_ind][band1][band2][band3][band4][spin][spin_p]
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
                    Ham_(row_ind, row_ind) += 1.0*BlochEigvals[spin][band][k_sublattices[kset_ind][k_ind]]
                                              + Parameters_.MagField_ZeemanSplitting*(spin);
                    //Ham_(row_ind, row_ind) += DispersionTriangularLattice(k_sublattices[kset_ind][k_ind]);
                }
            }
        }


        //Hartree+Fock Offsets
        for(int spin=0;spin<2;spin++){
            for(int band1=0;band1<Nbands;band1++){
            for(int band2=0;band2<Nbands;band2++){
                for(int k_ind=0;k_ind<k_sublattices[kset_ind].size();k_ind++){
                    row_ind = k_ind +
                              k_sublattices[kset_ind].size()*band1+
                              k_sublattices[kset_ind].size()*Nbands*spin;
                    col_ind = k_ind +
                              k_sublattices[kset_ind].size()*band2+
                              k_sublattices[kset_ind].size()*Nbands*spin;

                    Ham_(row_ind, col_ind) += (1.0*Hbar[k_sublattices[kset_ind][k_ind]][band1][band2][spin]
                                            +
                                           1.0*Fbar[k_sublattices[kset_ind][k_ind]][band1][band2][spin]);
                }
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
                    if(  !Parameters_.Imposing_SzZero || (spin==spin_p)){
                 Ham_(row_ind,col_ind) += -1.0*FockCoefficients[kset_ind][k2_ind][k3_ind][band2][band4][spin][spin_p];
                    }
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
for(int layer=0;layer<Parameters_.max_layer_ind;layer++){
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



void Hamiltonian::Kick_OParams(double kick){
    for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
    for(int row_val=0;row_val<OParams[kSL_ind].n_row();row_val++){
    for(int col_val=0;col_val<OParams[kSL_ind].n_col();col_val++){
    OParams[kSL_ind](row_val,col_val) = OParams[kSL_ind](row_val,col_val) +
                                        kick*Myrandom();

    }}}
}

void Hamiltonian::Update_OParams_SimpleMixing(){


for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
for(int row_val=0;row_val<OParams[kSL_ind].n_row();row_val++){
for(int col_val=0;col_val<OParams[kSL_ind].n_col();col_val++){
OParams[kSL_ind](row_val,col_val) = alpha_mixing*OParams_new[kSL_ind](row_val,col_val) + 
                                    (1.0-alpha_mixing)*OParams[kSL_ind](row_val,col_val);

}}}




}

void Hamiltonian::Update_OrderParameters_AndersonMixing(int iter){

    bool with_SVD=false;
    int Offset_;
    int m_;
    int row_, col_;
    int old_ind;
    int OP_size, OP_Real_size, OP_Imag_size;
    Mat_1_int NewInd_to_kSL_ind;
    Mat_1_int NewInd_to_row_ind;
    Mat_1_int NewInd_to_col_ind;
    NewInd_to_kSL_ind.clear();
    NewInd_to_row_ind.clear();
    NewInd_to_col_ind.clear();

    OP_size=0;
    OP_Real_size=0;
    OP_Imag_size=0;
    for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
    assert(OParams[kSL_ind].n_row()==OParams[kSL_ind].n_col());//square matrix
    OP_size += 2*int(((OParams[kSL_ind].n_row()*(OParams[kSL_ind].n_row()-1))+0.5)/2.0)
               + OParams[kSL_ind].n_row();
    }
    //Vectorization order
    //col=>row
    //all real values for col>=row first in order m_11, m_12,..,m_1n, m21, m22,..m2n,m31,....
    //then imag values vol>row in order m12, m13,....,m1n, m23,...
    for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
        for(int row_i=0;row_i<OParams[kSL_ind].n_row();row_i++){
            for(int col_i=0;col_i<OParams[kSL_ind].n_col();col_i++){
                if(col_i>=row_i){//for real part
                  NewInd_to_kSL_ind.push_back(kSL_ind);
                  NewInd_to_row_ind.push_back(row_i);
                  NewInd_to_col_ind.push_back(col_i);
                  OP_Real_size++;
                }
            }
        }
    }
    for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
        for(int row_i=0;row_i<OParams[kSL_ind].n_row();row_i++){
            for(int col_i=0;col_i<OParams[kSL_ind].n_col();col_i++){
                if(col_i>row_i){//for imag part
                  NewInd_to_kSL_ind.push_back(kSL_ind);
                  NewInd_to_row_ind.push_back(row_i);
                  NewInd_to_col_ind.push_back(col_i);
                  OP_Imag_size++;
                }
            }
        }
    }

    assert(OP_size==NewInd_to_kSL_ind.size());



    if(iter==0){
        //        cout<<"Anderson mixing for iter "<<iter<<endl;

        x_k_.clear();x_k_.resize(OP_size);
        for(int i=0;i<OP_size;i++){
            if(i<OP_Real_size){
                x_k_[i] = OParams[NewInd_to_kSL_ind[i]](NewInd_to_row_ind[i], NewInd_to_col_ind[i]).real();
            }
            else{
                x_k_[i] = OParams[NewInd_to_kSL_ind[i]](NewInd_to_row_ind[i], NewInd_to_col_ind[i]).imag();
            }
        }

        f_k_.clear();f_k_.resize(OP_size);
        for(int i=0;i<OP_size;i++){
            if(i<OP_Real_size){
                f_k_[i] = OParams_new[NewInd_to_kSL_ind[i]](NewInd_to_row_ind[i], NewInd_to_col_ind[i]).real()
                        - OParams[NewInd_to_kSL_ind[i]](NewInd_to_row_ind[i], NewInd_to_col_ind[i]).real();
            }
            else{
                f_k_[i] = OParams_new[NewInd_to_kSL_ind[i]](NewInd_to_row_ind[i], NewInd_to_col_ind[i]).imag()
                        - OParams[NewInd_to_kSL_ind[i]](NewInd_to_row_ind[i], NewInd_to_col_ind[i]).imag();
            }
        }
        //assert(OParams.value.size() == MFParams_.OParams_.value.size());

        //f_k = OParams.value;
        //x_k = MFParams_.OParams_.value;

        for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
            for(int row_i=0;row_i<OParams[kSL_ind].n_row();row_i++){
                for(int col_i=0;col_i<OParams[kSL_ind].n_col();col_i++){

            OParams[kSL_ind](row_i, col_i) = (1.0 - alpha_mixing)*OParams[kSL_ind](row_i, col_i) +
                                              alpha_mixing*OParams_new[kSL_ind](row_i, col_i);
        }}}

        x_km1_=x_k_;
        X_mat.resize(0,0);

        f_km1_=f_k_;
        F_mat.resize(0,0);
        //f_km1=

    }
    else{
        //      cout<<"Anderson mixing for iter "<<iter<<endl;
        x_k_.clear();x_k_.resize(OP_size);

        for(int i=0;i<OP_size;i++){
            if(i<OP_Real_size){
                x_k_[i] = OParams[NewInd_to_kSL_ind[i]](NewInd_to_row_ind[i], NewInd_to_col_ind[i]).real();
            }
            else{
                x_k_[i] = OParams[NewInd_to_kSL_ind[i]](NewInd_to_row_ind[i], NewInd_to_col_ind[i]).imag();
            }
        }


        f_k_.clear();f_k_.resize(OP_size);
        for(int i=0;i<OP_size;i++){
            if(i<OP_Real_size){
                f_k_[i] = OParams_new[NewInd_to_kSL_ind[i]](NewInd_to_row_ind[i], NewInd_to_col_ind[i]).real()
                        - OParams[NewInd_to_kSL_ind[i]](NewInd_to_row_ind[i], NewInd_to_col_ind[i]).real();
            }
            else{
                f_k_[i] = OParams_new[NewInd_to_kSL_ind[i]](NewInd_to_row_ind[i], NewInd_to_col_ind[i]).imag()
                        - OParams[NewInd_to_kSL_ind[i]](NewInd_to_row_ind[i], NewInd_to_col_ind[i]).imag();
            }
        }

        Del_x_km1.clear();Del_x_km1.resize(OP_size);
        Del_f_km1.clear();Del_f_km1.resize(OP_size);
        for(int i=0;i<OP_size;i++){
            Del_x_km1[i] = x_k_[i] - x_km1_[i];
            Del_f_km1[i] = f_k_[i] - f_km1_[i];
        }


        m_=min(Parameters_.AM_m,iter);
        gamma_k_.clear();
        gamma_k_.resize(m_);

        //updating X_mat_k
        Matrix <double> Xmat_temp;
        Xmat_temp.resize(X_mat.n_row(),X_mat.n_col());
        Xmat_temp=X_mat;
        X_mat.resize(OP_size,m_);

        if(iter<=Parameters_.AM_m){
            for(col_=0;col_<Xmat_temp.n_col();col_++){
                for(row_=0;row_<Xmat_temp.n_row();row_++){
                    X_mat(row_,col_) = Xmat_temp(row_,col_);
                }
            }

            for(col_=m_-1;col_<m_;col_++){
                for(row_=0;row_<OP_size;row_++){
                    X_mat(row_,col_) = Del_x_km1[row_];
                }
            }
        }
        else{
            for(col_=1;col_<Xmat_temp.n_col();col_++){
                for(row_=0;row_<Xmat_temp.n_row();row_++){
                    X_mat(row_,col_-1) = Xmat_temp(row_,col_);
                }
            }
            for(row_=0;row_<OP_size;row_++){
                X_mat(row_,m_-1) = Del_x_km1[row_];
            }
        }



        //updating F_mat_k
        Matrix <double> Fmat_temp;
        Fmat_temp.resize(F_mat.n_row(),F_mat.n_col());
        Fmat_temp=F_mat;
        F_mat.resize(OP_size,m_);

        if(iter<=Parameters_.AM_m){
            for(col_=0;col_<Fmat_temp.n_col();col_++){
                for(row_=0;row_<Fmat_temp.n_row();row_++){
                    F_mat(row_,col_) = Fmat_temp(row_,col_);
                }
            }

            for(col_=m_-1;col_<m_;col_++){
                for(row_=0;row_<OP_size;row_++){
                    F_mat(row_,col_) = Del_f_km1[row_];
                }
            }
        }
        else{
            for(col_=1;col_<Fmat_temp.n_col();col_++){
                for(row_=0;row_<Fmat_temp.n_row();row_++){
                    F_mat(row_,col_-1) = Fmat_temp(row_,col_);
                }
            }
            for(row_=0;row_<OP_size;row_++){
                F_mat(row_,m_-1) = Del_f_km1[row_];
            }
        }

        //cout<<"here 1"<<endl;

        //Update gamma_k using Total least sqaure minimaztion (using SVD of F_mat)
        if(with_SVD==false){
            for(int i=0;i<m_;i++){
                gamma_k_[i] = 1.0/(1.0*m_);
            }
        }
        else{
            int r_;
            r_=min(OP_size,m_);
            Matrix<double> A_;  //nxm; n=OP_size
            Matrix<double> VT_; //mxm
            Matrix<double> U_;  //nxn
            vector<double> Sigma_; //atmost non-zero min(n,m) values
            A_.resize(F_mat.n_row(), F_mat.n_col());
            A_=F_mat;

            //cout<<"here 2"<<endl;
            Perform_SVD(A_,VT_,U_,Sigma_);
            //cout<<"here 2.5"<<endl;

            Matrix<double> UT_f;
            Matrix<double> Sinv_UT_f;

            UT_f.resize(r_,1);
            for(int i=0;i<r_;i++){
                for(int j=0;j<OP_size;j++){
                    UT_f(i,0) += U_(j,i)*f_k_[j];
                }
            }

            Sinv_UT_f.resize(r_,1);//S-inv in Pseudoinverse of Sigma_
            for(int i=0;i<r_;i++){
                if(abs(Sigma_[i])>=0.001){
                    Sinv_UT_f(i,0) = (1.0/Sigma_[i])*UT_f(i,0);
                }
                else{
                    Sinv_UT_f(i,0)=0.0;
                }
            }

            double sum_gamma=0.0;
            for(int i=0;i<m_;i++){
                gamma_k_[i]=0.0;
                for(int j=0;j<r_;j++){
                    gamma_k_[i] += VT_(j,i)*Sinv_UT_f(j,0);
                }
                sum_gamma += abs(gamma_k_[i]);

            }

            if(sum_gamma>1){
                for(int i=0;i<m_;i++){
                    gamma_k_[i] = gamma_k_[i]*(1.0/sum_gamma);
                }
            }

        }


        //cout<<"here 3"<<endl;


        //Mat_1_doub Temp_F_gamma_k, Temp_X_gamma_k;
        xbar_k_.clear();fbar_k_.clear();
        xbar_k_.resize(OP_size);
        fbar_k_.resize(OP_size);
        double temp_f, temp_x;
        for(int i=0;i<OP_size;i++){
            temp_f=0.0;
            temp_x=0.0;
            for(int j=0;j<m_;j++){
                temp_f +=F_mat(i,j)*gamma_k_[j];
                temp_x +=X_mat(i,j)*gamma_k_[j];
            }
            xbar_k_[i] = x_k_[i] - 1.0*temp_x;
            fbar_k_[i] = f_k_[i] - 1.0*temp_f;
        }


        x_kp1_.clear();
        x_kp1_.resize(OP_size);
        for(int i=0;i<OP_size;i++){
            if(iter==1){
            x_kp1_[i] = (1.0 - 1.0*alpha_mixing)*xbar_k_[i]  +
                    alpha_mixing*fbar_k_[i];
            }
            else{
            x_kp1_[i] = (1.0 - 0.0*alpha_mixing)*xbar_k_[i]  +
                        alpha_mixing*fbar_k_[i];
            }
        }


        for(int i=0;i<OP_size;i++){
            if(i<OP_Real_size){
                OParams[NewInd_to_kSL_ind[i]](NewInd_to_row_ind[i], NewInd_to_col_ind[i]).real(x_kp1_[i]);
            }
            else{
                OParams[NewInd_to_kSL_ind[i]](NewInd_to_row_ind[i], NewInd_to_col_ind[i]).imag(x_kp1_[i]);
            }
        }


        for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
            for(int row_i=0;row_i<OParams[kSL_ind].n_row();row_i++){
                OParams[kSL_ind](row_i, row_i).imag(0);
                for(int col_i=0;col_i<OParams[kSL_ind].n_col();col_i++){
             if(col_i>row_i){
            OParams[kSL_ind](col_i, row_i) = conj(OParams[kSL_ind](row_i, col_i));
        }
                }}}



        //---saving arrays for next iteration-----
        x_km1_=x_k_;
        f_km1_=f_k_;

    }


/*
    if(iter%20==0 && iter>1){

        //complex<double>(Myrandom(),Myrandom())
        for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
            for(int row_i=0;row_i<OParams[kSL_ind].n_row();row_i++){
                for(int col_i=0;col_i<OParams[kSL_ind].n_col();col_i++){
                    if(col_i==row_i){
                      OParams[kSL_ind](col_i, row_i) += (1.0/(1.0*iter*iter))*complex<double>(Myrandom(),0);
                    }
                    if(col_i>row_i){
                     OParams[kSL_ind](col_i, row_i) +=  (1.0/(1.0*iter*iter))*complex<double>(Myrandom(),Myrandom());
                      OParams[kSL_ind](row_i, col_i) = conj(OParams[kSL_ind](col_i, row_i));
                    }
                }}}
    }
    */


}


void Hamiltonian::Imposing_ZeroSz(){

vector<Matrix<complex<double>>> OParams_temp;
OParams_temp=OParams;

int Spin_up=0;
int Spin_dn=1;
int row_val, row_val_p;
int col_val, col_val_p;
int spin1_bar, spin2_bar;

int k_ind_val, k_ind1, k_ind2;
int minus_k_ind1, minus_k_ind2;
int G1_ind_temp, G2_ind_temp;
int minus_k_ind_val;
int minus_k_ind_SL, minus_k_ind;

complex<double> val_avg;

//Time reversal symmetry
for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
for(int k_ind=0;k_ind<k_sublattices[kSL_ind].size();k_ind++){

    k_ind_val = k_sublattices[kSL_ind][k_ind];
    k_ind1=Coordinates_.indx_cellwise(k_ind_val);
    k_ind2=Coordinates_.indy_cellwise(k_ind_val);

    Folding_to_BrillouinZone(-1*k_ind1, -1*k_ind2, minus_k_ind1, minus_k_ind2, G1_ind_temp, G2_ind_temp);
    minus_k_ind_val=minus_k_ind1 + minus_k_ind2*l1_;

    minus_k_ind_SL = Inverse_kSublattice_mapping[minus_k_ind_val].first;
    minus_k_ind = Inverse_kSublattice_mapping[minus_k_ind_val].second;


for(int band1=0;band1<Nbands;band1++){
for(int band2=0;band2<Nbands;band2++){

for(int spin1=0;spin1<2;spin1++){
spin1_bar=1-spin1;
for(int spin2=0;spin2<2;spin2++){
spin2_bar=1-spin2;

row_val = k_ind +
          k_sublattices[kSL_ind].size()*band1 +
          k_sublattices[kSL_ind].size()*Nbands*spin1;

col_val = k_ind +
          k_sublattices[kSL_ind].size()*band2 +
          k_sublattices[kSL_ind].size()*Nbands*spin2;

row_val_p = minus_k_ind +
        k_sublattices[minus_k_ind_SL].size()*band1 +
        k_sublattices[minus_k_ind_SL].size()*Nbands*spin1_bar;

col_val_p = minus_k_ind +
        k_sublattices[minus_k_ind_SL].size()*band2 +
        k_sublattices[minus_k_ind_SL].size()*Nbands*spin2_bar;

val_avg = 0.5*(OParams_temp[kSL_ind](row_val, col_val) + OParams_temp[minus_k_ind_SL](row_val_p, col_val_p));

OParams[kSL_ind](row_val, col_val)=val_avg;

}}

}}
}}



OParams_temp=OParams;

//Inversion symmetry
for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
for(int k_ind=0;k_ind<k_sublattices[kSL_ind].size();k_ind++){

    k_ind_val = k_sublattices[kSL_ind][k_ind];
    k_ind1=Coordinates_.indx_cellwise(k_ind_val);
    k_ind2=Coordinates_.indy_cellwise(k_ind_val);

    Folding_to_BrillouinZone(-1*k_ind1, -1*k_ind2, minus_k_ind1, minus_k_ind2, G1_ind_temp, G2_ind_temp);
    minus_k_ind_val=minus_k_ind1 + minus_k_ind2*l1_;

    minus_k_ind_SL = Inverse_kSublattice_mapping[minus_k_ind_val].first;
    minus_k_ind = Inverse_kSublattice_mapping[minus_k_ind_val].second;


for(int band1=0;band1<Nbands;band1++){
for(int band2=0;band2<Nbands;band2++){

for(int spin1=0;spin1<2;spin1++){
spin1_bar=spin1;
for(int spin2=0;spin2<2;spin2++){
spin2_bar=spin2;

row_val = k_ind +
          k_sublattices[kSL_ind].size()*band1 +
          k_sublattices[kSL_ind].size()*Nbands*spin1;

col_val = k_ind +
          k_sublattices[kSL_ind].size()*band2 +
          k_sublattices[kSL_ind].size()*Nbands*spin2;

row_val_p = minus_k_ind +
        k_sublattices[minus_k_ind_SL].size()*band1 +
        k_sublattices[minus_k_ind_SL].size()*Nbands*spin1_bar;

col_val_p = minus_k_ind +
        k_sublattices[minus_k_ind_SL].size()*band2 +
        k_sublattices[minus_k_ind_SL].size()*Nbands*spin2_bar;

val_avg = 0.5*(OParams_temp[kSL_ind](row_val, col_val) + OParams_temp[minus_k_ind_SL](row_val_p, col_val_p));

OParams[kSL_ind](row_val, col_val)=val_avg;

}}

}}
}}



/*
complex<double> val_Sp, val_Sm;


for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){

for(int k_ind=0;k_ind<k_sublattices[kSL_ind].size();k_ind++){
for(int band1=0;band1<Nbands;band1++){

row_val_up = k_ind +
          k_sublattices[kSL_ind].size()*band1 +
          k_sublattices[kSL_ind].size()*Nbands*Spin_up;

row_val_dn = k_ind +
          k_sublattices[kSL_ind].size()*band1 +
          k_sublattices[kSL_ind].size()*Nbands*Spin_dn;

for(int k_ind_p=0;k_ind_p<k_sublattices[kSL_ind].size();k_ind_p++){
for(int band2=0;band2<Nbands;band2++){
col_val_up = k_ind_p +
          k_sublattices[kSL_ind].size()*band2 +
          k_sublattices[kSL_ind].size()*Nbands*Spin_up;
col_val_dn = k_ind_p +
          k_sublattices[kSL_ind].size()*band2 +
          k_sublattices[kSL_ind].size()*Nbands*Spin_dn;


val_Sp = complex<double> (0.5*OParams[kSL_ind](row_val_dn, col_val_up).real() + 0.5*OParams[kSL_ind](row_val_up, col_val_dn).real(), 0.0);
val_Sm = complex<double> (0.5*OParams[kSL_ind](row_val_dn, col_val_up).real() + 0.5*OParams[kSL_ind](row_val_up, col_val_dn).real(), 0.0);

OParams[kSL_ind](row_val_up, col_val_dn) = 0;//val_Sp;
OParams[kSL_ind](col_val_dn, row_val_up) = 0;//conj(val_Sp);

OParams[kSL_ind](row_val_dn, col_val_up) = 0;//val_Sm;
OParams[kSL_ind](col_val_up, row_val_dn) = 0;//conj(val_Sm);
}}}}}

*/




}

void Hamiltonian::Perform_SVD(Matrix<double> & A_, Matrix<double> & VT_, Matrix<double> & U_, vector<double> & Sigma_){


    char jobz='A'; //A,S,O,N

    int m=A_.n_row();
    int n=A_.n_col();
    int lda=A_.n_row();
    int ldu=A_.n_row();
    int ldvt=n;

    Sigma_.clear();
    Sigma_.resize(min(m,n));

    U_.resize(ldu,m);

    VT_.resize(ldvt,n);


    vector<double> work(3);
    int info;
    int lwork= -1;
    vector<int> iwork(8*min(m,n));

    // query:
    dgesdd_(&jobz, &m, &n, &(A_(0,0)),&lda, &(Sigma_[0]),&(U_(0,0)), &ldu, &(VT_(0,0)), &ldvt,
            &(work[0]), &lwork, &(iwork[0]), &info);
    //lwork = int(real(work[0]))+1;
    lwork = int((work[0]));
    work.resize(lwork);
    // real work:
    dgesdd_(&jobz, &m, &n, &(A_(0,0)),&lda, &(Sigma_[0]),&(U_(0,0)), &ldu, &(VT_(0,0)), &ldvt,
            &(work[0]), &lwork, &(iwork[0]), &info);
    if (info!=0) {
        if(info>0){
            std::cerr<<"info="<<info<<"\n";
            perror("diag: zheev: failed with info>0.\n");}
        if(info<0){
            std::cerr<<"info="<<info<<"\n";
            perror("diag: zheev: failed with info<0.\n");
        }
    }

    // Ham_.print();



}


void Hamiltonian::Perform_SVD_complex(Matrix<complex<double>> & A_, Matrix<complex<double>> & VT_, Matrix<complex<double>> & U_, vector<double> & Sigma_){


    char jobz='A'; //A,S,O,N

    int m=A_.n_row();
    int n=A_.n_col();
    int lda=A_.n_row();
    int ldu=A_.n_row();
    int ldvt=n;

    Sigma_.clear();
    Sigma_.resize(min(m,n));

    U_.resize(ldu,m);

    VT_.resize(ldvt,n);


    vector<complex<double>> work(3);
    int info;
    int lwork= -1;
    vector<int> iwork(8*min(m,n));
    int lrwork = max( (5*min(m,n)*min(m,n)) + 5*min(m,n), (2*max(m,n)*min(m,n)) + (2*min(m,n)*min(m,n)) + min(m,n) );
    vector<double> rwork(lrwork);

    // query:
    zgesdd_(&jobz, &m, &n, &(A_(0,0)),&lda, &(Sigma_[0]),&(U_(0,0)), &ldu, &(VT_(0,0)), &ldvt,
            &(work[0]), &lwork, &(rwork[0]), &(iwork[0]), &info);
    //lwork = int(real(work[0]))+1;
    lwork = int((work[0]).real());
    work.resize(lwork);
    // real work:
    zgesdd_(&jobz, &m, &n, &(A_(0,0)),&lda, &(Sigma_[0]),&(U_(0,0)), &ldu, &(VT_(0,0)), &ldvt,
            &(work[0]), &lwork, &(rwork[0]), &(iwork[0]), &info);
    if (info!=0) {
        if(info>0){
            std::cerr<<"info="<<info<<"\n";
            perror("diag: zheev: failed with info>0.\n");}
        if(info<0){
            std::cerr<<"info="<<info<<"\n";
            perror("diag: zheev: failed with info<0.\n");
        }
    }

    // Ham_.print();



}


void Hamiltonian::Initialize_OParams(){

    if(Parameters_.Read_OPs_bool){

        if(Parameters_.OP_Read_Type=="KSpace"){
        string line;
        int kSL_ind_temp, row_val_temp, col_val_temp;
        double OP_real, OP_imag;
//        char temp_char[50];
//        sprintf(temp_char, "%.10f", Parameters_.Temperature);

//        string inputfile = Parameters_.OP_input_file;
        ifstream fileOPin(Parameters_.OP_input_file.c_str());
        //fileOPin>>line;
        getline(fileOPin, line);
        //cout<<"'"<<Parameters_.OP_input_file<<"'"<<endl;
        //cout<<line<<endl;

        for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
            OParams[kSL_ind].resize(k_sublattices[kSL_ind].size()*2*Nbands,k_sublattices[kSL_ind].size()*2*Nbands);
        }
        for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
            for(int row_val=0;row_val<OParams[kSL_ind].n_row();row_val++){
            for(int col_val=0;col_val<OParams[kSL_ind].n_col();col_val++){
            fileOPin>>kSL_ind_temp>>row_val_temp>>col_val_temp>>OP_real>>OP_imag;
            cout<<kSL_ind_temp<<"  "<<row_val_temp<<"  "<<col_val_temp<<endl;
            cout<<kSL_ind<<"  "<<row_val<<"  "<<col_val<<endl;
            assert(kSL_ind_temp==kSL_ind);
            assert(row_val_temp==row_val);
            assert(col_val_temp==col_val);
            OParams[kSL_ind](row_val,col_val) = complex<double>(OP_real, OP_imag);
            }}
        }

        }//KSpaceType

       if(Parameters_.OP_Read_Type=="RealSpace"){
        Create_PMat();

        int k1_diff, G1_diff, k2_diff, G2_diff ;
        for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
            OParams[kSL_ind].resize(k_sublattices[kSL_ind].size()*2*Nbands,k_sublattices[kSL_ind].size()*2*Nbands);

            for(int spin=0;spin<2;spin++){
            for(int k_ind=0;k_ind<k_sublattices[kSL_ind].size();k_ind++){
            for(int band1=0;band1<Nbands;band1++){
            int row_val = k_ind +
                      k_sublattices[kSL_ind].size()*band1 +
                      k_sublattices[kSL_ind].size()*Nbands*spin;
            int k_ind_val = k_sublattices[kSL_ind][k_ind];
            int k_s_ind = k_ind_val + spin*l1_*l2_;

            for(int spin_p=0;spin_p<2;spin_p++){
            for(int k_ind_p=0;k_ind_p<k_sublattices[kSL_ind].size();k_ind_p++){
            for(int band2=0;band2<Nbands;band2++){
            int col_val = k_ind_p +
                      k_sublattices[kSL_ind].size()*band2 +
                      k_sublattices[kSL_ind].size()*Nbands*spin_p;
            int k_ind_p_val = k_sublattices[kSL_ind][k_ind_p];
            int k_s_p_ind = k_ind_p_val + spin_p*l1_*l2_;

            k1_diff =Coordinates_.indx_cellwise(k_ind_p_val)-Coordinates_.indx_cellwise(k_ind_val);
            k1_diff = k1_diff + (l1_-1);
            k2_diff =Coordinates_.indy_cellwise(k_ind_p_val)-Coordinates_.indy_cellwise(k_ind_val);
            k2_diff = k2_diff + (l2_-1);

            OParams[kSL_ind](row_val,col_val) = 0.0;

            for(int g1_1=0;g1_1<G_grid_L1;g1_1++){
                for(int g1_2=0;g1_2<G_grid_L2;g1_2++){
                    for(int layer1=0;layer1<Parameters_.max_layer_ind;layer1++){
            int g1_l1_ind=HamiltonianCont_.Coordinates_.Nbasis(g1_1, g1_2, layer1);

            for(int g2_1=0;g2_1<G_grid_L1;g2_1++){
                   for(int g2_2=0;g2_2<G_grid_L2;g2_2++){
                   for(int layer2=0;layer2<Parameters_.max_layer_ind;layer2++){
                    int g2_l2_ind=HamiltonianCont_.Coordinates_.Nbasis(g2_1, g2_2, layer2);

           G1_diff = g2_1 - g1_1 + (G_grid_L1-1);
           G2_diff = g2_2 - g1_2 + (G_grid_L2-1);

           OParams[kSL_ind](row_val,col_val) += BlochStates[spin][band1][k_ind_val][g1_l1_ind]*
                                                conj(BlochStates[spin_p][band2][k_ind_p_val][g2_l2_ind])*
                                               Pmat[k1_diff][k2_diff][G1_diff][G2_diff][spin][spin_p][layer1][layer2];

                   }}}

                }}}
            //BlochStates[spin][n][i1+i2*l1_][comp];

            }}}
            }}}

        }



       }


    }
    else{
for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
    OParams[kSL_ind].resize(k_sublattices[kSL_ind].size()*2*Nbands,k_sublattices[kSL_ind].size()*2*Nbands);
for(int row_val=0;row_val<OParams[kSL_ind].n_row();row_val++){
for(int col_val=0;col_val<OParams[kSL_ind].n_col();col_val++){
    if(row_val!=col_val){
        if(row_val>col_val){
OParams[kSL_ind](row_val,col_val) = complex<double>(Myrandom(),Myrandom());
OParams[kSL_ind](col_val,row_val) = conj(OParams[kSL_ind](row_val,col_val));
        }
    }
    else{
OParams[kSL_ind](row_val,col_val) = complex<double>(Myrandom(),0.0);
    }
}}}

    }


/*
    if(Parameters_.Imposing_SzZero){
        for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
            OParams[kSL_ind].resize(k_sublattices[kSL_ind].size()*2*Nbands,k_sublattices[kSL_ind].size()*2*Nbands);
        for(int row_val=0;row_val<OParams[kSL_ind].n_row();row_val++){
        for(int col_val=0;col_val<OParams[kSL_ind].n_col();col_val++){
            if(row_val!=col_val){
                if(row_val>col_val){
        OParams[kSL_ind](row_val,col_val) = 0.0;
        OParams[kSL_ind](col_val,row_val) = 0.0;
                }
            }
            else{
        OParams[kSL_ind](row_val,col_val) = complex<double>(Myrandom(),0.0);
            }
        }}}
    }
*/


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


    Initialize_OParams();
    if(Parameters_.Imposing_SzZero){
      Imposing_ZeroSz();
    }
    Update_Hartree_Coefficients();
    Update_Fock_Coefficients();



    for(int Temp_no=0;Temp_no<Parameters_.Temperature_points.size();Temp_no++){

        Parameters_.Temperature = Parameters_.Temperature_points[Temp_no];
        Temperature = Parameters_.Temperature;
        beta_=1.0/(KB_*Temperature);

        cout<<"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"<<endl;
        cout<<"Temperature = "<<Parameters_.Temperature<<endl;

        char temp_char[50];
        sprintf(temp_char, "%.10f", Parameters_.Temperature);

    string file_conv="HF_out_Temp" + string(temp_char) +".txt";
    ofstream Convergence_out(file_conv.c_str());
    Convergence_out<<"#iter   OP_diff   mu  QuantEnergy.real  .imag  ClassEnergy.real  .imag  Total_n_up    Total_n_dn   Energy_diff"<<endl;

    int N_Particles;
    N_Particles = int((nu_holes_target*ns_)+0.5);
    double diff_=1000;
    int iter=0;
    double mu_old;

    double Energy_old =0.0;
    double Energy_new;
    double Energy_diff=10000.0;

    while( !(iter>HF_max_iterations || (diff_<HF_convergence_error && Energy_diff<HF_convergence_error)) ){

    for(int kset_ind=0;kset_ind<k_sublattices.size();kset_ind++){
        
        Create_Hamiltonian(kset_ind);
        char Dflag='V';

//         string filename1 = "Iter_"+to_string(iter)+"_kSL_ind_"+to_string(kset_ind)+"_Hamil.txt";
//         Print_Spectrum(kset_ind, filename1);

        Diagonalize(Dflag);
        AppendEigenspectrum(kset_ind);

//         string filename = "Iter_"+to_string(iter)+"_kSL_ind_"+to_string(kset_ind)+"_Spectrum.txt";
//         Print_Spectrum(kset_ind, filename);

    }

    if(iter==0){
        //near bottom of first band
        mu_old = EigValues[0][0] + 0.5;
        cout<<"initial mu = "<<mu_old<<endl;
    }
    
    
    mu_=chemicalpotential(mu_old,N_Particles);

    Calculate_OParams_and_diff(diff_);


    Calculate_Total_Energy();
    Energy_new = Total_QuantEnergy.real() + Total_ClassEnergy.real();
    Energy_diff = abs(Energy_new - Energy_old);
    if(Convergence_technique=="SimpleMixing"){
    Update_OParams_SimpleMixing();}
    else{
    assert(Convergence_technique=="AndersonMixing");
    Update_OrderParameters_AndersonMixing(iter);
    }

    if(Parameters_.Imposing_SzZero){
      Imposing_ZeroSz();
    }

    Update_Hartree_Coefficients();
    Update_Fock_Coefficients();
    mu_old=mu_;

    Convergence_out<<iter<<"   "<<diff_<<"   ";
    Convergence_out<<setprecision(10)<< scientific;
    Convergence_out<<mu_<<"   "<<Total_QuantEnergy.real()<<"  "<<Total_QuantEnergy.imag()<<"  "<<Total_ClassEnergy.real()<<"   "<<Total_ClassEnergy.imag()<<"   "<<Total_n_up<<"  "<<Total_n_dn<<"   "<<Energy_diff<<endl;

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


    Energy_old=Energy_new;
    iter++;
    }




    //One extra iteration
    if(false){
    Update_Hartree_Coefficients();
    Update_Fock_Coefficients();
    mu_old=mu_;
    for(int kset_ind=0;kset_ind<k_sublattices.size();kset_ind++){

        Parameters_.MagField_ZeemanSplitting=0.0;
        Create_Hamiltonian(kset_ind);
        char Dflag='V';

//         string filename1 = "Iter_"+to_string(iter)+"_kSL_ind_"+to_string(kset_ind)+"_Hamil.txt";
//         Print_Spectrum(kset_ind, filename1);

        Diagonalize(Dflag);
        AppendEigenspectrum(kset_ind);

//         string filename = "Iter_"+to_string(iter)+"_kSL_ind_"+to_string(kset_ind)+"_Spectrum.txt";
//         Print_Spectrum(kset_ind, filename);

    }
    mu_=chemicalpotential(mu_old,N_Particles);
    Calculate_OParams_and_diff(diff_);
    }


    string file_OP="Temperature_"+ string(temp_char) +Parameters_.OP_out_file;
    ofstream file_OP_out(file_OP.c_str());
    file_OP_out<<"#k_sublattice  row_val   col_val  value.real  value.imag"<<endl;
for(int kSL_ind=0;kSL_ind<k_sublattices.size();kSL_ind++){
for(int row_val=0;row_val<OParams[kSL_ind].n_row();row_val++){
for(int col_val=0;col_val<OParams[kSL_ind].n_col();col_val++){
    file_OP_out<<kSL_ind<<"  "<<row_val<<"  "<<col_val<<"  "<<OParams[kSL_ind](row_val,col_val).real()<<"  "<<OParams[kSL_ind](row_val,col_val).imag()<<endl;
}}}


    Calculate_Total_Spin();
    Calculate_layer_resolved_densities();

    string DOSFILE = "DOS_"+string(temp_char)+".txt";
    Print_SPDOS(DOSFILE);

    string EigenFile = "Eigenvalues"+string(temp_char)+".txt";
    Write_ordered_spectrum(EigenFile);

    Print_HF_Bands();

    if(abs(Parameters_.NMat_det)==1){
    Calculate_ChernNumbers_HFBands();
    cout<<"-------Non Abelian Chern number---------"<<endl;
    Calculate_NonAbelianChernNumbers_HFBands();
    cout<<"-----------------------------------------"<<endl;

    Calculate_QuantumGeometry_using_Projectors();

    }


    //Print_HF_Band_Projected_Interaction();
    //Print_HF_Band_Projected_Interaction_TR_and_Inversion_imposed();


    string Oparams1_str = "Temp_"+string(temp_char)+"RealSpace_OParams_moiresites.txt";
    string Oparams2_str = "Temp_"+string(temp_char)+"RealSpace_OParams.txt";

   // Calculate_RealSpace_OParams_important_positions_new3(Oparams1_str);
   // Calculate_RealSpace_OParams_new3(Oparams2_str);

    //Here
    //Kick_OParams(0.01);


    //Calculate_RealSpace_OParams_new("RealSpace_OParams_new.txt");

    //Calculate_RealSpace_OParams("RealSpace_OParams.txt","RealSpace_OParams2.txt");

    }//Temperature


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

    if(spin==spinp && n==np){
    BO[n][spin][k_ind][np][spinp][kp_ind] += conj(BlochStates_old_[spin][n][k_ind1+k_ind2*(l1_+1)][comp])*
                                         (BlochStates_old_[spinp][np][kp_ind1+kp_ind2*(l1_+1)][comp]);
        }
}

}}}}
}}}}



}


void Hamiltonian::Saving_PBZ_BlochState_Overlaps(){

int k_ind, kp_ind;
for(int n=0;n<Nbands;n++){
for(int spin=0;spin<2;spin++){
for(int k_ind1=0;k_ind1<(l1_);k_ind1++){
for(int k_ind2=0;k_ind2<(l2_);k_ind2++){
k_ind = k_ind1 + k_ind2*(l1_);

for(int np=0;np<Nbands;np++){
for(int spinp=0;spinp<2;spinp++){
for(int kp_ind1=0;kp_ind1<(l1_);kp_ind1++){
for(int kp_ind2=0;kp_ind2<(l2_);kp_ind2++){
kp_ind = kp_ind1 + kp_ind2*(l1_);

BO_PBZ[n][spin][k_ind][np][spinp][kp_ind] =0.0;

for(int comp=0;comp<G_grid_L1*G_grid_L2*2;comp++){
    //cout<<"Here: "<<n<<" "<<spin<<" "<<k_ind<<" "<<np<<" "<<spinp<<" "<<kp_ind<<" "<<comp<<"  "<<endl;

    if(spin==spinp && n==np){
    BO_PBZ[n][spin][k_ind][np][spinp][kp_ind] += conj(BlochStates[spin][n][k_ind1+k_ind2*(l1_)][comp])*
                                         (BlochStates[spinp][np][kp_ind1+kp_ind2*(l1_)][comp]);
        }
}

}}}}
}}}}



}

complex<double> Hamiltonian::Determinant(Matrix<complex<double>> & A_){

    complex<double> det_;
    int ncol, nrow;
    ncol=A_.n_col();
    nrow=A_.n_row();
    assert(ncol==2);
    assert(nrow==2);

    det_ = A_(0,0)*A_(1,1) - A_(0,1)*A_(1,0);

    return det_;

}


complex<double> Hamiltonian::Get_U_mat_NonAbelian(int mx_left, int my_left, int mx_right, int my_right, Mat_1_int Bands_){


    int n_left_HF, n_right_HF, nx_left_HF, ny_left_HF, nx_right_HF, ny_right_HF;
    nx_left_HF=mx_left%l1_; nx_right_HF=mx_right%l1_;
    ny_left_HF=my_left%l2_; ny_right_HF=my_right%l2_;
    n_right_HF = nx_right_HF + ny_right_HF*(l1_);
    n_left_HF = nx_left_HF + ny_left_HF*(l1_);

    int n_left, n_right;
    n_right = mx_right + my_right*(l1_);
    n_left = mx_left + my_left*(l1_);

    complex<double> U_mat_val=0.0;
    int NBands = Bands_.size();

    Matrix<complex<double>> Overlap_Mat;
    Overlap_Mat.resize(NBands,NBands);


    for(int row_=0;row_<NBands;row_++){
    for(int col_=0;col_<NBands;col_++){
    Overlap_Mat(row_,col_)=0.0;


    int q_kSL_ind_left = Inverse_kSublattice_mapping[n_left_HF].first;
    int q_kSL_internal_ind_left = Inverse_kSublattice_mapping[n_left_HF].second;
    int q_kSL_ind_right = Inverse_kSublattice_mapping[n_right_HF].first;
    int q_kSL_internal_ind_right = Inverse_kSublattice_mapping[n_right_HF].second;

    for(int band_n=0;band_n<Nbands;band_n++){
    for(int spin=0;spin<2;spin++){
    for(int band_np=0;band_np<Nbands;band_np++){
    for(int spin_p=0;spin_p<2;spin_p++){

    int col_val_left =  q_kSL_internal_ind_left +
                    k_sublattices[q_kSL_ind_left].size()*band_n +
                    k_sublattices[q_kSL_ind_left].size()*Nbands*spin;

    int col_val_right =  q_kSL_internal_ind_right +
                    k_sublattices[q_kSL_ind_right].size()*band_np +
                    k_sublattices[q_kSL_ind_right].size()*Nbands*spin_p;


     Overlap_Mat(row_, col_) += BO_PBZ[band_n][spin][n_left_HF][band_np][spin_p][n_right_HF]*
            conj(EigVectors[q_kSL_ind_left](col_val_left,Bands_[row_]))*
            (EigVectors[q_kSL_ind_right](col_val_right,Bands_[col_]));
    }}
    }}


    }}


    U_mat_val = Determinant(Overlap_Mat);

    return U_mat_val;
}



void Hamiltonian::Calculate_Band_Projector(Mat_1_int Bands_, int nx_, int ny_, Mat_2_Complex_doub & Projector_full){


    int n_ind = nx_ + ny_*(l1_);
    int q_kSL_ind = Inverse_kSublattice_mapping[n_ind].first;
    int q_kSL_internal_ind = Inverse_kSublattice_mapping[n_ind].second;

//Projector_band_resolved.resize(Bands_.size());
//for(int m=0;m<Bands_.size();m+){
//   Projector_band_resolved[m].resize(G_grid_L1*G_grid_L2*2);
//   for(int comp1_=0;comp1_<G_grid_L1*G_grid_L2*2;comp1_++){
//   Projector_band_resolved[m][comp1_].resize(G_grid_L1*G_grid_L2*2);
//   }
//}


Projector_full.resize(G_grid_L1*G_grid_L2*2);
for(int comp1_=0;comp1_<G_grid_L1*G_grid_L2*2;comp1_++){
Projector_full[comp1_].resize(G_grid_L1*G_grid_L2*2);
for(int comp2_=0;comp2_<G_grid_L1*G_grid_L2*2;comp2_++){
  Projector_full[comp1_][comp2_]=0.0;
}
}

for(int m=0;m<Bands_.size();m++){
    for(int comp1_=0;comp1_<G_grid_L1*G_grid_L2*2;comp1_++){
    for(int comp2_=0;comp2_<G_grid_L1*G_grid_L2*2;comp2_++){

    //Projector_band_resolved[m][comp1_][comp2_]=0.0;
    for(int n=0;n<Nbands;n++){
    for(int spin=0;spin<2;spin++){
        int col_val =  q_kSL_internal_ind +
                            k_sublattices[q_kSL_ind].size()*n +
                            k_sublattices[q_kSL_ind].size()*Nbands*spin;

        for(int n_p=0;n_p<Nbands;n_p++){
        for(int spin_p=0;spin_p<2;spin_p++){
        int col_val_p =  q_kSL_internal_ind +
                            k_sublattices[q_kSL_ind].size()*n_p +
                            k_sublattices[q_kSL_ind].size()*Nbands*spin_p;

    Projector_full[comp1_][comp2_] +=
            BlochStates[spin_p][n_p][nx_+ny_*(l1_)][comp1_]*
            conj(BlochStates[spin][n][nx_+ny_*(l1_)][comp2_])*
            conj(EigVectors[q_kSL_ind](col_val,Bands_[m]))*
            (EigVectors[q_kSL_ind](col_val_p,Bands_[m]));



        }}
    }}
}}

}


}


void Hamiltonian::Calculate_QuantumGeometry_using_Projectors(){

    assert(abs(Parameters_.NMat_det)==1);

    int L1_,L2_;
    L1_=l1_; //along G1 (b6)
    L2_=l2_; //along G2 (b2)


//    Mat_2_Complex_doub g_metric;
//    g_metric.resize(2);
//    for(int alpha=0;alpha<2;alpha++){
//    g_metric[alpha].resize(2);
//    }

    int NBands_in_a_set=2;
    int N_bands_Chern =  Nbands*2; //2 here for spin;
    int N_Sets=int(N_bands_Chern/NBands_in_a_set);

    int n_p_alpha_x, n_p_alpha_y;
    int n_p_beta_x,  n_p_beta_y;

    Mat_2_Complex_doub P_k, P_k_plus_alpha, P_k_plus_beta;
    Mat_2_Complex_doub del_alpha_P, del_beta_P;
    Mat_2_Complex_doub prod_P;
    Mat_2_Complex_doub prod_P_Berry;


    del_alpha_P.resize(G_grid_L1*G_grid_L2*2);
    del_beta_P.resize(G_grid_L1*G_grid_L2*2);
    prod_P.resize(G_grid_L1*G_grid_L2*2);
    prod_P_Berry.resize(G_grid_L1*G_grid_L2*2);
    for(int comp1_=0;comp1_<G_grid_L1*G_grid_L2*2;comp1_++){
    del_alpha_P[comp1_].resize(G_grid_L1*G_grid_L2*2);
    del_beta_P[comp1_].resize(G_grid_L1*G_grid_L2*2);
    prod_P[comp1_].resize(G_grid_L1*G_grid_L2*2);
    prod_P_Berry[comp1_].resize(G_grid_L1*G_grid_L2*2);
    }

    for(int alpha=0;alpha<2;alpha++){
    for(int beta=0;beta<2;beta++){


    for (int band_set = 0; band_set < N_Sets; band_set++)
    {

        string File_metric_str = "QGeometry_"+to_string(alpha)+"_"+to_string(beta)+
                                 "BandSet_" +to_string(band_set)+ ".txt";
        ofstream File_metric_out(File_metric_str.c_str());

        complex<double> g_trace=0.0;
        complex<double> b_trace=0.0;


        Mat_1_int Bands_;
        for(int b=0;b<NBands_in_a_set;b++){
        Bands_.push_back(band_set*NBands_in_a_set + b);
        }


        for (int nx = 0; nx < L1_; nx++)
        {
            for (int ny = 0; ny < L2_; ny++)
            {

                    n_p_alpha_x = (nx + (1-alpha))%L1_;
                    n_p_alpha_y = (ny + (alpha))%L2_;

                        n_p_beta_x = (nx + (1-beta))%L1_;
                        n_p_beta_y = (ny + (beta))%L2_;


                        Calculate_Band_Projector(Bands_,nx,ny,P_k);
                        Calculate_Band_Projector(Bands_,n_p_alpha_x,n_p_alpha_y,P_k_plus_alpha);
                        Calculate_Band_Projector(Bands_,n_p_beta_x,n_p_beta_y,P_k_plus_beta);


                        for(int comp1_=0;comp1_<G_grid_L1*G_grid_L2*2;comp1_++){
                            for(int comp2_=0;comp2_<G_grid_L1*G_grid_L2*2;comp2_++){
                        del_alpha_P[comp1_][comp2_] = P_k_plus_alpha[comp1_][comp2_] - P_k[comp1_][comp2_];
                        del_beta_P[comp1_][comp2_] = P_k_plus_beta[comp1_][comp2_] - P_k[comp1_][comp2_];
                            }
                        }


                        g_trace=0.0;
                        for(int comp1_=0;comp1_<G_grid_L1*G_grid_L2*2;comp1_++){
                             prod_P[comp1_][comp1_]=0.0;
                            for(int comp3_=0;comp3_<G_grid_L1*G_grid_L2*2;comp3_++){
                            prod_P[comp1_][comp1_] += del_alpha_P[comp1_][comp3_]*del_beta_P[comp3_][comp1_];
                            }
                         g_trace +=0.5*prod_P[comp1_][comp1_];
                        }


                        b_trace=0.0;
                        for(int comp1_=0;comp1_<G_grid_L1*G_grid_L2*2;comp1_++){
                             prod_P_Berry[comp1_][comp1_]=0.0;
                         for(int comp2_=0;comp2_<G_grid_L1*G_grid_L2*2;comp2_++){
                            for(int comp3_=0;comp3_<G_grid_L1*G_grid_L2*2;comp3_++){
                            prod_P_Berry[comp1_][comp1_] += P_k[comp1_][comp2_]*del_alpha_P[comp2_][comp3_]*del_beta_P[comp3_][comp1_]
                                                            - P_k[comp1_][comp2_]*del_beta_P[comp2_][comp3_]*del_alpha_P[comp3_][comp1_];
                            }
                         }
                         b_trace +=0.5*prod_P_Berry[comp1_][comp1_];
                        }

                        File_metric_out.precision(10);

                       File_metric_out<<nx<<"  "<<ny<<"  "<<g_trace.real()<<"  "<<g_trace.imag()
                                     <<"  "<<b_trace.real()<<"  "<<b_trace.imag()<<endl;

                       if(ny==(L2_-1)){//For pm3d corners2color c1
                           File_metric_out<<nx<<"  "<<ny+1<<"  "<<g_trace.real()<<"  "<<g_trace.imag()
                                            <<"  "<<b_trace.real()<<"  "<<b_trace.imag()<<endl;

                       }


            }


            File_metric_out<<endl;
            if(nx==(L1_-1)){
                for (int ny = 0; ny < L2_+1; ny++)
                {
                    File_metric_out<<nx+1<<"  "<<ny<<"  "<<g_trace.real()<<"  "<<g_trace.imag()
                                     <<"  "<<b_trace.real()<<"  "<<b_trace.imag()<<endl;
                }
            }

        }


    }

}}


}

void Hamiltonian::Calculate_NonAbelianChernNumbers_HFBands(){


    assert(abs(Parameters_.NMat_det)==1);

    int L1_,L2_;
    L1_=l1_; //along G1 (b6)
    L2_=l2_; //along G2 (b2)

    int mbz_factor=1;
    Saving_PBZ_BlochState_Overlaps();


    int NBands_in_a_set=2;
    int N_bands_Chern =  Nbands*2; //2 here for spin;
    int N_Sets=int(N_bands_Chern/NBands_in_a_set);

//    int q_kSL_ind_left, q_kSL_internal_ind_left;
//    int q_kSL_ind_right, q_kSL_internal_ind_right;

   Matrix<complex<double>> F_mat; //F1, F2, F3, F4, F5;
   F_mat.resize(N_bands_Chern, ((L1_))*((L2_)));

   complex<double> Ux_k, Uy_k, Ux_kpy, Uy_kpx;
   vector<complex<double>> F_bands;
   F_bands.resize(N_bands_Chern);
   vector<complex<double>> Chern_num;
   Chern_num.resize(N_bands_Chern);

   vector<complex<double>> F_bands_orgnl;
   F_bands_orgnl.resize(N_bands_Chern);
   vector<complex<double>> Chern_num_orgnl;
   Chern_num_orgnl.resize(N_bands_Chern);
   for (int band_set = 0; band_set < N_Sets; band_set++)
   {

       Mat_1_int Bands_;
       for(int b=0;b<NBands_in_a_set;b++){
       Bands_.push_back(band_set*NBands_in_a_set + b);
       }

       string file_Fk="Fk_band_NonAbelian"+to_string(band_set)+ ".txt";
       ofstream fl_Fk_out(file_Fk.c_str());
       fl_Fk_out<<"#nx  ny  tilde_F(nx,ny).real()  tilde_F(nx,ny).imag()  ArgofLog.real()  ArgofLog.imag()"<<endl;
       fl_Fk_out<<"#Extra momentum point for pm3d corners2color c1"<<endl;


       F_bands[band_set] = 0.0;
       F_bands_orgnl[band_set] = 0.0;
       // for (int nx = L1_/2; nx < 3*L1_/2; nx++)
       // {
       //     for (int ny = L2_/2; ny < 3*L2_/2; ny++)
       //     {

       for (int nx = 0; nx < L1_; nx++)
       {
           for (int ny = 0; ny < L2_; ny++)
           {

               int n_ind = nx + ny*(L1_); //for k=( 2*nx*Pi/(Lx/2),  2*ny*Pi/(Ly/2) )

               int mx=nx;
               int my=ny;
               int mx_left, my_left, mx_right, my_right;

               //U1_k
               mx_left = mx;
               my_left = my;
               mx_right = (mx + 1);// % (mbz_factor*L1_);
               my_right = my;
               //Ux_k = Get_U_mat_old(mx_left, my_left, mx_right, my_right, band);
               Ux_k = Get_U_mat_NonAbelian(mx_left, my_left, mx_right, my_right, Bands_);
               Ux_k = Ux_k * (1.0 / abs(Ux_k));


               //U2_kpx
               mx_left = (mx + 1);// % (mbz_factor*L1_);
               my_left = my;
               mx_right = mx_left;
               my_right = (my_left + 1);// % (mbz_factor*L2_);
               //Uy_kpx = Get_U_mat_old(mx_left, my_left, mx_right, my_right, band);
               Uy_kpx = Get_U_mat_NonAbelian(mx_left, my_left, mx_right, my_right, Bands_);
               Uy_kpx = Uy_kpx * (1.0 / abs(Uy_kpx));


               //U1_kpy
               mx_left = mx;
               my_left = (my + 1);// % (mbz_factor*L2_);
               mx_right = (mx_left + 1);// % (mbz_factor*L1_);
               my_right = my_left;
               //Ux_kpy = Get_U_mat_old(mx_left, my_left, mx_right, my_right,band);
               Ux_kpy = Get_U_mat_NonAbelian(mx_left, my_left, mx_right, my_right, Bands_);
               Ux_kpy = Ux_kpy * (1.0 / abs(Ux_kpy));

               //U2_k
               Uy_k = 0;
               mx_left = mx;
               my_left = my;
               mx_right = mx_left;
               my_right = (my_left + 1);// % (mbz_factor*L2_);
               //Uy_k = Get_U_mat_old(mx_left, my_left, mx_right, my_right, band);
               Uy_k = Get_U_mat_NonAbelian(mx_left, my_left, mx_right, my_right, Bands_);
               Uy_k = Uy_k * (1.0 / abs(Uy_k));

               // Calculating tilde F12
               F_mat(band_set, n_ind) = log((Ux_k) *
                                    (Uy_kpx) *
                                    conj(Ux_kpy) * conj(Uy_k));

               F_bands[band_set] += (F_mat(band_set, n_ind));


               fl_Fk_out.precision(10);

               fl_Fk_out<<nx<<"  "<<ny<<"  "<<F_mat(band_set, n_ind).real()<<"  "<<F_mat(band_set, n_ind).imag()<<
                          "  "<<(Ux_k*Uy_kpx*conj(Ux_kpy)*conj(Uy_k)).real()<<
                          "  "<<(Ux_k*Uy_kpx*conj(Ux_kpy)*conj(Uy_k)).imag()<<endl;




               if(abs((abs(F_mat(band_set, n_ind).imag()) - PI))<0.001){
                   cout<<ny<<"  "<<nx<<"  gives Pi for band"<< band_set <<endl;
                   // assert (abs((abs(F_mat(band, n).imag()) - M_PI))>0.0000001);
               }


               if(ny==(L2_-1)){//For pm3d corners2color c1
                   fl_Fk_out<<nx<<"  "<<ny+1<<"  "<<F_mat(band_set, n_ind).real()<<"  "<<F_mat(band_set, n_ind).imag()<<
                              "  "<<(Ux_k*Uy_kpx*conj(Ux_kpy)*conj(Uy_k)).real()<<
                              "  "<<(Ux_k*Uy_kpx*conj(Ux_kpy)*conj(Uy_k)).imag()<<endl;
               }
           }

           if(nx==(L1_)-1){//For pm3d corners2color c1
               fl_Fk_out<<endl;
               for(int ny_=0;ny_<L2_;ny_++){
                   int n_ = nx + L1_*ny_;
                   fl_Fk_out<<nx+1<<"  "<<ny_<<"  "<<F_mat(band_set, n_).real()<<"  "<<F_mat(band_set, n_).imag()<<
                              "  "<<(Ux_k*Uy_kpx*conj(Ux_kpy)*conj(Uy_k)).real()<<
                              "  "<<(Ux_k*Uy_kpx*conj(Ux_kpy)*conj(Uy_k)).imag()<<endl;

                  if(ny_==(L2_)-1){//For pm3d corners2color c1
                   fl_Fk_out<<nx+1<<"  "<<ny_+1<<"  "<<F_mat(band_set, n_).real()<<"  "<<F_mat(band_set, n_).imag()<<
                              "  "<<(Ux_k*Uy_kpx*conj(Ux_kpy)*conj(Uy_k)).real()<<
                              "  "<<(Ux_k*Uy_kpx*conj(Ux_kpy)*conj(Uy_k)).imag()<<endl;
               }
               }
           }


           fl_Fk_out<<endl;
       }


       Chern_num[band_set] = (-1.0 * iota_complex / (2 * PI *mbz_factor*mbz_factor)) * F_bands[band_set];
       // Chern_num_orgnl[band] = (-1.0 * iota_complex / (2 * PI)) * F_bands_orgnl[band];
       fl_Fk_out<<"#Chern no*2pi*Iota= "<<F_bands[band_set].real()<<"  "<<F_bands[band_set].imag()<<endl;
       cout << "tilde Chern number [" << band_set << "] = " << Chern_num[band_set].real() << "        " << Chern_num[band_set].imag() << endl;
       //  cout << "Chern number [" << band << "] = " << Chern_num_orgnl[band].real() << " " << Chern_num_orgnl[band].imag() << endl;

   }

cout<<"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"<<endl;



}



void Hamiltonian::Calculate_ChernNumbers_HFBands(){

    assert(abs(Parameters_.NMat_det)==1);


    int mbz_factor=1;
    Saving_PBZ_BlochState_Overlaps();
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


//                 Ux_k += BO[band_n][spin][n_left][band_np][spin_p][n_right]*
//                        conj(EigVectors[q_kSL_ind_left](col_val_left,band))*
//                        (EigVectors[q_kSL_ind_right](col_val_right,band));
                Ux_k += BO_PBZ[band_n][spin][n_left_HF][band_np][spin_p][n_right_HF]*
                       conj(EigVectors[q_kSL_ind_left](col_val_left,band))*
                       (EigVectors[q_kSL_ind_right](col_val_right,band));

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


//                 Uy_kpx += BO[band_n][spin][n_left][band_np][spin_p][n_right]*
//                        conj(EigVectors[q_kSL_ind_left](col_val_left,band))*
//                        (EigVectors[q_kSL_ind_right](col_val_right,band));
                Uy_kpx += BO_PBZ[band_n][spin][n_left_HF][band_np][spin_p][n_right_HF]*
                          conj(EigVectors[q_kSL_ind_left](col_val_left,band))*
                          (EigVectors[q_kSL_ind_right](col_val_right,band));

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


//                 Ux_kpy += BO[band_n][spin][n_left][band_np][spin_p][n_right]*
//                        conj(EigVectors[q_kSL_ind_left](col_val_left,band))*
//                        (EigVectors[q_kSL_ind_right](col_val_right,band));

                Ux_kpy += BO_PBZ[band_n][spin][n_left_HF][band_np][spin_p][n_right_HF]*
                       conj(EigVectors[q_kSL_ind_left](col_val_left,band))*
                       (EigVectors[q_kSL_ind_right](col_val_right,band));

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


//                Uy_k += BO[band_n][spin][n_left][band_np][spin_p][n_right]*
//                        conj(EigVectors[q_kSL_ind_left](col_val_left,band))*
//                        (EigVectors[q_kSL_ind_right](col_val_right,band));

                Uy_k += BO_PBZ[band_n][spin][n_left_HF][band_np][spin_p][n_right_HF]*
                        conj(EigVectors[q_kSL_ind_left](col_val_left,band))*
                        (EigVectors[q_kSL_ind_right](col_val_right,band));

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


    char temp_char[50];
    sprintf(temp_char, "%.10f", Temperature);


string file_bands="Bands_HF"+string(temp_char)+".txt";
ofstream file_bands_out(file_bands.c_str());

double overlap_top, overlap_bottom;
int q_ind1, q_ind2, q_ind;
int q1_kSL_ind, q1_kSL_internal_ind;
Mat_1_intpair k_path_;
double kx_val, ky_val;
k_path_ = Get_k_path(2);

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

for(int layer=0;layer<Parameters_.max_layer_ind;layer++){
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

if(Parameters_.max_layer_ind==2){
overlap_top=overlaps_[TOP_];
overlap_bottom=overlaps_[BOTTOM_];
}
if(Parameters_.max_layer_ind==1){
overlap_bottom=overlaps_[0];
overlap_top=0.0;
}

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




        //path no. =2
        if(path_no==2){
            for(int n1_=0;n1_<L1_;n1_++){
             for(int n2_=0;n2_<L2_;n2_++){
            temp_pair.first = n1_;
            temp_pair.second = n2_;
            k_path.push_back(temp_pair);
             }}
        }


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


//LambdaNew_[spin][n1][n2][k1][k2][q1][q2][g1]


cout<<"Calculating and Saving FormFactors"<<endl;

int k1_1_ind, k1_2_ind, k2_1_ind, k2_2_ind;
int g1_ind, g2_ind;
for(int spin=0;spin<2;spin++){
for(int n1=0;n1<Nbands;n1++){
for(int n2=0;n2<Nbands;n2++){
/*
 string file_FF="Lambda_k_band_spin"+to_string(spin)+"_bands"+to_string(n1)+"_"+to_string(n2)+"Version1.txt";
 ofstream fl_FF_out(file_FF.c_str());
 fl_FF_out<<"#k1(=kx+ky*l1_)   k2    Lambda[spin][n1][n2][k1][k2].real()   Lambda[spin][n1][n2][k1][k2].imag()"<<endl;
*/

for(int k1_1_=-(l1_-1);k1_1_<=l1_-1;k1_1_++){
k1_1_ind = k1_1_ + (l1_-1);
for(int k1_2_=-(l2_-1);k1_2_<=l2_-1;k1_2_++){
k1_2_ind = k1_2_ + (l2_-1);

for(int k2_1_=0;k2_1_<=2*l1_-2;k2_1_++){
k2_1_ind = k2_1_;
for(int k2_2_=0;k2_2_<=2*l2_-2;k2_2_++){
k2_2_ind = k2_2_;

for(int g1_=Lambda_G_grid_L1_min;g1_<=Lambda_G_grid_L1_max;g1_++){
g1_ind = g1_ - Lambda_G_grid_L1_min;
for(int g2_=Lambda_G_grid_L2_min;g2_<=Lambda_G_grid_L2_max;g2_++){
g2_ind = g2_ - Lambda_G_grid_L2_min;

//cout<<spin<<"(2) "<<n1<<"("<<Nbands<<") "<<n2<<"("<<Nbands<<") "<<k1_ind<<"("<<ns_<<") "<<k2_ind<<"("<<ns_<<")"<<endl;

LambdaNew_[spin][n1][n2][k1_1_ind][k1_2_ind][k2_1_ind][k2_2_ind][g1_ind][g2_ind]=FormFactor(spin, n1, n2, k1_1_, k1_2_, k2_1_ + (g1_*l1_), k2_2_ + (g2_*l2_) );


}}
//fl_FF_out<<k1_ind<<"  "<<k2_ind<<"  "<<Lambda_[spin][n1][n2][k1_ind][k2_ind].real()<<"  "<<Lambda_[spin][n1][n2][k1_ind][k2_ind].imag()<<endl;
}}
//fl_FF_out<<endl;
}}

}
}
}


}


void Hamiltonian::PrintFormFactors_PBZ(int band1, int band2, int spin){


    for(int k1_1=0;k1_1<l1_;k1_1++){
    for(int k1_2=0;k1_2<l2_;k1_2++){
    int k1 = k1_1 + k1_2*l1_;

    for(int q_1=0;q_1<l1_;q_1++){
    for(int q_2=0;q_2<l2_;q_2++){
    int q = q_1 + q_2*l1_;


    string file_FF1="LambdaPBZ_k2pq_band_spin"+to_string(spin)+"_bands"+to_string(band1)+"_"+to_string(band2)+ "_k2_q_" +to_string(k1) +"_" + to_string(q) +  ".txt";
    ofstream fl_FF_out1(file_FF1.c_str());
    fl_FF_out1<<"#g1   g2   Lambda[spin][k2][q][g1,g2].real()   Lambda[spin][k2][q][g1,g2].imag()"<<endl;

    string file_FF="LambdaPBZ_k1mq_band_spin"+to_string(spin)+"_bands"+to_string(band1)+"_"+to_string(band2)+ "_k1_q_" +to_string(k1) +"_" + to_string(q) +  ".txt";
    ofstream fl_FF_out(file_FF.c_str());
    fl_FF_out<<"#g1   g2   Lambda[spin][k1][q][g1,g2].real()   Lambda[spin][k1][q][g1,g2].imag()"<<endl;


    for(int g_ind1=Lambda_G_grid_L1_min;g_ind1<=Lambda_G_grid_L1_max;g_ind1++){
    int g1_ind = g_ind1 - Lambda_G_grid_L1_min;
    for(int g_ind2=Lambda_G_grid_L2_min;g_ind2<=Lambda_G_grid_L2_max;g_ind2++){
    int g2_ind = g_ind2 - Lambda_G_grid_L2_min;

    fl_FF_out<<g_ind1<<"   "<<g_ind2<<"  "<<LambdaPBZ_k1_m_q[spin][band1][band2][k1_1][k1_2][q_1][q_2][g1_ind][g2_ind].real()<< "  "<<LambdaPBZ_k1_m_q[spin][band1][band2][k1_1][k1_2][q_1][q_2][g1_ind][g2_ind].imag()<<endl;

    fl_FF_out1<<g_ind1<<"   "<<g_ind2<<"  "<<LambdaPBZ_k2_p_q[spin][band1][band2][k1_1][k1_2][q_1][q_2][g1_ind][g2_ind].real()<< "  "<<LambdaPBZ_k2_p_q[spin][band1][band2][k1_1][k1_2][q_1][q_2][g1_ind][g2_ind].imag()<<endl;



    }
    fl_FF_out<<endl;
    fl_FF_out1<<endl;
    }

    }}
    }}

}

void Hamiltonian::PrintFormFactors2(int band1, int band2, int spin){



// int k1_1, k1_2, k2_1, k2_2;
// k1_1=1; k1_2=0;
// k2_1=0; k2_2=0;

  for(int k1_1=0;k1_1<l1_;k1_1++){
  for(int k1_2=0;k1_2<l2_;k1_2++){
  int k1 = k1_1 + k1_2*l1_;

  for(int k2_1=0;k2_1<l1_;k2_1++){
  for(int k2_2=0;k2_2<l2_;k2_2++){
  int k2 = k2_1 + k2_2*l1_;


   string file_FF="Lambda_k_band_spin"+to_string(spin)+"_bands"+to_string(band1)+"_"+to_string(band2)+ "_k1_k2_" +to_string(k1) +"_" + to_string(k2) +  ".txt";
   ofstream fl_FF_out(file_FF.c_str());
   fl_FF_out<<"#g1   g2   Lambda[spin][k1][k2][g1,g2].real()   Lambda[spin][k1][k2][g1,g2].imag()"<<endl;

 int k1_1_ind = k1_1 + (l1_-1);
 int k1_2_ind = k1_2 + (l2_-1);

 int k2_1_ind = k2_1;
 int k2_2_ind = k2_2;


     for(int g_ind1=Lambda_G_grid_L1_min;g_ind1<=Lambda_G_grid_L1_max;g_ind1++){
     int g1_ind = g_ind1 - Lambda_G_grid_L1_min;
     for(int g_ind2=Lambda_G_grid_L2_min;g_ind2<=Lambda_G_grid_L2_max;g_ind2++){
     int g2_ind = g_ind2 - Lambda_G_grid_L2_min;

     fl_FF_out<<g_ind1<<"   "<<g_ind2<<"  "<<LambdaNew_[spin][band1][band2][k1_1_ind][k1_2_ind][k2_1_ind][k2_2_ind][g1_ind][g2_ind].real()<< "  "<<LambdaNew_[spin][band1][band2][k1_1_ind][k1_2_ind][k2_1_ind][k2_2_ind][g1_ind][g2_ind].imag()<<endl;



     }
     fl_FF_out<<endl;
     }


  }}}}

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
for(int layer=0;layer<Parameters_.max_layer_ind;layer++){
    
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


void Hamiltonian::PrintBlochStates_old(){

int comp, k_ind;
for(int spin=0;spin<2;spin++){
    for(int layer=0;layer<Parameters_.max_layer_ind;layer++){
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
