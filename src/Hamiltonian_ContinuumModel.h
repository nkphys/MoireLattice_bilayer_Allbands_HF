#include <algorithm>
#include <functional>
#include <math.h>
#include "tensor_type.h"
#include "ParametersEngine.h"
#include "Coordinates_ContinuumModel.h"
#define PI acos(-1.0)

#ifndef Hamiltonian_ContinuumModel_class
#define Hamiltonian_ContinuumModel_class

extern "C" void   zheev_(char *,char *,int *,std::complex<double> *, int *, double *,
                         std::complex<double> *,int *, double *, int *);
//zheev_(&jobz,&uplo,&n,&(Ham_(0,0)),&lda,&(eigs_[0]),&(work[0]),&lwork,&(rwork[0]),&info);

class Hamiltonian_ContinuumModel {
public:

    Hamiltonian_ContinuumModel(Parameters& Parameters__, Coordinates_ContinuumModel&  Coordinates__)
        :Parameters_(Parameters__),Coordinates_(Coordinates__)

    {
        Initialize();
    }


    void Initialize();    //::DONE
    
    void Check_Hermiticity();  //::DONE
    void HTBCreate();   //::DONE
    void HTBCreate_MoTe2Homobilayer();
    void HTBCreate_MoTe2WSe2Bilayer();
    void HTBCreate_WSe2WS2Bilayer();
    void HTBCreate_GammaValleyHomobilayer();
    void HTBCreate_GammaValleyHomobilayerWithDisplacementField();
    double NIEnergy(double kx_val, double ky_val);

    void Diagonalize(char option);   //::DONE
    void copy_eigs(int i);  //::DONE
    void Get_Overlap_layers(int state_ind);

    void Getting_kx_ky_in_Primitive_BZ(double & _kx_, double & _ky_, int n1, int n2, int &n1_new, int &n2_new);
    void Getting_n1_n2_in_Primitive_BZ(int n1, int n2, int &n1_new, int &n2_new, int &g1_off, int &g2_off);
    void Saving_NonInteractingSpectrum();
    void Saving_NonInteractingSpectrum_PrimBZ();
    void Create_PBZ_map();
    void NonInteractingSpectrum_AlongPath(Mat_1_intpair k_path);
    void Calculate_ChernNumbers();
    Mat_1_intpair Get_k_path(int path_no);

    Parameters &Parameters_;
    Coordinates_ContinuumModel &Coordinates_;
    int ns_, l1_, l2_;
    double kx_, ky_;
    double k_plusx, k_minusx, k_plusy, k_minusy;
    double k_plusx_p, k_minusx_p, k_plusy_p, k_minusy_p;
    double kx_offset, ky_offset;
    int valley;
    int mbz_factor;

    Mat_2_int PBZ_map_n1, PBZ_map_n2;


    Matrix<complex<double>> HTB_;
    Matrix<complex<double>> Ham_;
    Matrix<double> Tx,Ty,Tpxpy,Tpxmy;
    vector<double> eigs_,eigs_saved_,sx_,sy_,sz_;

    
    Mat_4_Complex_doub BlochStates; //[valley(spin)][band][k][G,l]
    Mat_4_Complex_doub BlochStates_PBZ; //[valley(spin)][band][k][G,l]
    Mat_3_doub eigvals; //[valley(spin)][band][k]
    Mat_3_doub eigvals_PBZ; //[valley(spin)][band][k]
    double Overlap_bottom, Overlap_top;

    //real space  effective H params
    int L1_eff, L2_eff;
    Mat_2_Complex_doub Tij;
    Mat_2_Complex_doub Uij;

};


void Hamiltonian_ContinuumModel::Initialize(){

     

    l1_=Parameters_.Grid_moireRL_L1;
    l2_=Parameters_.Grid_moireRL_L2;
    ns_ = l1_*l2_;

    int space=ns_*Coordinates_.n_orbs_;

    HTB_.resize(space,space);
    Ham_.resize(space,space);
    eigs_.resize(space);
    eigs_saved_.resize(space);


    k_plusx = (-1.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire);
    k_plusy = (-1.0/3.0)*(2.0*PI/Parameters_.a_moire);


    k_minusx = (-1.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire);
    k_minusy = (1.0/3.0)*(2.0*PI/Parameters_.a_moire);

    k_plusx_p = (1.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire);
    k_plusy_p = (-1.0/3.0)*(2.0*PI/Parameters_.a_moire);

    k_minusx_p = (1.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire);
    k_minusy_p = (1.0/3.0)*(2.0*PI/Parameters_.a_moire);

    ky_offset=(-4.0*PI)/(3.0*Parameters_.a_moire); 
    kx_offset=0; 

    //real space  effective H params
    L1_eff=12;L2_eff=12;
    Tij.resize(L1_eff*L2_eff);
    for(int i=0;i<L1_eff*L2_eff;i++){
        Tij[i].resize(L1_eff*L2_eff);
    }
    Uij.resize(L1_eff*L2_eff);
    for(int i=0;i<L1_eff*L2_eff;i++){
        Uij[i].resize(L1_eff*L2_eff);
    }


    valley= 1;//int(Parameters_.ValleyTau);
	cout<<"VALLEY = "<<valley<<endl;

    mbz_factor=1;

    PBZ_map_n1.resize(Parameters_.moire_BZ_L1);
    for(int i1=0;i1<Parameters_.moire_BZ_L1;i1++){
    PBZ_map_n1[i1].resize(Parameters_.moire_BZ_L2);
    }

    PBZ_map_n2.resize(Parameters_.moire_BZ_L1);
    for(int i1=0;i1<Parameters_.moire_BZ_L1;i1++){
    PBZ_map_n2[i1].resize(Parameters_.moire_BZ_L2);
    }
} // ----------


void Hamiltonian_ContinuumModel::Calculate_ChernNumbers(){


    
     int L1_,L2_;
     L1_=Parameters_.moire_BZ_L1; //along G1 (b6)
     L2_=Parameters_.moire_BZ_L2; //along G2 (b2)
     int N_bands_Chern = Parameters_.N_bands_HF;


    for(int Spin_=0;Spin_<=1;Spin_++){
        cout<<"XXXXXXXXXXXXXXX FOR SPIN = "<<Spin_<<" XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"<<endl;
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
        string file_Fk="Fk_band"+to_string(band)+"_spin_" + to_string(Spin_) + ".txt";
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

                //U1_k
                Ux_k = 0;
                n_left = n;
                nx_right = (nx + 1);// % (mbz_factor*L1_);
                ny_right = ny;
                n_right = nx_right + ny_right*((mbz_factor*L1_)+1);
                for (int comp = 0; comp < ns_*2; comp++)
                {
                    Ux_k += conj(BlochStates[Spin_][band][n_left][comp])*
                            BlochStates[Spin_][band][n_right][comp];
                }
                Ux_k = Ux_k * (1.0 / abs(Ux_k));

                //U2_kpx
                Uy_kpx = 0;
                nx_left = (nx + 1);// % (mbz_factor*L1_);
                ny_left = ny;
                n_left = nx_left + ny_left*((mbz_factor*L1_)+1);
                nx_right = nx_left;
                ny_right = (ny_left + 1);// % (mbz_factor*L2_);
                n_right = nx_right + ny_right*((mbz_factor*L1_)+1);
                for (int comp = 0; comp < 2*ns_; comp++)
                {
                    Uy_kpx += conj(BlochStates[Spin_][band][n_left][comp])*
                            BlochStates[Spin_][band][n_right][comp];
                }
                Uy_kpx = Uy_kpx * (1.0 / abs(Uy_kpx));

                //U1_kpy
                Ux_kpy = 0;
                nx_left = nx;
                ny_left = (ny + 1);// % (mbz_factor*L2_);
                n_left = nx_left + ny_left*((mbz_factor*L1_)+1);
                nx_right = (nx_left + 1);// % (mbz_factor*L1_);
                ny_right = ny_left;
                n_right = nx_right + ny_right*((mbz_factor*L1_)+1);
                for (int comp = 0; comp < 2*ns_; comp++)
                {
                    Ux_kpy += conj(BlochStates[Spin_][band][n_left][comp])*
                            BlochStates[Spin_][band][n_right][comp];
                }
                Ux_kpy = Ux_kpy * (1.0 / abs(Ux_kpy));

                //U2_k
                Uy_k = 0;
                nx_left = nx;
                ny_left = ny;
                n_left = nx_left + ny_left*((mbz_factor*L1_)+1);
                nx_right = nx_left;
                ny_right = (ny_left + 1);// % (mbz_factor*L2_);
                n_right = nx_right + ny_right*((mbz_factor*L1_)+1);
                for (int comp = 0; comp < 2*ns_; comp++)
                {
                    Uy_k += conj(BlochStates[Spin_][band][n_left][comp])*
                            BlochStates[Spin_][band][n_right][comp];
                }
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

}




void Hamiltonian_ContinuumModel::NonInteractingSpectrum_AlongPath(Mat_1_intpair k_path){


    mbz_factor=1;
    int L1_,L2_;
    L1_=Parameters_.moire_BZ_L1; //along G1 (b6)
    L2_=Parameters_.moire_BZ_L2; //along G2 (b2)
    int N_bands = Parameters_.N_bands_HF;

    int comp;


    eigvals.resize(2);
    for(int spin=0;spin<2;spin++){
    eigvals[spin].resize(N_bands);
        for(int n=0;n<N_bands;n++){
            eigvals[spin][n].resize(((mbz_factor*L1_)+1)*((mbz_factor*L2_)+1));
        }
    }


    int k_ind;

    for(int spin=0;spin<=1;spin++){
    string file_bands_out="Bands_energy_AlongPath_spin" + to_string(spin)+".txt";
    ofstream FileBandsOut(file_bands_out.c_str());
    FileBandsOut<<"#n1   n2   E0(n1,n2)   E1(n1,n2)  ..."<<endl;

        valley = 2*spin -1;
    for(int n_=0;n_<k_path.size();n_++){
        int n1=k_path[n_].first;
        int n2=k_path[n_].second;
        FileBandsOut<<n_<<"   "<<n1<<"  "<<n2<<"  ";
        k_ind = n1 + ((mbz_factor*L1_)+1)*n2;
    //kx_=(2.0*PI/Parameters_.a_moire)*((n1-(mbz_factor*L1_/2))*(1.0/(sqrt(3)*L1_))  +  (n2-(mbz_factor*L2_/2))*(1.0/(sqrt(3)*L2_)));
    //ky_=(2.0*PI/Parameters_.a_moire)*((n1-(mbz_factor*L1_/2))*(-1.0/(L1_))  +  (n2-(mbz_factor*L2_/2))*(1.0/(L2_)));

    kx_=(2.0*PI/Parameters_.a_moire)*((n1)*(1.0/(sqrt(3)*L1_))  +  (n2)*(1.0/(sqrt(3)*L2_)));
    ky_=(2.0*PI/Parameters_.a_moire)*((n1)*(-1.0/(L1_))  +  (n2)*(1.0/(L2_)));

    //    Getting_kx_ky_in_Primitive_BZ(kx_, ky_, n1, n2);

    HTBCreate();
    char Dflag='V';
    Diagonalize(Dflag);

    for(int n=0;n<N_bands;n++){
       FileBandsOut<<eigs_[n]<<"  ";
    }
    FileBandsOut<<endl;
    }
    }



}

void Hamiltonian_ContinuumModel::Create_PBZ_map(){

    int L1_=Parameters_.moire_BZ_L1; //along G1 (b6)
    int L2_=Parameters_.moire_BZ_L2; //along G2 (b2)
    int n1_new, n2_new;
    double kx_temp, ky_temp;
    for(int n1=0;n1<L1_;n1++){
        for(int n2=0;n2<L2_;n2++){
         Getting_kx_ky_in_Primitive_BZ(kx_temp, ky_temp, n1, n2, n1_new, n2_new);
        PBZ_map_n1[n1][n2]=n1_new;
        PBZ_map_n2[n1][n2]=n2_new;
        }}

}



void Hamiltonian_ContinuumModel::Getting_n1_n2_in_Primitive_BZ(int n1, int n2, int &n1_new, int &n2_new, int &g1_off, int &g2_off){


    int L1_,L2_;
   L1_=Parameters_.moire_BZ_L1; //along G1 (b6)
   L2_=Parameters_.moire_BZ_L2; //along G2 (b2)

   // kx_=(2.0*PI/Parameters_.a_moire)*((n1)*(1.0/(sqrt(3)*L1_))  +  (n2)*(1.0/(sqrt(3)*L2_)));
   // ky_=(2.0*PI/Parameters_.a_moire)*((n1)*(-1.0/(L1_))  +  (n2)*(1.0/(L2_)));

   int g1_temp, g2_temp;
   double kx_temp, ky_temp;
   double dis_temp;
   double kx_final, ky_final;
   kx_final=(2.0*PI/Parameters_.a_moire)*((n1 + 0*L1_)*(1.0/(sqrt(3)*L1_))  +  (n2+ 0*L2_)*(1.0/(sqrt(3)*L2_)));
   ky_final=(2.0*PI/Parameters_.a_moire)*((n1 + 0*L1_)*(-1.0/(L1_))  +  (n2+ 0*L2_)*(1.0/(L2_)));
   double dis=(kx_final*kx_final  + ky_final*ky_final);
   int n1_new_final=n1;
   int n2_new_final=n2;
    g1_temp=0;
    g2_temp=0;

   for(int g1_=-1;g1_<=0;g1_++){
       for(int g2_=-1;g2_<=0;g2_++){
        kx_temp=(2.0*PI/Parameters_.a_moire)*((n1 + g1_*L1_)*(1.0/(sqrt(3)*L1_))  +  (n2+ g2_*L2_)*(1.0/(sqrt(3)*L2_)));
        ky_temp=(2.0*PI/Parameters_.a_moire)*((n1 + g1_*L1_)*(-1.0/(L1_))  +  (n2+ g2_*L2_)*(1.0/(L2_)));
        dis_temp=(kx_temp*kx_temp  + ky_temp*ky_temp);
        if(dis_temp<dis){
        dis = dis_temp;
        kx_final=kx_temp;
        ky_final=ky_temp;
        n1_new_final = n1 + g1_*L1_;
        n2_new_final = n2 + g2_*L2_;
        g1_temp = g1_;
        g2_temp = g2_;
        }
       }
   }


   g1_off=g1_temp;
   g2_off=g2_temp;

   n1_new = n1_new_final;
   n2_new = n2_new_final;


}

void Hamiltonian_ContinuumModel::Getting_kx_ky_in_Primitive_BZ(double & _kx_, double & _ky_, int n1, int n2, int &n1_new, int &n2_new){


    int L1_,L2_;
   L1_=Parameters_.moire_BZ_L1; //along G1 (b6)
   L2_=Parameters_.moire_BZ_L2; //along G2 (b2)

   // kx_=(2.0*PI/Parameters_.a_moire)*((n1)*(1.0/(sqrt(3)*L1_))  +  (n2)*(1.0/(sqrt(3)*L2_)));
   // ky_=(2.0*PI/Parameters_.a_moire)*((n1)*(-1.0/(L1_))  +  (n2)*(1.0/(L2_)));

   int min_g1, min_g2;
   double kx_temp, ky_temp;
   double dis_temp;
   double kx_final, ky_final;
   kx_final=(2.0*PI/Parameters_.a_moire)*((n1 + 0*L1_)*(1.0/(sqrt(3)*L1_))  +  (n2+ 0*L2_)*(1.0/(sqrt(3)*L2_)));
   ky_final=(2.0*PI/Parameters_.a_moire)*((n1 + 0*L1_)*(-1.0/(L1_))  +  (n2+ 0*L2_)*(1.0/(L2_)));
   double dis=(kx_final*kx_final  + ky_final*ky_final);
   int n1_new_final=n1;
   int n2_new_final=n2;

   for(int g1_=-1;g1_<=0;g1_++){
       for(int g2_=-1;g2_<=0;g2_++){
        kx_temp=(2.0*PI/Parameters_.a_moire)*((n1 + g1_*L1_)*(1.0/(sqrt(3)*L1_))  +  (n2+ g2_*L2_)*(1.0/(sqrt(3)*L2_)));
        ky_temp=(2.0*PI/Parameters_.a_moire)*((n1 + g1_*L1_)*(-1.0/(L1_))  +  (n2+ g2_*L2_)*(1.0/(L2_)));
        dis_temp=(kx_temp*kx_temp  + ky_temp*ky_temp);
        if(dis_temp<dis){
        dis=dis_temp;
        kx_final=kx_temp;
        ky_final=ky_temp;
        n1_new_final = n1 + g1_*L1_;
        n2_new_final = n2 + g2_*L2_;
        }
       }
   }


   _kx_=kx_final;
   _ky_=ky_final;

   n1_new = n1_new_final;
   n2_new = n2_new_final;

   //overwritten
  // kx_=(2.0*PI/Parameters_.a_moire)*((n1 + 0*L1_)*(1.0/(sqrt(3)*L1_))  +  (n2+ 0*L2_)*(1.0/(sqrt(3)*L2_)));
  // ky_=(2.0*PI/Parameters_.a_moire)*((n1 + 0*L1_)*(-1.0/(L1_))  +  (n2+ 0*L2_)*(1.0/(L2_)));

}


void Hamiltonian_ContinuumModel::Saving_NonInteractingSpectrum_PrimBZ(){



        mbz_factor=1;

         int L1_,L2_;
        L1_=Parameters_.moire_BZ_L1; //along G1 (b6)
        L2_=Parameters_.moire_BZ_L2; //along G2 (b2)
        int N_bands = Parameters_.N_bands_HF;

        int comp;

        //REMEMBER TO USE ONLY INSIDE BZ k points for HF not the extra k1, k2 line.
        BlochStates_PBZ.resize(2);
        for(int spin=0;spin<2;spin++){
          BlochStates_PBZ[spin].resize(N_bands);
          for(int n=0;n<N_bands;n++){
            BlochStates_PBZ[spin][n].resize(((mbz_factor*L1_))*((mbz_factor*L2_)));
            for(int i=0;i<((mbz_factor*L1_))*((mbz_factor*L2_));i++){  //k_ind
               BlochStates_PBZ[spin][n][i].resize(ns_*Coordinates_.n_orbs_); //G_ind*layer
            }
          }
        }

        eigvals_PBZ.resize(2);
        for(int spin=0;spin<2;spin++){
        eigvals_PBZ[spin].resize(N_bands);
            for(int n=0;n<N_bands;n++){
                eigvals_PBZ[spin][n].resize(((mbz_factor*L1_))*((mbz_factor*L2_)));
            }
        }


        int k_ind;
        int n1_new, n2_new;
        for(int spin=0;spin<=1;spin++){
        string file_bands_out="PBZ_Bands_energy_FullSpectrum_spin" + to_string(spin)+".txt";
        ofstream FileBandsOut(file_bands_out.c_str());
        FileBandsOut<<"#n1   n2   E0(n1,n2)   E1(n1,n2)  ..."<<endl;

            valley = 2*spin -1;
        for(int n1=0;n1<mbz_factor*L1_;n1++){
            for(int n2=0;n2<mbz_factor*L2_;n2++){
                FileBandsOut<<n1<<"  "<<n2<<"  ";
            k_ind = n1 + ((mbz_factor*L1_))*n2;

        //kx_=(2.0*PI/Parameters_.a_moire)*((n1-(mbz_factor*L1_/2))*(1.0/(sqrt(3)*L1_))  +  (n2-(mbz_factor*L2_/2))*(1.0/(sqrt(3)*L2_)));
        //ky_=(2.0*PI/Parameters_.a_moire)*((n1-(mbz_factor*L1_/2))*(-1.0/(L1_))  +  (n2-(mbz_factor*L2_/2))*(1.0/(L2_)));

        Getting_kx_ky_in_Primitive_BZ(kx_, ky_, n1, n2, n1_new, n2_new);
       // kx_=(2.0*PI/Parameters_.a_moire)*((n1)*(1.0/(sqrt(3)*L1_))  +  (n2)*(1.0/(sqrt(3)*L2_)));
       // ky_=(2.0*PI/Parameters_.a_moire)*((n1)*(-1.0/(L1_))  +  (n2)*(1.0/(L2_)));

        HTBCreate();
//        string filename = "Hamil_ContModel_n1_n2_" + to_string(n1) +"_"+to_string(n2)+ "_spin" + to_string(spin) +".txt";
//        Ham_.print_in_file(filename, 4);

        char Dflag='V';
        Diagonalize(Dflag);

        for(int n=0;n<N_bands;n++){
            FileBandsOut<<eigs_[n]<<"  ";
           eigvals_PBZ[spin][n][k_ind] = eigs_[n];

    // string file_BlochState="PBZ_BlochState_spin"+to_string(spin)+"_band"+to_string(n)+"k_mbz_"+to_string(n1)+"_"+to_string(n2)+".txt";
  //ofstream fl_BlochState_out(file_BlochState.c_str());
 //fl_BlochState_out<<"#G1  G2   unk(layer=0).real  imag   unk(layer=1).real  imag"<<endl;
            for(int i1=0;i1<l1_;i1++){
                for(int i2=0;i2<l2_;i2++){
   //              fl_BlochState_out<<i1<<" "<<i2<<"  ";
                for(int orb=0;orb< Coordinates_.n_orbs_ ;orb++){//layer
                comp=Coordinates_.Nbasis(i1, i2, orb);
                BlochStates_PBZ[spin][n][k_ind][comp]=Ham_(comp,n);
     //            fl_BlochState_out<<BlochStates_PBZ[spin][n][k_ind][comp].real()<<"  "<<BlochStates_PBZ[spin][n][k_ind][comp].imag()<<"  ";
                }
       //          fl_BlochState_out<<endl;
            }
         //    fl_BlochState_out<<endl;
        }
        }
        FileBandsOut<<endl;

        cout<<n1<<"("<<((mbz_factor*L1_))<<")  "<<n2<<"("<<((mbz_factor*L2_))<<")  "<<spin<<"(2)  done"<<endl;
        }
        FileBandsOut<<endl;
        }
        }


}


void Hamiltonian_ContinuumModel::Saving_NonInteractingSpectrum(){

        

        mbz_factor=1;

         int L1_,L2_;
        L1_=Parameters_.moire_BZ_L1; //along G1 (b6)
        L2_=Parameters_.moire_BZ_L2; //along G2 (b2)
        int N_bands = Parameters_.N_bands_HF;

        int comp;

        //REMEMBER TO USE ONLY INSIDE BZ k points for HF not the extra k1, k2 line.
        BlochStates.resize(2);
        for(int spin=0;spin<2;spin++){
          BlochStates[spin].resize(N_bands);
          for(int n=0;n<N_bands;n++){
            BlochStates[spin][n].resize(((mbz_factor*L1_)+1)*((mbz_factor*L2_)+1));
            for(int i=0;i<((mbz_factor*L1_)+1)*((mbz_factor*L2_)+1);i++){  //k_ind
               BlochStates[spin][n][i].resize(ns_*Coordinates_.n_orbs_); //G_ind*layer
            }
          }
        }

        eigvals.resize(2);
        for(int spin=0;spin<2;spin++){
        eigvals[spin].resize(N_bands);
            for(int n=0;n<N_bands;n++){
                eigvals[spin][n].resize(((mbz_factor*L1_)+1)*((mbz_factor*L2_)+1));
            }
        }
       

        int k_ind;

        for(int spin=0;spin<=1;spin++){
        string file_bands_out="Bands_energy_FullSpectrum_spin" + to_string(spin)+".txt";
        ofstream FileBandsOut(file_bands_out.c_str());
        FileBandsOut<<"#n1   n2   E0(n1,n2)   E1(n1,n2)  ..."<<endl;

            valley = 2*spin -1;
        for(int n1=0;n1<mbz_factor*L1_+1;n1++){
            for(int n2=0;n2<mbz_factor*L2_+1;n2++){
                FileBandsOut<<n1<<"  "<<n2<<"  ";
            k_ind = n1 + ((mbz_factor*L1_)+1)*n2;

        //kx_=(2.0*PI/Parameters_.a_moire)*((n1-(mbz_factor*L1_/2))*(1.0/(sqrt(3)*L1_))  +  (n2-(mbz_factor*L2_/2))*(1.0/(sqrt(3)*L2_)));
        //ky_=(2.0*PI/Parameters_.a_moire)*((n1-(mbz_factor*L1_/2))*(-1.0/(L1_))  +  (n2-(mbz_factor*L2_/2))*(1.0/(L2_)));
        
       // Getting_kx_ky_in_Primitive_BZ(kx_, ky_, n1, n2);
        kx_=(2.0*PI/Parameters_.a_moire)*((n1)*(1.0/(sqrt(3)*L1_))  +  (n2)*(1.0/(sqrt(3)*L2_)));
        ky_=(2.0*PI/Parameters_.a_moire)*((n1)*(-1.0/(L1_))  +  (n2)*(1.0/(L2_)));

        HTBCreate();
//        string filename = "Hamil_ContModel_n1_n2_" + to_string(n1) +"_"+to_string(n2)+ "_spin" + to_string(spin) +".txt";
//        Ham_.print_in_file(filename, 4);

        char Dflag='V';
        Diagonalize(Dflag);

        for(int n=0;n<N_bands;n++){
            FileBandsOut<<eigs_[n]<<"  ";
           eigvals[spin][n][k_ind] = eigs_[n];

   //  string file_BlochState="BlochState_spin"+to_string(spin)+"_band"+to_string(n)+"k_mbz_"+to_string(n1)+"_"+to_string(n2)+".txt";
//  ofstream fl_BlochState_out(file_BlochState.c_str());
// fl_BlochState_out<<"#G1  G2   unk(layer=0).real  imag   unk(layer=1).real  imag"<<endl;
            for(int i1=0;i1<l1_;i1++){
                for(int i2=0;i2<l2_;i2++){
                   // fl_BlochState_out<<i1<<" "<<i2<<"  "; 
                for(int orb=0;orb< Coordinates_.n_orbs_ ;orb++){//layer
                comp=Coordinates_.Nbasis(i1, i2, orb);
                BlochStates[spin][n][k_ind][comp]=Ham_(comp,n);
                // fl_BlochState_out<<BlochStates[spin][n][k_ind][comp].real()<<"  "<<BlochStates[spin][n][k_ind][comp].imag()<<"  ";
                }
                // fl_BlochState_out<<endl;
            }
            // fl_BlochState_out<<endl;
        }
        }
        FileBandsOut<<endl;

        cout<<n1<<"("<<((mbz_factor*L1_)+1)<<")  "<<n2<<"("<<((mbz_factor*L2_)+1)<<")  "<<spin<<"(2)  done"<<endl;
        }
        FileBandsOut<<endl;
        }
        }


}

double Hamiltonian_ContinuumModel::NIEnergy(double kx_val, double ky_val){

    double energy_;
    //energy_ = -1.0*(Parameters_.RedPlanckConst*Parameters_.RedPlanckConst*(kx_val*kx_val  + ky_val*ky_val))*(0.5/Parameters_.MStar);
    energy_ = -1.0*((3.809842*1000)*(kx_val*kx_val  + ky_val*ky_val));

    return energy_;
}




void Hamiltonian_ContinuumModel::Check_Hermiticity()

{
    complex<double> temp(0,0);
    complex<double>temp2;

    for(int i=0;i<Ham_.n_row();i++) {
        for(int j=0;j<Ham_.n_row();j++) {
            if(
                    abs(Ham_(i,j) - conj(Ham_(j,i)))>0.00001
                    ) {
                cout<<Ham_(i,j)<<endl;
                cout<<conj(Ham_(j,i))<<endl;

            }
            assert(
                        abs(Ham_(i,j) - conj(Ham_(j,i)))<0.00001
                        ); //+ Ham_(i+orbs_*ns_,j) + Ham_(i,j+orbs_*ns_);
            //temp +=temp2*conj(temp2);
        }
    }

    // cout<<"Hermiticity: "<<temp<<endl;
}



void Hamiltonian_ContinuumModel::Diagonalize(char option){

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


void Hamiltonian_ContinuumModel::Get_Overlap_layers(int state_ind){
 //Ham_(comp,eigen_no)

int comp;
int Bottom_, Top_;
    Bottom_=0;Top_=1;
  Overlap_bottom=0.0;Overlap_top=0.0;
  
    
  for(int i1=0;i1<l1_;i1++){
  for(int i2=0;i2<l2_;i2++){    
    
    comp=Coordinates_.Nbasis(i1, i2, Bottom_);
    Overlap_bottom += abs(Ham_(comp,state_ind))*abs(Ham_(comp,state_ind));

    if(Coordinates_.n_orbs_>1){
    comp=Coordinates_.Nbasis(i1, i2, Top_);
    Overlap_top += abs(Ham_(comp,state_ind))*abs(Ham_(comp,state_ind));
    }
	}
	}

	
}


void Hamiltonian_ContinuumModel::HTBCreate(){
    if(Parameters_.MaterialName=="MoTe2Homobilayer"){
        HTBCreate_MoTe2Homobilayer();
    }
    if(Parameters_.MaterialName=="MoTe2WSe2Bilayer"){
        HTBCreate_MoTe2WSe2Bilayer();
    }
    if(Parameters_.MaterialName=="WSe2WS2Bilayer"){
        HTBCreate_WSe2WS2Bilayer();
    }
    if(Parameters_.MaterialName=="MoSe2Homobilayer" || Parameters_.MaterialName=="MoS2Homobilayer" ||
       Parameters_.MaterialName=="WS2Homobilayer"){
       HTBCreate_GammaValleyHomobilayer();
    }
    if(Parameters_.MaterialName=="MoSe2HomobilayerWithDisplacementField" || Parameters_.MaterialName=="MoS2HomobilayerWithDisplacementField" ||
       Parameters_.MaterialName=="WS2HomobilayerWithDisplacementField"){
       HTBCreate_GammaValleyHomobilayerWithDisplacementField();
    }

    if(!(Parameters_.MaterialName=="WSe2WS2Bilayer"  || Parameters_.MaterialName=="MoTe2WSe2Bilayer" || Parameters_.MaterialName=="MoTe2Homobilayer" ||
         Parameters_.MaterialName=="MoSe2Homobilayer" || Parameters_.MaterialName=="MoS2Homobilayer" ||
         Parameters_.MaterialName=="WS2Homobilayer" || Parameters_.MaterialName=="MoSe2HomobilayerWithDisplacementField" || Parameters_.MaterialName=="MoS2HomobilayerWithDisplacementField" ||
         Parameters_.MaterialName=="WS2HomobilayerWithDisplacementField")
    ){
        cout<<"Requires correct MaterialName in input file"<<endl;
        assert(false);
    }

    // cout<<"XXXXXXXXXXXXXXXXXXXXXXXXXXX"<<endl;
    // Ham_.print();
    // cout<<"XXXXXXXXXXXXXXXXXXXXXXXXXXX"<<endl;
}

void Hamiltonian_ContinuumModel::HTBCreate_MoTe2WSe2Bilayer(){


    //LATER CHANGE IT TO along G1 and G2

    Ham_.resize(ns_*2,ns_*2);
    double b1x_, b1y_, b2x_, b2y_;
    b1x_=(2.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire);
    b1y_=(0.0)*(2.0*PI/Parameters_.a_moire);
    b2x_=(1.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire);
    b2y_=(1.0)*(2.0*PI/Parameters_.a_moire);

    int Bottom_, Top_;
    Bottom_=0;Top_=1;

    //l1_/2,l2_/2 is the k-point

    double kx_local, ky_local;

    int row, col;
    int i1_neigh, i2_neigh;
    for(int i1=0;i1<l1_;i1++){
        for(int i2=0;i2<l2_;i2++){
            kx_local = kx_ + (-(l1_/2)+i1)*(b1x_) + (-(l2_/2)+i2)*(b2x_);
            ky_local = ky_ + (-(l1_/2)+i1)*(b1y_) + (-(l2_/2)+i2)*(b2y_);
            for(int orb=0;orb<2;orb++){
                row=Coordinates_.Nbasis(i1, i2, orb);
                if(orb==Bottom_){

                    //1
                    col = row;
    //                Ham_(row,col) += ((1.0/Parameters_.MStar_bottom)*NIEnergy(kx_local - valley*k_minusx_p , ky_local - valley*k_minusy_p )) + (1.0*Parameters_.Vz_bottom);
		      Ham_(row,col) += ((1.0/Parameters_.MStar_bottom)*NIEnergy(kx_local , ky_local )) + (1.0*Parameters_.Vz_bottom);


                    //2 i.e +/- b1
                    i1_neigh = i1 + 1;
                    i2_neigh = i2;
                    if(i1_neigh<l1_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += Parameters_.V_param_bottom*exp(-iota_complex*Parameters_.Psi_param);
                    }
                    i1_neigh = i1 - 1;
                    i2_neigh = i2;
                    if(i1_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += Parameters_.V_param_bottom*exp(iota_complex*Parameters_.Psi_param);
                    }


                    //3 i.e +/- b3
                    i1_neigh = i1 - 1;
                    i2_neigh = i2 + 1;
                    if(i1_neigh>=0 && i2_neigh<l2_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += Parameters_.V_param_bottom*exp(-iota_complex*Parameters_.Psi_param);
                    }
                    i1_neigh = i1 + 1;
                    i2_neigh = i2 - 1;
                    if(i1_neigh<l1_ && i2_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += Parameters_.V_param_bottom*exp(iota_complex*Parameters_.Psi_param);
                    }


                    //4 i.e +/- b5
                    i1_neigh = i1;
                    i2_neigh = i2 - 1;
                    if(i2_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += Parameters_.V_param_bottom*exp(-iota_complex*Parameters_.Psi_param);
                    }
                    i1_neigh = i1;
                    i2_neigh = i2 + 1;
                    if(i2_neigh<l2_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += Parameters_.V_param_bottom*exp(iota_complex*Parameters_.Psi_param);
                    }


                    //5
                    col = Coordinates_.Nbasis(i1, i2, Top_);
                    Ham_(row,col) += valley*Parameters_.omega_param;

                    //6
                    i1_neigh = i1;
                    i2_neigh = i2-(valley*1);
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
			(i1_neigh<l1_ && i1_neigh>=0)
			){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += valley*Parameters_.omega_param*
                                         exp(iota_complex*(valley*2.0)*(PI/3.0));
                    }

                    //7
                    i1_neigh = i1+(valley*1);
                    i2_neigh = i2-(valley*1);
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
                        (i1_neigh<l1_ && i1_neigh>=0)
			){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += valley*Parameters_.omega_param*
                                         exp(iota_complex*(valley*4.0)*(PI/3.0));
                    }


                }
                else{//i.e. orb=Top_


                    //1
                    col = row;
   //                 Ham_(row,col) += ((1.0/Parameters_.MStar_top)*NIEnergy(kx_local - valley*k_plusx_p , ky_local - valley*k_plusy_p) ) + (1.0*Parameters_.Vz_top);
		    Ham_(row,col) += ((1.0/Parameters_.MStar_top)*NIEnergy(kx_local - valley*kx_offset, ky_local - valley*ky_offset) ) + (1.0*Parameters_.Vz_top);

                    
                    //5
                    col = Coordinates_.Nbasis(i1, i2, Bottom_);
                    Ham_(row,col) += valley*Parameters_.omega_param;

                    //6
                    i1_neigh = i1;
                    i2_neigh = i2+(valley*1);
                    if(
			(i2_neigh<l2_ && i2_neigh>=0) &&
                        (i1_neigh<l1_ && i1_neigh>=0)
			){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += valley*Parameters_.omega_param*
                                          exp(-1.0*iota_complex*(valley*2.0)*(PI/3.0));
                    }

                    //7
                    i1_neigh = i1-(valley*1);
                    i2_neigh = i2+(valley*1);
                    if(
			(i2_neigh<l2_ && i2_neigh>=0) &&
                        (i1_neigh<l1_ && i1_neigh>=0)
			){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += valley*Parameters_.omega_param*
                                        exp(-1.0*iota_complex*(valley*4.0)*(PI/3.0));
                    }

                }

            }
        }
    }



} // ----------

void Hamiltonian_ContinuumModel::HTBCreate_MoTe2Homobilayer(){


    //This is written in hole operators


    Ham_.resize(ns_*2,ns_*2); //here 2 is for layers

    double b1x_, b1y_, b6x_, b6y_, b2x_, b2y_;

    //G1+G2
    b1x_=(2.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire); //G1+G2
    b1y_=(0.0)*(2.0*PI/Parameters_.a_moire);

    //G1
    b6x_=(1.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire); //G1
    b6y_=(-1.0)*(2.0*PI/Parameters_.a_moire);


    //G2
    b2x_=(1.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire); //G2
    b2y_=(1.0)*(2.0*PI/Parameters_.a_moire);  //G2

    int Bottom_, Top_;
    Bottom_=0;Top_=1;

    //l1_/2,l2_/2 is the k-point

    double kx_local, ky_local;

    int row, col;
    int i1_neigh, i2_neigh;
    for(int i1=0;i1<l1_;i1++){
        for(int i2=0;i2<l2_;i2++){
            kx_local = kx_ + (-(l1_/2)+i1)*(b6x_) + (-(l2_/2)+i2)*(b2x_);
            ky_local = ky_ + (-(l1_/2)+i1)*(b6y_) + (-(l2_/2)+i2)*(b2y_);
            for(int orb=0;orb<2;orb++){
                row=Coordinates_.Nbasis(i1, i2, orb);
                if(orb==Bottom_){

                    if(true){

                    //1
                    col = row;
                    Ham_(row,col) += ((-1.0/Parameters_.MStar_bottom)*NIEnergy(kx_local - valley*k_plusx, ky_local - valley*k_plusy)) + (-1.0*Parameters_.Vz_bottom);

                    //2 i.e +/- b1  (V1)
                    i1_neigh = i1 + 1;
                    i2_neigh = i2 + 1;
                    if(i1_neigh<l1_ && i2_neigh<l2_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_bottom*exp(iota_complex*Parameters_.Psi_param);
                    }
                    i1_neigh = i1 - 1;
                    i2_neigh = i2 - 1;
                    if(i1_neigh>=0 && i2_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_bottom*exp(-iota_complex*Parameters_.Psi_param);
                    }


                    //3 i.e +/- b3  (V1)
                    i1_neigh = i1 - 1;
                    i2_neigh = i2;
                    if(i1_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_bottom*exp(iota_complex*Parameters_.Psi_param);
                    }
                    i1_neigh = i1 + 1;
                    i2_neigh = i2;
                    if(i1_neigh<l1_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_bottom*exp(-iota_complex*Parameters_.Psi_param);
                    }


                    //4 i.e +/- b5 (V1)
                    i1_neigh = i1;
                    i2_neigh = i2 - 1;
                    if(i2_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_bottom*exp(iota_complex*Parameters_.Psi_param);
                    }
                    i1_neigh = i1;
                    i2_neigh = i2 + 1;
                    if(i2_neigh<l2_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_bottom*exp(-iota_complex*Parameters_.Psi_param);
                    }




                    //+/- (1,2)=g^{2}_{1}  (V2)
                    i1_neigh = i1 + 1;
                    i2_neigh = i2 + 2;
                    if(i1_neigh<l1_ && i2_neigh<l2_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V2_param_bottom;
                    }
                    i1_neigh = i1 - 1;
                    i2_neigh = i2 - 2;
                    if(i1_neigh>=0 && i2_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V2_param_bottom;
                    }


                    //+/- (-2,-1)=g^{2}_{3}  (V2)
                    i1_neigh = i1 - 2;
                    i2_neigh = i2 - 1;
                    if( (i1_neigh<l1_ && i2_neigh<l2_)  &&
                         (i1_neigh>=0 && i2_neigh>=0)   ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V2_param_bottom;
                    }
                    i1_neigh = i1 + 2;
                    i2_neigh = i2 + 1;
                    if( (i1_neigh<l1_ && i2_neigh<l2_)  &&
                         (i1_neigh>=0 && i2_neigh>=0)   ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V2_param_bottom;
                    }


                    //+/- (1,-1)=g^{2}_{5}  (V2)
                    i1_neigh = i1 + 1;
                    i2_neigh = i2 - 1;
                    if( (i1_neigh<l1_ && i2_neigh<l2_)  &&
                         (i1_neigh>=0 && i2_neigh>=0)   ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V2_param_bottom;
                    }
                    i1_neigh = i1 - 1;
                    i2_neigh = i2 + 1;
                    if( (i1_neigh<l1_ && i2_neigh<l2_)  &&
                         (i1_neigh>=0 && i2_neigh>=0)   ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V2_param_bottom;
                    }



                    //old version w
                    //col = Coordinates_.Nbasis(i1, i2, Top_);
                    //Ham_(row,col) += -1.0*Parameters_.omega_param;

                    // 0 (w1)
                    i1_neigh = i1+(valley*0);
                    i2_neigh = i2+(valley*0);
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega_param;
                    }



                    // (0,1)=g^{1}_{2} (w1)
                    i1_neigh = i1;
                    i2_neigh = i2+(valley*1);
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
			(i1_neigh<l1_ && i1_neigh>=0)
			){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega_param;
                    }

                    // (-1,0)=g^{1}_{3} (w1)
                    i1_neigh = i1-(valley*1);
                    i2_neigh = i2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
                        (i1_neigh<l1_ && i1_neigh>=0)
			){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega_param;
                    }




                    // (-1,1) (w2)
                    i1_neigh = i1-(valley*1);
                    i2_neigh = i2+(valley*1);
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }


                    // (1,1) (w2)
                    i1_neigh = i1+(valley*1);
                    i2_neigh = i2+(valley*1);
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }


                    // (-1,-1) (w2)
                    i1_neigh = i1-(valley*1);
                    i2_neigh = i2-(valley*1);
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }



                }
                }
                else{//i.e. orb=Top_
                    if(true){

                    //1
                    col = row;
                    Ham_(row,col) += ((-1.0/Parameters_.MStar_top)*NIEnergy(kx_local - valley*k_minusx, ky_local - valley*k_minusy)) + (-1.0*Parameters_.Vz_top);
//		    Ham_(row,col) += NIEnergy(kx_local - k_minusx, ky_local - k_minusy) - (0.5*Parameters_.Vz_);

                    //2 i.e +/- b1
                    i1_neigh = i1 + 1;
                    i2_neigh = i2 + 1;
                    if(i1_neigh<l1_ && i2_neigh<l2_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_top*exp(-iota_complex*Parameters_.Psi_param);
                    }
                    i1_neigh = i1 - 1;
                    i2_neigh = i2 - 1;
                    if(i1_neigh>=0 && i2_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_top*exp(iota_complex*Parameters_.Psi_param);
                    }


                    //3 i.e +/- b3
                    i1_neigh = i1 - 1;
                    i2_neigh = i2;
                    if(i1_neigh>=0 && i2_neigh<l2_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_top*exp(-iota_complex*Parameters_.Psi_param);
                    }
                    i1_neigh = i1 + 1;
                    i2_neigh = i2;
                    if(i1_neigh<l1_ && i2_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_top*exp(iota_complex*Parameters_.Psi_param);
                    }


                    //4 i.e +/- b5
                    i1_neigh = i1;
                    i2_neigh = i2 - 1;
                    if(i2_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_top*exp(-iota_complex*Parameters_.Psi_param);
                    }
                    i1_neigh = i1;
                    i2_neigh = i2 + 1;
                    if(i2_neigh<l2_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_top*exp(iota_complex*Parameters_.Psi_param);
                    }




                    //+/- (1,2)=g^{2}_{1}  (V2)
                    i1_neigh = i1 + 1;
                    i2_neigh = i2 + 2;
                    if(i1_neigh<l1_ && i2_neigh<l2_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V2_param_bottom;
                    }
                    i1_neigh = i1 - 1;
                    i2_neigh = i2 - 2;
                    if(i1_neigh>=0 && i2_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V2_param_bottom;
                    }


                    //+/- (-2,-1)=g^{2}_{3}  (V2)
                    i1_neigh = i1 - 2;
                    i2_neigh = i2 - 1;
                    if( (i1_neigh<l1_ && i2_neigh<l2_)  &&
                         (i1_neigh>=0 && i2_neigh>=0)   ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V2_param_bottom;
                    }
                    i1_neigh = i1 + 2;
                    i2_neigh = i2 + 1;
                    if( (i1_neigh<l1_ && i2_neigh<l2_)  &&
                         (i1_neigh>=0 && i2_neigh>=0)   ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V2_param_bottom;
                    }


                    //+/- (1,-1)=g^{2}_{5}  (V2)
                    i1_neigh = i1 + 1;
                    i2_neigh = i2 - 1;
                    if( (i1_neigh<l1_ && i2_neigh<l2_)  &&
                         (i1_neigh>=0 && i2_neigh>=0)   ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V2_param_bottom;
                    }
                    i1_neigh = i1 - 1;
                    i2_neigh = i2 + 1;
                    if( (i1_neigh<l1_ && i2_neigh<l2_)  &&
                         (i1_neigh>=0 && i2_neigh>=0)   ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V2_param_bottom;
                    }




                    // old verion of w1
                    //col = Coordinates_.Nbasis(i1, i2, Bottom_);
                    //Ham_(row,col) += -1.0*Parameters_.omega_param;





                    // 0(w1)
                    i1_neigh = i1-(valley*0);
                    i2_neigh = i2-(valley*0);
                    if(
            (i2_neigh<l2_ && i2_neigh>=0) &&
                        (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega_param;
                    }


                    //(0,1)=g^{1}_{2} (w1)
                    i1_neigh = i1;
                    i2_neigh = i2-(valley*1);
                    if(
			(i2_neigh<l2_ && i2_neigh>=0) &&
                        (i1_neigh<l1_ && i1_neigh>=0)
			){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega_param;
                    }

                    //(-1,0)=g^{1}_{3} (w1)
                    i1_neigh = i1+(valley*1);
                    i2_neigh = i2;
                    if(
			(i2_neigh<l2_ && i2_neigh>=0) &&
                        (i1_neigh<l1_ && i1_neigh>=0)
			){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega_param;
                    }



                    //(-1,1) (w2)
                    i1_neigh = i1+(valley*1);
                    i2_neigh = i2-(valley*1);
                    if(
            (i2_neigh<l2_ && i2_neigh>=0) &&
                        (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }


                    //(1,1) (w2)
                    i1_neigh = i1-(valley*1);
                    i2_neigh = i2-(valley*1);
                    if(
            (i2_neigh<l2_ && i2_neigh>=0) &&
                        (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }


                    //(-1,-1) (w2)
                    i1_neigh = i1+(valley*1);
                    i2_neigh = i2+(valley*1);
                    if(
            (i2_neigh<l2_ && i2_neigh>=0) &&
                        (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }



                }
                }

            }
        }
    }



} // ----------




void Hamiltonian_ContinuumModel::HTBCreate_WSe2WS2Bilayer(){

    //This is written in hole operators

    Ham_.resize(ns_,ns_);
    double b1x_, b1y_, b6x_, b6y_, b2x_, b2y_;

    //G1+G2
    b1x_=(2.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire); //G1+G2
    b1y_=(0.0)*(2.0*PI/Parameters_.a_moire);

    //G1
    b6x_=(1.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire); //G1
    b6y_=(-1.0)*(2.0*PI/Parameters_.a_moire);


    //G2
    b2x_=(1.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire); //G2
    b2y_=(1.0)*(2.0*PI/Parameters_.a_moire);  //G2


    int Bottom_, Top_;
    Bottom_=0;Top_=1;

    double fac_;

    //l1_/2,l2_/2 is the k-point

    Mat_2_int neigh_G_shell_1, neigh_G_shell_2;
    neigh_G_shell_1.resize(3);neigh_G_shell_2.resize(3);
    for(int s=0;s<3;s++){
    neigh_G_shell_1[s].resize(6);
    neigh_G_shell_2[s].resize(6);
    }

    //shell=1
    neigh_G_shell_1[0][0]=1;neigh_G_shell_2[0][0]=1; //g1
    neigh_G_shell_1[0][1]=0;neigh_G_shell_2[0][1]=1;
    neigh_G_shell_1[0][2]=-1;neigh_G_shell_2[0][2]=0;
    neigh_G_shell_1[0][3]=-1;neigh_G_shell_2[0][3]=-1;
    neigh_G_shell_1[0][4]=0;neigh_G_shell_2[0][4]=-1;
    neigh_G_shell_1[0][5]=1;neigh_G_shell_2[0][5]=0;

    //shell=2
    neigh_G_shell_1[1][0]=1;neigh_G_shell_2[1][0]=2;
        neigh_G_shell_1[1][1]=-1;neigh_G_shell_2[1][1]=1;
        neigh_G_shell_1[1][2]=-2;neigh_G_shell_2[1][2]=-1;
        neigh_G_shell_1[1][3]=-1;neigh_G_shell_2[1][3]=-2;
        neigh_G_shell_1[1][4]=1;neigh_G_shell_2[1][4]=-1;
        neigh_G_shell_1[1][5]=2;neigh_G_shell_2[1][5]=1;

    //shell=3
        neigh_G_shell_1[2][0]=2;neigh_G_shell_2[2][0]=2;
        neigh_G_shell_1[2][1]=0;neigh_G_shell_2[2][1]=2;
        neigh_G_shell_1[2][2]=-2;neigh_G_shell_2[2][2]=0;
        neigh_G_shell_1[2][3]=-2;neigh_G_shell_2[2][3]=-2;
        neigh_G_shell_1[2][4]=0;neigh_G_shell_2[2][4]=-2;
        neigh_G_shell_1[2][5]=2;neigh_G_shell_2[2][5]=0;




    Mat_1_doub VParams;
    VParams.resize(3);
    VParams[0]=Parameters_.V1_param;
    VParams[1]=Parameters_.V2_param;
    VParams[2]=Parameters_.V3_param;


    double kx_local, ky_local;

    int row, col;
    int i1_neigh, i2_neigh;
    for(int i1=0;i1<l1_;i1++){
        for(int i2=0;i2<l2_;i2++){
            kx_local = kx_ + (-(l1_/2)+i1)*(b6x_) + (-(l2_/2)+i2)*(b2x_);
            ky_local = ky_ + (-(l1_/2)+i1)*(b6y_) + (-(l2_/2)+i2)*(b2y_);

                row=Coordinates_.Nbasis(i1, i2, 0);

                    //1
                    col = row;
                    Ham_(row,col) += -(1.0/Parameters_.MStar_bottom)*NIEnergy(kx_local, ky_local);

                    //cout<<row<<" "<<col<<" "<<-(1.0/Parameters_.MStar_bottom)*NIEnergy(kx_local, ky_local)<<endl;

             for (int s=0;s<1;s++){

            for(int neigh_ind=0;neigh_ind<6;neigh_ind++){


                if(neigh_ind==0 || neigh_ind==2 || neigh_ind==4){
                    fac_=-1.0;
                }
                else{
                    fac_=1.0;
                }

            i1_neigh = i1 + neigh_G_shell_1[s][neigh_ind];
            i2_neigh = i2 + neigh_G_shell_2[s][neigh_ind];

            if( (i1_neigh<l1_)  && (i2_neigh<l2_)  && (i1_neigh>=0)  && (i2_neigh>=0)  ){
            col = Coordinates_.Nbasis(i1_neigh, i2_neigh, 0);
                        Ham_(row,col) += -1.0*VParams[s]*exp(fac_*iota_complex*Parameters_.Psi_param);
            }

            }


             }


        }
    }



} // ----------




void Hamiltonian_ContinuumModel::HTBCreate_GammaValleyHomobilayerWithDisplacementField(){


    //This is written in hole operators


    Ham_.resize(ns_*2,ns_*2); //here 2 is for layers

    double b1x_, b1y_, b6x_, b6y_, b2x_, b2y_;

    //G1+G2
    b1x_=(2.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire); //G1+G2
    b1y_=(0.0)*(2.0*PI/Parameters_.a_moire);

    //G1
    b6x_=(1.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire); //G1
    b6y_=(-1.0)*(2.0*PI/Parameters_.a_moire);


    //G2
    b2x_=(1.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire); //G2
    b2y_=(1.0)*(2.0*PI/Parameters_.a_moire);  //G2

    int Bottom_, Top_;
    Bottom_=0;Top_=1;

    //l1_/2,l2_/2 is the k-point

    double kx_local, ky_local;

    int row, col;
    int i1_neigh, i2_neigh;
    for(int i1=0;i1<l1_;i1++){
        for(int i2=0;i2<l2_;i2++){
            kx_local = kx_ + (-(l1_/2)+i1)*(b6x_) + (-(l2_/2)+i2)*(b2x_);
            ky_local = ky_ + (-(l1_/2)+i1)*(b6y_) + (-(l2_/2)+i2)*(b2y_);
            for(int orb=0;orb<2;orb++){
                row=Coordinates_.Nbasis(i1, i2, orb);
                if(orb==Bottom_){

                    if(true){

                    //1
                    col = row;
                    Ham_(row,col) += ((-1.0/Parameters_.MStar_bottom)*NIEnergy(kx_local, ky_local)) + (-1.0*Parameters_.Vz_bottom);

                    //2 i.e +/- b1  (V1)
                    i1_neigh = i1 + 1;
                    i2_neigh = i2 + 1;
                    if(i1_neigh<l1_ && i2_neigh<l2_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_bottom*exp(iota_complex*Parameters_.Psi_param);
                    }
                    i1_neigh = i1 - 1;
                    i2_neigh = i2 - 1;
                    if(i1_neigh>=0 && i2_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_bottom*exp(-iota_complex*Parameters_.Psi_param);
                    }


                    //3 i.e +/- b3  (V1)
                    i1_neigh = i1 - 1;
                    i2_neigh = i2;
                    if(i1_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_bottom*exp(iota_complex*Parameters_.Psi_param);
                    }
                    i1_neigh = i1 + 1;
                    i2_neigh = i2;
                    if(i1_neigh<l1_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_bottom*exp(-iota_complex*Parameters_.Psi_param);
                    }


                    //4 i.e +/- b5 (V1)
                    i1_neigh = i1;
                    i2_neigh = i2 - 1;
                    if(i2_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_bottom*exp(iota_complex*Parameters_.Psi_param);
                    }
                    i1_neigh = i1;
                    i2_neigh = i2 + 1;
                    if(i2_neigh<l2_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_bottom*exp(-iota_complex*Parameters_.Psi_param);
                    }



                    //w0
                    col = Coordinates_.Nbasis(i1, i2, Top_);
                    Ham_(row,col) += -1.0*Parameters_.omega0_param;


                    // +/- b1 (w1)
                    i1_neigh = i1 + 1;
                    i2_neigh = i2 + 1;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega1_param;
                    }
                    i1_neigh = i1 - 1;
                    i2_neigh = i2 - 1;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega1_param;
                    }

                    // +/-b3  (w1)
                    i1_neigh = i1 - 1;
                    i2_neigh = i2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega1_param;
                    }
                    i1_neigh = i1 + 1;
                    i2_neigh = i2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega1_param;
                    }

                    // +b5 (w1)
                    i1_neigh = i1;
                    i2_neigh = i2 - 1;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega1_param;
                    }
                    i1_neigh = i1;
                    i2_neigh = i2 + 1;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega1_param;
                    }


                    // +/-2b1 (w2)
                    i1_neigh = i1 + 2;
                    i2_neigh = i2 + 2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }
                    i1_neigh = i1 - 2;
                    i2_neigh = i2 - 2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }


                    // +/-2b3  (w2)
                    i1_neigh = i1 - 2;
                    i2_neigh = i2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }
                    i1_neigh = i1 + 2;
                    i2_neigh = i2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }

                    // +/- 2b5 (w2)
                    i1_neigh = i1;
                    i2_neigh = i2 - 2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }
                    i1_neigh = i1;
                    i2_neigh = i2 + 2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Top_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }





                }
                }
                else{//i.e. orb=Top_
                    if(true){

                    //1
                    col = row;
                    Ham_(row,col) += ((-1.0/Parameters_.MStar_top)*NIEnergy(kx_local, ky_local )) + (-1.0*Parameters_.Vz_top);

                    //2 i.e +/- b1
                    i1_neigh = i1 + 1;
                    i2_neigh = i2 + 1;
                    if(i1_neigh<l1_ && i2_neigh<l2_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_top*exp(-iota_complex*Parameters_.Psi_param);
                    }
                    i1_neigh = i1 - 1;
                    i2_neigh = i2 - 1;
                    if(i1_neigh>=0 && i2_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_top*exp(iota_complex*Parameters_.Psi_param);
                    }


                    //3 i.e +/- b3
                    i1_neigh = i1 - 1;
                    i2_neigh = i2;
                    if(i1_neigh>=0 && i2_neigh<l2_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_top*exp(-iota_complex*Parameters_.Psi_param);
                    }
                    i1_neigh = i1 + 1;
                    i2_neigh = i2;
                    if(i1_neigh<l1_ && i2_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_top*exp(iota_complex*Parameters_.Psi_param);
                    }


                    //4 i.e +/- b5
                    i1_neigh = i1;
                    i2_neigh = i2 - 1;
                    if(i2_neigh>=0){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_top*exp(-iota_complex*Parameters_.Psi_param);
                    }
                    i1_neigh = i1;
                    i2_neigh = i2 + 1;
                    if(i2_neigh<l2_){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, orb);
                        Ham_(row,col) += -1.0*Parameters_.V_param_top*exp(iota_complex*Parameters_.Psi_param);
                    }




                    // w0
                    col = Coordinates_.Nbasis(i1, i2, Bottom_);
                    Ham_(row,col) += -1.0*Parameters_.omega0_param;



                    // +b1 (w1)
                    i1_neigh = i1-1;
                    i2_neigh = i2-1;
                    if(
            (i2_neigh<l2_ && i2_neigh>=0) &&
                        (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega1_param;
                    }
                    i1_neigh = i1+1;
                    i2_neigh = i2+1;
                    if(
            (i2_neigh<l2_ && i2_neigh>=0) &&
                        (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega1_param;
                    }


                    // +b3  (w1)
                    i1_neigh = i1 + 1;
                    i2_neigh = i2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega1_param;
                    }
                    i1_neigh = i1 - 1;
                    i2_neigh = i2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega1_param;
                    }

                    // +b5 (w1)
                    i1_neigh = i1;
                    i2_neigh = i2 + 1;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega1_param;
                    }
                    i1_neigh = i1;
                    i2_neigh = i2 - 1;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega1_param;
                    }


                    // +2b1 (w2)
                    i1_neigh = i1 - 2;
                    i2_neigh = i2 - 2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }
                    i1_neigh = i1 + 2;
                    i2_neigh = i2 + 2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }

                    // +2b3  (w2)
                    i1_neigh = i1 + 2;
                    i2_neigh = i2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }
                    i1_neigh = i1 - 2;
                    i2_neigh = i2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }


                    // +2b5 (w2)
                    i1_neigh = i1;
                    i2_neigh = i2 + 2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }
                    i1_neigh = i1;
                    i2_neigh = i2 - 2;
                    if( (i2_neigh<l2_ && i2_neigh>=0) &&
            (i1_neigh<l1_ && i1_neigh>=0)
            ){//OBC
                        col = Coordinates_.Nbasis(i1_neigh, i2_neigh, Bottom_);
                        Ham_(row,col) += -1.0*Parameters_.omega2_param;
                    }



                }
                }

            }
        }
    }





}

void Hamiltonian_ContinuumModel::HTBCreate_GammaValleyHomobilayer(){

    //This is written in hole operators

    Ham_.resize(ns_,ns_);
    double b1x_, b1y_, b6x_, b6y_, b2x_, b2y_;

    //G1+G2
    b1x_=(2.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire); //G1+G2
    b1y_=(0.0)*(2.0*PI/Parameters_.a_moire);

    //G1
    b6x_=(1.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire); //G1
    b6y_=(-1.0)*(2.0*PI/Parameters_.a_moire);


    //G2
    b2x_=(1.0/sqrt(3.0))*(2.0*PI/Parameters_.a_moire); //G2
    b2y_=(1.0)*(2.0*PI/Parameters_.a_moire);  //G2


    int Bottom_, Top_;
    Bottom_=0;Top_=1;

    //l1_/2,l2_/2 is the k-point

    Mat_2_int neigh_G_shell_1, neigh_G_shell_2;
    neigh_G_shell_1.resize(3);neigh_G_shell_2.resize(3);
    for(int s=0;s<3;s++){
    neigh_G_shell_1[s].resize(6);
    neigh_G_shell_2[s].resize(6);
    }

    //shell=1
    neigh_G_shell_1[0][0]=1;neigh_G_shell_2[0][0]=1;
    neigh_G_shell_1[0][1]=0;neigh_G_shell_2[0][1]=1;
    neigh_G_shell_1[0][2]=-1;neigh_G_shell_2[0][2]=0;
    neigh_G_shell_1[0][3]=-1;neigh_G_shell_2[0][3]=-1;
    neigh_G_shell_1[0][4]=0;neigh_G_shell_2[0][4]=-1;
    neigh_G_shell_1[0][5]=1;neigh_G_shell_2[0][5]=0;

    //shell=2
    neigh_G_shell_1[1][0]=1;neigh_G_shell_2[1][0]=2;
        neigh_G_shell_1[1][1]=-1;neigh_G_shell_2[1][1]=1;
        neigh_G_shell_1[1][2]=-2;neigh_G_shell_2[1][2]=-1;
        neigh_G_shell_1[1][3]=-1;neigh_G_shell_2[1][3]=-2;
        neigh_G_shell_1[1][4]=1;neigh_G_shell_2[1][4]=-1;
        neigh_G_shell_1[1][5]=2;neigh_G_shell_2[1][5]=1;

    //shell=3
        neigh_G_shell_1[2][0]=2;neigh_G_shell_2[2][0]=2;
        neigh_G_shell_1[2][1]=0;neigh_G_shell_2[2][1]=2;
        neigh_G_shell_1[2][2]=-2;neigh_G_shell_2[2][2]=0;
        neigh_G_shell_1[2][3]=-2;neigh_G_shell_2[2][3]=-2;
        neigh_G_shell_1[2][4]=0;neigh_G_shell_2[2][4]=-2;
        neigh_G_shell_1[2][5]=2;neigh_G_shell_2[2][5]=0;




    Mat_1_doub VParams;
    VParams.resize(3);
    VParams[0]=Parameters_.V1_param;
    VParams[1]=Parameters_.V2_param;
    VParams[2]=Parameters_.V3_param;


    double kx_local, ky_local;

    int row, col;
    int i1_neigh, i2_neigh;
    for(int i1=0;i1<l1_;i1++){
        for(int i2=0;i2<l2_;i2++){
            kx_local = kx_ + (-(l1_/2)+i1)*(b6x_) + (-(l2_/2)+i2)*(b2x_);
            ky_local = ky_ + (-(l1_/2)+i1)*(b6y_) + (-(l2_/2)+i2)*(b2y_);

                row=Coordinates_.Nbasis(i1, i2, 0);

                    //1
                    col = row;
                    Ham_(row,col) += -(1.0/Parameters_.MStar_bottom)*NIEnergy(kx_local, ky_local);

                    //cout<<row<<" "<<col<<" "<<-(1.0/Parameters_.MStar_bottom)*NIEnergy(kx_local, ky_local)<<endl;

             for (int s=0;s<3;s++){

            for(int neigh_ind=0;neigh_ind<6;neigh_ind++){
            i1_neigh = i1 + neigh_G_shell_1[s][neigh_ind];
            i2_neigh = i2 + neigh_G_shell_2[s][neigh_ind];

            if( (i1_neigh<l1_)  && (i2_neigh<l2_)  && (i1_neigh>=0)  && (i2_neigh>=0)  ){
            col = Coordinates_.Nbasis(i1_neigh, i2_neigh, 0);
                        Ham_(row,col) += -1.0*VParams[s]*exp(iota_complex*Parameters_.Psi_param);
            }

            }


             }


        }
    }



} // ----------


Mat_1_intpair Hamiltonian_ContinuumModel::Get_k_path(int path_no){

int n1, n2;
        Mat_1_intpair k_path;
        k_path.clear();

        pair_int temp_pair;
        int L1_,L2_;
        L1_=Parameters_.moire_BZ_L1; //along G1 (b6)
        L2_=Parameters_.moire_BZ_L2; //along G2 (b2)


        if(path_no==0){
            //Gamma to  K=(n1=-L1/3, n2=L2/3)
            n1=0;
            n2=0;
                while(n2<=int(L2_/3)){
            temp_pair.first = n1;
                    temp_pair.second = n2;
                    k_path.push_back(temp_pair);
            n1--;
            n2++;
                //cout<<n1<<"  "<<n2<<endl;
            }


            //K to M=(n1=0, n2 = L2/2)
            n2 = int(L2_/3)+1;
                n1 = -int(L1_/3) +2;
            while(n2<=int(L2_/2)){
            temp_pair.first = n1;
                temp_pair.second = n2;
                k_path.push_back(temp_pair);
                n2 = n2+1;
            n1 = n1+2;
            //cout<<n1<<"  "<<n2<<endl;
            }

            //assert(n1==0);
            n1=0;
            n2=int(L2_/2);
            //M to Gamma
            n2 = n2-1;
            while(n2>=0){
                temp_pair.first = n1;
                temp_pair.second = n2;
                k_path.push_back(temp_pair);
                n2 = n2-1;
                }
        }


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


        if (path_no==2){
        //gamma to M (n1=0,n2=l2/2)
         n1=0;
         n2=0;
         while(n2<(L2_/2)){
             temp_pair.first = n1;
             temp_pair.second = n2;
             k_path.push_back(temp_pair);
             n2++;
             n1=0;
         }

         //M  to K
         //n2=int(2*L2_/3);
         //n1=int(L1_/3);
          //+(2,1)
         //(L1/3 , L2/6)
         n1=0;
         n2=L2_/2;
         while(n1<(L1_/3)){
             temp_pair.first = n1;
             temp_pair.second = n2;
             k_path.push_back(temp_pair);
             n2+=1;
             n1+=2;
         }

         //K to Gamma
         n1=int(L1_/3);
         n2=int(2*L2_/3);
         while(n1>=0){
             temp_pair.first = n1;
             temp_pair.second = n2;
             k_path.push_back(temp_pair);
             n2=n2-2;
             n1=n1-1;
         }

        }

        //path no. =2
        if(path_no==3){
            for(int n1_=0;n1_<L1_;n1_++){
             for(int n2_=0;n2_<L2_;n2_++){
            temp_pair.first = n1_;
            temp_pair.second = n2_;
            k_path.push_back(temp_pair);
             }}
        }


return k_path;

}


void Hamiltonian_ContinuumModel::copy_eigs(int i){

    int space=2*ns_;

    if (i == 0) {
        for(int j=0;j<space;j++) {
            eigs_[j] = eigs_saved_[j];
        }
    }
    else {
        for(int j=0;j<space;j++) {
            eigs_saved_[j] = eigs_[j];
        }
    }

}


#endif
