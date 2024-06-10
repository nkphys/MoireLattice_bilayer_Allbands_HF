#include <iostream>
#include <iomanip>
#include <fstream>
#include <stdio.h>
#include <vector>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <random>
#include <complex>
#include <cmath>
#include <cassert>
using namespace std;
#include "Matrix.h"
#include "ParametersEngine.h"
#include "Coordinates_ContinuumModel.h"
#include "Hamiltonian_ContinuumModel.h"
#include "Coordinates.h"
#include "Hamiltonian_HF.h"

#include "random"


int main(int argc, char *argv[]) {

    string ex_string_original =argv[0];

    string ex_string;
    //ex_string.substr(ex_string_original.length()-5);
    ex_string=ex_string_original.substr (2);
    cout<<"'"<<ex_string<<"'"<<endl;

    bool PerformHartreeFock=true;



    if(ex_string=="MoireBands" || true){


        string model_inputfile = argv[1];

        if (argc<2) { throw std::invalid_argument("USE:: executable inputfile"); }

        Parameters Parameters_;
        Parameters_.Initialize(model_inputfile);


        int n1, n2;
        Mat_1_intpair k_path, k_path2, k_path3, k_path_choosen;
        k_path.clear();
        k_path2.clear();
        k_path3.clear();
        pair_int temp_pair;
        int L1_,L2_;
        L1_=Parameters_.moire_BZ_L1; //along G1 (b6)
        L2_=Parameters_.moire_BZ_L2; //along G2 (b2)

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


        //k_path2
        //K+' to K-'
        n1=int(2*L1_/3);
        n2=int(L2_/3);
        while(n2<=int((2*L2_)/3)){
            temp_pair.first = n1;
            temp_pair.second = n2;
            k_path2.push_back(temp_pair);
            n1=n1-1;
            n2=n2+1;
        }

        //K-' to Gamma
        n1=int(L1_/3)-1;
        n2=int((2*L2_)/3)-2;
        while(n1>=0){
            temp_pair.first = n1;
            temp_pair.second = n2;
            k_path2.push_back(temp_pair);
            n1=n1-1;
            n2=n2-2;
        }



         //k_path3
        //K-' to K+'
        n1=int(L1_/3);
        n2=int((2*L2_)/3);
        while(n1<=int((2*L2_)/3)){
            temp_pair.first = n1;
            temp_pair.second = n2;
            k_path3.push_back(temp_pair);
            n1=n1+1;
            n2=n2-1;
        }

        //K+' to Gamma
        n1=int((2*L1_)/3)-2;
        n2=int(L2_/3)-1;
        while(n1>=0){
            temp_pair.first = n1;
            temp_pair.second = n2;
            k_path3.push_back(temp_pair);
            n1=n1-2;
            n2=n2-1;
        }

        int Coord_norbs=Parameters_.max_layer_ind;




        Coordinates_ContinuumModel Coordinates_(Parameters_.Grid_moireRL_L1, Parameters_.Grid_moireRL_L2, Coord_norbs);
        Hamiltonian_ContinuumModel Hamiltonian_(Parameters_, Coordinates_);

         Hamiltonian_.Saving_NonInteractingSpectrum();
         Hamiltonian_.NonInteractingSpectrum_AlongPath(Hamiltonian_.Get_k_path(2));

         //assert(false);

         Hamiltonian_.Calculate_ChernNumbers();

         Coordinates Coordinates_HF_(Parameters_.moire_BZ_L1, Parameters_.moire_BZ_L2, 1);

        // cout<<"CHECK Coordinates"<<endl;
        // for(int i=0;i<Parameters_.moire_BZ_L1*Parameters_.moire_BZ_L2;i++){
        // cout<<i<<"  "<<Coordinates_HF_.indx_cellwise(i)<<"  "<<Coordinates_HF_.indy_cellwise(i)<<endl;
        // }

        if(PerformHartreeFock){
         mt19937_64 Generator_(Parameters_.RandomSeed);
         Hamiltonian Hamiltonian_HF_(Parameters_, Coordinates_HF_,Hamiltonian_, Generator_);
       
       
         //Hamiltonian_HF_.PrintBlochStates();
        // Hamiltonian_HF_.Print_Vint();
        //   Hamiltonian_HF_.Calculate_FormFactors();
         // Hamiltonian_HF_.PrintFormFactors(0, 0, 0);
         // Hamiltonian_HF_.PrintFormFactors(0, 0, 1);
        //  Hamiltonian_HF_.PrintFormFactors(0, 1, 0);
        //  Hamiltonian_HF_.PrintFormFactors(1, 0, 0);
        //  Hamiltonian_HF_.PrintFormFactors(0, 0, 1);
         // Hamiltonian_HF_.Print_Interaction_value2(0,0);
         // Hamiltonian_HF_.Print_Interaction_value2(2,3);

        Hamiltonian_HF_.RunSelfConsistency();
        }

        //assert(false);

        k_path_choosen=k_path;
        
        for(int spin=0;spin<=1;spin++){
        Hamiltonian_.valley = 2*spin -1;
        string file_bands_out="Bands_energy_spin" +to_string(spin)+ "_alongPath.txt";
        ofstream FileBandsOut(file_bands_out.c_str());
        FileBandsOut<<"#index kx_value ky_value E0(k) Overlap_bottom Overlap_top E1(k) Overlap_bottom Overlap_top  E2(k) ....."<<endl;
        for(int index=0;index<k_path_choosen.size();index++){
        n1=k_path_choosen[index].first;
        n2=k_path_choosen[index].second;


        Hamiltonian_.kx_=(2.0*PI/Parameters_.a_moire)*(n1*(1.0/(sqrt(3)*L1_))  +  n2*(1.0/(sqrt(3)*L2_)));
        Hamiltonian_.ky_=(2.0*PI/Parameters_.a_moire)*(n1*(-1.0/(L1_))  +  n2*(1.0/(L2_)));
        Hamiltonian_.HTBCreate();
        char Dflag='V';
        Hamiltonian_.Diagonalize(Dflag);


        cout << k_path_choosen.size()<<"  "<<index <<"  "<<Hamiltonian_.Ham_.n_col()<<endl;
        FileBandsOut<<index<<"  "<<Hamiltonian_.kx_<<"  "<<Hamiltonian_.ky_<<"   ";
        for(int band=0;band<Hamiltonian_.eigs_.size();band++){
	    Hamiltonian_.Get_Overlap_layers(band);
            FileBandsOut<<Hamiltonian_.eigs_[band]<<"  "<<Hamiltonian_.Overlap_bottom<<"  "<<Hamiltonian_.Overlap_top<<"  ";
        }
        FileBandsOut<<endl;
        }
    }

    }






    cout << "--------THE END--------" << endl;
} // main
