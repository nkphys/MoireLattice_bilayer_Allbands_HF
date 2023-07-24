#ifndef Parameters_class
#define Parameters_class
#include "tensor_type.h"

class Parameters{

public:

    double V_param_top,V_param_bottom, Psi_param, omega_param, a_moire;
    double TwistTheta, a_monolayer, MStar_top, MStar_bottom, eps_DE;
    int Grid_moireRL_L1, Grid_moireRL_L2;
    int moire_BZ_L1, moire_BZ_L2;
    string MUC_row1, MUC_row2;
    double Vz_top, Vz_bottom;
    double ValleyTau;
    int N_bands_HF;
    double d_gate;
    Matrix<int> NMat_MUC;
    int NMat_det;
    bool FixingMu;
    double MuValueFixed;


    double Temperature; //in Kelvin
    double holesdensity_per_moire_unit_cell;  //per moire unit cell
    int Max_HFIterations;
    double HF_convergence_error;
    double SimpleMixing_alpha;


    int RandomSeed;
    string MaterialName;    
    void Initialize(string inputfile_);
    double matchstring(string file,string match);
    string matchstring2(string file,string match);


};


void Parameters::Initialize(string inputfile_){


    cout << "____________________________________" << endl;
    cout << "Reading the inputfile name: " << inputfile_ << endl;
    cout << "____________________________________" << endl;

    MaterialName=matchstring2(inputfile_, "MaterialName");


    holesdensity_per_moire_unit_cell=matchstring(inputfile_, "holedensity_per_moire_unit_cell");
    Temperature = matchstring(inputfile_, "Temperature_in_Kelvin");
    HF_convergence_error = matchstring(inputfile_, "HF_convergence_error");
    SimpleMixing_alpha = matchstring(inputfile_, "SimpleMixing_alpha");
    Max_HFIterations = int(matchstring(inputfile_, "Max_HFIterations"));
    RandomSeed = int(matchstring(inputfile_, "RandomSeed"));

    V_param_top = matchstring(inputfile_,"V_param_top_in_meV");
    V_param_bottom = matchstring(inputfile_,"V_param_bottom_in_meV");
    Psi_param = matchstring(inputfile_,"Psi_param_in_radians");
    omega_param = matchstring(inputfile_,"omega_param_in_meV");
    Grid_moireRL_L1 = int(matchstring(inputfile_,"Grid_moireReciprocalLattice_L1"));
    Grid_moireRL_L2 = int(matchstring(inputfile_,"Grid_moireReciprocalLattice_L2"));
    moire_BZ_L1 = int(matchstring(inputfile_,"moire_BZ_L1"));
    moire_BZ_L2 = int(matchstring(inputfile_,"moire_BZ_L2"));
    
    MUC_row1 = matchstring2(inputfile_,"MagneticUnitCell_NMat_row1");
    MUC_row2 = matchstring2(inputfile_,"MagneticUnitCell_NMat_row2");

    N_bands_HF = int(matchstring(inputfile_,"No_of_bands_used_for_HartreeFock"));

    a_monolayer = matchstring(inputfile_,"a_monolayer_in_angstorm");
    d_gate = matchstring(inputfile_,"distance_to_gate_in_angstorm");
    TwistTheta = matchstring(inputfile_,"Twist_Theta_in_radians");
    MStar_top = matchstring(inputfile_,"MStar_top_in_RestMass");
    MStar_bottom = matchstring(inputfile_,"MStar_bottom_in_RestMass");
    eps_DE = matchstring(inputfile_, "eps_DE");
    Vz_top=matchstring(inputfile_,"Layer_Potential_top_Vz_in_meV");
    Vz_bottom=matchstring(inputfile_,"Layer_Potential_bottom_Vz_in_meV");

    
    stringstream MUC_row1_ss(MUC_row1);
    stringstream MUC_row2_ss(MUC_row2);
    NMat_MUC.resize(2,2);
    MUC_row1_ss>>NMat_MUC(0,0)>>NMat_MUC(0,1);
    MUC_row2_ss>>NMat_MUC(1,0)>>NMat_MUC(1,1);

    NMat_det = (NMat_MUC(0,0)*NMat_MUC(1,1) - NMat_MUC(0,1)*NMat_MUC(1,0));

    if(abs(NMat_det)<=0.0000001){
        cout<<"NMat_MUC determinant cannot be 0, please change the NMat"<<endl;
        assert(false);
    }

    if( (moire_BZ_L1*moire_BZ_L2)%(abs(NMat_det)) !=0){
        cout<<"no. of moire sites must be multiple of NMat_determinant, Please change the NMat_determinant"<<endl;
        assert(false);
    }
    else{
        cout<<"No.of atoms in single Magnetic Unit cell = NMat_determinant = "<<abs(NMat_det)<<endl;
        cout<<"Total no. of magnetic unit cells = "<< (moire_BZ_L1*moire_BZ_L2)/(abs(NMat_det))<<endl;
    }

    if((NMat_MUC(1,1)*moire_BZ_L1)%abs(NMat_det) != 0){
    cout<<"(NMat_MUC(1,1)*moire_BZ_L1)%abs(NMat_det) == 0 FAILED"<<endl;
    assert((NMat_MUC(1,1)*moire_BZ_L1)%abs(NMat_det) == 0);
    }
    if((NMat_MUC(0,0)*moire_BZ_L2)%abs(NMat_det) != 0){
    cout<<"(NMat_MUC(0,0)*moire_BZ_L2)%abs(NMat_det) == 0 FAILED"<<endl;
    assert((NMat_MUC(0,0)*moire_BZ_L2)%abs(NMat_det) == 0);
    }
    if((NMat_MUC(1,0)*moire_BZ_L2)%abs(NMat_det) != 0){
        cout<<"(NMat_MUC(1,0)*moire_BZ_L2)%abs(NMat_det) == 0 FAILED"<<endl;
    assert((NMat_MUC(1,0)*moire_BZ_L2)%abs(NMat_det) == 0);
    }
    if((NMat_MUC(0,1)*moire_BZ_L1)%abs(NMat_det) != 0){
        cout<<"(NMat_MUC(0,1)*moire_BZ_L1)%abs(NMat_det) == 0 FAILED"<<endl;
    assert((NMat_MUC(0,1)*moire_BZ_L1)%abs(NMat_det) == 0);
    }


    cout<<"Emergent Magnetic unit cell (MUC) recirocal lattice vectors"<<endl;
    cout<<"in units of moire brilloun zone lattice points"<<endl;
    cout<<"G1_tilde = "<<(NMat_MUC(1,1)*moire_BZ_L1)/NMat_det<<" , "<<-1*(NMat_MUC(1,0)*moire_BZ_L2)/NMat_det<<endl;
    cout<<"G2_tilde = "<<-1*(NMat_MUC(0,1)*moire_BZ_L1)/NMat_det<<" , "<<(NMat_MUC(0,0)*moire_BZ_L2)/NMat_det<<endl;


    // for(int i=0;i<2;i++){
    //     for(int j=0;j<2;j++){
    // assert(NMat_MUC(i,j)>=0);
    // }
    // }


    ValleyTau=1;
    assert(abs(ValleyTau)==1);

    if(MaterialName=="MoTe2Homobilayer"){
    a_moire = a_monolayer/abs(TwistTheta);
    }
    if(MaterialName=="MoTe2WSe2Bilayer"){
    double a_bottom, a_top;
    a_bottom = 3.575; //in Angstorm
    a_top = 3.32;
    a_moire = (a_bottom*a_top)/(a_bottom-a_top);
    }
    cout<<"a_moire (in Angstorm)= "<<a_moire<<endl;



    //To be added later in input file
    FixingMu=false;
    MuValueFixed=-10; //in meV
    
    //assert(false);

}


double Parameters::matchstring(string file,string match) {
    string test;
    string line;
    ifstream readFile(file);
    double amount;
    bool pass=false;
    while (std::getline(readFile, line)) {
        std::istringstream iss(line);
        if (std::getline(iss, test, '=') && pass==false) {
            // ---------------------------------
            if (iss >> amount && test==match) {
                // cout << amount << endl;
                pass=true;
            }
            else {
                pass=false;
            }
            // ---------------------------------
            if(pass) break;
        }
    }
    if (pass==false) {
        string errorout=match;
        errorout+="= argument is missing in the input file!";
        throw std::invalid_argument(errorout);
    }
    cout << match << " = " << amount << endl;
    return amount;
}

string Parameters::matchstring2(string file,string match) {

    string line;
    ifstream readFile(file);
    string amount;
    int offset;

    if(readFile.is_open())
    {
        while(!readFile.eof())
        {
            getline(readFile,line);

            if ((offset = line.find(match, 0)) != string::npos) {
                amount = line.substr (offset+match.length()+1);				}

        }
        readFile.close();
    }
    else
    {cout<<"Unable to open input file while in the Parameters class."<<endl;}




    cout << match << " = " << amount << endl;
    return amount;
}

#endif



