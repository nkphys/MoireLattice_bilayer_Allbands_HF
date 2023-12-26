set term epslatex standalone color 
unset key
# unset xtics
# unset ytics

#set cbrange [-1.0:1.0] 
set output 'plot_TL2.tex'

#A1_Len=60
#A2_Len=200

#set border 4095 lw 3 

#unset border 
#unset xtics
#unset ytics

#n1y=sqrt(3.0)*0.5*A2_Len
#n1x=0.5*A2_Len
#n2x=n1x+A1_Len
#n3x=n2x-n1x

#x_off=4
#y_off=4

#xrange_val_min=0-x_off
#xrange_val_max=n2x+x_off

#yrange_val_min=0-y_off
#yrange_val_max=n1y+y_off

#set xrange [xrange_val_min:xrange_val_max]
#set yrange [yrange_val_min:yrange_val_max]
#set yrange [xrange_val_min:xrange_val_max]

#ratio=(yrange_val_max-yrange_val_min+4.7)/(xrange_val_max-xrange_val_min)

#ratio=0.6
ratio=1.0

set size 1,ratio
set style arrow 1 head filled size screen 0.0,20 lw 10 lc pal 
set style arrow 3 head filled size screen 0.1,1,1  lw 1 lc pal 


#set arrow 4 nohead from 0,0 to n1x,n1y lw 2 lc "black"
#set arrow 5 nohead from n1x,n1y to n2x,n1y lw 2 lc "black"
#set arrow 6 nohead from n2x,n1y to n3x,0 lw 2 lc "black"
#set arrow 7 nohead from n3x,0 to 0,0 lw 2 lc "black"

scale = 1.0 
#p 'MySkyrmion.txt' u (( $1 + (0.5*$2)  )-0.4*$4):(( sqrt(3.0)*0.5*($2) )-0.4*$5):(($4)*scale):(($5)*scale):(($3)) w vec arrowstyle 3 notitle

set yr [-6:7]
set xr [-1:12]
unset cbr
#set cbr [-0.2:0.2]
set palette define (0 "red", 0.3 "red", 1.0 "blue")

#p "RealSpace_OParams.txt" u ($3):($4):($8*10000):($7*10000):(sqrt($8*$8 + $7*$7 +$6*$6)) w vec arrowstyle 3 notitle
#p "~/Desktop/Data_ACF/MoTe2Bilayer_HF_Runs/Filling1.0/TwistAngle2.5/N_MoireCells6x6/MagneticUC_N00_1_N01_0_N10_0_N11_1/N_HF_BANDS3/DistanceToGate200/epsilon4.0/RandomSeed1/layer_0_RealSpace_OParams.txt" u ($3):($4):($6*1200):($8*1200):(sqrt($8*$8 + $7*$7 + $6*$6)*1200) w vec arrowstyle 3 notitle
#layer_0_RealSpace_OParams_moiresites.txt

#p "/home/nitin/Desktop/Telperion/data4/home/nn7/Moire_HFRuns/Filling3.0/TwistAngle1.0/N_MoireCells6x6/MagneticUC_N00_6_N01_0_N10_0_N11_6/N_HF_BANDS3/DistanceToGate200_MagSplit0.00/epsilon10.0/RandomSeed1/layer_0_RealSpace_OParams.txt" u ($3):($4):($6*5000000):($7*500000):(($6)*50000) w vec arrowstyle 3 notitle
#1:2:($2*0+10) with circles

#p "/home/nitin/Desktop/Lisa1/GammaValleyRuns/Filling2.0/TwistAngle4.0/N_MoireCells6x6/MagneticUC_N00_1_N01_0_N10_0_N11_1/N_HF_BANDS2/DistanceToGate200_MagSplit0.00/epsilon15.0/RandomSeed1/layer_0_RealSpace_OParams.txt" u ($3):($4):($6*100):($7*100):(($6)*50) w vec arrowstyle 3 notitle
#1:2:($2*0+10) with circles

p "layer_0_RealSpace_OParams.txt" u ($3):($4):($8*400):($7*400):((1*$6*$6 + 1*$7*$7 + 1*$8*$8)) w vec arrowstyle 3 notitle
#p "layer_0_RealSpace_OParams_moiresites.txt" u ($5):($6):($10*1000000):($9*1000000):((1*$8*$8 + 1*$9*$9 + 1*$10*$10)) w vec arrowstyle 3 notitle

#set style fill transparent solid 0.6 noborder
#p "layer_1_RealSpace_OParams.txt" u ($3):($4):(($5)*50.0):($5) with circles lc palette notitle


set palette defined (0 "white", 1.0 "blue")
set pm3d map
set pm3d corners2color c1
set pm3d interpolate 1,1

#sp "/home/nitin/Desktop/Telperion/data4/home/nn7/Moire_HFRuns/Filling3.0/TwistAngle1.0/N_MoireCells6x6/MagneticUC_N00_6_N01_0_N10_0_N11_6/N_HF_BANDS3/DistanceToGate200_MagSplit0.00/epsilon10.0/RandomSeed1/layer_0_RealSpace_OParams.txt" u ($3):($4):(($5)*500.0) w pm3d map lc palette notitle
#sp "/home/nitin/Desktop/Telperion/data4/home/nn7/GammaValleyRuns/Filling0.5/TwistAngle1.0/N_MoireCells6x6/MagneticUC_N00_6_N01_0_N10_0_N11_6/N_HF_BANDS2/DistanceToGate200_MagSplit0.00/epsilon4.0/RandomSeed3/layer_0_RealSpace_OParams.txt" u ($3):($4):(($5)*500.0) w pm3d map lc palette notitle

#sp "/home/nitin/Desktop/Lisa1/GammaValleyRuns/Filling2.0/TwistAngle1.0/N_MoireCells6x6/MagneticUC_N00_3_N01_0_N10_0_N11_3/N_HF_BANDS2/DistanceToGate200_MagSplit0.00/epsilon80.0/RandomSeed3/layer_0_RealSpace_OParams.txt" u ($3):($4):(($5)*500.0) w pm3d map lc palette notitle

#sp "/home/nitin/Desktop/Lisa2/GammaValleyRuns/Filling1.0/TwistAngle1.0/N_MoireCells6x6/MagneticUC_N00_6_N01_0_N10_0_N11_6/N_HF_BANDS2/DistanceToGate200_MagSplit0.00/epsilon10.0/RandomSeed1/layer_0_RealSpace_OParams.txt" u ($3):($4):(($5)*500000000.0) w pm3d map lc palette notitle

#set cbr [0:1]
#sp "layer_0_RealSpace_OParams.txt" u ($3):($4):((($5))*50.0) w pm3d map lc palette notitle

#p "layer_0_RealSpace_OParams_moiresites.txt" u ($5 - 0.5*$8*0.2):($6 - 0.5*$10*0.2):($9*4000.2):($10*4000.2):(sqrt($8*$8 + $9*$9 +$10*$10)) w vec arrowstyle 3 notitle

#p "RealSpace_OParams_moiresites.txt" u ($5 - (0.0*$10) ):($6 -  ($9*0.0) ):($9):($10):($8) w vec arrowstyle 3 notitle
#p "RealSpace_OParams.txt" u ($3):($4):($6*8000*1):($7*8000*1):($5) w vec arrowstyle 3 notitle 
