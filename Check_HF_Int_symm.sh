k1_=1
mk1_=1

k2_=1
mk2_=1


for q in {0..3}
do
grep "^${k1_}  ${k2_}  ${q} " Interation_spins0_0_bands_0_0_0_0.txt
grep "^${mk2_}  ${mk1_}  ${q} " Interation_spins1_1_bands_0_0_0_0.txt
echo ""
done
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"

for q in {0..3}
do 
grep "^2  2   2   2  0  0  ${k1_}  ${k2_}  ${q} " HF_Band_projected_Interaction_bands_2_3.txt
grep "^3  3   3   3  1  1  ${mk2_}  ${mk1_}  ${q} " HF_Band_projected_Interaction_bands_2_3.txt
echo ""
 done

 echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXx"

 for q in {0..3}
do
grep "^2  2   2   2  0  0  ${k1_}  ${k2_}  ${q} " HF_Band_projected_Interaction_bands_2_3_TR_and_Inversion_imposed.txt
grep "^3  3   3   3  1  1  ${mk2_}  ${mk1_}  ${q} " HF_Band_projected_Interaction_bands_2_3_TR_and_Inversion_imposed.txt
echo ""
 done

