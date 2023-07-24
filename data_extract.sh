rm v2.txt

for i2 in {..47}
do
for i1 in {24..47}
do

i=$(echo "${i1}+(${i2}*3*24)" | bc -l)

grep "^0  ${i} " Lambda_k_band_spin0_bands0_0_old.txt >> v2.txt


done
done
