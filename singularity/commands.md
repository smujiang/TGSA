# follow this tutorial
# https://github.com/JDACS4C-IMPROVE/Singularity/tree/6fba67a60623ca60eff65d96580846170dbedae5 

git clone https://github.com/JDACS4C-IMPROVE/Singularity.git
cd Singularity
rm config/improve.env
./setup

rm definitions/*.def
cp ../TGSA.def definitions/

singularity exec --bind /home/ac.jjiang/data_dir/TGSA/benchmark_dataset_pilot1_generator:/candle_data_dir ./images/TGSA.sif python /usr/local/TGSA/train.py




