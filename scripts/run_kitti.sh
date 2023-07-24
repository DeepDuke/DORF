#! /bin/bash

# make 5 directories
result_path="${HOME}/melodic_workspace/DORF/results/";
mkdir -p ${result_path};
cd ${result_path};

dir_set=("00" "01" "02" "05" "07");

# dataset path
pcd_paths=(
    "${HOME}/melodic_workspace/DORF/dataset/kitti/pcd/00_4390_to_4530_w_interval2_voxel_0.200000.pcd"
    "${HOME}/melodic_workspace/DORF/dataset/kitti/pcd/01_150_to_250_w_interval1_voxel_0.200000.pcd"
    "${HOME}/melodic_workspace/DORF/dataset/kitti/pcd/02_860_to_950_w_interval2_voxel_0.200000.pcd"
    "${HOME}/melodic_workspace/DORF/dataset/kitti/pcd/05_2350_to_2670_w_interval2_voxel_0.200000.pcd"
    "${HOME}/melodic_workspace/DORF/dataset/kitti/pcd/07_630_to_820_w_interval2_voxel_0.200000.pcd"
)

for ((i=0; i< 5; i++))  
do  
# run REMO
python ${HOME}/melodic_workspace/DORF/src/main.py --dataset=kitti --seq=${dir_set[$i]} --config_path /home/spacex/melodic_workspace/DORF/config/kitti.yaml --n_proc=1

# kill process
sudo pkill -9 python

done

# analysis
cd ${result_path};
for ((i=0; i< 5; i++))  
do  
static_map_path="seq_${dir_set[$i]}_final_static_map.pcd";
python ${HOME}/melodic_workspace/DORF/src/analysis.py --gt ${pcd_paths[$i]} --est "./${dir_set[$i]}/$static_map_path"
done