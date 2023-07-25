#! /bin/bash

# make 5 directories
result_path="/home/spacex/melodic_workspace/DORF/results/";
mkdir -p ${result_path};
cd ${result_path};

dir_set=("ped50" "ped100" "ped150");

# dataset path
pcd_paths=(
    "/home/spacex/melodic_workspace/DORF/dataset/gazebo/pcd/ped_50_raw_voxel_0.2.pcd"
    "/home/spacex/melodic_workspace/DORF/dataset/gazebo/pcd/ped_100_raw_voxel_0.2.pcd"
    "/home/spacex/melodic_workspace/DORF/dataset/gazebo/pcd/ped_150_raw_voxel_0.2.pcd"
)

for ((i=0; i< 3; i++))  
do  
# run REMO
python /home/spacex/melodic_workspace/DORF/main.py --dataset=gazebo --seq=${dir_set[$i]} --config_path /home/spacex/melodic_workspace/DORF/config/gazebo.yaml --n_proc=10

# kill process
sudo pkill -9 python

done

# analysis
cd ${result_path};
for ((i=0; i< 5; i++))  
do  
static_map_path="seq_${dir_set[$i]}_final_static_map.pcd";
python /home/spacex/melodic_workspace/DORF/tools/analysis.py --gt ${pcd_paths[$i]} --est "./${dir_set[$i]}/$static_map_path"
done