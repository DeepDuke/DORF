# DORF

## Environment
- Ubuntu 18.04 LTS
- ROS Melodic

## Requirements
### ROS Setting 
- Install ROS on your PC
```Bash
# Setup your sources.list
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# Set up your keys
sudo apt install curl # if you haven't already installed curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo apt update
# Full installation
sudo apt install ros-melodic-desktop-full
```

### Python Setting
- The metric calculation for PR/RR code is implemented by Python2.7 (borrowed from [ERASOR](https://github.com/LimHyungTae/ERASOR))
- To run the code, the following python packages are necessary: [pypcd](https://github.com/dimatura/pypcd), [numpy](https://numpy.org/), [scikit-learn](https://scikit-learn.org/stable/index.html), [bresenham](https://pypi.org/project/bresenham/), [tabulate](https://github.com/gregbanks/python-tabulate)

```Python2.7
pip install pypcd
pip install numpy
pip install scikit-learn
pip install bresenham
pip install tabulate
```

We have already wrapped the DORF into a python wheel for simplely installing dependencices, the wheel file `dorf-0.0.1-py2-none-any.whl` is in the `Releases Section` in this repo. Please download this wheel file and run:
```bash
pip install dorf-0.0.1-py2-none-any.whl
```
then clone this repo:
```bash
git clone git@github.com:DeepDuke/DORF.git
```
## Prepared Dataset
- preprocessed kitti dataset (from [ERASOR](https://github.com/LimHyungTae/ERASOR))
- preprocessed gazebo crowd dataset (also cpnverted into same format with [ERASOR](https://github.com/LimHyungTae/ERASOR))


## Run

- Step 1, uncompress the dataset into `dataset` folder, it should have this structure:
```bash
.
├── gazebo
│   ├── bag
│   │   ├── node_ped_100_voxel_0.2.bag
│   │   ├── node_ped_150_voxel_0.2.bag
│   │   └── node_ped_50_voxel_0.2.bag
│   └── pcd
│       ├── ped_100_raw_voxel_0.2.pcd
│       ├── ped_150_raw_voxel_0.2.pcd
│       └── ped_50_raw_voxel_0.2.pcd
└── kitti
    ├── bag
    │   ├── 00_4390_to_4530_w_interval_2_node.bag
    │   ├── 01_150_to_250_w_interval_1_node.bag
    │   ├── 02_860_to_950_w_interval_2_node.bag
    │   ├── 05_2350_to_2670_w_interval_2_node.bag
    │   └── 07_630_to_820_w_interval_2_node.bag
    └── pcd
        ├── 00_4390_to_4530_w_interval2_voxel_0.200000.pcd
        ├── 01_150_to_250_w_interval1_voxel_0.200000.pcd
        ├── 02_860_to_950_w_interval2_voxel_0.200000.pcd
        ├── 05_2350_to_2670_w_interval2_voxel_0.200000.pcd
        └── 07_630_to_820_w_interval2_voxel_0.200000.pcd
```

- Step 2, run on single sequence, please modify the dataset path in `kitti.yaml` and `gazebo.yaml`
```bash
cd DORF
# Example: test sequence 00 of kitti dataset
python main.py --dataset=kitti --seq=00  --config_path ./config/kitti.yaml --n_proc=10

# Example: test sequence ped_50 of gazebo pedestrian dataset (ped_50 contains 50 pedestrians) 
python main.py --dataset=gazebo --seq=ped50  --config_path ./config/gazebo.yaml --n_proc=10
``` 

We have already provide scripts to run all sequences for these datasets, please run:
```bash
# run on all sequences of kitti
bash ./scripts/run_kitti.sh

# run on all sequences of gazebo
bash ./scripts/run_gazebo.sh
``` 

The results should be in `DORF/results` folder.

## Calculate the metrics
```bash
python dorf/tools/analysis.py --gt /path/to/gt_truth.pcd  --est /path/to/estimated_static_map.pcd 
```

Don't worry, we have also already provide the ground truth and result pcd files. Then you directly reproduce the metrics in this work. Please download them at [dorf_pcd_files](https://drive.google.com/file/d/1wdjOwMGXblpk9CUYGP2mIqrrWqtSCc9I/view?usp=sharing).


## Acknowledgements
Thanks for the code of the following work.

[1] Lim, Hyungtae et al. “ERASOR: Egocentric Ratio of Pseudo Occupancy-Based Dynamic Object Removal for Static 3D Point Cloud Map Building.” IEEE Robotics and Automation Letters 6 (2021): 2272-2279.

[2] Kim, Giseop and Ayoung Kim. “Remove, then Revert: Static Point cloud Map Construction using Multiresolution Range Images.” 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (2020): 10758-10765.
