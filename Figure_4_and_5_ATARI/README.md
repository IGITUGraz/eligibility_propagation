## Installation guide and system requirements
The scripts for <i>reward-based e-prop</i> were tested for tensorflow 1.14.0, python 3.7.7, gcc 8.3.0 using >12 CPU cores, >32GB RAM and also a GPU of the type NVIDIA GeForce GTX 1080Ti.

Setting up the environment for reproduction should take no more than 10 minutes by carrying out the following steps:

- Install conda (version 4.6.12 or above)
- Create a new environment with all dependencies by:
```
conda create --name reward-based-e-prop --file conda_requirements.txt
conda activate reward-based-e-prop
pip install -r requirements.txt
```
- Download the [Arcade-Learning-Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) (ALE) by:
```
git clone https://github.com/mgbellemare/Arcade-Learning-Environment
```
- Compile the ALE and export its binaries into a new directory `ale`:
```
cd Arcade-Learning-Environment
mkdir build && cd build
cmake -DUSE_SDL=OFF -DUSE_RLGLUE=OFF -DBUILD_EXAMPLES=ON -DCMAKE_INSTALL_PREFIX=../../ale ..
make -j 4
make install
cd ../..
```
- Compile the custom tensorflow op that embeds ALE into tensorflow graphs: 
```
python compile.py ale
```

## Training a spiking agent with <i>reward-based symmetric e-prop</i>
To perform training use the command:
```
LD_LIBRARY_PATH=ale/lib python main.py --result_dir results
```
This script should print a status output after every 5M processed frames along with plots in the results directory and it is expected to finish completely with a score ~10 after ~3 days.

## Acknowledgements

This code is inspired by [https://github.com/deepmind/scalable-agent] and [https://github.com/dudevil/tf-ale-op]
