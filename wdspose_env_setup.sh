mkdir downloaded_weights
# mkdir -p ~/wdspose_env && cd ~/wdspose_env

# MINICONDA_INSTALLER_SCRIPT=Miniconda3-4.5.4-Linux-x86_64.sh
# MINICONDA_PREFIX=/usr/local
# wget -nc https://repo.continuum.io/miniconda/$MINICONDA_INSTALLER_SCRIPT
# chmod +x $MINICONDA_INSTALLER_SCRIPT
# ./$MINICONDA_INSTALLER_SCRIPT -b -f -p $MINICONDA_PREFIX

# conda update -y -n base conda

# # git clone https://github.com/cocodataset/cocoapi.git
# git clone https://github.com/facebookresearch/Detectron.git
# git clone https://github.com/leoxiaobin/deep-high-resolution-net.pytorch.git
# git clone https://github.com/vegesm/wdspose.git




# download some weights
wget -nc -O ./downloaded_weights/best_generalization_net_G.pth http://www.cs.cornell.edu/projects/megadepth/dataset/models/best_generalization_net_G.pth
mv ./downloaded_weights/best_generalization_net_G.pth ./wdspose/best_generalization_net_G.pth

wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1zYC7go9EV0XaSlSBjMaiyE_4TcHc_S38" -O ./downloaded_weights/pose_hrnet_w32_256x192.pth && rm -rf /tmp/cookies.txt

wget -nc --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1azqyC3mnryVMm9rbHFcAlk9ERBl13aOR' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1azqyC3mnryVMm9rbHFcAlk9ERBl13aOR" -O ./downloaded_weights/model_data.zip && rm -rf /tmp/cookies.txt
unzip -o ./downloaded_weights/model_data.zip -d ./wdspose



# cp  ~/3dpose_setup/wdspose_custom_files/wdspose_script.py ~/wdspose_env/wdspose
# cp  ~/3dpose_setup/wdspose_custom_files/show_images.py ~/wdspose_env/wdspose
# touch ~/wdspose_env/deep-high-resolution-net.pytorch/lib/core/__init__.py


conda create -n wdspose -y python=2.7

source activate wdspose



conda install -y pytorch==1.1.0 torchvision==0.3.0 cudatoolkit=10.0 -c pytorch
# conda install -y pytorch==1.0.0 torchvision==0.2.1 cuda80 -c pytorch

conda install -y -c conda-forge opencv
# conda install -y -c confa-forge ros-noetic-desktop

pip install numpy pyyaml==3.12 matplotlib setuptools Cython mock scipy six future protobuf json_tricks yacs pycocotools pathlib pathlib2


# geht ueber pip:
    # cd cocoapi/PythonAPI/ && make install
    # cd ../..

# cd deep-high-resolution-net.pytorch/lib && make
# cd ../..

# cd Detectron && make
# cd ..

mkdir -p ./deep-high-resolution-net.pytorch/lib/models/pytorch/pose_coco
mv ./downloaded_weights/pose_hrnet_w32_256x192.pth ./deep-high-resolution-net.pytorch/lib/models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth

# cp ~/3dpose_setup/wdspose_custom_files/megadepth.py ~/wdspose_env/wdspose/scripts/megadepth.py
# cp ~/3dpose_setup/wdspose_custom_files/maskrcnn.py ~/wdspose_env/wdspose/scripts/maskrcnn.py
# cp ~/3dpose_setup/wdspose_custom_files/torch_tools.py ~/wdspose_env/wdspose/training/torch_tools.py
# cp ~/3dpose_setup/wdspose_custom_files/utils.py ~/wdspose_env/deep-high-resolution-net.pytorch/lib/utils/utils.py
# cp ~/3dpose_setup/wdspose_custom_files/hrnet.py ~/wdspose_env/wdspose/scripts/hrnet.py
# cp ~/3dpose_setup/wdspose_custom_files/predict.py ~/wdspose_env/wdspose/scripts/predict.py
# cp ~/3dpose_setup/wdspose_custom_files/preprocess.py ~/wdspose_env/wdspose/training/preprocess.py


# cp ~/3dpose_setup/wdspose_custom_files/metadata.csv ~/wdspose_env/wdspose/examples/metadata.csv



# cd ~/wdspose_env/wdspose
# python wdspose_script.py
