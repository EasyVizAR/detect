# EasyVizAR Object Detection

# Installation

## Installation of the Snap Package

The snap package is automatically built and released to the snap store on each
commit to the main branch. It is currently built for amd64 and arm64 platforms.
Unfortunately, it will probably only work with CPU and not Cuda because of
difficulties packaging Cuda libraries in the snap.

Install the stable version:

    sudo snap install easyvizar-detect

Install the latest build:

    sudo snap install --edge easyvizar-detect

The install will take some time to complete, as it downloads and installs
PyTorch as part of the install operation.

If this will be running on the same machine as the easyvizar-edge snap
(recommended), you should connect the easyvizar-edge:data interface to allow
easyvizar-detect to read image files stored by easyvizar-edge.

    sudo snap connect easyvizar-detect:data easyvizar-edge:data

## Installation on Jetson

In order to take advantage of the GPU, it is recommended to run the Python code
directly on the Jetson board without Docker or Snap confinement. Start by using
the Nvidia SDK tools to set up the required drivers.

Find a version of the PyTorch wheel that is compatible with your Jetson.
<https://forums.developer.nvidia.com/t/pytorch-for-jetson-version-1-11-now-available/72048>

Find a version of torchvision that is compatible.
<https://pypi.org/project/torchvision/>
<https://qengineering.eu/install-pytorch-on-jetson-nano.html>

Be careful to use a compatible version of Python that is compatible with the
dependencies as well as Nvidia's PyTorch wheel. Python 3.6 with torch 1.10.0 and
torchvision 0.11.0 seems to work.

    sudo apt-get install libopenblas-base libopenmpi-dev
    python3.6 -m pip install --upgrade pip
    python3.6 -m pip install -r requirements.txt
    python3.6 -m pip install torch-1.10.0-cp36-cp36m-linux_aarch64.whl
    python3.6 -m pip install torchvision-0.11.0a0+fa347eb-cp36-cp36m-linux_aarch64.whl

If everything is set up correctly, the following command should print "True".

    python3.6 -c "import torch; print(torch.cuda.is_available())"

Finally, run the detector:

    python3.6 -m detect

## Using Docker

    docker build -t detect
    docker run -d --gpus all --add-host=host.docker.internal:host-gateway -e VIZAR_SERVER="http://host.docker.internal:5000" --restart always --name detect detect

## Preparing a Model File

Download a pretrained model file.

    https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-seg.pt

Use the yolo export script.

    yolo export model=yolov8m-seg.pt format=onnx imgsz=428,760 opset=16

Use the augment script to add an NMS layer.

    python3 scripts/augment_model.py yolov8m-seg.onnx

Use netron to verify that the model looks correct.

    netron yolov8m-seg-nms.onnx
