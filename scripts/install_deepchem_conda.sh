#!/usr/bin/env bash

CMDNAME=`basename $0`
if [ $# -ne 2 ]; then
    echo "Please set two arguments."
    echo "Usage) $CMDNAME python_version cpu_or_gpu" 1>&2
    echo "Example) $CMDNAME 3.6 gpu" 1>&2
    exit 1
fi

# create deepchem environment
conda config --set always_yes yes
conda create --name deepchem python=$0
conda install -c conda-forge conda-merge

if [ "$0" = "gpu" ];
then
    # We expect the CUDA vesion is 10.1.
    conda-merge $PWD/env.common.yml $PWD/env.gpu.yml > $PWD/env.yml
    echo "Installing DeepChem in the GPU environment"
else
    if [ "$(uname)" == 'Darwin' ];
    then
        conda-merge $PWD/env.common.yml $PWD/env.mac.cpu.yml > $PWD/env.yml
    else
        conda-merge $PWD/env.common.yml $PWD/env.cpu.yml > $PWD/env.yml
    fi
    echo "Installing DeepChem in the CPU environment"
fi

# install all dependencies
conda env update --file $PWD/env.yml
conda activate deepchem

# # Fixed packages
# tensorflow=2.3.0
# tensorflow_probability==0.11.0
# torch=1.6.0
# torchvision=0.7.0
# pyg_torch=1.6.0

# # Install TensorFlow dependencies
# pip install tensorflow==$tensorflow tensorflow-probability==$tensorflow_probability

# # Install PyTorch dependencies
# if [ "$(uname)" == 'Darwin' ];
# then
#     # For MacOSX
#     pip install torch==$torch torchvision==$torchvision
# else
#     pip install torch==$torch+$cuda torchvision==$torchvision+$cuda -f https://download.pytorch.org/whl/torch_stable.html
# fi

# # Install PyTorch Geometric and DGL dependencies
# pip install torch-scatter==latest+$cuda -f https://pytorch-geometric.com/whl/torch-$pyg_torch.html
# pip install torch-sparse==latest+$cuda -f https://pytorch-geometric.com/whl/torch-$pyg_torch.html
# pip install torch-cluster==latest+$cuda -f https://pytorch-geometric.com/whl/torch-$pyg_torch.html
# pip install torch-spline-conv==latest+$cuda -f https://pytorch-geometric.com/whl/torch-$pyg_torch.html
# pip install torch-geometric
# pip install $dgl_pkg
# # install transformers package
# pip install transformers
