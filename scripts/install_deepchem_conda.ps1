#!/usr/bin/env pwsh

$CMDNAME = $myInvocation.MyCommand.name
if ($Args.Count -eq 2)
{
    echo "Please set two arguments."
    echo "Usage) $CMDNAME python_version cpu_or_gpu" 1>&2
    echo "Example) $CMDNAME 3.6 gpu" 1>&2
    exit 1
}

# create deepchem environment
conda config --set always_yes yes
conda create --name deepchem python=$args[0]
conda install -c conda-forge conda-merge

if($args[0] -eq "gpu")
{
    # We expect the CUDA vesion is 10.1.
    conda-merge $PWD/env.common.yml $PWD/env.gpu.yml > $PWD/env.yml
    echo "Installing DeepChem in the GPU environment"
}
else
{
    conda-merge $PWD/env.common.yml $PWD/env.cpu.yml > $PWD/env.yml
    echo "Installing DeepChem in the CPU environment"
}

# install all dependencies
conda env update --file $PWD/env.yml
conda activate deepchem
