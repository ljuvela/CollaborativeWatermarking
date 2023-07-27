# AdversarialWatermarking


## Setup


Create Conda environment 
```bash
conda create -n adversarial-watermarking python=3.10
conda activate adversarial-watermarking

conda install -c pytorch -c conda-forge pytorch torchaudio pytest conda-build matplotlib scipy pysoundfile tensorboard

conda install -c pytorch-nightly -c nvidia torchaudio


dataset = torchaudio.prototype.datasets.Musan(root='/scratch/elec/t412-speechcom/ljuvela/DATA/MUSAN', subset='noise', download=True)

```

Install submodule dependencies and link them to the conda python evironment. You could use `PYTHONPATH` but this keeps the environment contained 

```bash
git submodule update --init --recursive
conda develop third_party/hifi-gan
conda develop third_party/asvspoof-2021/LA/Baseline-LFCC-LCNN
# conda develop third_party/asvspoof-2021/LA/
```

Can uninstall with
```bash
conda develop -u <path-to-python-module>
```


Download pretrained ASVSpoof models

```bash
cd third_party/asvspoof-2021/LA/Baseline-LFCC-LCNN/project
sh 00_download.sh
```


## Naming

Collaboratove WaterMarking (CoWaMa)

Adversarial Watermarking (AdWaMa)


## VSCODE config for development

```json
{
    "terminal.integrated.env.linux" : {"PYTHONPATH": "${workspaceFolder}/third_party/hifi-gan;${workspaceFolder}/tests"},
    "terminal.integrated.env.osx" :   {"PYTHONPATH": "${workspaceFolder}/third_party/hifi-gan;${workspaceFolder}/tests"},

    "python.analysis.extraPaths": [
        "third_party/hifi-gan",
        "third_party/asvspoof-2021/LA/Baseline-LFCC-LCNN"]
}
```


### Noise agumentation

Musan dataset is available as direct download from
https://www.openslr.org/17/


Torchaudio has a nice wrapper, but only available in the nightly prototype at the moment 
https://pytorch.org/audio/master/generated/torchaudio.prototype.datasets.Musan.html


### Room impulse repsonse augmentation

http://mcdermottlab.mit.edu/Reverb/IR_Survey.html



