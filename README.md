# Collaborative Watermarking for Adversarial Speech Synthesis

This repository contains source code for "Collaborative Watermarking for Adversarial Speech Synthesis", submitted to ICASSP 2024.

Listen to audio samples at the demo page https://ljuvela.github.io/CollaborativeWatermarkingDemo/


## Environment setup

Create a `conda` environment and install dependencies  
```bash
conda create -n adversarial-watermarking python=3.10
conda activate adversarial-watermarking
conda install -c pytorch -c conda-forge pytorch torchaudio pytest conda-build matplotlib scipy pysoundfile tensorboard
```

Install submodule dependencies and link them to the conda python evironment. You could use `PYTHONPATH` instead, but we recommend `conda` to keep the environment contained 
```bash
git submodule update --init --recursive
conda develop third_party/hifi-gan
conda develop third_party/asvspoof-2021/LA/Baseline-LFCC-LCNN
```


## Pretrained models

To replicate experiments, download pretrained ASVSpoof models

```bash
cd third_party/asvspoof-2021/LA/Baseline-LFCC-LCNN/project
sh 00_download.sh
```

We do not currently distribute pre-trained HiFi GAN models. In the paper, we trained a wide range of configurations for 100k iterations, which still leaves a quality gap to the official Hifi GAN trained for 2.5M iterations (https://github.com/jik876/hifi-gan). If there is demand for a production-quality watermarked HiFi GAN, we will definitely consider training one.

### Noise augmentation

Musan dataset is available as direct download from
https://www.openslr.org/17/

Torchaudio has a nice wrapper, but only available in the nightly prototype at the moment 
https://pytorch.org/audio/master/generated/torchaudio.prototype.datasets.Musan.html

Using the nightly build provides a convenient method for download, but be careful not to break your environment
```bash
conda install -c pytorch-nightly -c nvidia torchaudio
dataset = torchaudio.prototype.datasets.Musan(root='/path/on/your/system/MUSAN', subset='noise', download=True)
```

### Room impulse repsonse augmentation

The repository also implements room impulse response augmentation. The paper didn't have room to evaluate this, but you're welcome to experiment

Download the MIT RIR dataset from
http://mcdermottlab.mit.edu/Reverb/IR_Survey.html


## VSCode config

The following config snippet helps VSCode find the dependencies for development
```json
{
    "terminal.integrated.env.linux" : {"PYTHONPATH": "${workspaceFolder}/third_party/hifi-gan;${workspaceFolder}/tests"},
    "terminal.integrated.env.osx" :   {"PYTHONPATH": "${workspaceFolder}/third_party/hifi-gan;${workspaceFolder}/tests"},

    "python.analysis.extraPaths": [
        "third_party/hifi-gan",
        "third_party/asvspoof-2021/LA/Baseline-LFCC-LCNN"]
}
```




