import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

from modules.watermark import WatermarkModelEnsemble
from modules.augment import NoiseAugment, RandomTimeStretch, ReverbAugment
from modules.audiodataset import AudioDataset
from modules.metrics import WatermarkMetric

import soundfile as sf

import pandas as pd


torch.backends.cudnn.benchmark = True


def get_dataset_filelist(filelist:str, wavs_dir: str, ext='.wav'):

    with open(filelist, 'r', encoding='utf-8') as fi:
        files = [os.path.join(wavs_dir, x.split('|')[0] + ext)
                          for x in fi.read().split('\n') if len(x) > 0]

    return files


def render(args, h):

    print(f"Arguments: {args}")

    print(f"Config: {h}")

    torch.cuda.manual_seed(h.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    generator = Generator(h, input_channels=h.num_mels)
    generator = generator.to(device)

    print("checkpoints directory : ", args.checkpoint_path)

    if os.path.isdir(args.checkpoint_path):
        cp_g = scan_checkpoint(args.checkpoint_path, 'g_')

    state_dict_g = load_checkpoint(cp_g, device)
    generator.load_state_dict(state_dict_g['generator'])


    eval_filelist = get_dataset_filelist(
        filelist=args.test_filelist,
        wavs_dir=args.wavs_dir,
        ext=args.wavefile_ext
        )


    testset = MelDataset(
        eval_filelist, 
        segment_size=h.segment_size,
        n_fft=h.n_fft, num_mels=h.num_mels,
        hop_size=h.hop_size, win_size=h.win_size,
        sampling_rate=h.sampling_rate,
        fmin=h.fmin, fmax=h.fmax,
        split=False, shuffle=False, 
        n_cache_reuse=0, fmax_loss=h.fmax_for_loss,
        device=device)
    test_loader = DataLoader(
        testset, num_workers=0, shuffle=False,
        sampler=None, batch_size=1, pin_memory=True,
        drop_last=True)

    # Evaluation
    generator.eval()
    torch.cuda.empty_cache()
    val_err_tot = 0

    max_wav_files = args.max_wav_files_out
    if max_wav_files is None:
        max_wav_files = len(test_loader)

    print("Rendering waveforms")
    os.makedirs(f"{args.output_dir}/audio", exist_ok=True)
    with torch.no_grad():
        for j, batch in enumerate(test_loader):

            if j >= max_wav_files:
                break

            x, y, filename, y_mel = batch
            y_g_hat = generator(x.to(device))
            y_g_np = y_g_hat.detach().cpu().squeeze().numpy()
            bname = os.path.splitext(os.path.basename(filename[0]))[0]

            print(f"Rendering file '{bname}', ({j} / {len(test_loader)})")

            sf.write(f"{args.output_dir}/audio/{bname}.wav", y_g_np, h.sampling_rate)




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--wavs_dir')
    parser.add_argument('--test_filelist', default='experiments/filelists/vctk-local/vctk-test.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--wavefile_ext', default='.wav', type=str)
    parser.add_argument('--output_dir')
    parser.add_argument('--max_wav_files_out', default=None, type=int)

    args = parser.parse_args()

    with open(args.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    render(args, h)


if __name__ == '__main__':
    main()
