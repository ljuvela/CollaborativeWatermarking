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
from metrics import DiscriminatorMetrics
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

from wrappers.watermark import WatermarkModelEnsemble
from wrappers.augment import NoiseAugment, RandomTimeStretch, ReverbAugment
from wrappers.audiodataset import AudioDataset

import soundfile as sf

import pandas as pd


torch.backends.cudnn.benchmark = True


def get_dataset_filelist(filelist:str, wavs_dir: str, ext='.wav'):

    with open(filelist, 'r', encoding='utf-8') as fi:
        files = [os.path.join(wavs_dir, x.split('|')[0] + ext)
                          for x in fi.read().split('\n') if len(x) > 0]

    return files


def eval(a, h):

    print(f"Arguments: {a}")

    print(f"Config: {h}")


    torch.cuda.manual_seed(h.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    generator = Generator(h, input_channels=h.num_mels)
    generator = generator.to(device)
    mpd = MultiPeriodDiscriminator().to(device)
    msd = MultiScaleDiscriminator().to(device)

    watermark = WatermarkModelEnsemble(
        model_type=h.watermark_model,
        sample_rate=h.sampling_rate,
        config=h
        ).to(device)
    
    
    print(generator)
    print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
        cp_wm = scan_checkpoint(a.checkpoint_path, 'wm_')

    steps = 0
    if cp_g is None or cp_do is None:
        state_dict_do = None
        last_epoch = -1
    else:
        state_dict_g = load_checkpoint(cp_g, device)
        state_dict_do = load_checkpoint(cp_do, device)
        generator.load_state_dict(state_dict_g['generator'])
        mpd.load_state_dict(state_dict_do['mpd'])
        msd.load_state_dict(state_dict_do['msd'])
        steps = state_dict_do['steps'] + 1
        last_epoch = state_dict_do['epoch']

    if cp_wm is None:
        state_dict_wm = None
    else:
        state_dict_wm = load_checkpoint(cp_wm, device)
        watermark.load_state_dict(state_dict_wm['watermark'])

    eval_filelist = get_dataset_filelist(
        filelist=a.test_filelist,
        wavs_dir=a.wavs_dir,
        ext=a.wavefile_ext
        )

    noise_filelist = get_dataset_filelist(
        filelist=a.noise_input_eval_file,
        wavs_dir=a.noise_input_wavs_dir,
        ext=a.wavefile_ext)
    aug_noise = NoiseAugment(
        AudioDataset(
            split=True,
            training_files=noise_filelist,
            sampling_rate=h.sampling_rate, 
            segment_size=h.segment_size,
            resample=True, device=device),
        batch_size=h.batch_size,
        snr=10, num_workers=0).to(device)
    
    reverb_filelist = get_dataset_filelist(
        filelist=a.noise_input_eval_file,
        wavs_dir=a.noise_input_wavs_dir,
        ext=a.wavefile_ext)
    aug_reverb = ReverbAugment(
        AudioDataset(
            training_files=reverb_filelist, 
            sampling_rate=h.sampling_rate,
            split=False,
            resample=True,
            device=device),
        num_workers=0).to(device)

    aug_stretch = RandomTimeStretch(min_scale_factor=0.9, max_scale_factor=1.1).to(device)
    

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
    
    testset_batch = MelDataset(
        eval_filelist, 
        segment_size=h.segment_size,
        n_fft=h.n_fft, num_mels=h.num_mels,
        hop_size=h.hop_size, win_size=h.win_size,
        sampling_rate=h.sampling_rate,
        fmin=h.fmin, fmax=h.fmax,
        split=True, shuffle=False,
        n_cache_reuse=0, device=device)
    test_loader_batch = DataLoader(
        testset_batch, num_workers=h.num_workers, shuffle=False,
        sampler=None, batch_size=h.batch_size, pin_memory=True,
        drop_last=True)


    # Evaluation
    generator.eval()
    mpd.eval()
    msd.eval()
    watermark.eval()
    torch.cuda.empty_cache()
    val_err_tot = 0


    class Condition():

        def __init__(self, tag:str,
                    model: torch.nn.Module, 
                    augmentation: torch.nn.Module):
            self.tag = tag
            self.augmentation = augmentation
            self.model = model
            self.metrics = [DiscriminatorMetrics() for i in range(watermark.get_num_models())]

        def accumulate(self, input_real, input_fake, model):

            real_aug = self.augmentation(input_real)
            fake_aug = self.augmentation(input_fake)

            score_real, score_fake = self.model(real_aug, fake_aug)

            for metrics, y_wm_real_i, y_wm_fake_i in zip(self.metrics, score_real, score_fake):
                metrics.accumulate(
                    disc_real_outputs = y_wm_real_i,
                    disc_fake_outputs = y_wm_fake_i
                )

        def get_eer(self):
            out_val = []
            model_labels = self.model.get_labels()
            for model_label, metric in zip(model_labels, self.metrics):
                out_val.append((model_label, self.tag, metric.eer))
            return out_val


    conditions = [
        Condition('clean', model=watermark, augmentation=torch.nn.Identity()),
        Condition('noise', model=watermark, augmentation=aug_noise),
        Condition('stretch', model=watermark, augmentation=aug_stretch),
        Condition('reverb', model=watermark, augmentation=aug_reverb),
        Condition('stretch + noise + reverb', model=watermark, 
            augmentation=torch.nn.Sequential(aug_stretch, aug_noise, aug_reverb))
    ]

    with torch.no_grad():

        for j, batch in enumerate(test_loader_batch):

            print(f"Calculating metrics for batch ({j+1}/{len(test_loader_batch)})")
            x, y, _, y_mel = batch
            y_g_hat = generator(x.to(device))

            # Apply augmentation and batch validation samples
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            for cond in conditions:
                cond.accumulate(input_real=y, input_fake=y_g_hat, model=watermark)

            break

        watermark_role = h['watermark_role']
        train_aug_types = h.get('augmentation_types', None)
        if train_aug_types is None:
            augmentation = 'none'
        else:
            augmentation = ' + '.join(train_aug_types)
        pretrained = 'true' if a.watermark_was_pretrained else 'false'

        df_list = []
        for cond in conditions:
            for subcond in cond.get_eer():
                model_tag, cond_tag, eer = subcond
                df_list.append(pd.DataFrame(
                    {
                    'model': model_tag,
                    'condition': cond_tag,
                    'eer': eer,
                    'role': watermark_role,
                    'augmentation': augmentation,
                    'pretrained': pretrained
                    }, index=[0]
                ))

        # Save metrics dataframe 
        df = pd.concat(df_list, ignore_index=True)
        df.to_csv(os.path.join(a.output_dir, 'results.csv'))
        
        os.makedirs(f"{a.output_dir}/audio", exist_ok=True)
        # Render waveforms for listening
        for j, batch in enumerate(test_loader):


            # y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            # y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
            #                                 h.hop_size, h.win_size,
            #                                 h.fmin, h.fmax_for_loss)
            # val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

            if j <= a.max_wav_files_out:

                x, y, filename, y_mel = batch
                y_g_hat = generator(x.to(device))
                y_g_np = y_g_hat.detach().cpu().squeeze().numpy()
                bname = os.path.splitext(os.path.basename(filename[0]))[0]
                sf.write(f"{a.output_dir}/audio/{bname}.wav", y_g_np, h.sampling_rate)


        # val_err = val_err_tot / (j+1)
        # sw.add_scalar("validation/mel_spec_error", val_err, steps)



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--test_filelist', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--config', default='')
    parser.add_argument('--wavefile_ext', default='.wav', type=str)
    parser.add_argument('--noise_input_eval_file', default='experiments/filelists/musan/test_list.txt')
    parser.add_argument('--noise_input_wavs_dir', default='../../DATA/musan/noise/free-sound')
    parser.add_argument('--reverb_input_eval_file', default='experiments/filelists/mit-rir/test_list.txt')
    parser.add_argument('--reverb_input_wavs_dir', default='../../DATA/MIT-RIR/Audio')
    parser.add_argument('--watermark_was_pretrained', default=False, type=bool)
    parser.add_argument('--output_dir')
    parser.add_argument('--max_wav_files_out', default=10, type=int)

    a = parser.parse_args()

    with open(a.config) as f:
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

    eval(a, h)


if __name__ == '__main__':
    main()
