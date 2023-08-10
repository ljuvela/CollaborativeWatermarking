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
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from modules.metrics import DiscriminatorMetrics
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

from modules.watermark import WatermarkModelEnsemble
from modules.augment import get_augmentations


torch.backends.cudnn.benchmark = True


def train(rank, a, h):

    print(f"Arguments: {a}")

    print(f"Config: {h}")

    if h.num_gpus > 1:
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    if torch.cuda.is_available():
        device = torch.device('cuda:{:d}'.format(rank))
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
    
    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
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

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)
        watermark = DistributedDataParallel(watermark, device_ids=[rank]).to(device)

    if a.pretrained_watermark_path is not None:
        state_dict = torch.load(a.pretrained_watermark_path, map_location='cpu')
        watermark.load_pretrained_state_dict(state_dict)

    if a.freeze_watermark_weights:
        for param in watermark.parameters():
            param.requires_grad_(False)
        if h.get('watermark_unfreeze_output_layer', True):
            watermark.output_layer_requires_grad_(True)
        watermark.eval()
        watermark.train_rnns() # enable gradients for RNNs

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_wm = torch.optim.AdamW(watermark.parameters(), 
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    if state_dict_wm is not None:
        optim_wm.load_state_dict(state_dict_wm['optim_wm'])
    elif last_epoch > -1:
        for pg in optim_wm.param_groups:
            pg['initial_lr'] = pg['lr']

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_wm = torch.optim.lr_scheduler.ExponentialLR(optim_wm, gamma=h.lr_decay, last_epoch=last_epoch)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    trainset = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)

    train_sampler = DistributedSampler(trainset) if h.num_gpus > 1 else None

    train_loader = DataLoader(trainset, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    augmentation_train, augmentation_valid = get_augmentations(h, a, device=device)

    if rank == 0:
        validset = MelDataset(
            validation_filelist, 
            segment_size=h.segment_size,
            n_fft=h.n_fft, num_mels=h.num_mels,
            hop_size=h.hop_size, win_size=h.win_size,
            sampling_rate=h.sampling_rate,
            fmin=h.fmin, fmax=h.fmax,
            split=False, shuffle=False, 
            n_cache_reuse=0, fmax_loss=h.fmax_for_loss,
            device=device, fine_tuning=a.fine_tuning,
            base_mels_path=a.input_mels_dir)
        validation_loader = DataLoader(
            validset, num_workers=0, shuffle=False,
            sampler=None, batch_size=1, pin_memory=True,
            drop_last=True)
        
        validset_batch = MelDataset(
            validation_filelist, 
            segment_size=h.segment_size,
            n_fft=h.n_fft, num_mels=h.num_mels,
            hop_size=h.hop_size, win_size=h.win_size,
            sampling_rate=h.sampling_rate,
            fmin=h.fmin, fmax=h.fmax,
            split=True, shuffle=False,
            n_cache_reuse=0, fmax_loss=h.fmax_for_loss,
            device=device, fine_tuning=a.fine_tuning,
            base_mels_path=a.input_mels_dir)
        validation_loader_batch = DataLoader(
            validset_batch, num_workers=h.num_workers, shuffle=False,
            sampler=None, batch_size=h.batch_size, pin_memory=True,
            drop_last=True)


        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))

    generator.train()
    mpd.train()
    msd.train()
    if a.freeze_watermark_weights:
        watermark.eval()
        watermark.train_rnns() # enable gradients for RNNs
    else:
        watermark.train()


    # generator = torch.compile(generator)
    # mpd = torch.compile(mpd)
    # msd = torch.compile(msd)
    for epoch in range(max(0, last_epoch), a.training_epochs):

        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)

        for i, batch in enumerate(train_loader):

            if rank == 0:
                start_b = time.time()
            x, y, _, y_mel = batch # mel, audio, filename, mel_loss
            x = torch.autograd.Variable(x.to(device, non_blocking=True))
            y = torch.autograd.Variable(y.to(device, non_blocking=True))
            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
            y = y.unsqueeze(1)

            y_g_hat = generator(x)

            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft,
                                          h.num_mels, h.sampling_rate,
                                          h.hop_size, h.win_size,
                                          h.fmin, h.fmax_for_loss)

            optim_d.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            optim_d.step()

            # Generator
            optim_g.zero_grad()
            optim_wm.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            # Adversary 
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

            # Watermark
            y_r_wm_input = y
            wm_role = h.get('watermark_role', 'collaborator')
            if wm_role == 'collaborator':
                y_g_wm_input = y_g_hat
            elif wm_role == 'observer':
                y_g_wm_input = y_g_hat.detach()
            if augmentation_train is not None:
                y_g_wm_input = augmentation_train(y_g_wm_input)
                y_r_wm_input = augmentation_train(y)
            y_wm_real, y_wm_fake = watermark(y_r_wm_input, y_g_wm_input)
            loss_wm_total = 0.0
            wm_losses_r = []
            wm_losses_f = []
            for y_wm_real_i, y_wm_fake_i in zip(y_wm_real, y_wm_fake):
                wm_loss, wm_loss_r, wm_loss_f = discriminator_loss(
                    disc_real_outputs=y_wm_real_i, disc_generated_outputs=y_wm_fake_i)
                loss_wm_total += wm_loss
                wm_losses_r.append(wm_loss_r)
                wm_losses_f.append(wm_loss_f)

            # Adversarial (S, F), Feature matching (S, F), Mel, Collaborative
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_wm_total

            loss_gen_all.backward()
            optim_g.step()
            optim_wm.step()

            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        mel_error = F.l1_loss(y_mel, y_g_hat_mel).item()

                    print('Steps : {:d}, Gen Loss Total : {:4.3f}, Mel-Spec. Error : {:4.3f}, s/b : {:4.3f}'.
                          format(steps, loss_gen_all, mel_error, time.time() - start_b))

                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    checkpoint_path = "{}/g_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict()})
                    checkpoint_path = "{}/do_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'mpd': (mpd.module if h.num_gpus > 1
                                                         else mpd).state_dict(),
                                     'msd': (msd.module if h.num_gpus > 1
                                                         else msd).state_dict(),
                                     'optim_g': optim_g.state_dict(), 'optim_d': optim_d.state_dict(), 'steps': steps,
                                     'epoch': epoch})
                    checkpoint_path = "{}/wm_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'watermark': (watermark.module if h.num_gpus > 1
                                                         else watermark).state_dict(),
                                     'optim_wm': optim_wm.state_dict()
                                     })


                # Tensorboard summary logging
                if steps % a.summary_interval == 0:
                    sw.add_scalar("training/gen_loss_total", loss_gen_all, steps)
                    sw.add_scalar("training/mel_spec_error", mel_error, steps)
                    # Framed Discriminator losses
                    sw.add_scalar("training_gan/disc_f_r", sum(losses_disc_f_r), steps)
                    sw.add_scalar("training_gan/disc_f_g", sum(losses_disc_f_g), steps)
                    # Multiscale Discriminator losses
                    sw.add_scalar("training_gan/disc_s_r", sum(losses_disc_s_r), steps)
                    sw.add_scalar("training_gan/disc_s_g", sum(losses_disc_s_g), steps)
                    # Framed Generator losses
                    sw.add_scalar("training_gan/gen_f", sum(losses_gen_f), steps)
                    # Multiscale Generator losses
                    sw.add_scalar("training_gan/gen_s", sum(losses_gen_s), steps)
                    # Feature Matching losses
                    sw.add_scalar("training_gan/loss_fm_f", loss_fm_f, steps)
                    sw.add_scalar("training_gan/loss_fm_s", loss_fm_s, steps)

                    # WATERMARK LOSSES
                    for label, losses in zip(watermark.get_labels(), wm_losses_r):
                        sw.add_scalar(f"training_watermark/{label}_real", sum(losses), steps)
                    for label, losses in zip(watermark.get_labels(), wm_losses_f):
                        sw.add_scalar(f"training_watermark/{label}_fake", sum(losses), steps)

                    # log minibatch EER
                    if a.log_training_eer:
                        watermark_metrics = [DiscriminatorMetrics() for i in range(watermark.get_num_models())]
                        with torch.no_grad():
                            for metrics, y_wm_real_i, y_wm_fake_i in zip(watermark_metrics, y_wm_real, y_wm_fake):
                                metrics.accumulate(
                                    disc_real_outputs = y_wm_real_i,
                                    disc_fake_outputs = y_wm_fake_i
                                )
                        for label, metric in zip(watermark.get_labels(), watermark_metrics):
                            sw.add_scalar(f"training_watermark/{label}_equal_error_rate", metric.eer, steps)

                # Validation
                if steps % a.validation_interval == 0:
                # if steps % a.validation_interval == 0 and steps != 0:
                    print(f"Validating at step {steps}")
                    generator.eval()
                    mpd.eval()
                    msd.eval()
                    watermark.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0

                    # Validation set metrics
                    mpd_adversary_metrics = DiscriminatorMetrics()
                    msd_adversary_metrics = DiscriminatorMetrics()
                    watermark_metrics = [DiscriminatorMetrics() for i in range(watermark.get_num_models())]
  
                    with torch.no_grad():

                        for j, batch in enumerate(validation_loader_batch):

                            x, y, _, y_mel = batch
                            y_g_hat = generator(x.to(device))

                            # Apply augmentation and batch validation samples
                            y = torch.autograd.Variable(y.to(device, non_blocking=True))
                            y = y.unsqueeze(1)
                            if augmentation_valid is not None:
                                y_g_wm_input = augmentation_valid(y_g_hat)
                                y_r_wm_input = augmentation_valid(y)
                            else:
                                y_g_wm_input = y_g_hat
                                y_r_wm_input = y

                            # Watermark EER
                            y_wm_real, y_wm_fake = watermark(y_r_wm_input, y_g_wm_input)
                            for metrics, y_wm_real_i, y_wm_fake_i in zip(watermark_metrics, y_wm_real, y_wm_fake):
                                metrics.accumulate(
                                    disc_real_outputs = y_wm_real_i,
                                    disc_fake_outputs = y_wm_fake_i
                                )

                            if a.log_discriminator_validation_eer:
                                # Calculate discriminator EER
                                y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat)
                                mpd_adversary_metrics.accumulate(
                                    disc_real_outputs = y_df_hat_r,
                                    disc_fake_outputs = y_df_hat_g
                                )

                                y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat)
                                msd_adversary_metrics.accumulate(
                                    disc_real_outputs = y_ds_hat_r,
                                    disc_fake_outputs = y_ds_hat_g
                                )

                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            y_g_hat = generator(x.to(device))

                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size,
                                                          h.fmin, h.fmax_for_loss)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()

                            if j <= 4:
                                if steps == 0:
                                    sw.add_audio('gt/y_{}'.format(j), y[0], steps, h.sampling_rate)
                                    sw.add_figure('gt/y_spec_{}'.format(j), plot_spectrogram(x[0]), steps)

                                sw.add_audio('generated/y_hat_{}'.format(j), y_g_hat[0], steps, h.sampling_rate)
                                y_hat_spec = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate, h.hop_size, h.win_size,
                                                             h.fmin, h.fmax)
                                sw.add_figure('generated/y_hat_spec_{}'.format(j),
                                              plot_spectrogram(y_hat_spec.squeeze(0).cpu().numpy()), steps)

                        val_err = val_err_tot / (j+1)
                        sw.add_scalar("validation/mel_spec_error", val_err, steps)

                        for label, metric in zip(watermark.get_labels(), watermark_metrics):
                            sw.add_scalar(f"validation/watermark_{label}_equal_error_rate", metric.eer, steps)

                        if a.log_discriminator_validation_eer:
                            sw.add_scalar("validation/mpd_adversary_equal_error_rate", mpd_adversary_metrics.eer, steps)
                            sw.add_scalar("validation/msd_adversary_equal_error_rate", msd_adversary_metrics.eer, steps)

                    generator.train()
                    mpd.train()
                    msd.train()
                    if a.freeze_watermark_weights:
                        watermark.eval()
                        watermark.train_rnns() # enable gradients for RNNs
                    else:
                        watermark.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        scheduler_wm.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))


def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()

    parser.add_argument('--group_name', default=None)
    parser.add_argument('--input_wavs_dir', default='LJSpeech-1.1/wavs')
    parser.add_argument('--input_mels_dir', default='ft_dataset')
    parser.add_argument('--input_training_file', default='LJSpeech-1.1/training.txt')
    parser.add_argument('--input_validation_file', default='LJSpeech-1.1/validation.txt')
    parser.add_argument('--checkpoint_path', default='cp_hifigan')
    parser.add_argument('--pretrained_watermark_path', default=None)
    parser.add_argument('--freeze_watermark_weights', default=False, type=bool)
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--log_training_eer', default=False, type=bool)
    parser.add_argument('--log_discriminator_validation_eer', default=False)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--wavefile_ext', default='.wav', type=str)
    parser.add_argument('--use_augmentation', default=False, type=bool)
    parser.add_argument('--noise_input_training_file', default='experiments/filelists/musan/train_list.txt')
    parser.add_argument('--noise_input_validation_file', default='experiments/filelists/musan/valid_list.txt')
    parser.add_argument('--noise_input_wavs_dir', default='../../DATA/musan/noise/free-sound')
    parser.add_argument('--reverb_input_training_file', default='experiments/filelists/mit-rir/train_list.txt')
    parser.add_argument('--reverb_input_validation_file', default='experiments/filelists/mit-rir/valid_list.txt')
    parser.add_argument('--reverb_input_wavs_dir', default='../../DATA/MIT-RIR/Audio')


    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()
