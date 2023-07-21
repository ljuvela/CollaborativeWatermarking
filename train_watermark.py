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
from metrics import DiscriminatorMetrics
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

from wrappers.models import LFCC_LCNN


torch.backends.cudnn.benchmark = True


def train(rank, a, h):

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

    mpd_watermark = MultiPeriodDiscriminator().to(device)
    msd_watermark = MultiScaleDiscriminator().to(device)

    mpd_adversary_metrics = DiscriminatorMetrics()
    msd_adversary_metrics = DiscriminatorMetrics()
    mpd_collab_metrics = DiscriminatorMetrics()
    msd_collab_metrics = DiscriminatorMetrics()

    if rank == 0:
        print(generator)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    if os.path.isdir(a.checkpoint_path):
        cp_g = scan_checkpoint(a.checkpoint_path, 'g_')
        cp_do = scan_checkpoint(a.checkpoint_path, 'do_')
        cp_do_wm = scan_checkpoint(a.checkpoint_path, 'do_wm_') 

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

    if cp_do_wm is None:
        state_dict_do_wm = None
    else:
        state_dict_do_wm = load_checkpoint(cp_do_wm, device)
        mpd_watermark.load_state_dict(state_dict_do_wm['mpd'])
        msd_watermark.load_state_dict(state_dict_do_wm['msd'])

    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        mpd = DistributedDataParallel(mpd, device_ids=[rank]).to(device)
        msd = DistributedDataParallel(msd, device_ids=[rank]).to(device)
        mpd_watermark = DistributedDataParallel(mpd_watermark, device_ids=[rank]).to(device)
        msd_watermark = DistributedDataParallel(msd_watermark, device_ids=[rank]).to(device)

    optim_g = torch.optim.AdamW(generator.parameters(), h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()),
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2])
    optim_d_wm = torch.optim.AdamW(itertools.chain(msd_watermark.parameters(), mpd_watermark.parameters()), 
                                h.learning_rate, betas=[h.adam_b1, h.adam_b2]) # TODO: lump with generator?

    if state_dict_do is not None:
        optim_g.load_state_dict(state_dict_do['optim_g'])
        optim_d.load_state_dict(state_dict_do['optim_d'])

    if state_dict_do_wm is not None:
        optim_d_wm.load_state_dict(state_dict_do_wm['optim_d'])
    elif last_epoch > -1:
        for pg in optim_d_wm.param_groups:
            pg['initial_lr'] = pg['lr']

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_d_wm = torch.optim.lr_scheduler.ExponentialLR(optim_d_wm, gamma=h.lr_decay, last_epoch=last_epoch)

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

    if rank == 0:
        validset = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)
        validation_loader = DataLoader(validset, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)

        sw = SummaryWriter(os.path.join(a.checkpoint_path, 'logs'))


    generator.train()
    mpd.train()
    msd.train()
    mpd_watermark.train()
    msd_watermark.train()

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
            optim_d_wm.zero_grad()

            # MPD
            y_df_hat_r, y_df_hat_g, _, _ = mpd(y, y_g_hat.detach())
            loss_disc_f, losses_disc_f_r, losses_disc_f_g = discriminator_loss(y_df_hat_r, y_df_hat_g)
            # MSD
            y_ds_hat_r, y_ds_hat_g, _, _ = msd(y, y_g_hat.detach())
            loss_disc_s, losses_disc_s_r, losses_disc_s_g = discriminator_loss(y_ds_hat_r, y_ds_hat_g)
            loss_disc_all = loss_disc_s + loss_disc_f
            loss_disc_all.backward()
            optim_d.step()

            # # MPD Watermark
            # y_df_hat_wm_r, y_df_hat_wm_g, _, _ = mpd_watermark(y, y_g_hat.detach())
            # loss_disc_f_wm, losses_disc_f_wm_r, losses_disc_f_wm_g = discriminator_loss(
            #     real=y_df_hat_r, generated = y_df_hat_g)
            # # MSD Watermark
            # y_ds_hat_wm_r, y_ds_hat_wm_g, _, _ = msd_watermark(y, y_g_hat.detach())
            # loss_disc_s_wm, losses_disc_s_wm_r, losses_disc_s_wm_g = discriminator_loss(
            #     real=y_ds_hat_r, generated=y_ds_hat_g)
            # # Aggregate losses and apply optimization step
            # loss_disc_wm_all = loss_disc_s_wm + loss_disc_f_wm
            # loss_disc_wm_all.backward()
            # optim_d_wm.step()


            # Generator
            optim_g.zero_grad()
            optim_d_wm.zero_grad()

            # L1 Mel-Spectrogram Loss
            loss_mel = F.l1_loss(y_mel, y_g_hat_mel) * 45

            # Adversary 
            y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = mpd(y, y_g_hat)
            y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = msd(y, y_g_hat)
            loss_fm_f = feature_loss(fmap_f_r, fmap_f_g)
            loss_fm_s = feature_loss(fmap_s_r, fmap_s_g)
            loss_gen_f, losses_gen_f = generator_loss(y_df_hat_g)
            loss_gen_s, losses_gen_s = generator_loss(y_ds_hat_g)

            # Collaborator (watermark), Generator is aligned with Discriminator
            y_df_hat_wm_r, y_df_hat_wm_g, _, _ = mpd_watermark(y, y_g_hat)
            loss_disc_f_wm, losses_disc_f_wm_r, losses_disc_f_wm_g = discriminator_loss(
                disc_real_outputs=y_df_hat_wm_r, disc_generated_outputs=y_df_hat_wm_g)
            y_ds_hat_wm_r, y_ds_hat_wm_g, _, _ = msd_watermark(y, y_g_hat)
            loss_disc_s_wm, losses_disc_s_wm_r, losses_disc_s_wm_g = discriminator_loss(
                disc_real_outputs=y_ds_hat_wm_r, disc_generated_outputs=y_ds_hat_wm_g)

            # Adversarial (S, F), Feature matching (S, F), Mel, Collaborative
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel + loss_disc_f_wm + loss_disc_s_wm

            loss_gen_all.backward()
            optim_g.step()
            optim_d_wm.step()

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
                    checkpoint_path = "{}/do_wm_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path, 
                                    {'mpd': (mpd_watermark.module if h.num_gpus > 1
                                                         else mpd_watermark).state_dict(),
                                     'msd': (msd_watermark.module if h.num_gpus > 1
                                                         else msd_watermark).state_dict(),
                                     'optim_d': optim_d_wm.state_dict()
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
                    # Framed Discriminator losses
                    sw.add_scalar("training_watermark/disc_f_r", sum(losses_disc_f_wm_r), steps)
                    sw.add_scalar("training_watermark/disc_f_g", sum(losses_disc_f_wm_g), steps)
                    # Multiscale Discriminator losses
                    sw.add_scalar("training_watermark/disc_s_r", sum(losses_disc_s_wm_r), steps)
                    sw.add_scalar("training_watermark/disc_s_g", sum(losses_disc_s_wm_g), steps)

                # Validation
                if steps % a.validation_interval == 0:  # and steps != 0:
                    generator.eval()
                    mpd.eval()
                    msd.eval()
                    mpd_watermark.eval()
                    msd_watermark.eval()
                    torch.cuda.empty_cache()
                    val_err_tot = 0

                    # Validation set metrics
                    mpd_adversary_metrics = DiscriminatorMetrics()
                    msd_adversary_metrics = DiscriminatorMetrics()
                    mpd_collab_metrics = DiscriminatorMetrics()
                    msd_collab_metrics = DiscriminatorMetrics()
  
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            x, y, _, y_mel = batch
                            y_g_hat = generator(x.to(device))

                            y_mel = torch.autograd.Variable(y_mel.to(device, non_blocking=True))
                            y_g_hat_mel = mel_spectrogram(y_g_hat.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                          h.hop_size, h.win_size,
                                                          h.fmin, h.fmax_for_loss)
                            val_err_tot += F.l1_loss(y_mel, y_g_hat_mel).item()


                            # TODO: calculate discriminator EER
                            y = torch.autograd.Variable(y.to(device, non_blocking=True))
                            y = y.unsqueeze(1)

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

                            y_df_hat_r, y_df_hat_g, _, _ = mpd_watermark(y, y_g_hat)
                            mpd_collab_metrics.accumulate(
                                disc_real_outputs = y_df_hat_r,
                                disc_fake_outputs = y_df_hat_g
                            )

                            y_ds_hat_r, y_ds_hat_g, _, _ = msd_watermark(y, y_g_hat)
                            msd_collab_metrics.accumulate(
                                disc_real_outputs = y_ds_hat_r,
                                disc_fake_outputs = y_ds_hat_g
                            )

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

                        sw.add_scalar("validation/mpd_adversary_accuracy", mpd_adversary_metrics.accuracy, steps)
                        sw.add_scalar("validation/msd_adversary_accuracy", msd_adversary_metrics.accuracy, steps)
                        sw.add_scalar("validation/mpd_collab_accuracy", mpd_collab_metrics.accuracy, steps)
                        sw.add_scalar("validation/msd_collab_accuracy", msd_collab_metrics.accuracy, steps)

                        sw.add_scalar("validation/mpd_adversary_equal_error_rate", mpd_adversary_metrics.eer, steps)
                        sw.add_scalar("validation/msd_adversary_equal_error_rate", msd_adversary_metrics.eer, steps)
                        sw.add_scalar("validation/mpd_collab_equal_error_rate", mpd_collab_metrics.eer, steps)
                        sw.add_scalar("validation/msd_collab_equal_error_rate", msd_collab_metrics.eer, steps)


                    generator.train()
                    mpd.train()
                    msd.train()
                    mpd_watermark.train()
                    msd_watermark.train()

            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
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
    parser.add_argument('--config', default='')
    parser.add_argument('--training_epochs', default=3100, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int)
    parser.add_argument('--checkpoint_interval', default=5000, type=int)
    parser.add_argument('--summary_interval', default=100, type=int)
    parser.add_argument('--validation_interval', default=1000, type=int)
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--wavefile_ext', default='.wav', type=str)

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
