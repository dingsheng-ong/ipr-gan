from experiments.base import Experiment
from experiments.util import ImageWriter
from pytorch_msssim import ssim as ssim_fn
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from tqdm import tqdm
import datasets
import json
import math
import models
import numpy as np
import os
import tools
import torch

class ImageSuperResolution(Experiment):
    def __init__(self, config):
        print('IMAGE SUPER-RESOLUTION EXPERIMENT\n')
        super(ImageSuperResolution, self).__init__(config)
        self.configure_dataset()
        self.configure_model()
        self.configure_protection()

    def configure_dataset(self):
        print('*** DATASET ***')
        name = self.config.dataset.name
        self.data_loader = getattr(datasets, name)(
            path=self.config.dataset.path,
            size=self.config.dataset.size,
            batch_size=self.config.hparam.bsz,
            num_workers=self.config.resource.worker,
            drop_last=True,
            test=False
        )
        print(f'Name: {name.upper()}')
        print(f'# samples: {len(self.data_loader)}\n')

    def configure_model(self):
        model_conf = self.config.model
        model_fn = getattr(models, model_conf.type)
        self.model = model_fn(model_conf, device=self.device)

        params_g = self.model.optG.param_groups[0]['params']
        params_d = self.model.optD.param_groups[0]['params']

        print('*** MODEL ***')
        print(f'G: {model_conf.G}')
        print(f'# params: {sum(map(lambda p: p.numel(), params_g))}')
        print(f'D: {model_conf.D}')
        print(f'# params: {sum(map(lambda p: p.numel(), params_d))}\n')

    def configure_protection(self):
        self.bbox = False
        self.wbox = False

        wm_conf = self.config.get('protection', None)
        if wm_conf:
            bbox = wm_conf.get('bbox', None)
            wbox = wm_conf.get('wbox', None)
            if bbox:
                print('*** BLACK-BOX ***')
                
                bbox['normalized'] = False
                bbox['input_var'] = 'low_res'
                bbox['output_var'] = 'super_res'
                bbox['target'] = 'G'
                self.model = models.BlackBoxWrapper(self.model, bbox)
                
                print(f'Input f(x): {bbox.fn_inp}')
                print(f'Output f(x): {bbox.fn_out}')
                print(f'lambda: {bbox["lambda"]}')
                print(f'Loss: {bbox.loss_fn}\n')
                self.bbox = True

            if wbox:
                print('*** WHITE-BOX ***')
                
                wbox['target'] = 'G'
                self.model = models.WhiteBoxWrapper(self.model, wbox)

                print(f'Gamma0: {wbox.gamma_0}')
                print(f'Signature: {wbox.string}\n')
                self.wbox = True

    def train(self):

        pretrain_iter = self.config.hparam.pretrain_iter
        halfway = pretrain_iter + (self.config.hparam.iteration // 2)
        if self._step == halfway and pretrain_iter > 0:
            self.model.optG.param_groups[0]['lr'] *= 0.1
            self.model.optD.param_groups[0]['lr'] *= 0.1

        if self._step <= pretrain_iter:
            lr, hr = next(self.data_loader)

            data = {'low_res': lr, 'high_res': hr, 'pretrain': True}
            data['inhibit_bbox'] = True
            self.model.update_g(data)
        else:
            d_iter = self.config.hparam.get('d_iter', 1)
            g_iter = self.config.hparam.get('g_iter', 1)
            
            for _ in range(g_iter):
                lr, hr = next(self.data_loader)
                data = {'low_res': lr, 'high_res': hr, 'pretrain': False}
                # data['inhibit_bbox'] = self._step < halfway
                self.model.update_g(data)

            for _ in range(d_iter):
                data = {
                    'high_res': self.model.high_res,
                    'super_res': self.model.super_res
                }
                self.model.update_d(data)

    def checkpoint(self):
        if self._step == 'end':
            state_dict = self.model.state_dict()
            state_dict['step'] = 'END'

            ckpt_path = os.path.join(self.config.log.path, 'checkpoint.pt')
            torch.save(state_dict, ckpt_path)
            return

        metrics = self.model.get_metrics()
        self.logger.write_scalar(metrics, self._step)

        if self._step % self.config.log.freq == 0:
            if not hasattr(self, 'fixed_lr'):
                lr, _ = next(self.data_loader)
                self.fixed_lr = lr
                if self.bbox:
                    with torch.no_grad():
                        bsz = self.config.hparam.bsz
                        lr = lr[:(bsz // 2), ...]
                        xwm = self.model.fn_inp(lr).detach().cpu()
                        lr = torch.cat([lr, xwm], dim=0)
                    self.fixed_lr = lr
            
            with torch.no_grad():
                self.model.G.eval()
                sr = self.model.G(self.fixed_lr)
                self.model.G.train()
                sr = torch.clamp(sr, 0, 1).detach().cpu()

            self.logger.save_images(sr, self._step)

            state_dict = self.model.state_dict()
            state_dict['step'] = self._step

            ckpt_path = os.path.join(self.config.log.path, 'checkpoint.pt')
            torch.save(state_dict, ckpt_path)

            if self._step == self.config.hparam.pretrain_iter:
                ckpt_path = os.path.join(self.config.log.path, 'pretrain.pt')
                torch.save(state_dict, ckpt_path)

    def evaluate(self, fpath):
        rgb2luma = lambda arg: np.uint8(
            (((np.float64(arg) @ [65.481, 128.553, 24.966]) / 255.) + 16.).round()
        )
        tensor2numpy = lambda arg: np.uint8((
            arg.clamp_(0, 1).detach().cpu().numpy().transpose(1, 2, 0) * 255
        ).round())

        if self.bbox:
            fn_out_conf = self.model.fn_out.module.config
            fn_out_conf['opaque'] = True
            apply_mask = self.model.fn_out.module.__class__(
                fn_out_conf, normalized=True
            ).apply_mask

        torch.manual_seed(self.config.seed)

        print('*** EVALUATION ***')

        self.model.G.eval()

        if self.wbox:
            bit_err_rate = self.model.loss_model.compute_ber(self.model.G)
        else:
            bit_err_rate = float('nan')

        sample_dir = self.config.get('sample_dir', None)
        if sample_dir:
            image_writer = ImageWriter(sample_dir)

        metrics = {}
        for data in self.config.evaluation.data:
            loader = getattr(datasets, data['name'])(
                path=data['path'],
                size=data['size'],
                batch_size=data['bsz'],
                num_workers=1,
                test=True,
                shuffle=False,
                drop_last=False
            )
            stats = {'psnr': [], 'ssim': []}
            if self.bbox:
                stats['q'] = []
                stats['p'] = []
                stats['m'] = []
            for lr, hr in tqdm(
                loader,
                desc=data['name'],
                leave=False,
                total=int(math.ceil(len(loader)/data['bsz']))
            ):
                with torch.no_grad():
                    sr = self.model.G(lr).clamp_(0, 1)

                    if sample_dir:
                        for i in range(sr.size(0)):
                            image_writer(sr[i].cpu(), suffix='gen')

                    if self.bbox:
                        zwm = self.model.fn_inp(lr)
                        xwm = self.model.G(zwm).clamp_(0, 1)
                        zwm = zwm.clamp_(0, 1)
                        ywm = self.model.fn_out(sr)
                        if sample_dir:
                            for i in range(xwm.size(0)):
                                image_writer(zwm[i].cpu(), suffix='z')
                                image_writer(xwm[i].cpu(), suffix='wm')

                        wm_x = apply_mask(xwm.cpu())
                        wm_y = apply_mask(ywm.cpu())

                        ssim = ssim_fn(wm_x, wm_y, data_range=1, size_average=False)
                        p_value = tools.compute_matching_prob(wm_x, wm_y)
                        match = p_value < self.config.evaluation.p_thres

                        stats['q'].append(ssim.detach().cpu())
                        stats['p'].append(p_value)
                        stats['m'].append(match)

                    sr = rgb2luma(tensor2numpy(sr[0]))[4:-4, 4:-4]
                    hr = rgb2luma(tensor2numpy(hr[0]))[4:-4, 4:-4]

                    psnr = peak_signal_noise_ratio(hr, sr)
                    ssim = structural_similarity(hr, sr)
                    
                    stats['psnr'].append(torch.FloatTensor([psnr]))
                    stats['ssim'].append(torch.FloatTensor([ssim]))
            for k in stats: stats[k] = torch.cat(stats[k], dim=0).numpy()

            psnr = np.mean(stats['psnr'])
            ssim = np.mean(stats['ssim'])

            metrics[data['name']] = {
                'PSNR': f'{psnr:.2f}',
                'SSIM': f'{ssim:.4f}',
            }

            ssim_wm = np.mean(stats['q']) if self.bbox else float('nan')
            p_value = np.mean(stats['p']) if self.bbox else float('nan')
            match   = np.sum(stats['m']) if self.bbox else float('nan')
            sample_size = len(loader)

            if self.wbox:
                metrics[data['name']]['WBOX'] = f'{bit_err_rate:.4f}'

            if self.bbox:
                metrics[data['name']]['BBOX'] = {
                    'Q_WM': f'{ssim_wm:.4f}',
                    'P': f'{p_value:.3e}',
                    'MATCH': f'{match:d}/{sample_size:d}'
                }

            print(
                f'Dataset: {data["name"]}'
                f'\n\tPSNR: {psnr:.2f}'
                f'\n\tSSIM: {ssim:.4f}'
                f'\n\tWBOX: {bit_err_rate:.4f}'
                f'\n\tBBOX:'
                f'\n\t\tQ_WM: {ssim_wm:.4f}'
                f'\n\t\tP: {p_value:.3e}'
                f'\n\t\tMATCH: {match/sample_size:.4f}'
            )

        json.dump(metrics, open(fpath, 'w'), indent=2, sort_keys=True)