from experiments.base import Experiment
from experiments.util import ImageWriter
from experiments.util import calculate_frechet_distance
from experiments.util import calculate_inception_score
from networks import InceptionActivations
from pytorch_msssim import ssim as ssim_fn
from tqdm import tqdm
import datasets
import json
import math
import models
import numpy as np
import os
import tools
import torch

class ImageGeneration(Experiment):
    def __init__(self, config):
        print('IMAGE GENERATION EXPERIMENT\n')
        super(ImageGeneration, self).__init__(config)
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
            drop_last=True
        )
        print(f'Name: {name.upper()}')
        print(f'# samples: {len(self.data_loader)}\n')

    def configure_model(self):
        model_conf = self.config.model
        model_fn = getattr(models, model_conf.type)
        self.model = model_fn(model_conf, device=self.device)

        params_g = sum(map(lambda p: p.numel(), self.model.G.parameters()))
        params_d = sum(map(lambda p: p.numel(), self.model.D.parameters()))

        print('*** MODEL ***')
        print(f'G: {model_conf.G}')
        print(f'# params: {params_g}')
        print(f'D: {model_conf.D}')
        print(f'# params: {params_d}\n')

    def configure_protection(self):
        self.bbox = False
        self.wbox = False

        wm_conf = self.config.get('protection', None)
        if wm_conf:
            bbox = wm_conf.get('bbox', None)
            wbox = wm_conf.get('wbox', None)
            if bbox:
                print('*** BLACK-BOX ***')
                
                bbox['normalized'] = True
                # G(latent) -> fake_sample
                bbox['input_var'] = 'latent'
                bbox['output_var'] = 'generated'
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

        d_iter = self.config.hparam.get('d_iter', 1)
        g_iter = self.config.hparam.get('g_iter', 1)
        
        # fetch data
        for _ in range(d_iter):
            x, _ = next(self.data_loader)
            z = torch.randn(x.size(0), 128)

            data = {'real_sample': x, 'latent': z}
            self.model.update_d(data)

        for _ in range(g_iter):
            data = { 'fake_sample': self.model.fake_sample }
            self.model.update_g(data)

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
            postproc = lambda x: (x.clamp_(-1, 1) + 1.) / 2.
            if not hasattr(self, 'fixed_z'):
                bsz = self.config.hparam.bsz
                self.fixed_z = torch.randn(bsz, 128)
                if self.bbox:
                    with torch.no_grad():
                        z = torch.randn(bsz // 2, 128)
                        zwm = self.model.fn_inp(z).detach().cpu()
                        z = torch.cat([z, zwm], dim=0)
                    self.fixed_z = z
            
            with torch.no_grad():
                self.model.G.eval()
                fake_sample = self.model.G(self.fixed_z)
                self.model.G.train()
                img = postproc(fake_sample).detach().cpu()

            self.logger.save_images(img, self._step)

            state_dict = self.model.state_dict()
            state_dict['step'] = self._step

            ckpt_path = os.path.join(self.config.log.path, 'checkpoint.pt')
            torch.save(state_dict, ckpt_path)

    def evaluate(self, fpath):
        postproc = lambda x: (x.clamp_(-1, 1) + 1.) / 2.

        if self.bbox:
            fn_out_conf = self.model.fn_out.module.config
            fn_out_conf['opaque'] = True
            apply_mask = self.model.fn_out.module.__class__(
                fn_out_conf, normalized=True
            ).apply_mask

        torch.manual_seed(self.config.seed)

        print('*** EVALUATION ***')

        inception = torch.nn.DataParallel(
            InceptionActivations().to(self.device[0]),
            device_ids=[k.index for k in self.device]
        )
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
                num_workers=self.config.resource.worker,
                shuffle=False,
                drop_last=False
            )
            stats = {'fx': [], 'fy': [], 'prob': []}
            if self.bbox:
                stats['q'] = []
                stats['p'] = []
                stats['m'] = []
            for y, _ in tqdm(
                loader,
                desc=data['name'],
                leave=False,
                total=int(math.ceil(len(loader)/data['bsz']))
            ):
                with torch.no_grad():
                    z = torch.randn(y.size(0), 128)
                    x = self.model.G(z)

                    if sample_dir:
                        for i in range(x.size(0)):
                            image_writer(postproc(x[i]).cpu(), suffix='gen')

                    if self.bbox:
                        zwm = self.model.fn_inp(z)
                        xwm = self.model.G(zwm)
                        ywm = self.model.fn_out(x)

                        if sample_dir:
                            for i in range(xwm.size(0)):
                                image_writer(postproc(xwm[i]).cpu(), suffix='wm')

                        wm_x = postproc(apply_mask(xwm.cpu()))
                        wm_y = postproc(apply_mask(ywm.cpu()))

                        ssim = ssim_fn(wm_x, wm_y, data_range=1, size_average=False)
                        p_value = tools.compute_matching_prob(wm_x, wm_y)
                        match = p_value < self.config.evaluation.p_thres

                        stats['q'].append(ssim.detach().cpu())
                        stats['p'].append(p_value)
                        stats['m'].append(match)

                    fx, prob  = inception(x.detach())
                    fy, _ = inception(y)
                    stats['fx'].append(fx.detach().cpu())
                    stats['fy'].append(fy.detach().cpu())
                    stats['prob'].append(prob.detach().cpu())

            for k in stats: stats[k] = torch.cat(stats[k], dim=0).numpy()
            
            mu1 = np.mean(stats['fx'], axis=0)
            mu2 = np.mean(stats['fy'], axis=0)
            sig1 = np.cov(stats['fx'], rowvar=False)
            sig2 = np.cov(stats['fy'], rowvar=False)
            
            fid = calculate_frechet_distance(mu1, sig1, mu2, sig2)
            is_mean, is_std = calculate_inception_score(stats['prob'])
            ssim_wm = np.mean(stats['q']) if self.bbox else float('nan')
            p_value = np.mean(stats['p']) if self.bbox else float('nan')
            match   = np.sum(stats['m']) if self.bbox else float('nan')
            sample_size = len(loader)

            metrics[data['name']] = {
                'FID': f'{fid:.4f}',
                'IS_MEAN': f'{is_mean:.4f}',
                'IS_STD': f'{is_std:.4f}'
            }

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
                f'\n\tFID: {fid:.2f}'
                f'\n\tIS: {is_mean:.4f} +/- {is_std:.4f}'
                f'\n\tWBOX: {bit_err_rate:.4f}'
                f'\n\tBBOX:'
                f'\n\t\tQ_WM: {ssim_wm:.4f}'
                f'\n\t\tP: {p_value:.3e}'
                f'\n\t\tMATCH: {match/sample_size:.4f}'
            )

        json.dump(metrics, open(fpath, 'w'), indent=2, sort_keys=True)