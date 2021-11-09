from experiments.base import Experiment
from experiments.util import ImageWriter
from pytorch_msssim import ssim as ssim_fn
from torchvision import transforms
from tqdm import tqdm
import datasets
import json
import math
import models
import numpy as np
import os
import tools
import torch

class ImageTranslation(Experiment):
    def __init__(self, config):
        print('IMAGE TRANSLATION EXPERIMENT\n')
        super(ImageTranslation, self).__init__(config)
        self.configure_dataset()
        self.configure_model()
        self.configure_protection()

    def configure_dataset(self):
        print('*** DATASET ***')
        name = self.config.dataset.name
        self.data_loader = getattr(datasets, name)(
            path=self.config.dataset.path,
            load=self.config.dataset.load,
            crop=self.config.dataset.crop,
            batch_size=self.config.hparam.bsz,
            num_workers=self.config.resource.worker,
            drop_last=False,
            test=False
        )
        print(f'Name: {name.upper()}')
        print(f'# samples: {len(self.data_loader)}\n')

        n = math.ceil(len(self.data_loader) / self.config.hparam.bsz)
        self.config.hparam.iteration *= n
        self.config.log.freq *= n

    def configure_model(self):
        model_conf = self.config.model
        model_conf.epoch = self.config.hparam.iteration // self.config.log.freq
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
                
                bbox['normalized'] = True
                bbox['input_var'] = 'real_B'
                bbox['output_var'] = 'fake_A'
                bbox['target'] = 'GB'
                self.model = models.BlackBoxWrapper(self.model, bbox)
                
                print(f'Input f(x): {bbox.fn_inp}')
                print(f'Output f(x): {bbox.fn_out}')
                print(f'lambda: {bbox["lambda"]}')
                print(f'Loss: {bbox.loss_fn}\n')
                self.bbox = True

            if wbox:
                print('*** WHITE-BOX ***')
                
                wbox['target'] = 'GB'
                self.model = models.WhiteBoxWrapper(self.model, wbox)

                print(f'Gamma0: {wbox.gamma_0}')
                print(f'Signature: {wbox.string}\n')
                self.wbox = True

    def train(self, **kwargs):
        d_iter = self.config.hparam.get('d_iter', 1)
        g_iter = self.config.hparam.get('g_iter', 1)
        
        # update lr at start of every epoch
        is_attack = not (self.config.get('attack_mode', None) is None)
        if self._step % self.config.log.freq == 1 and not is_attack:
            if self._step > 1:
                self.model.update_lr()

        for _ in range(g_iter):
            real_A, real_B = next(self.data_loader)
            data = {'real_A': real_A, 'real_B': real_B}
            self.model.update_g(data)

        for _ in range(d_iter):
            data = {
                'real_A': self.model.real_A,
                'real_B': self.model.real_B,
                'fake_A': self.model.fake_A.detach(),
                'fake_B': self.model.fake_B.detach()
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
            if not (hasattr(self, 'fixed_A') and hasattr(self, 'fixed_B')):
                real_A, real_B = next(self.data_loader)
                if self.bbox:
                    with torch.no_grad():
                        xwm = self.model.fn_inp(real_B).detach().cpu()
                        real_B = torch.cat([real_B, xwm], dim=0)

                self.fixed_A = real_A
                self.fixed_B = real_B
            
            with torch.no_grad():
                self.model.GA.eval()
                self.model.GB.eval()
                fake_B = self.model.GA(self.fixed_A)
                fake_A = self.model.GB(self.fixed_B)
                self.model.GA.train()
                self.model.GB.train()
                fake_A = torch.clamp((fake_A + 1) / 2., 0, 1).detach().cpu()
                fake_B = torch.clamp((fake_B + 1) / 2., 0, 1).detach().cpu()

            samples = torch.cat([fake_A, fake_B], dim=0)
            self.logger.save_images(samples, self._step // self.config.log.freq)

            state_dict = self.model.state_dict()
            state_dict['step'] = self._step

            ckpt_path = os.path.join(self.config.log.path, 'checkpoint.pt')
            torch.save(state_dict, ckpt_path)

    def evaluate(self, fpath):
        if self.bbox:
            fn_out_conf = self.model.fn_out.module.config
            fn_out_conf['opaque'] = True
            apply_mask = self.model.fn_out.module.__class__(
                fn_out_conf, normalized=True
            ).apply_mask

        to_pil_image = transforms.ToPILImage()

        torch.manual_seed(self.config.seed)
        print('*** EVALUATION ***')

        if self.wbox:
            bit_err_rate = self.model.loss_model.compute_ber(self.model.GB)
        else:
            bit_err_rate = float('nan')

        dirname = self.config.get('attack_mode', 'samples')
        img_dir_root = os.path.join(os.path.dirname(fpath), dirname)
        os.makedirs(img_dir_root, exist_ok=True)

        sample_dir = self.config.get('sample_dir', None)
        if sample_dir:
            image_writer = ImageWriter(sample_dir)

        metrics = {}
        self.model.GA.eval()
        self.model.GB.eval()
        for data in self.config.evaluation.data:
            loader = getattr(datasets, data['name'])(
                path=data['path'],
                load=data['load'],
                crop=data['crop'],
                batch_size=data['bsz'],
                num_workers=self.config.resource.worker,
                drop_last=False,
                test=True
            )
            
            img_dir = os.path.join(img_dir_root, data['name'])
            os.makedirs(img_dir, exist_ok=True)
            
            if self.bbox:
                stats = {'p': [], 'q': [], 'm': []}
            count = 0
            for _, real_B in tqdm(
                loader,
                desc=data['name'],
                leave=False,
                total=int(math.ceil(len(loader)/data['bsz']))
            ):
                fake_A = self.model.GB(real_B)
                fake_A = torch.clamp((fake_A + 1) / 2., 0, 1).detach().cpu()

                if sample_dir:
                    for i in range(fake_A.size(0)):
                        image_writer(fake_A[i], suffix='gen')

                if self.bbox:
                    zwm = self.model.fn_inp(real_B)
                    xwm = self.model.GB(zwm)
                    zwm = torch.clamp((zwm + 1) / 2., 0, 1).detach()
                    xwm = torch.clamp((xwm + 1) / 2., 0, 1).detach()
                    ywm = self.model.fn_out(fake_A)
                    ywm = torch.clamp((ywm + 1) / 2., 0, 1).detach()
                    wm_x = apply_mask(xwm.cpu())
                    wm_y = apply_mask(ywm.cpu())

                    if sample_dir:
                        for i in range(xwm.size(0)):
                            image_writer(zwm[i], suffix='z')
                            image_writer(xwm[i], suffix='wm')

                    ssim = ssim_fn(wm_x, wm_y, data_range=1, size_average=False)
                    p_value = tools.compute_matching_prob(wm_x, wm_y)
                    match = p_value < self.config.evaluation.p_thres

                    stats['q'].append(ssim.detach().cpu())
                    stats['p'].append(p_value)
                    stats['m'].append(match)
                to_pil_image(fake_A[0]).save(os.path.join(img_dir, f'{count}.png'))
                count += 1
            
            metrics[data['name']] = {}

            if self.bbox:
                for k in stats:
                    stats[k] = torch.cat(stats[k], dim=0).numpy()

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
                f'\n\tWBOX: {bit_err_rate:.4f}'
                f'\n\tBBOX:'
                f'\n\t\tQ_WM: {ssim_wm:.4f}'
                f'\n\t\tP: {p_value:.3e}'
                f'\n\t\tMATCH: {match/sample_size:.4f}'
            )

        json.dump(metrics, open(fpath, 'w'), indent=2, sort_keys=True)