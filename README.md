# GAN IPR Protection

### Official pytorch implementation of the paper: "Protecting Intellectual Property of Generative Adversarial Networks from Ambiguity Attack"

#### CVPR 2021 | [PDF](https://openaccess.thecvf.com/content/CVPR2021/html/Ong_Protecting_Intellectual_Property_of_Generative_Adversarial_Networks_From_Ambiguity_Attacks_CVPR_2021_paper.html)

(Released on September 2021)

Updated on September 26, 2022

## Updates
1. Fix bugs
2. Our framework on DNN IP protection is accepted in TPAMI 2022, see [here](https://github.com/kamwoh/DeepIPR).
3. Our framework on RNN IP protection is accepted in AACL IJCNLP 2022, see [here](https://github.com/zhiqin1998/RecurrentIPR).
4. Our framework on Multi-modal IP protection is accepted in Pattern Recognition 2022, see [here](https://github.com/jianhanlim/ipr-imagecaptioning).

## Description
<p align="justify">
  Ever since Machine Learning as a Service (MLaaS) emerges as a viable business that utilizes deep learning models to generate lucrative revenue, Intellectual Property Right (IPR) has become a major concern because these deep learning models can easily be replicated, shared, and re-distributed by any unauthorized third parties. To the best of our knowledge, one of the prominent deep learning models - Generative Adversarial Networks (GANs) which has been widely used to create photorealistic image are totally unprotected despite the existence of pioneering IPR protection methodology for Convolutional Neural Networks (CNNs). This paper therefore presents a complete protection framework in both black-box and white-box settings to enforce IPR protection on GANs. Empirically, we show that the proposed method does not compromise the original GANs performance (i.e. image generation, image super-resolution, style transfer), and at the same time, it is able to withstand both removal and ambiguity attacks against embedded watermarks.
</p>
<p align="center"><img src="overview.png" width="50%"></p>
<p align="center">Figure 1: Overview of our proposed GANs protection framework in black-box setting.</p>

## How to run

The code is tested on `Python 3.8.8` and `PyTorch 1.8.0`.

### Dependencies
```bash
$ pip install -r requirements.txt
```

### Train
Modify the config files in `configs/`, then run the following command:
```bash
$ python train.py -c configs/<path-to-yaml-file>
```

### Tensorboard
```bash
$ tensorboard --logdir log/
```

### Evaluate
```bash
$ python eval.py -l log/<directory> -s sample/
```
To evaluate CycleGAN on CityScapes, you can use the code in `scripts/`. You need to rename the trained log directory `log/CycleGAN-XXXX-CITY-X` to `scripts/log` and run `run.py`. Next, you need to download the `fcn-8s-cityscapes.caffemodel` and put it in `scripts/caffemodel` directory (Please see the original [CycleGAN repo](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)). Finally, you should put the cityscapes ground-truth images in `scripts/gt/`. The file structure should be as shown below:
```
scripts/
├─ caffemodel/
│  ├─ deploy.prototxt
│  └─ fcn-8s-cityscapes.caffemodel
├─ gt/
│  ├─ 1.png
│  ├─ ...
│  └─ 499.png
├─ log/
│  ├─ samples
│  │  ├─ 1.png
│  │  ├─ ...
│  │  └─ 499.png
│  ├─ metrics.json
│  └─ ...
├─ Dockerfile
├─ labels.py
└─ run.py
```
Then, run `python run.py` to start the evaluation.

### Attack
#### Fine-tuning and Overwriting
```bash
$ python attack.py -l log/<directory> -m <finetune/overwrite> -w <path-to-new-watermark> -d <to-load-discriminator>
```
#### Pruning
```bash
$ python prune.py -l log/<directory> -s sample/
``` 
#### Ambiguity Attack
```bash
$ python sign_flip.py -l log/<directory> -s sample/
```

## Citation
If you find this work useful for your research, please cite
```bibtex
@inproceedings{GanIPR,
  title={Protecting Intellectual Property of Generative Adversarial Networks from Ambiguity Attack},
  author={Ong, Ding Sheng and Chan, Chee Seng and Ng, Kam Woh and Fan, Lixin and Yang, Qiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2021},
}
```

## Feedback
Suggestions and opinions on this work (both positive and negative) are greatly welcomed. Please contact the authors by sending an email to
`sheng970303@gmail.com` or `cs.chan@um.edu.my`.

## License and Copyright
The project is open source under BSD-3 license (see the ``` LICENSE ``` file).

&#169;2021 Universiti Malaya and WeBank.
