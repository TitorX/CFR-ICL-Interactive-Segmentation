## [CFR-ICL: Cascade-Forward Refinement with Iterative Click Loss for Interactive Image Segmentation](https://ojs.aaai.org/index.php/AAAI/article/view/28306)

<p align="center">
  <img src="./assets/img/flowchart.png" alt="drawing", width="650"/>
</p>

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cfr-icl-cascade-forward-refinement-with/interactive-segmentation-on-berkeley)](https://paperswithcode.com/sota/interactive-segmentation-on-berkeley?p=cfr-icl-cascade-forward-refinement-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cfr-icl-cascade-forward-refinement-with/interactive-segmentation-on-davis)](https://paperswithcode.com/sota/interactive-segmentation-on-davis?p=cfr-icl-cascade-forward-refinement-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cfr-icl-cascade-forward-refinement-with/interactive-segmentation-on-pascal-voc)](https://paperswithcode.com/sota/interactive-segmentation-on-pascal-voc?p=cfr-icl-cascade-forward-refinement-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cfr-icl-cascade-forward-refinement-with/interactive-segmentation-on-sbd)](https://paperswithcode.com/sota/interactive-segmentation-on-sbd?p=cfr-icl-cascade-forward-refinement-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cfr-icl-cascade-forward-refinement-with/interactive-segmentation-on-grabcut)](https://paperswithcode.com/sota/interactive-segmentation-on-grabcut?p=cfr-icl-cascade-forward-refinement-with)

## Environment
Training and evaluation environment: Python 3.9, PyTorch 1.13.1, CUDA 11.0. Run the following command to install required packages.
```
pip3 install -r requirements.txt
```

You need to configue the paths to the datasets in `config.yml` before training or testing. 

## Dataset

A script `download_datasets.sh` is prepared to download and organize required datasets.

| Dataset   |                      Description             |           Download Link              |
|-----------|----------------------------------------------|:------------------------------------:|
|MS COCO    |  118k images with 1.2M instances (train)     |  [official site][MSCOCO]             |
|LVIS v1.0  |  100k images with 1.2M instances (total)     |  [official site][LVIS]               |
|COCO+LVIS* |  99k images with 1.5M instances (train)      |  [original LVIS images][LVIS] + <br> [combined annotations][COCOLVIS_annotation] |
|SBD        |  8498 images with 20172 instances for (train)<br>2857 images with 6671 instances for (test) |[official site][SBD]|
|Grab Cut   |  50 images with one object each (test)       |  [GrabCut.zip (11 MB)][GrabCut]      |
|Berkeley   |  96 images with 100 instances (test)         |  [Berkeley.zip (7 MB)][Berkeley]     |
|DAVIS      |  345 images with one object each (test)      |  [DAVIS.zip (43 MB)][DAVIS]          |
|Pascal VOC |  1449 images with 3417 instances (test)      |  [official site][PascalVOC]          |
|COCO_MVal  |  800 images with 800 instances (test)        |  [COCO_MVal.zip (127 MB)][COCO_MVal] |

[MSCOCO]: https://cocodataset.org/#download
[LVIS]: https://www.lvisdataset.org/dataset
[SBD]: http://home.bharathh.info/pubs/codes/SBD/download.html
[GrabCut]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/GrabCut.zip
[Berkeley]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/Berkeley.zip
[DAVIS]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/DAVIS.zip
[PascalVOC]: http://host.robots.ox.ac.uk/pascal/VOC/
[COCOLVIS_annotation]: https://github.com/saic-vul/ritm_interactive_segmentation/releases/download/v1.0/cocolvis_annotation.tar.gz
[COCO_MVal]: https://github.com/saic-vul/fbrs_interactive_segmentation/releases/download/v1.0/COCO_MVal.zip

## Demo
<p align="center">
  <img src="./assets/img/demo1.gif" alt="drawing", width="500"/>
</p>

An example script to run the demo. 
```
python demo.py --checkpoint=weights/cocolvis_icl_vit_huge.pth --gpu 0
```

## Evaluation

Before evaluation, please download the datasets and models, and then configure the path in `config.yml`.

Download our model, please download below 3 zipped files and extract before use:

- [cocolvis_icl_vit_huge.pth.7z.001](https://github.com/TitorX/CFR-ICL-Interactive-Segmentation/releases/download/v1.0/cocolvis_icl_vit_huge.pth.7z.001)
- [cocolvis_icl_vit_huge.pth.7z.002](https://github.com/TitorX/CFR-ICL-Interactive-Segmentation/releases/download/v1.0/cocolvis_icl_vit_huge.pth.7z.002)
- [cocolvis_icl_vit_huge.pth.7z.003](https://github.com/TitorX/CFR-ICL-Interactive-Segmentation/releases/download/v1.0/cocolvis_icl_vit_huge.pth.7z.003)




Use the following code to evaluate the huge model.

```
python scripts/evaluate_model.py NoBRS \
    --gpu=0 \
    --checkpoint=cocolvis_icl_vit_huge.pth \
    --datasets=GrabCut,Berkeley,DAVIS,PascalVOC,SBD \\
    --cf-n=4 \\
    --acf

# cf-n: CFR steps
# acf: adaptive CFR
```

## Training

Before training, please download the [MAE](https://github.com/facebookresearch/mae) pretrained weights (click to download: [ViT-Base](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth), [ViT-Large](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large.pth), [ViT-Huge](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth)) and configure the dowloaded path in `config.yml`

Please also download the pretrained SimpleClick models from [here](https://github.com/uncbiag/SimpleClick).

Use the following code to train a huge model on C+L: 
```
python train.py models/plainvit_huge448_cocolvis.py \
    --batch-size=32 \
    --ngpus=4
```

## Citation

```
@article{sun2023cfricl,
      title={CFR-ICL: Cascade-Forward Refinement with Iterative Click Loss for Interactive Image Segmentation}, 
      author={Shoukun Sun and Min Xian and Fei Xu and Tiankai Yao and Luca Capriotti},
      year={2023},
      eprint={2303.05620},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgement
Our project is developed based on [RITM](https://github.com/SamsungLabs/ritm_interactive_segmentation) and [SimpleClick](https://github.com/uncbiag/SimpleClick)
