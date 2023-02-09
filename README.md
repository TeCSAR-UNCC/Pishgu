# Pishgu: : Universal Path Prediction Network Architecture for Real-time Cyber-physical Edge Systems

This repo contains the official Pytorch implementation of paper "Pishgu: : Universal Path Prediction Network Architecture for Real-time Cyber-physical Edge Systems" accepted to be published and presented at [IEEE/ACM ICCPS 2023](https://iccps.acm.org/2023/). We present a universal architecture for trajectory prediction for pedestrians and vehicles from various points of view specifically designed for CPS applications. Pishgu is lightweight and captures the interdependencies between subjects using Graph Isomorphism Network (GIN). On top of that, by utilizing attentive convolutional layers, Pishgu is able to focus on more informative features and define a novel approach for trajectory prediction. 

## Domains and Datasets
- Vehicle Bird's-eye View: NGSIM Dataset
- Pedestrian Bird's-eye View: UCY and ETH Datasets
- Pedestrian High-angle View: VIRAT/ActEV Dataset

You can download the preprocessed data from this [link](https://drive.google.com/file/d/1BnhGtGgiafV6LP9rnIJA6b5Lp4GxmPeB/view?usp=share_link). 

## Installation 
```
git clone https://github.com/TeCSAR-UNCC/Pishgu.git
cd Pishgu
pip install -r requirments.txt
```

## Training and Testing
Each domain has a corresponding Confif file in configs folder. For training and saving the model in the Training section just set the "save_model" and "train" fields to True and use the following command:
```
python3 main.py --config {path_to_the_confif_file}
```

For testing, just give the path to desired model in the config file and set "save_model" and "train" fields to False and use the same command:
```
python3 main.py --config {path_to_the_confif_file}
```

We also provide the trained weights in all domains in the the "model" folder. 

## Citation
If you found Pishgu helpful and used it in your research, please use the folllowing BibTeX entry.
```
@article{noghre2022pishgu,
  title={Pishgu: Universal Path Prediction Architecture through Graph Isomorphism and Attentive Convolution},
  author={Noghre, Ghazal Alinezhad and Katariya, Vinit and Pazho, Armin Danesh and Neff, Christopher and Tabkhi, Hamed},
  journal={arXiv preprint arXiv:2210.08057},
  year={2022}
}
```
## Refrences

This repo is based on these awesome works:
- [CARPe_Postum] (https://github.com/TeCSAR-UNCC/CARPe_Posterum)
- [Social GAN] (https://github.com/agrimgupta92/sgan)

