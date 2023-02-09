# This file is based on the following git repository: https://github.com/agrimgupta92/sgan

# It loads the dataset presented in thier paper Social-GAN https://arxiv.org/abs/1803.10892

# The paper is cited as follows:
#@inproceedings{gupta2018social,
#  title={Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks},
#  author={Gupta, Agrim and Johnson, Justin and Fei-Fei, Li and Savarese, Silvio and Alahi, Alexandre},
#  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
#  number={CONF},
#  year={2018}
#}

from torch.utils.data import DataLoader

from utils.trajectories import TrajectoryDataset, seq_collate


def data_loader(path, batch_size=64, obs_len=8, pred_len=12, delim='space', debug=False):
    dset = TrajectoryDataset(
        path,
        obs_len,
        pred_len,
        skip=1,
        delim=delim)
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=seq_collate)
    return dset, loader
