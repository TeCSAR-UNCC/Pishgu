# Some of the functions in this file based on the following git repository: https://github.com/agrimgupta92/sgan
# These include relative_to_abs, l2_loss, displacement_error, and final_displacement_error

# It is for operations from the data presented in thier paper Social-GAN https://arxiv.org/abs/1803.10892

# The paper is cited as follows:
#@inproceedings{gupta2018social,
#  title={Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks},
#  author={Gupta, Agrim and Johnson, Justin and Fei-Fei, Li and Savarese, Silvio and Alahi, Alexandre},
#  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
#  number={CONF},
#  year={2018}
#}

#util.py
import torch
import os
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch as tgb

def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    displacement = torch.cumsum(rel_traj, dim=1)
    start_pos = torch.unsqueeze(start_pos, dim=1)
    abs_traj = displacement + start_pos
    return abs_traj



def l2_loss(pred_traj, pred_traj_gt, loss_mask, random=0, mode='average'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    loss = (loss_mask.unsqueeze(dim=2) *
            (pred_traj_gt.squeeze(dim=1) - pred_traj)**2)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data)
    elif mode == 'raw':
        return loss.sum(dim=2).sum(dim=1)

def rmse(pred_traj, pred_traj_gt, consider_ped=None, mode='raw'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = (pred_traj_gt.squeeze(dim=1) - pred_traj)**2
    if consider_ped is not None:
        loss = (torch.sqrt(loss.sum(dim=2)).sum(dim=0) * consider_ped)/loss.shape[0]
    else:
        loss = (torch.sqrt(loss.sum(dim=2)).sum(dim=0))/loss.shape[0]
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

def horiz_eval(loss_total, n_horiz):
    loss_total = loss_total.cpu().numpy()
    avg_res = np.zeros(n_horiz)
    n_all = loss_total.shape[0]
    n_frames = n_all//n_horiz
    for i in range(n_horiz):
        if i == 0:
            st_id = 0
        else:
            st_id = n_frames*i

        if i == n_horiz-1:
            en_id = n_all-1
        else:
            en_id = n_frames*i + n_frames - 1

        avg_res[i] = np.mean(loss_total[st_id:en_id+1])

    return avg_res


def displacement_error(pred_traj, pred_traj_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = (pred_traj_gt.squeeze(dim=1) - pred_traj)**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'raw':
        return loss

def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - pred_pos
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=1)) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)

def getGraphDataList(obs_traj, obs_traj_rel, seq_start_end):
    data_list = []
    for (start, end) in seq_start_end:
        x1=obs_traj[start:end,:,:,:].reshape(end-start, int(obs_traj.shape[2]*obs_traj.shape[3]))
        x2=obs_traj_rel[start:end,:,:,:].reshape(end-start, int(obs_traj_rel.shape[2]*obs_traj_rel.shape[3]))
        x = torch.cat((x1,x2),dim=1)

        NUM_NODES = x.shape[0]
        num_edges = int((NUM_NODES*(NUM_NODES-1)))

        edges_per_node = int(num_edges/NUM_NODES)
        edge_list1 = []
        edge_list2 = []
        nodeList = [node for node in range(0,NUM_NODES)]
        for n in range(0,NUM_NODES):
            for e in range(0,edges_per_node):
                edge_list1.append(n)
            for k in nodeList:
                if(k != n):
                    edge_list2.append(k)

        edge_index = torch.tensor([edge_list1,
                                edge_list2], dtype=torch.long)
    
        data = Data(x=x, edge_index=edge_index, num_nodes=NUM_NODES)
        data_list.append(data)
    return data_list

def train(model, train_loader, optimizer, device, obs_step):
    losses = []
    model.train()
    for batch in train_loader:
        batch = [tensor.to(device) for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
            non_linear_ped, loss_mask, seq_start_end, _ ) = batch
        optimizer.zero_grad()
        
        data_list = getGraphDataList(obs_traj,obs_traj_rel, seq_start_end)
        graph_batch = tgb.from_data_list(data_list)

        pred_traj = model(obs_traj_rel, graph_batch.x.to(device), graph_batch.edge_index.to(device))
        pred_traj = pred_traj.reshape(pred_traj.shape[0],train_loader.dataset.pred_len,2)

        pred_traj_real = relative_to_abs(pred_traj, obs_traj[:,:,-1,:].squeeze(1))

        loss_mask = loss_mask[:, obs_step:]
        loss = l2_loss(pred_traj_real, pred_traj_gt, loss_mask, mode='average')
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    return losses

def test(model, test_loader, device):
    total_traj = 0
    ade_batches, fde_batches = [], []
    model.eval()
    for batch in test_loader:
        batch = [tensor.to(device) for tensor in batch]
        (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
            non_linear_ped, loss_mask, seq_start_end, _) = batch
        total_traj += pred_traj_gt.size(0)

        data_list = getGraphDataList(obs_traj,obs_traj_rel, seq_start_end)
        graph_batch = tgb.from_data_list(data_list)       
        
        pred_traj = model(obs_traj_rel, graph_batch.x.to(device), graph_batch.edge_index.to(device))
        pred_traj = pred_traj.reshape(pred_traj.shape[0],test_loader.dataset.pred_len,2).detach()

        pred_traj_real = relative_to_abs(pred_traj, obs_traj[:,:,-1,:].squeeze(1))

        ade_batches.append(torch.sum(displacement_error(pred_traj_real, pred_traj_gt, mode='raw')).detach().item())
        fde_batches.append(torch.sum(final_displacement_error(pred_traj_real[:,-1,:], pred_traj_gt[:,:,-1,:].squeeze(1), mode='raw')).detach().item())
    ade = sum(ade_batches) / (total_traj * test_loader.dataset.pred_len)
    fde = sum(fde_batches) / (total_traj)
    return ade, fde
