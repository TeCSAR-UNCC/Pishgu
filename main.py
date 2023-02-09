import torch
import time
import os
import argparse
from statistics import mean
from thop import profile
from torch_geometric.data.batch import Batch as tgb
import utils.loader as dl
import utils.network as net
import utils.util as ut
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Training and validation parameters.')
    parser.add_argument('--config', help='config file path')
    args = parser.parse_args()
    return args

args = parse_args()
with open(args.config, "r") as file:
    config = yaml.safe_load(file)

output_size = config['input_data']['points_per_position']*config['input_data']['prtediction_step']
num_features = config['input_data']['points_per_position']*config['input_data']['observed_steps']

# make sure to change name of the datasets models and this line
if config['input_data']['dataset'] == ["VIRAT_ActEV"]:
    delim = 'space'
else:
    delim = 'tab'

if config['training']['train']:
    saveFolder = config['training']['save_folder'] 
    lr = config['training']['learning_rate']
    for test_file in config['input_data']['dataset']:
        print("Test file: " + test_file)
        if config['training']['save_model']:
            if not os.path.exists("models/"+str(saveFolder)+"/"):
                    os.makedirs("models/"+str(saveFolder)+"/")

        device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        model = net.NetGINConv(num_features, output_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config['training']['weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['training']['milestones'], gamma=0.1)

        data_dir = 'datasets/'+test_file+'/'
        _, train_loader = dl.data_loader(data_dir+config['input_data']['train_folder'], 
                                        batch_size=config['training']['batch_size'],
                                        obs_len=config['input_data']['observed_steps'],
                                        pred_len=config['input_data']['prtediction_step'],
                                        delim=delim)

        _, val_loader = dl.data_loader(data_dir+config['input_data']['val_folder'], 
                                        batch_size=config['training']['batch_size'],
                                        obs_len=config['input_data']['observed_steps'],
                                        pred_len=config['input_data']['prtediction_step'],
                                        delim=delim)

        
        best_info = [1000.0, 1000.0, 0]
        for epoch in range(0, config['training']['epoches']):
            losses = ut.train(model, train_loader, optimizer, device, obs_step=config['input_data']['observed_steps'])
            if(epoch%config['training']['validation_interval']==0):
                ade, fde = ut.test(model, val_loader, device)
                print("ADE: " + str(ade) + "  FDE: " + str(fde) + "   Epoch: " + str(epoch))
                if ade < best_info[0]:
                    best_info[0] = ade
                    best_info[1] = fde
                    best_info[2] = epoch
                    if (config['training']['save_model']):
                        model_path = "models/"+str(saveFolder)+"/"+ test_file+".pt"
                        torch.save(model.state_dict(), model_path)
        print(test_file + "   Best ADE: " + str(best_info[0]) + "   FDE: " + str(best_info[1]) + "   Epoch: " + str(best_info[2]) + "   lr: " + str(lr) + "\n\n")

else:
    all_ops = []
    times = []
    res = []
    for test_file in config['input_data']["dataset"]:
        ls = []
        total_traj = 0
        device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        model = net.NetGINConv(num_features, output_size).to(device)
        model_folder = 'models/TRAINED/'
        model.load_state_dict(torch.load(os.path.join(config['training']['model_dir'], test_file)+".pt", map_location='cpu'))
        data_dir = 'datasets/'+test_file+'/'
        if config['input_data']['test_folder'] != 'None':
            _, test_loader = dl.data_loader(data_dir+config['input_data']['test_folder'], 
                                            batch_size=1,
                                            obs_len=config['input_data']['observed_steps'],
                                            pred_len=config['input_data']['prtediction_step'],
                                            delim=delim)
        else:
            _, test_loader = dl.data_loader(data_dir+config['input_data']['val_folder'], 
                                            batch_size=1,
                                            obs_len=config['input_data']['observed_steps'],
                                            pred_len=config['input_data']['prtediction_step'],
                                            delim=delim)

        ade_batches, fde_batches = [], []
        if  config['input_data']['subject'] == 'vehicle':
            rmse_batch = torch.full((25,1), 0.0)
            rmse_batch = rmse_batch.squeeze(dim=1)
            rmse_batch = rmse_batch.to(device)
            number_of_traj = 0

        model.eval()
        for batch in test_loader:
            batch = [tensor.to(device) for tensor in batch]
            (obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel,
                non_linear_ped, loss_mask, seq_start_end, frame_id) = batch
            total_traj += pred_traj_gt.size(0)

            data_list = ut.getGraphDataList(obs_traj,obs_traj_rel, seq_start_end)
            graph_batch = tgb.from_data_list(data_list)

            start = time.time()
            pred_traj = model(obs_traj_rel, graph_batch.x.to(device), graph_batch.edge_index.to(device))
            end = time.time()
            times.append(end-start)

            pred_traj = pred_traj.reshape(pred_traj.shape[0], config['input_data']['prtediction_step'],config['input_data']['points_per_position']).detach()

            pred_traj_real = ut.relative_to_abs(pred_traj, obs_traj[:,:,-1,:].squeeze(1))

            ade_batches.append(torch.sum(ut.displacement_error(pred_traj_real, pred_traj_gt, mode='raw')).detach().item())
            fde_batches.append(torch.sum(ut.final_displacement_error(pred_traj_real[:,-1,:], pred_traj_gt[:,:,-1,:].squeeze(1), mode='raw')).detach().item())
       
            if  config['input_data']['subject'] == 'vehicle':
                rmse_batch += (ut.rmse(pred_traj_real, pred_traj_gt, mode='raw')) #.detach().item()
                number_of_traj+=1

            # ops, params = profile(model, inputs=(obs_traj_rel, graph_batch.x.to(device), graph_batch.edge_index.to(device)))
            # all_ops.append(ops)

        ade = sum(ade_batches) / (total_traj * config['input_data']['prtediction_step'])
        fde = sum(fde_batches) / (total_traj)

        if  config['input_data']['subject'] == 'vehicle':
            rmse = rmse_batch/number_of_traj

        print(test_file + "  ADE: " + str(ade) + "  FDE: " + str(fde))
        if  config['input_data']['subject'] == 'vehicle':
            print("RMSE: 1s,2s,3s,4s,5s")
            pred_fde_horiz = ut.horiz_eval(rmse, 5)
            print(pred_fde_horiz)
        ls.append(test_file)
        ls.append(ade)
        ls.append(fde)
        res.append(ls)
    avg_fde = 0
    avg_ade = 0
    for data_st in res:
        avg_ade = avg_ade + data_st[1]
        avg_fde = avg_fde + data_st[2]
    avg_ade = avg_ade/len(res)
    avg_fde = avg_fde/len(res)

    print("Average Execution Time: " + str(mean(times)) + " sec")
    # print("Params: " + str(params))
    # print("Average OPs: " + str(mean(all_ops)))
    print (res)
    print ("Average ADE: ", str(avg_ade))
    print ("Average FDE: ", str(avg_fde))
