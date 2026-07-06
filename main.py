import glob, os, time, math, warnings, copy, re
from datetime import datetime
import argparse
import random
import pickle
import gc

# OpenMP / MKL thread config — set before torch import
NUM_CPUS = os.cpu_count() or 4
os.environ["OMP_NUM_THREADS"] = str(NUM_CPUS)
os.environ["MKL_NUM_THREADS"] = str(NUM_CPUS)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_CPUS)

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

torch.set_num_threads(NUM_CPUS)
torch.set_num_interop_threads(1)  # inter-op parallelism

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

from rnn import load_model
from rnn.utils import num_trainable_params
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='robocup2D')
parser.add_argument('--data_dir', type=str, default='robocup2d_data')
parser.add_argument('--n_roles', type=int, default=23)
parser.add_argument('--burn_in', type=int, default=10)
parser.add_argument('-t_step', '--totalTimeSteps', type=int, default=20)
parser.add_argument('--overlap', type=int, default=0)
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--n_epoch', type=int, default=1)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('-ev_th','--event_threshold', type=int, default=50)
parser.add_argument('--fs', type=int, default=1)
parser.add_argument('--cont', action='store_true')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--numProcess', type=int, default=4)
parser.add_argument('--TEST', action='store_true') 
parser.add_argument('--challenge_data', type=str, default=None)
parser.add_argument('--Sanity', action='store_true')
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--finetune', action='store_true')
parser.add_argument('--drop_ind', action='store_true')
parser.add_argument('--use_perturbation', action='store_true')
parser.add_argument('--pert_noise_scale', type=float, default=None)
parser.add_argument('--pert_p_event', type=float, default=None)
parser.add_argument('--pf_alpha', type=float, default=None)
parser.add_argument('--pf_beta', type=float, default=None)
parser.add_argument('--pf_gamma', type=float, default=None)
parser.add_argument('--use_clifford', action='store_true')
parser.add_argument('--clifford_layers', type=int, default=2)
parser.add_argument('--use_enkf', action='store_true')
parser.add_argument('--enkf_r', type=float, default=1.0)
parser.add_argument('--enkf_adaptive', action='store_true')
parser.add_argument('--use_volterra_mc', action='store_true')
parser.add_argument('--use_pc', action='store_true')
parser.add_argument('--pf_num_particles', type=int, default=4)
parser.add_argument('--field_scale', type=float, default=105.0)
parser.add_argument('--use_recursive_memory', action='store_true')
parser.add_argument('--recursive_alpha', type=float, default=0.7)
parser.add_argument('--use_interception', action='store_true')
parser.add_argument('--intercept_lambda', type=float, default=0.2)
parser.add_argument('--use_wavelet', action='store_true')
parser.add_argument('--wavelet_level', type=int, default=1)
parser.add_argument('--wavelet_family', type=str, default='db4')
parser.add_argument('--use_intercept', action='store_true')
parser.add_argument('--intercept_beta', type=float, default=0.7)
parser.add_argument('--intercept_horizon', type=int, default=5)
parser.add_argument('--intercept_weight', type=float, default=0.5)
parser.add_argument('--use_pn_intercept', action='store_true')
parser.add_argument('--pn_beta', type=float, default=0.7)
parser.add_argument('--pn_N', type=float, default=4.0)
parser.add_argument('--pn_k_lateral', type=float, default=2.0)
parser.add_argument('--use_fluid_ball', action='store_true')
parser.add_argument('--fluid_ball_gamma', type=float, default=0.8)
parser.add_argument('--fluid_ball_sigma', type=float, default=0.5)
parser.add_argument('--fluid_ball_gamma_target', type=float, default=None)
parser.add_argument('--fluid_ball_gamma_tau', type=float, default=3.0)
parser.add_argument('--use_hybrid_ball', action='store_true')
parser.add_argument('--hybrid_gamma', type=float, default=0.6)
parser.add_argument('--hybrid_linear_speed', type=float, default=0.3)
parser.add_argument('--hybrid_fluid_accel', type=float, default=0.5)
parser.add_argument('--use_ocsvm_ball', action='store_true')
parser.add_argument('--ocsvm_model_path', type=str, default='weights/ocsvm_ball.pkl')
parser.add_argument('--integrator', type=str, default='heun', choices=['legacy','euler','heun','simpson','ab2'])
parser.add_argument('--use_dynamic_fallback', action='store_true')
parser.add_argument('--trajectory_select', action='store_true')
parser.add_argument('--fallback_w_dist', type=float, default=0.5)
parser.add_argument('--fallback_w_speed', type=float, default=0.3)
parser.add_argument('--fallback_w_horizon', type=float, default=0.2)
parser.add_argument('--fallback_w_accel', type=float, default=0.0)
parser.add_argument('--fallback_accel_max', type=float, default=30.0)
parser.add_argument('--accel_clamp', type=float, default=0.0)
parser.add_argument('--use_event_head', action='store_true')
parser.add_argument('--event_loss_weight', type=float, default=0.2)
args, _ = parser.parse_known_args()

path_init = './weights/' 
if args.challenge_data is not None:
    args.Challenge = True
else:
    args.Challenge = False
    
if args.Challenge:
    args.TEST = True

def run_epoch(train, rollout, hp):
    loader = train_loader if train == 1 else val_loader if train == 0 else test_loader
 
    losses = {} 
    losses2 = {}
    sample_nan = 0
    total_samples = 0

    # Desativa gradientes na validação/teste
    if train == 0 or train == -1:
        torch.set_grad_enabled(False)

    for batch_idx, (data) in enumerate(tqdm(loader, desc="Processing batches")):
        if args.cuda:
            data = data.cuda()
        data = data.permute(2, 1, 0, 3)

        if torch.isnan(data).any():
            nan_mask = torch.isnan(data).any(dim=1).any(dim=2).any(dim=0)
            sample_nan += torch.sum(nan_mask)
            data = data[:, :, ~nan_mask]

        current_batch_size = data.size(2)
        total_samples += current_batch_size

        if train == 1:
            batch_losses, batch_losses2 = model(data, rollout, train, hp=hp)
            optimizer.zero_grad()
            total_loss = sum(batch_losses.values())
            total_loss.backward()
            if hp['model'] != 'RNN_ATTENTION': 
                nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        else:
            _, batch_losses, batch_losses2 = model.sample(data, rollout=True, burn_in=hp['burn_in'])
        
        for key in batch_losses:
            if batch_idx == 0:
                losses[key] = batch_losses[key].item() * current_batch_size
            else:
                losses[key] += batch_losses[key].item() * current_batch_size
        
        for key in batch_losses2:
            if batch_idx == 0:
                losses2[key] = batch_losses2[key].item() * current_batch_size
            else:
                losses2[key] += batch_losses2[key].item() * current_batch_size

    effective_samples = total_samples - sample_nan
    if effective_samples > 0:
        for key in losses:
            losses[key] /= effective_samples
        for key in losses2:
            losses2[key] /= effective_samples
    else:
        for key in losses:
            losses[key] = 0.0
        for key in losses2:
            losses2[key] = 0.0

    if train == 0 or train == -1:
        torch.set_grad_enabled(True)

    return losses, losses2

def loss_str(losses):
    ret = ''
    for key in losses:
        if 'vel' in key:
            ret += ' {}: {:.3f} |'.format(key, losses[key])
        else: 
            ret += ' {}: {:.3f} |'.format(key, losses[key])
    return ret[:-2]

def run_sanity(args, test_loader):
    data = []
    for batch_idx, batch in enumerate(test_loader):
        data.append(batch.numpy())
    data = np.concatenate(data, axis=0)
    data = data[:,0] 

    n_agents = args.n_agents
    batchSize, _, _ = data.shape
    n_feat = args.n_feat
    burn_in = args.burn_in
    fs = args.fs
    GT = data.copy()
    losses = {}
    losses['e_pos'] = np.zeros(batchSize)
    losses['e_vel'] = np.zeros(batchSize)
    losses['e_e_p'] = np.zeros(batchSize)
    losses['e_e_v'] = np.zeros(batchSize)

    for t in range(args.horizon):
        for i in range(n_agents):
            current_pos = data[:, t, n_feat*i+0:n_feat*i+2]
            current_vel = data[:, burn_in, n_feat*i+2:n_feat*i+4]
            next_pos0 = GT[:, t+1, n_feat*i+0:n_feat*i+2]
            next_vel0 = GT[:, t+1, n_feat*i+2:n_feat*i+4]

            if t >= burn_in: 
                next_pos = current_pos + current_vel*fs      
                next_vel = current_vel 
                losses['e_pos'] += batch_error(next_pos, next_pos0)
                losses['e_vel'] += batch_error(next_vel, next_vel0)
                data[:, t+1, n_feat*i+0:n_feat*i+2] = next_pos
            if t == args.horizon-1:
                losses['e_e_p'] += batch_error(next_pos, next_pos0)
                losses['e_e_v'] += batch_error(next_vel, next_vel0)

    losses['e_pos'] /= (args.horizon-burn_in)*n_agents 
    losses['e_vel'] /= (args.horizon-burn_in)*n_agents
    losses['e_e_p'] /= n_agents 
    losses['e_e_v'] /= n_agents

    avgL2_m = {}
    avgL2_sd = {}
    for key in losses:
        avgL2_m[key] = np.mean(losses[key])
        avgL2_sd[key] = np.std(losses[key])

    print('Velocity (Sanity Check)')
    print('Mean:')
    print('  Position Error: {:.2f} ± {:.2f}'.format(avgL2_m['e_pos'], avgL2_sd['e_pos']))
    print('  Velocity Error: {:.2f} ± {:.2f}'.format(avgL2_m['e_vel'], avgL2_sd['e_vel']))
    print('Endpoint:')
    print('  Position Error: {:.2f} ± {:.2f}'.format(avgL2_m['e_e_p'], avgL2_sd['e_e_p']))
    print('  Velocity Error: {:.2f} ± {:.2f}'.format(avgL2_m['e_e_v'], avgL2_sd['e_e_v']))
    
    losses['e_pos'] = np.mean(losses['e_pos'])
    losses['e_e_p'] = np.mean(losses['e_e_p'])
    return losses

def batch_error(predict, true):
    error = np.sqrt(np.sum((predict[:,:2] - true[:,:2])**2, 1))
    return error

if __name__ == '__main__':
    numProcess = args.numProcess  
    os.environ["OMP_NUM_THREADS"] = str(numProcess) 
    TEST = args.TEST

    if not torch.cuda.is_available():
        args.cuda = False
        print('cuda is not used')
    else:
        args.cuda = True
        print('cuda is used')

    args.seed = 42
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    args.filter = True
    global fs
    fs = 1/args.fs
    if args.data == 'soccer':
        n_pl = 11
        
    event_threshold = args.event_threshold
    n_roles = args.n_roles
    batchSize = args.batchsize
    overlapWindow = args.overlap
    totalTimeSteps = args.totalTimeSteps

    file_paths = [os.path.join(args.data_dir, file) for file in os.listdir(args.data_dir)]
    os.makedirs("./metadata", exist_ok=True)

    def create_metadata(file_paths, total_time_steps, overlap, output_path):
        total_time_steps += 1
        metadata = []
        for file_path in file_paths:
            playmode_col = pd.read_csv(file_path, usecols=['playmode']).playmode
            play_on_indices = playmode_col[playmode_col == 'play_on'].index.tolist()
            start_idx = play_on_indices[0]
            for i in range(1, len(play_on_indices)):
                if play_on_indices[i] != play_on_indices[i - 1] + 1:
                    end_idx = play_on_indices[i - 1]
                    if end_idx - start_idx + 1 >= total_time_steps:
                        for j in range(start_idx, end_idx - total_time_steps + 1, total_time_steps - overlap):
                            metadata.append({
                                'file_path': file_path,
                                'start_idx': j,
                                'end_idx': j + total_time_steps
                            })
                    start_idx = play_on_indices[i]
            end_idx = play_on_indices[-1]
            if end_idx - start_idx + 1 >= total_time_steps:
                for j in range(start_idx, end_idx - total_time_steps + 1, total_time_steps - overlap):
                    metadata.append({
                        'file_path': file_path,
                        'start_idx': j,
                        'end_idx': j + total_time_steps
                    })
        with open(output_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"Metadata saved to {output_path}")
        return metadata

    if not os.path.exists("./metadata/metadata.pkl"):
        print("Creating metadata...")
        metadata = create_metadata(file_paths, args.totalTimeSteps, args.overlap, output_path="./metadata/metadata.pkl")
    else:
        print("Loading metadata...")
        with open("./metadata/metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        unique_file_paths = set(item['file_path'] for item in metadata)
        if len(file_paths) != len(unique_file_paths):
            print("Updating metadata...")
            metadata = create_metadata(file_paths, args.totalTimeSteps, args.overlap, output_path="./metadata/metadata.pkl")

    def split_metadata_by_date(metadata, val_ratio=0.1, test_ratio=0.1, val_games=None, test_games=None):
        pattern = re.compile(r"(\d{2})(\d{2})-(\d{4}).*?(202\d)")
        def extract_datetime(file_name):
            match = pattern.search(file_name)
            if match:
                month, day, time, year = match.groups()
                hour, minute = divmod(int(time), 100)
                return datetime(int(year), int(month), int(day), hour, minute)
            else:
                raise ValueError(f"Invalid file name format: {file_name}")
        file_times = {}
        for item in metadata:
            file_name = item['file_path'].split('/')[-1]
            file_times[item['file_path']] = extract_datetime(file_name)
        sorted_files = sorted(file_times.items(), key=lambda x: x[1])
        files = [file for file, _ in sorted_files]
        if val_ratio is not None and test_ratio is not None:
            train_files, temp_files = train_test_split(files, test_size=val_ratio + test_ratio, shuffle=False)
            val_files, test_files = train_test_split(temp_files, test_size=test_ratio / (val_ratio + test_ratio), shuffle=False)
        elif val_games is not None and test_games is not None:
            train_files, temp_files = train_test_split(files, test_size=(val_games + test_games)/len(files), shuffle=False)
            val_files, test_files = train_test_split(temp_files, test_size=test_games / (val_games + test_games), shuffle=False)
        train_metadata = [item for item in metadata if item['file_path'] in train_files]
        val_metadata = [item for item in metadata if item['file_path'] in val_files]
        test_metadata = [item for item in metadata if item['file_path'] in test_files]
        random.shuffle(train_metadata)
        return train_metadata, val_metadata, test_metadata

    if len(metadata) < 5000:
        train_metadata, val_metadata, test_metadata = split_metadata_by_date(
            metadata, val_ratio=0.1, test_ratio=0.1)
    else:
        train_metadata, val_metadata, test_metadata = split_metadata_by_date(
            metadata, val_ratio=None, test_ratio=None, val_games=1, test_games=1)
        
    print('train: '+str(len(train_metadata))+' val:'+str(len(val_metadata))+' test: '+str(len(test_metadata)))

    args.agents = [f'l{i}' for i in range(1, 12)] + [f'r{i}' for i in range(1, 12)] + ['b']
    args.Modify_Velocity = True
    print('Modify_Velocity:'+str(args.Modify_Velocity))

    def extract_sequence_tensor(metadata_item, challenge_item=None, no_truncate=False):
        if challenge_item is None:
            file_path, start_idx, end_idx = metadata_item['file_path'], metadata_item['start_idx'], metadata_item['end_idx']
            chunk = pd.read_csv(file_path, skiprows=range(1, start_idx), nrows=end_idx - start_idx)
        else:
            chunk = challenge_item

        data = []
        for agent in args.agents:
            agent_data = chunk[[f'{agent}_x', f'{agent}_y', f'{agent}_vx', f'{agent}_vy']].values
            if challenge_item is not None and not no_truncate:
                agent_data = agent_data[-args.totalTimeSteps:]
            data.append(agent_data)

        tensor = torch.tensor(np.array(data), dtype=torch.float32)
        if args.Modify_Velocity:
            vel = (tensor[:,1:,0:2] - tensor[:,:-1,0:2]) * args.fs
            tensor[:,:-1,2:4] = vel
        tensor = tensor.permute(1, 0, 2)
        tensor = tensor.reshape(tensor.size(0), -1)
        return tensor

    def compute_train_stats(train_metadata):
        sum_tensor = None
        sumsq_tensor = None
        total_steps = 0

        for item in train_metadata:
            tensor = extract_sequence_tensor(item)
            if sum_tensor is None:
                sum_tensor = tensor.sum(dim=0)
                sumsq_tensor = (tensor ** 2).sum(dim=0)
            else:
                sum_tensor += tensor.sum(dim=0)
                sumsq_tensor += (tensor ** 2).sum(dim=0)
            total_steps += tensor.size(0)

        train_mean = sum_tensor / total_steps
        train_var = torch.clamp(sumsq_tensor / total_steps - train_mean ** 2, min=1e-6)
        train_std = torch.sqrt(train_var)
        return train_mean, train_std

    train_mean, train_std = compute_train_stats(train_metadata)
    print('Using train-set normalization')

    class Dataset(Dataset):
        def __init__(self, args, metadata, feature_mean, feature_std, challenge_data=None):
            self.metadata = metadata
            self.args = args
            self.feature_mean = feature_mean
            self.feature_std = feature_std
            self.challenge_data = challenge_data

        def __len__(self):
            if self.challenge_data is None:
                return len(self.metadata)
            else:
                return len(self.challenge_data) 

        def __getitem__(self, idx):
            if self.challenge_data is None:
                item = self.metadata[idx]
                tensor = extract_sequence_tensor(item)
            else:
                tensor = extract_sequence_tensor(None, challenge_item=self.challenge_data[idx], no_truncate=True)

            tensor = (tensor - self.feature_mean) / (self.feature_std + 1e-6)
            tensor = tensor.unsqueeze(0)
            return tensor

    if args.Challenge:
        urls_challenge = sorted(os.listdir(args.challenge_data))
        challenge_data, challenge_cycle, challenge_names = [], [], []
        for url in urls_challenge:
            if url == '@eaDir' or not url.endswith('.tracking.csv'):
                continue
            challenge_data.append(pd.read_csv(args.challenge_data + os.sep + url))
            challenge_cycle.append(challenge_data[-1].shape[0])
            # Preserva nome original (ex: HELIOS2024_4-vs-Mars_0)
            challenge_names.append(url.replace('.tracking.csv', ''))
        test_metadata = None
        len_seqs_test = len(challenge_data)
        # Process each challenge file individually (different lengths)
        batchSize_test = 1
    else:
        challenge_data = None
        len_seqs_test = len(test_metadata)
        batchSize_test = args.batchsize

    num_workers = args.num_workers
    kwargs = {'num_workers': num_workers, 'pin_memory': True} if args.cuda else {}
    print('num_workers:'+str(num_workers))

    if not TEST or args.Challenge:    
        if not TEST:
            train_loader = DataLoader(Dataset(args, train_metadata, train_mean, train_std),
                    batch_size=args.batchsize, shuffle=True, **kwargs)
            val_loader = DataLoader(Dataset(args, val_metadata, train_mean, train_std),
                    batch_size=args.batchsize, shuffle=False, drop_last=False, **kwargs)
    
        test_loader = DataLoader(Dataset(args, test_metadata, train_mean, train_std, challenge_data=challenge_data),
                batch_size=batchSize_test if args.Challenge else args.batchsize, shuffle=False, drop_last=False, **kwargs)

    activeRoleInd = range(n_roles)
    activeRole = [str(n) for n in range(n_roles)]
    args.n_agents = len(activeRole)

    outputlen0 = 2
    n_feat = 4
    featurelen = 4*23  
    args.n_feat = n_feat
    args.fs = fs
    args.horizon = totalTimeSteps

    if args.Sanity:
        losses = run_sanity(args, test_loader)

    init_filename0 = path_init + args.model + '_' + args.data + '/'
    init_filename0 = init_filename0 + str(batchSize) + '_' + str(totalTimeSteps)      
    if args.drop_ind:
        init_filename0 = init_filename0 + '_drop_ind' 

    if not os.path.isdir(init_filename0):
        os.makedirs(init_filename0)
    init_pthname = '{}_state_dict'.format(init_filename0)
    print('model: '+init_filename0)

    if not os.path.isdir(init_pthname):
        os.makedirs(init_pthname)

    args.dataset = args.data
    args.start_lr = 1e-3 
    args.min_lr = 1e-3 
    clip = True
    save_every = 1
    args.batch_size = batchSize
    args.x_dim = outputlen0
    args.y_dim = featurelen
    args.z_dim = 64 
    args.h_dim = 64
    args.rnn_dim = 100
    args.n_layers = 2
    args.rnn_micro_dim = args.rnn_dim
    args.n_all_agents = 22 if args.data == 'soccer' else 10 
    ball_dim = 4 
    temperature = 1 if args.data == 'soccer' else 1 

    # Compact arch for CPU
    args.h_dim = 16
    args.z_dim = 16
    args.rnn_dim = 32
        
    params = {
        'model' : args.model,
        'dataset' : args.dataset,
        'x_dim' : args.x_dim,
        'y_dim' : args.y_dim,
        'z_dim' : args.z_dim,
        'h_dim' : args.h_dim,
        'rnn_dim' : args.rnn_dim,
        'n_layers' : args.n_layers, 
        'len_seq' : totalTimeSteps,  
        'n_agents' : args.n_agents,    
        'min_lr' : args.min_lr,
        'start_lr' : args.start_lr,
        'seed' : args.seed,
        'cuda' : args.cuda,
        'n_feat' : n_feat,
        'fs' : fs,
        'embed_size' : 8,
        'embed_ball_size' : 8,
        'burn_in' : args.burn_in,
        'horizon' : args.horizon,
        'rnn_micro_dim' : args.rnn_micro_dim,
        'ball_dim' : ball_dim,
        'n_all_agents' : args.n_all_agents,
        'temperature' : temperature,
        'drop_ind' : args.drop_ind,
        'num_particles': args.pf_num_particles,
        'PREDICT_ACCELERATION': True,
        'LATENT_DIM': args.h_dim,
        'ROLLOUT_LOSS_WEIGHT': 0.2,
        'ROLLOUT_STEPS': 2,
        'VELOCITY_LOSS_WEIGHT': 0.5,
        'ACCELERATION_LOSS_WEIGHT': 0.1,
        'MEMORY_STEPS': 3,
        'TRANSFORMER_HEADS': 2,
        'TRANSFORMER_LAYERS': 1,
        'TRANSFORMER_FF_DIM': 32,
        'USE_RK2': False,
        'USE_AB3': True,
        'USE_PERTURBATION': args.use_perturbation,
        'NOISE_SCALE': args.pert_noise_scale if args.pert_noise_scale is not None else 0.02,
        'P_EVENT': args.pert_p_event if args.pert_p_event is not None else 0.3,
        'PF_ALPHA': args.pf_alpha if args.pf_alpha is not None else 1.0,
        'PF_BETA': args.pf_beta if args.pf_beta is not None else 0.5,
        'PF_GAMMA': args.pf_gamma if args.pf_gamma is not None else 2.0,
        'SCHEDULED_SAMPLING_START': 0.0,
        'SCHEDULED_SAMPLING_MAX': 0.9,
        'SCHEDULED_SAMPLING_ANNEAL_EPOCHS': 15,
        'USE_CLIFFORD': args.use_clifford,
        'CLIFFORD_LAYERS': args.clifford_layers,
        'USE_ENKF': args.use_enkf,
        'ENKF_R': args.enkf_r,
        'ENKF_ADAPTIVE': args.enkf_adaptive,
        'USE_VOLTERRA_MC': args.use_volterra_mc,
        'USE_PC': args.use_pc,
        'FIELD_SCALE': args.field_scale,
        'USE_RECURSIVE_MEMORY': args.use_recursive_memory,
        'RECURSIVE_ALPHA': args.recursive_alpha,
        'USE_INTERCEPTION': args.use_interception,
        'INTERCEPT_LAMBDA': args.intercept_lambda,
        'USE_WAVELET': args.use_wavelet,
        'WAVELET_LEVEL': args.wavelet_level,
        'WAVELET_FAMILY': args.wavelet_family,
        'USE_INTERCEPT': args.use_intercept,
        'INTERCEPT_BETA': args.intercept_beta,
        'INTERCEPT_HORIZON': args.intercept_horizon,
        'INTERCEPT_WEIGHT': args.intercept_weight,
        'USE_PN_INTERCEPT': args.use_pn_intercept,
        'PN_BETA': args.pn_beta,
        'PN_N': args.pn_N,
        'PN_K_LATERAL': args.pn_k_lateral,
        'USE_FLUID_BALL': args.use_fluid_ball,
        'FLUID_BALL_GAMMA': args.fluid_ball_gamma,
        'FLUID_BALL_SIGMA': args.fluid_ball_sigma,
        'FLUID_BALL_GAMMA_TARGET': args.fluid_ball_gamma_target,
        'FLUID_BALL_GAMMA_TAU': args.fluid_ball_gamma_tau,
        'USE_HYBRID_BALL': args.use_hybrid_ball,
        'HYBRID_GAMMA': args.hybrid_gamma,
        'HYBRID_LINEAR_SPEED': args.hybrid_linear_speed,
        'HYBRID_FLUID_ACCEL': args.hybrid_fluid_accel,
        'USE_OCSVM_BALL': args.use_ocsvm_ball,
        'OCSVM_MODEL_PATH': args.ocsvm_model_path,
        'INTEGRATOR': args.integrator,
        'USE_DYNAMIC_FALLBACK': args.use_dynamic_fallback,
        'FALLBACK_W_DIST': args.fallback_w_dist,
        'FALLBACK_W_SPEED': args.fallback_w_speed,
        'FALLBACK_W_HORIZON': args.fallback_w_horizon,
        'FALLBACK_W_ACCEL': args.fallback_w_accel,
        'FALLBACK_ACCEL_MAX': args.fallback_accel_max,
        'ACCEL_CLAMP': args.accel_clamp,
        'USE_EVENT_HEAD': args.use_event_head,
        'EVENT_LOSS_WEIGHT': args.event_loss_weight,
        'USE_TRAJECTORY_SELECT': args.trajectory_select,
    }

    # ================== CARREGAMENTO DO MODELO ==================
    if args.model == 'gtpa':
        from gtpa import GTPAModel
        model = GTPAModel(params, parser)
    else:
        model = load_model(args.model, params, parser)

    if args.cuda:
        model.cuda()
    params = model.params
    params['total_params'] = num_trainable_params(model)

    pickle.dump(params, open(init_filename0+'/params.p', 'wb'), protocol=2)

    if args.cont:
        if os.path.exists('{}_best.pth'.format(init_pthname)): 
            state_dict = torch.load('{}_best.pth'.format(init_pthname))
            model.load_state_dict(state_dict, strict=False)
            print('best model was loaded')
        else:
            print('args.cont = True but file did not exist')

    print('############################################################')

    best_val_loss = 0
    epochs_since_best = 0
    lr = max(args.start_lr, args.min_lr)
    epoch_first_best = -1

    pretrain_time = 0
    
    hyperparams = {
        'model': args.model,
        'burn_in': args.horizon,
        'pretrain': (0 < pretrain_time),
        'scheduled_sampling_prob': params.get('SCHEDULED_SAMPLING_START', 0.0),
        'rollout_steps': params.get('ROLLOUT_STEPS', 3),
    }
    
    if not TEST:
        for e in range(args.n_epoch):
            epoch = e+1
            print('epoch '+str(epoch))
            pretrain = (epoch <= pretrain_time)
            hyperparams['pretrain'] = pretrain

            # Anneal scheduled sampling probability
            ss_start = params.get('SCHEDULED_SAMPLING_START', 0.0)
            ss_max = params.get('SCHEDULED_SAMPLING_MAX', 0.3)
            ss_anneal = params.get('SCHEDULED_SAMPLING_ANNEAL_EPOCHS', 30)
            if pretrain:
                ss_prob = ss_start
            else:
                ss_epoch = epoch - pretrain_time
                ss_prob = min(ss_max, ss_start + (ss_max - ss_start) * ss_epoch / max(1, ss_anneal))
            hyperparams['scheduled_sampling_prob'] = ss_prob
            if epoch % max(1, args.n_epoch // 10) == 0 or epoch == 1:
                print('scheduled_sampling_prob: {:.3f}'.format(ss_prob))

            if epochs_since_best == 3:
                filename = '{}_best.pth'.format(init_pthname)
                state_dict = torch.load(filename)
                epochs_since_best = 0
                print('##### Best model is loaded #####')
            else:
                if not hyperparams['pretrain'] and not args.finetune:
                    print('########## lr {:.4e} ##########'.format(lr)) 
                    epochs_since_best += 1
                
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=lr,
                    weight_decay=1e-4)
            
            start_time = time.time()
            
            print('pretrain:'+str(hyperparams['pretrain']))
            hyperparams['burn_in'] = args.horizon
            train_loss, train_loss2 = run_epoch(train=1, rollout=False, hp=hyperparams)
            print('Train:\t'+loss_str(train_loss)+'|'+loss_str(train_loss2))

            torch.cuda.empty_cache()
            gc.collect()
            
            hyperparams['burn_in'] = args.burn_in
            val_loss, val_loss2 = run_epoch(train=0, rollout=True, hp=hyperparams)
            print('RO Val:\t'+loss_str(val_loss)+'|'+loss_str(val_loss2))

            total_val_loss = sum(val_loss.values())

            epoch_time = time.time() - start_time
            print('Time:\t {:.3f}'.format(epoch_time))

            torch.cuda.empty_cache()
            gc.collect()

            if e > epoch_first_best and (best_val_loss == 0 or total_val_loss < best_val_loss): 
                best_val_loss_prev = best_val_loss
                best_val_loss = total_val_loss
                epochs_since_best = 0

                filename = '{}_best.pth'.format(init_pthname)
                torch.save(model.state_dict(), filename)
                print('##### Best model #####')
                if epoch > pretrain_time and (best_val_loss_prev-best_val_loss)/best_val_loss < 0.0001 and best_val_loss_prev != 0:
                    print('best loss - current loss: ' + str(best_val_loss_prev) + ' - ' + str(best_val_loss))
                    break 

            if epoch % save_every == 0:
                filename = '{}_{}.pth'.format(init_pthname, epoch)
                torch.save(model.state_dict(), filename)
                print('########## Saved model ##########')
                           
        print('Best Val Loss: {:.4f}'.format(best_val_loss))
    
    # ================== TESTE / SUBMISSÃO ==================
    # Tenta carregar checkpoint treinado; se não existir, usa o modelo atual
    best_pth = '{}_best.pth'.format(init_pthname)
    if os.path.exists(best_pth):
        state_dict = torch.load(best_pth, map_location=lambda storage, loc: storage)
        model.load_state_dict(state_dict, strict=False)
        print('Loaded best checkpoint')
    else:
        print('No checkpoint found, using current model')

    print('test sample')

    if args.Challenge:
        CHALLENGE_PRED_STEPS = 30
        samples = np.zeros((CHALLENGE_PRED_STEPS, args.n_agents, len_seqs_test, n_feat))
        start_time = time.time()
        # Process each challenge file individually (lengths may differ)
        for seq in range(len_seqs_test):
            df = challenge_data[seq]
            # Extract features from ALL frames (no truncation)
            tensor = extract_sequence_tensor(None, challenge_item=df, no_truncate=True)
            tensor = (tensor - train_mean) / (train_std + 1e-6)
            # Shape: (T, 92) -> (1, T, 92)
            tensor = tensor.unsqueeze(0)
            # Shape: (1, T, 92) -> (T, 1, 1, 92)
            tensor = tensor.permute(1, 0, 2).unsqueeze(2)
            input_len = tensor.size(0)

            # Temporarily override RNN params for challenge prediction
            if args.model == 'RNN':
                orig_horizon = model.params.get('horizon')
                orig_len_seq = model.params.get('len_seq')
                orig_burn_in = model.params.get('burn_in')
                model.params['horizon'] = input_len + CHALLENGE_PRED_STEPS
                model.params['len_seq'] = input_len + CHALLENGE_PRED_STEPS
                model.params['burn_in'] = input_len
                model.len_seq = input_len + CHALLENGE_PRED_STEPS
            else:
                # GTPA: pad tensor with zeros for autoregressive prediction
                pad_tensor = torch.zeros(CHALLENGE_PRED_STEPS, tensor.size(1), tensor.size(2), tensor.size(3), device=tensor.device)
                tensor = torch.cat([tensor, pad_tensor], dim=0)

            with torch.no_grad():
                if hasattr(model, 'particle_filter'):
                    model.particle_filter.alpha_trace = []
                sample, _, _ = model.sample(tensor, rollout=True, burn_in=input_len, n_sample=1, TEST=True, Challenge=args.Challenge)

            # Dump alpha trace for analysis
            if hasattr(model, 'particle_filter') and model.particle_filter.alpha_trace:
                import json
                trace_path = f'results/test/alpha_trace_seq{seq}.json'
                os.makedirs('results/test', exist_ok=True)
                with open(trace_path, 'w') as f:
                    json.dump(model.particle_filter.alpha_trace, f, indent=2)

            # Restore RNN params if needed
            if args.model == 'RNN':
                model.params['horizon'] = orig_horizon
                model.params['len_seq'] = orig_len_seq
                model.params['burn_in'] = orig_burn_in

            sample_np = sample.detach().cpu().numpy()
            # For challenge, extract the last CHALLENGE_PRED_STEPS frames (predictions only)
            if sample_np.shape[0] >= input_len + CHALLENGE_PRED_STEPS:
                sample_np = sample_np[-CHALLENGE_PRED_STEPS:]  # last 30 non-zero predictions
            elif sample_np.shape[0] > CHALLENGE_PRED_STEPS:
                sample_np = sample_np[-CHALLENGE_PRED_STEPS:]
            else:
                sample_np = sample_np[:CHALLENGE_PRED_STEPS]
            sample_np = sample_np[:CHALLENGE_PRED_STEPS]

            # Sample output is (time, 1, batch, 92). Reshape to (time, 23, batch, 4).
            sample_np = sample_np.reshape(sample_np.shape[0], sample_np.shape[1], sample_np.shape[2], 23, n_feat)
            sample_np = sample_np.squeeze(1).transpose(0, 2, 1, 3)

            samples[:, :, seq:seq+1, :] = sample_np

            del sample, sample_np
        epoch_time = time.time() - start_time
        print('Time:\t {:.3f}'.format(epoch_time))

        # Export submission
        experiment_path = './results/test/submission'
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)

        feature_mean_np = train_mean.detach().cpu().numpy()
        feature_std_np = train_std.detach().cpu().numpy()

        for seq in range(samples.shape[2]):
            sample_full = samples[:, :, seq, :]  # (time, 23, 4)
            sample_full = sample_full * feature_std_np.reshape(1, 23, 4) + feature_mean_np.reshape(1, 23, 4)
            sample_ = sample_full[:, :, :2]  # (time, 23, 2)
            base_name = challenge_names[seq]
            sample_path = os.path.join(experiment_path, f'{base_name}.tracking.csv')
            df = pd.DataFrame(sample_.reshape(sample_.shape[0], -1),
                              columns=[f'agent_{a}_{c}' for a in range(23) for c in ['x','y']])
            rename_map = {f'agent_{i}': f'l{i+1}' for i in range(11)}
            rename_map.update({f'agent_{i}': f'r{i-10}' for i in range(11, 22)})
            rename_map['agent_22'] = 'b'
            for old, new in rename_map.items():
                df.columns = [col.replace(f'{old}_', f'{new}_') for col in df.columns]

            cycle = range(challenge_cycle[seq]+1, challenge_cycle[seq] + len(df) + 1)
            df.insert(0, '#', cycle)
            df.to_csv(sample_path, index=False)
        print('Samples saved to {}'.format(experiment_path))
    else:
        samples = np.zeros((args.totalTimeSteps, args.n_agents, len_seqs_test, n_feat))
        loader = test_loader
        losses = {}
        losses2 = {}

        start_time = time.time()
        for batch_idx, (data) in enumerate(loader):
            if args.cuda:
                data = data.cuda()
            data = data.permute(2, 1, 0, 3)
            
            with torch.no_grad():
                sample, output, output2 = model.sample(data, rollout=True, burn_in=args.burn_in, n_sample=1, TEST=True, Challenge=args.Challenge)

            sample_np = sample.detach().cpu().numpy()
            if sample_np.shape[0] > args.totalTimeSteps:
                sample_np = sample_np[:args.totalTimeSteps]
            elif sample_np.shape[0] < args.totalTimeSteps:
                pad = args.totalTimeSteps - sample_np.shape[0]
                sample_np = np.pad(sample_np, ((0, pad), (0,0), (0,0), (0,0)), mode='constant')
            
            # Sample output is (time, 1, batch, 92). Reshape to (time, 23, batch, 4).
            sample_np = sample_np.reshape(sample_np.shape[0], sample_np.shape[1], sample_np.shape[2], 23, n_feat)
            sample_np = sample_np.squeeze(1).transpose(0, 2, 1, 3)
            
            # Armazena no array de amostras
            current_batch = sample_np.shape[2]
            start_idx = batch_idx * batchSize_test
            end_idx = start_idx + current_batch
            samples[:sample_np.shape[0], :, start_idx:end_idx, :] = sample_np

            del sample, sample_np

            for key in output:
                if batch_idx == 0:
                    losses[key] = np.zeros(1)
                    losses2[key] = np.zeros((len_seqs_test))
                val = output[key].detach().cpu().numpy()
                if val.ndim == 0:
                    val = np.array([val])
                losses[key] += np.sum(val)
                losses2[key][start_idx:end_idx] = val[:current_batch]
                
            for key in output2:
                if batch_idx == 0:
                    losses[key] = np.zeros(1)
                    losses2[key] = np.zeros((len_seqs_test))
                val = output2[key].detach().cpu().numpy()
                if val.ndim == 0:
                    val = np.array([val])
                losses[key] += np.sum(val)
                losses2[key][start_idx:end_idx] = val[:current_batch]

            torch.cuda.empty_cache()
            gc.collect()

        epoch_time = time.time() - start_time
        print('Time:\t {:.3f}'.format(epoch_time)) 
        
        avgL2_m = {}
        avgL2_sd = {}
        for key in losses2:
            avgL2_m[key] = np.mean(losses2[key])
            avgL2_sd[key] = np.std(losses2[key])

        print(args.model)
        print('Mean:')
        print('  Position Error: {:.2f} ± {:.2f}'.format(avgL2_m['e_pos'], avgL2_sd['e_pos']))
        print('  Velocity Error: {:.2f} ± {:.2f}'.format(avgL2_m['e_vel'], avgL2_sd['e_vel']))
        print('Endpoint:')
        print('  Position Error: {:.2f} ± {:.2f}'.format(avgL2_m['e_e_p'], avgL2_sd['e_e_p']))
        print('  Velocity Error: {:.2f} ± {:.2f}'.format(avgL2_m['e_e_v'], avgL2_sd['e_e_v']))