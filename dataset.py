import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
import argparse
import csv
import torch
import re
from scipy.io import loadmat, savemat
from einops import rearrange, repeat
import torch.nn.functional as F
import nibabel as nib
from scipy.ndimage import zoom


class Datasets_SZ(Dataset):
    def __init__(self, argv,):
        super().__init__()
        self.argv = argv
        self.data_folder = 'processed_FNCandMSN_alldata'
        self.prior_file = f'biobert_biomarkers_Gem25FP0520_sim.pt'  # prior_file

        info = pd.read_csv(self.argv.data_path + f'{self.argv.datname}_info_select2.csv')
        info = info[~info['Sub_name'].duplicated()]

        path = os.path.join(self.argv.data_path, self.data_folder)  # processed_alltype_alldata
        file_list = os.listdir(path)
        file_ids = [re.split(r'[_\.]', file)[-2] for file in file_list]
        self.file_ids = [str(int(id)) if id.isdigit() else id for id in file_ids]
        self.file_ids = [id for id in self.file_ids if id in info['Sub_name'].to_numpy()]
        self.file_list = [
            file for file in file_list
            if (lambda x: str(int(x)) if x.isdigit() else x)(re.split(r'[_\.]', file)[-2]) in self.file_ids
        ]
        self.file_paths = [os.path.join(path, file) for file in self.file_list]

        self.msn = []
        self.fcn = []
        self.ids = []
        self.labels = []
        self.matched_info_rows = []
        for i in tqdm(range(len(self.file_paths)), ncols=80, desc=f'dataset loading'):
            file_path = os.path.join(self.file_paths[i])
            npy_data = np.load(file_path, allow_pickle=True).item()
            msn = npy_data['Sub_MSN']  # (116, 116)
            fcn = npy_data['Sub_FNC']  # (53, 53)

            id = (lambda x: str(int(x)) if x.isdigit() else x)(re.split(r'[_\.]', self.file_list[i])[-2])
            self.msn.append(torch.tensor(msn, dtype=torch.float32))
            self.fcn.append(torch.tensor(fcn, dtype=torch.float32))
            self.ids.append(id)
            self.labels.append(info[info['Sub_name'] == id]['DXGROUP2'].to_numpy()[0])  # 0=healthy, 1=patient

            matched_row = info[info['Sub_name'] == id]
            self.matched_info_rows.append(matched_row)

        self.labels = np.array(self.labels)

        # get_priori
        self.priori_dict = torch.load(os.path.join(self.argv.data_path, self.prior_file))
        print('done')


        # self.matched_info_rows to pd.DataFrame
        self.matched_info_rows = pd.concat(self.matched_info_rows)
        self.matched_info_rows_SZ1 = self.matched_info_rows[self.matched_info_rows['DXGROUP2'].isin([0, 1])]
        self.matched_info_rows_SZ2 = self.matched_info_rows[self.matched_info_rows['DXGROUP2'].isin([2, 3])]
        self.matched_info_rows_SZ3 = self.matched_info_rows[self.matched_info_rows['DXGROUP2'].isin([4, 5])]
        self.matched_info_rows_SZ4 = self.matched_info_rows[self.matched_info_rows['DXGROUP2'].isin([6, 7])]

        age_mean_SZ1_0 = self.matched_info_rows_SZ1[self.matched_info_rows_SZ1['DXGROUP2'] == 0]['age'].mean()
        age_std_SZ1_0 = self.matched_info_rows_SZ1[self.matched_info_rows_SZ1['DXGROUP2'] == 0]['age'].std()
        age_mean_SZ1_1 = self.matched_info_rows_SZ1[self.matched_info_rows_SZ1['DXGROUP2'] == 1]['age'].mean()
        age_std_SZ1_1 = self.matched_info_rows_SZ1[self.matched_info_rows_SZ1['DXGROUP2'] == 1]['age'].std()

        age_mean_SZ2_0 = self.matched_info_rows_SZ2[self.matched_info_rows_SZ2['DXGROUP2'] == 2]['age'].mean()
        age_std_SZ2_0 = self.matched_info_rows_SZ2[self.matched_info_rows_SZ2['DXGROUP2'] == 2]['age'].std()
        age_mean_SZ2_1 = self.matched_info_rows_SZ2[self.matched_info_rows_SZ2['DXGROUP2'] == 3]['age'].mean()
        age_std_SZ2_1 = self.matched_info_rows_SZ2[self.matched_info_rows_SZ2['DXGROUP2'] == 3]['age'].std()

        age_mean_SZ3_0 = self.matched_info_rows_SZ3[self.matched_info_rows_SZ3['DXGROUP2'] == 4]['age'].mean()
        age_std_SZ3_0 = self.matched_info_rows_SZ3[self.matched_info_rows_SZ3['DXGROUP2'] == 4]['age'].std()
        age_mean_SZ3_1 = self.matched_info_rows_SZ3[self.matched_info_rows_SZ3['DXGROUP2'] == 5]['age'].mean()
        age_std_SZ3_1 = self.matched_info_rows_SZ3[self.matched_info_rows_SZ3['DXGROUP2'] == 5]['age'].std()

        age_mean_SZ4_0 = self.matched_info_rows_SZ4[self.matched_info_rows_SZ4['DXGROUP2'] == 6]['age'].mean()
        age_std_SZ4_0 = self.matched_info_rows_SZ4[self.matched_info_rows_SZ4['DXGROUP2'] == 6]['age'].std()
        age_mean_SZ4_1 = self.matched_info_rows_SZ4[self.matched_info_rows_SZ4['DXGROUP2'] == 7]['age'].mean()
        age_std_SZ4_1 = self.matched_info_rows_SZ4[self.matched_info_rows_SZ4['DXGROUP2'] == 7]['age'].std()

        print(f'age_mean_SZ1_0: {age_mean_SZ1_0:.2f}±{age_std_SZ1_0:.2f}')
        print(f'age_mean_SZ1_1: {age_mean_SZ1_1:.2f}±{age_std_SZ1_1:.2f}')
        print(f'age_mean_SZ2_0: {age_mean_SZ2_0:.2f}±{age_std_SZ2_0:.2f}')
        print(f'age_mean_SZ2_1: {age_mean_SZ2_1:.2f}±{age_std_SZ2_1:.2f}')
        print(f'age_mean_SZ3_0: {age_mean_SZ3_0:.2f}±{age_std_SZ3_0:.2f}')
        print(f'age_mean_SZ3_1: {age_mean_SZ3_1:.2f}±{age_std_SZ3_1:.2f}')
        print(f'age_mean_SZ4_0: {age_mean_SZ4_0:.2f}±{age_std_SZ4_0:.2f}')
        print(f'age_mean_SZ4_1: {age_mean_SZ4_1:.2f}±{age_std_SZ4_1:.2f}')

        print('sex_counts_SZ1_0:', self.matched_info_rows_SZ1[self.matched_info_rows_SZ1['DXGROUP2'] == 0]['sex'].value_counts())
        print('sex_counts_SZ1_1:', self.matched_info_rows_SZ1[self.matched_info_rows_SZ1['DXGROUP2'] == 1]['sex'].value_counts())
        print('sex_counts_SZ2_0:', self.matched_info_rows_SZ2[self.matched_info_rows_SZ2['DXGROUP2'] == 2]['sex'].value_counts())
        print('sex_counts_SZ2_1:', self.matched_info_rows_SZ2[self.matched_info_rows_SZ2['DXGROUP2'] == 3]['sex'].value_counts())
        print('sex_counts_SZ3_0:', self.matched_info_rows_SZ3[self.matched_info_rows_SZ3['DXGROUP2'] == 4]['sex'].value_counts())
        print('sex_counts_SZ3_1:', self.matched_info_rows_SZ3[self.matched_info_rows_SZ3['DXGROUP2'] == 5]['sex'].value_counts())
        print('sex_counts_SZ4_0:', self.matched_info_rows_SZ4[self.matched_info_rows_SZ4['DXGROUP2'] == 6]['sex'].value_counts())
        print('sex_counts_SZ4_1:', self.matched_info_rows_SZ4[self.matched_info_rows_SZ4['DXGROUP2'] == 7]['sex'].value_counts())


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fcn = self.get_upper(self.fcn[idx])
        msn = self.get_upper(self.msn[idx])
        return {'id': self.ids[idx],
                'msn': self.msn[idx],
                'fcn': self.fcn[idx],
                'label': torch.tensor(self.labels[idx], dtype=torch.long),
                'fcn-msn': torch.cat([fcn, msn], dim=0)
                }

    def get_upper(self, mat: torch.Tensor) -> torch.Tensor:
        n = mat.shape[0]
        row_idx, col_idx = torch.triu_indices(n, n, offset=1)
        return mat[row_idx, col_idx]

    def save_mat(self, path=None):
        if path is None:
            path = os.path.join(self.argv.data_path, f'dataset_{self.argv.datname}.mat')
        savemat(path, {'id': self.ids,
                        'msn': torch.stack(self.msn, dim=0).detach().numpy().astype('float64'),
                        'fcn': torch.stack(self.fcn, dim=0).detach().numpy().astype('float64'),
                        'label': np.array(self.labels, dtype='float64'),
                       })
        print('save mat done')


class Datasets_ASD(Dataset):
    def __init__(self, argv,):
        super().__init__()
        self.argv = argv
        self.data_folder = 'processed_FNCandMSN_alldata'
        self.prior_file = f'biobert_biomarkers_Gem25FP0520_sim_3.pt'  # 3->10 2->5

        info = pd.read_csv(self.argv.data_path + f'{self.argv.datname}_info_select1.csv')
        info = info[~info['Sub_name'].duplicated()]

        path = os.path.join(self.argv.data_path, self.data_folder)  # processed_alltype_alldata
        file_list = os.listdir(path)
        file_ids = [re.split(r'[_\.]', file)[-2] for file in file_list]
        self.file_ids = [(int(id)) if id.isdigit() else id for id in file_ids]
        self.file_ids = [id for id in self.file_ids if id in info['Sub_name'].to_numpy()]
        self.file_list = [
            file for file in file_list
            if (lambda x: (int(x)) if x.isdigit() else x)(re.split(r'[_\.]', file)[-2]) in self.file_ids
        ]
        self.file_paths = [os.path.join(path, file) for file in self.file_list]

        self.msn = []
        self.fcn = []
        self.ids = []
        self.labels = []
        self.matched_info_rows = []
        for i in tqdm(range(len(self.file_paths)), ncols=80, desc=f'dataset loading'):
            file_path = os.path.join(self.file_paths[i])
            npy_data = np.load(file_path, allow_pickle=True).item()
            msn = npy_data['Sub_MSN']  # (116, 116)
            fcn = npy_data['Sub_FNC']  # (53, 53)

            id = (lambda x: (int(x)) if x.isdigit() else x)(re.split(r'[_\.]', self.file_list[i])[-2])
            self.msn.append(torch.tensor(msn, dtype=torch.float32))
            self.fcn.append(torch.tensor(fcn, dtype=torch.float32))
            self.ids.append(id)
            self.labels.append(info[info['Sub_name'] == id]['DXGROUP2'].to_numpy()[0])

            matched_row = info[info['Sub_name'] == id]
            self.matched_info_rows.append(matched_row)

        self.labels = np.array(self.labels)

        # get_priori
        self.priori_dict = torch.load(os.path.join(self.argv.data_path, self.prior_file))
        print('done')


        # self.matched_info_rows to pd.DataFrame
        self.matched_info_rows = pd.concat(self.matched_info_rows)
        self.matched_info_rows_ASD1 = self.matched_info_rows[self.matched_info_rows['DXGROUP2'] <= 37]
        self.matched_info_rows_ASD2 = self.matched_info_rows[self.matched_info_rows['DXGROUP2'] > 37]
        age_mean_ASD1_0 = self.matched_info_rows_ASD1[self.matched_info_rows_ASD1['DXGROUP'] == 0]['age'].mean()
        age_std_ASD1_0 = self.matched_info_rows_ASD1[self.matched_info_rows_ASD1['DXGROUP'] == 0]['age'].std()
        age_mean_ASD1_1 = self.matched_info_rows_ASD1[self.matched_info_rows_ASD1['DXGROUP'] == 1]['age'].mean()
        age_std_ASD1_1 = self.matched_info_rows_ASD1[self.matched_info_rows_ASD1['DXGROUP'] == 1]['age'].std()
        age_mean_ASD2_0 = self.matched_info_rows_ASD2[self.matched_info_rows_ASD2['DXGROUP'] == 0]['age'].mean()
        age_std_ASD2_0 = self.matched_info_rows_ASD2[self.matched_info_rows_ASD2['DXGROUP'] == 0]['age'].std()
        age_mean_ASD2_1 = self.matched_info_rows_ASD2[self.matched_info_rows_ASD2['DXGROUP'] == 1]['age'].mean()
        age_std_ASD2_1 = self.matched_info_rows_ASD2[self.matched_info_rows_ASD2['DXGROUP'] == 1]['age'].std()
        print(f'age_mean_ASD1_0: {age_mean_ASD1_0:.2f}±{age_std_ASD1_0:.2f}')
        print(f'age_mean_ASD1_1: {age_mean_ASD1_1:.2f}±{age_std_ASD1_1:.2f}')
        print(f'age_mean_ASD2_0: {age_mean_ASD2_0:.2f}±{age_std_ASD2_0:.2f}')
        print(f'age_mean_ASD2_1: {age_mean_ASD2_1:.2f}±{age_std_ASD2_1:.2f}')
        print('sex_counts_ASD1_0:', self.matched_info_rows_ASD1[self.matched_info_rows_ASD1['DXGROUP'] == 0]['sex'].value_counts())
        print('sex_counts_ASD1_1:', self.matched_info_rows_ASD1[self.matched_info_rows_ASD1['DXGROUP'] == 1]['sex'].value_counts())
        print('sex_counts_ASD1_0:', self.matched_info_rows_ASD2[self.matched_info_rows_ASD2['DXGROUP'] == 0]['sex'].value_counts())
        print('sex_counts_ASD1_1:', self.matched_info_rows_ASD2[self.matched_info_rows_ASD2['DXGROUP'] == 1]['sex'].value_counts())


    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fcn = self.get_upper(self.fcn[idx])
        msn = self.get_upper(self.msn[idx])
        return {'id': self.ids[idx],
                'msn': self.msn[idx],
                'fcn': self.fcn[idx],
                'label': torch.tensor(self.labels[idx], dtype=torch.long),
                'fcn-msn': torch.cat([fcn, msn], dim=0)
                }

    def get_upper(self, mat: torch.Tensor) -> torch.Tensor:
        n = mat.shape[0]
        row_idx, col_idx = torch.triu_indices(n, n, offset=1)
        return mat[row_idx, col_idx]

    def save_mat(self, path=None):
        if path is None:
            path = os.path.join(self.argv.data_path, f'dataset_{self.argv.datname}.mat')
        savemat(path, {'id': self.ids,
                        'msn': torch.stack(self.msn, dim=0).detach().numpy().astype('float64'),
                        'fcn': torch.stack(self.fcn, dim=0).detach().numpy().astype('float64'),
                        'label': np.array(self.labels, dtype='float64'),
                       })
        print('save mat done')


class Datasets_AnxDep(Dataset):
    def __init__(self, argv,):
        super().__init__()
        self.argv = argv
        self.data_folder = 'processed_FNCandMSN_alldata'
        self.prior_file = f'biobert_biomarkers_Gem25FP0520_sim.pt'  # 3->10 2->5

        info = pd.read_csv(self.argv.data_path + f'info_select_{self.argv.datname}.csv')
        info = info[~info['eid'].duplicated()]

        path = os.path.join(self.argv.data_path, self.data_folder)  # processed_alltype_alldata
        file_list = os.listdir(path)
        file_ids = [re.split(r'[_\.]', file)[-2] for file in file_list]
        self.file_ids = [(int(id)) if id.isdigit() else id for id in file_ids]
        self.file_ids = [id for id in self.file_ids if id in info['eid'].to_numpy()]
        self.file_list = [
            file for file in file_list
            if (lambda x: (int(x)) if x.isdigit() else x)(re.split(r'[_\.]', file)[-2]) in self.file_ids
        ]
        self.file_paths = [os.path.join(path, file) for file in self.file_list]

        self.msn = []
        self.fcn = []
        self.ids = []
        self.labels = []

        # self.hc = []
        self.matched_info_rows = []
        for i in tqdm(range(len(self.file_paths)), ncols=80, desc=f'dataset loading'):
            file_path = os.path.join(self.file_paths[i])
            npy_data = np.load(file_path, allow_pickle=True).item()
            msn = npy_data['Sub_MSN']  # (116, 116)
            fcn = npy_data['Sub_FNC']  # (53, 53)
            if np.isnan(msn).any() or np.isnan(fcn).any():
                print(f'nan found in {file_path}')
                continue

            id = (lambda x: (int(x)) if x.isdigit() else x)(re.split(r'[_\.]', self.file_list[i])[-2])
            self.msn.append(torch.tensor(msn, dtype=torch.float32))
            self.fcn.append(torch.tensor(fcn, dtype=torch.float32))
            self.ids.append(id)
            self.labels.append(info[info['eid'] == id]['DXGROUP'].to_numpy()[0])

            matched_row = info[info['eid'] == id]
            self.matched_info_rows.append(matched_row)

        self.labels = np.array(self.labels)

        # save self.hc self.sz self.anx self.dep to mat
        # savemat(os.path.join(self.argv.data_path, 'mat_UKHC.mat'), {'ukhc': self.hc})

        # get_priori
        self.priori_dict = torch.load(os.path.join(self.argv.data_path, self.prior_file))
        print('done')

        print(f'0: {len(self.labels[self.labels == 0])}')
        print(f'1: {len(self.labels[self.labels == 1])}')
        self.matched_info_rows = pd.concat(self.matched_info_rows)
        age_mean_Anx_0 = self.matched_info_rows[self.matched_info_rows['DXGROUP'] == 0]['21003-0.0'].mean()
        age_std_Anx_0 = self.matched_info_rows[self.matched_info_rows['DXGROUP'] == 0]['21003-0.0'].std()
        age_mean_Anx_1 = self.matched_info_rows[self.matched_info_rows['DXGROUP'] == 1]['21003-0.0'].mean()
        age_std_Anx_1 = self.matched_info_rows[self.matched_info_rows['DXGROUP'] == 1]['21003-0.0'].std()
        print(f'age_mean_0: {age_mean_Anx_0:.2f}±{age_std_Anx_0:.2f}')
        print(f'age_mean_1: {age_mean_Anx_1:.2f}±{age_std_Anx_1:.2f}')
        print('sex_counts_0:', self.matched_info_rows[self.matched_info_rows['DXGROUP'] == 0]['31-0.0'].value_counts())
        print('sex_counts_1:', self.matched_info_rows[self.matched_info_rows['DXGROUP'] == 1]['31-0.0'].value_counts())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fcn = self.get_upper(self.fcn[idx])
        msn = self.get_upper(self.msn[idx])
        return {'id': self.ids[idx],
                'msn': self.msn[idx],
                'fcn': self.fcn[idx],
                'label': torch.tensor(self.labels[idx], dtype=torch.long),
                'fcn-msn': torch.cat([fcn, msn], dim=0)
                }

    def get_upper(self, mat: torch.Tensor) -> torch.Tensor:
        n = mat.shape[0]
        row_idx, col_idx = torch.triu_indices(n, n, offset=1)
        return mat[row_idx, col_idx]

    def save_mat(self, path=None):
        if path is None:
            path = os.path.join(self.argv.data_path, f'dataset_{self.argv.datname}.mat')
        savemat(path, {'id': self.ids,
                        'msn': torch.stack(self.msn, dim=0).detach().numpy().astype('float64'),
                        'fcn': torch.stack(self.fcn, dim=0).detach().numpy().astype('float64'),
                        'label': np.array(self.labels, dtype='float64'),
                       })
        print('save mat done')


class Datasets_MCI(Dataset):
    def __init__(self, argv,):
        super().__init__()
        self.argv = argv
        self.data_folder = 'processed_FNCandMSN_alldata'
        self.prior_file = f'biobert_biomarkers_Gem25FP0520_sim.pt'  # 3->10 2->5

        info = pd.read_csv(self.argv.data_path + f'DXSUM_PTDEMOG_11Feb2026_select.csv')
        info = info[~info['Sub_name'].duplicated()]

        path = os.path.join(self.argv.data_path, self.data_folder)  # processed_alltype_alldata
        file_list = os.listdir(path)
        file_ids = [re.split(r'[_\.]', file)[-2] for file in file_list]
        self.file_ids = [str(int(id)) if id.isdigit() else id for id in file_ids]
        self.file_ids = [id for id in self.file_ids if id in info['Sub_name'].to_numpy()]
        self.file_list = [
            file for file in file_list
            if (lambda x: str(int(x)) if x.isdigit() else x)(re.split(r'[_\.]', file)[-2]) in self.file_ids
        ]
        self.file_paths = [os.path.join(path, file) for file in self.file_list]

        self.msn = []
        self.fcn = []
        self.ids = []
        self.labels = []
        self.matched_info_rows = []
        for i in tqdm(range(len(self.file_paths)), ncols=80, desc=f'dataset loading'):
            file_path = os.path.join(self.file_paths[i])
            npy_data = np.load(file_path, allow_pickle=True).item()
            msn = npy_data['Sub_MSN']  # (116, 116)
            fcn = npy_data['Sub_FNC']  # (53, 53)
            if np.isnan(msn).any() or np.isnan(fcn).any():
                print(f'nan found in {file_path}')
                continue

            id = (lambda x: str(int(x)) if x.isdigit() else x)(re.split(r'[_\.]', self.file_list[i])[-2])
            label = info[info['Sub_name'] == id]['DIAGNOSIS'].to_numpy()[0] - 1  # 0:HC, 1:MCI, 2:AD

            # AD
            if label == 2: continue

            self.msn.append(torch.tensor(msn, dtype=torch.float32))
            self.fcn.append(torch.tensor(fcn, dtype=torch.float32))
            self.ids.append(id)
            self.labels.append(label)

            matched_row = info[info['Sub_name'] == id]
            self.matched_info_rows.append(matched_row)

        self.labels = np.array(self.labels)

        # get_priori
        self.priori_dict = torch.load(os.path.join(self.argv.data_path, self.prior_file))
        print('done')

        print(f'0: {len(self.labels[self.labels == 0])}')
        print(f'1: {len(self.labels[self.labels == 1])}')
        print(f'2: {len(self.labels[self.labels == 2])}')
        self.matched_info_rows = pd.concat(self.matched_info_rows)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fcn = self.get_upper(self.fcn[idx])
        msn = self.get_upper(self.msn[idx])
        return {'id': self.ids[idx],
                'msn': self.msn[idx],
                'fcn': self.fcn[idx],
                'label': torch.tensor(int(self.labels[idx]), dtype=torch.long),
                'fcn-msn': torch.cat([fcn, msn], dim=0)
                }

    def get_upper(self, mat: torch.Tensor) -> torch.Tensor:
        n = mat.shape[0]
        row_idx, col_idx = torch.triu_indices(n, n, offset=1)
        return mat[row_idx, col_idx]

    def save_mat(self, path=None):
        if path is None:
            path = os.path.join(self.argv.data_path, f'dataset_{self.argv.datname}.mat')
        savemat(path, {'id': self.ids,
                        'msn': torch.stack(self.msn, dim=0).detach().numpy().astype('float64'),
                        'fcn': torch.stack(self.fcn, dim=0).detach().numpy().astype('float64'),
                        'label': np.array(self.labels, dtype='float64'),
                       })
        print('save mat done')


class Datasets_AD(Dataset):
    def __init__(self, argv,):
        super().__init__()
        self.argv = argv
        self.data_folder = 'processed_FNCandMSN_alldata'
        self.prior_file = f'biobert_biomarkers_Gem25FP0520_sim.pt'  # 3->10 2->5

        info = pd.read_csv(self.argv.data_path + f'DXSUM_PTDEMOG_11Feb2026_select.csv')
        info = info[~info['Sub_name'].duplicated()]

        path = os.path.join(self.argv.data_path, self.data_folder)  # processed_alltype_alldata
        file_list = os.listdir(path)
        file_ids = [re.split(r'[_\.]', file)[-2] for file in file_list]
        self.file_ids = [str(int(id)) if id.isdigit() else id for id in file_ids]
        self.file_ids = [id for id in self.file_ids if id in info['Sub_name'].to_numpy()]
        self.file_list = [
            file for file in file_list
            if (lambda x: str(int(x)) if x.isdigit() else x)(re.split(r'[_\.]', file)[-2]) in self.file_ids
        ]
        self.file_paths = [os.path.join(path, file) for file in self.file_list]

        self.msn = []
        self.fcn = []
        self.ids = []
        self.labels = []
        self.matched_info_rows = []
        for i in tqdm(range(len(self.file_paths)), ncols=80, desc=f'dataset loading'):
            file_path = os.path.join(self.file_paths[i])
            npy_data = np.load(file_path, allow_pickle=True).item()
            msn = npy_data['Sub_MSN']  # (116, 116)
            fcn = npy_data['Sub_FNC']  # (53, 53)
            if np.isnan(msn).any() or np.isnan(fcn).any():
                print(f'nan found in {file_path}')
                continue

            id = (lambda x: str(int(x)) if x.isdigit() else x)(re.split(r'[_\.]', self.file_list[i])[-2])
            label = info[info['Sub_name'] == id]['DIAGNOSIS'].to_numpy()[0] - 1  # 0:HC, 1:MCI, 2:AD

            # MCI
            if label == 1: continue
            if label == 2: label = 1

            self.msn.append(torch.tensor(msn, dtype=torch.float32))
            self.fcn.append(torch.tensor(fcn, dtype=torch.float32))
            self.ids.append(id)
            self.labels.append(label)

            matched_row = info[info['Sub_name'] == id]
            self.matched_info_rows.append(matched_row)

        self.labels = np.array(self.labels)

        # get_priori
        self.priori_dict = torch.load(os.path.join(self.argv.data_path, self.prior_file))
        print('done')

        print(f'0: {len(self.labels[self.labels == 0])}')
        print(f'1: {len(self.labels[self.labels == 1])}')
        print(f'2: {len(self.labels[self.labels == 2])}')
        self.matched_info_rows = pd.concat(self.matched_info_rows)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fcn = self.get_upper(self.fcn[idx])
        msn = self.get_upper(self.msn[idx])
        return {'id': self.ids[idx],
                'msn': self.msn[idx],
                'fcn': self.fcn[idx],
                'label': torch.tensor(int(self.labels[idx]), dtype=torch.long),
                'fcn-msn': torch.cat([fcn, msn], dim=0)
                }

    def get_upper(self, mat: torch.Tensor) -> torch.Tensor:
        n = mat.shape[0]
        row_idx, col_idx = torch.triu_indices(n, n, offset=1)
        return mat[row_idx, col_idx]

    def save_mat(self, path=None):
        if path is None:
            path = os.path.join(self.argv.data_path, f'dataset_{self.argv.datname}.mat')
        savemat(path, {'id': self.ids,
                        'msn': torch.stack(self.msn, dim=0).detach().numpy().astype('float64'),
                        'fcn': torch.stack(self.fcn, dim=0).detach().numpy().astype('float64'),
                        'label': np.array(self.labels, dtype='float64'),
                       })
        print('save mat done')


class Datasets_PD(Dataset):
    def __init__(self, argv,):
        super().__init__()
        self.argv = argv
        self.data_folder = 'processed_FNCandMSN_alldata'
        self.prior_file = f'biobert_biomarkers_Gem25FP0520_sim.pt'  # 3->10 2->5

        info = pd.read_csv(self.argv.data_path + f'info_PD.csv')
        info = info[~info['Sub_name'].duplicated()]

        path = os.path.join(self.argv.data_path, self.data_folder)  # processed_alltype_alldata
        file_list = os.listdir(path)
        file_ids = [re.split(r'[.]', file)[-2] for file in file_list]
        self.file_ids = [str(int(id)) if id.isdigit() else id for id in file_ids]
        self.file_ids = [id for id in self.file_ids if id in info['Sub_name'].to_numpy()]
        self.file_list = [
            file for file in file_list
            if (lambda x: str(int(x)) if x.isdigit() else x)(re.split(r'[.]', file)[-2]) in self.file_ids
        ]
        self.file_paths = [os.path.join(path, file) for file in self.file_list]

        self.msn = []
        self.fcn = []
        self.ids = []
        self.labels = []
        self.matched_info_rows = []
        for i in tqdm(range(len(self.file_paths)), ncols=80, desc=f'dataset loading'):
            file_path = os.path.join(self.file_paths[i])
            npy_data = np.load(file_path, allow_pickle=True).item()
            msn = npy_data['Sub_MSN']  # (116, 116)
            fcn = npy_data['Sub_FNC']  # (53, 53)
            if np.isnan(msn).any() or np.isnan(fcn).any():
                print(f'nan found in {file_path}')
                continue

            id = (lambda x: str(int(x)) if x.isdigit() else x)(re.split(r'[.]', self.file_list[i])[-2])
            label = info[info['Sub_name'] == id]['DXGROUP2'].to_numpy()[0]

            # 0=HC, 1=PDT, 2=PD, 3=PIPD, 4=PIHC
            if label == 2: label = 1
            if label == 3: label = 1
            if label == 4: label = 0

            self.msn.append(torch.tensor(msn, dtype=torch.float32))
            self.fcn.append(torch.tensor(fcn, dtype=torch.float32))
            self.ids.append(id)
            self.labels.append(label)

            matched_row = info[info['Sub_name'] == id]
            self.matched_info_rows.append(matched_row)

        self.labels = np.array(self.labels)

        # get_priori
        self.priori_dict = torch.load(os.path.join(self.argv.data_path, self.prior_file))
        print('done')

        print(f'0: {len(self.labels[self.labels == 0])}')
        print(f'1: {len(self.labels[self.labels == 1])}')
        self.matched_info_rows = pd.concat(self.matched_info_rows)
        self.matched_info_rows_PI = self.matched_info_rows[self.matched_info_rows['participant_id'].notna()]
        self.matched_info_rows_QDU = self.matched_info_rows[self.matched_info_rows['participant_id'].isna()]

        age_mean_PI_0 = self.matched_info_rows_PI[self.matched_info_rows_PI['DXGROUP2'] == 4]['age'].mean()
        age_std_PI_0 = self.matched_info_rows_PI[self.matched_info_rows_PI['DXGROUP2'] == 4]['age'].std()
        age_mean_PI_1 = self.matched_info_rows_PI[self.matched_info_rows_PI['DXGROUP2'] == 3]['age'].mean()
        age_std_PI_1 = self.matched_info_rows_PI[self.matched_info_rows_PI['DXGROUP2'] == 3]['age'].std()
        print(f'age_mean_PIHC: {age_mean_PI_0:.2f}±{age_std_PI_0:.2f}')
        print(f'age_mean_PIPD: {age_mean_PI_1:.2f}±{age_std_PI_1:.2f}')

        print('sex_counts_PIHC:', self.matched_info_rows_PI[self.matched_info_rows_PI['DXGROUP2'] == 4]['sex'].value_counts())
        print('sex_counts_PIPD:', self.matched_info_rows_PI[self.matched_info_rows_PI['DXGROUP2'] == 3]['sex'].value_counts())

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fcn = self.get_upper(self.fcn[idx])
        msn = self.get_upper(self.msn[idx])
        return {'id': self.ids[idx],
                'msn': self.msn[idx],
                'fcn': self.fcn[idx],
                'label': torch.tensor(int(self.labels[idx]), dtype=torch.long),
                'fcn-msn': torch.cat([fcn, msn], dim=0)
                }

    def get_upper(self, mat: torch.Tensor) -> torch.Tensor:
        n = mat.shape[0]
        row_idx, col_idx = torch.triu_indices(n, n, offset=1)
        return mat[row_idx, col_idx]

    def save_mat(self, path=None):
        if path is None:
            path = os.path.join(self.argv.data_path, f'dataset_{self.argv.datname}.mat')
        savemat(path, {'id': self.ids,
                        'msn': torch.stack(self.msn, dim=0).detach().numpy().astype('float64'),
                        'fcn': torch.stack(self.fcn, dim=0).detach().numpy().astype('float64'),
                        'label': np.array(self.labels, dtype='float64'),
                       })
        print('save mat done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='dataset')
    parser.add_argument('--datname', type=str, default='ASD', help='SZ, ASD')
    argv = parser.parse_args()

    argv.data_path = f'../process/{argv.datname}_Process/'

    if argv.datname == 'SZ':
        dataset = Datasets_SZ(argv)
    elif argv.datname == 'ASD':
        dataset = Datasets_ASD(argv)
    elif argv.datname == 'Anx':
        dataset = Datasets_AnxDep(argv)
    elif argv.datname == 'Dep':
        dataset = Datasets_AnxDep(argv)
    elif argv.datname == 'ADNI':
        dataset = Datasets_MCI(argv)
    elif argv.datname == 'AD':
        dataset = Datasets_AD(argv)
    elif argv.datname == 'PD':
        dataset = Datasets_PD(argv)
    else:
        assert False

    print(dataset.__getitem__(0).keys())
