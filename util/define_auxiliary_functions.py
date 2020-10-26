#!/usr/bin/env python3

import os
import sys
import datetime

import numpy as np
import pandas as pd
import pathlib

from enum import Enum

class Label(Enum):
    FAUST = 0
    HANDSPREIZEN = 1
    PALMARFLEXION = 2
    DORSALEXTENSION = 3
    SUPINATION = 4
    PRONATION = 5
    NEUTRALNULL = 6
    DAUMENHOCH = 7
    PISTOLE = 8
    PEACE = 9
    POMMESGABEL = 10
    MITTELFINGER = 11

    def __str__(self):
        return str(self.name)


def compute_calibrations(dataset, save_path='data'):
    feat_cols = [i for i in dataset.columns if 'electrode' in i]
    full_df_max = dataset[feat_cols].values.flatten().max()
    dataset_scaled = dataset.copy()
    dataset_scaled[feat_cols] = dataset[feat_cols] / full_df_max

    pd.options.mode.chained_assignment = None

    calib_cols = [w.replace('setup', 'calib') for w in feat_cols]

    ## Local Calibration
    calibrated = []
    sessions = dataset.session_uid.unique()
    for session in sessions:
        session_set = dataset_scaled.query("session_uid==@session")
        if len(session_set.iteration.unique()) > 1:
            for it in session_set.iteration.unique():
                iter_set = dataset_scaled.query("session_uid==@session and iteration==@it")
                new_vals = iter_set.loc[:, feat_cols] - iter_set.query("label==6").loc[:, feat_cols].mean()
                iter_set.loc[:, feat_cols] = new_vals
                calibrated.append(iter_set)
        else:
            session_set.loc[:, feat_cols] = session_set.loc[:, feat_cols] - session_set.query("label==6").loc[:, feat_cols].mean()
            calibrated.append(session_set)
    local_calib_set = pd.DataFrame().append(calibrated, ignore_index=True)

    ## Global Calibration
    calibrated = []
    sessions = dataset.session_uid.unique()
    for session in sessions:
        session_set = dataset_scaled.query("session_uid==@session")
        session_set.loc[:, feat_cols] = session_set.loc[:, feat_cols] - session_set.query("label==6 and iteration==0").loc[:, feat_cols].mean()
        calibrated.append(session_set)
    global_calib_set = pd.DataFrame().append(calibrated, ignore_index=True)

    ## Auto Calibration
    calib_dict = {w:w.replace('setup', 'calib') for w in feat_cols}

    calibrated = []
    sessions = dataset_scaled.session_uid.unique()
    for session in sessions:
        session_set = dataset_scaled.query("session_uid==@session")
        if len(session_set.iteration.unique()) > 1:
            for it in session_set.iteration.unique():
                iter_set = dataset_scaled.query("session_uid==@session and iteration==@it")
                new_vals = iter_set.loc[:, feat_cols] * 0 + iter_set.query("label==6").loc[:, feat_cols].mean().values
                iter_set[calib_cols] = new_vals.rename(columns=calib_dict)
                calibrated.append(iter_set)
        else:
            session_set[calib_cols] = iter_set.loc[:, feat_cols] * 0 + session_set.query("label==6").loc[:, feat_cols].mean().values
            calibrated.append(session_set)
    auto_calib_set = pd.DataFrame().append(calibrated, ignore_index=True)

        # Check if everything worked as expected
    assert local_calib_set.shape == global_calib_set.shape and global_calib_set.shape == dataset.shape and auto_calib_set.shape == (dataset.shape[0], dataset.shape[1] + len(feat_cols)), "Something went wrong during calibration!"

    print("Saving calibrated sets...")
    dataset_scaled.to_csv(os.path.join(save_path, "data_raw.csv"), index=False, header=True)
    local_calib_set.to_csv(os.path.join(save_path, "data_local.csv"), index=False, header=True)
    global_calib_set.to_csv(os.path.join(save_path, "data_global.csv"), index=False, header=True)
    auto_calib_set.to_csv(os.path.join(save_path, "data_auto.csv"), index=False, header=True)

    return dataset_scaled, global_calib_set, local_calib_set, auto_calib_set


def load_complete_dataset(directory_path: str):
    total_data = pd.DataFrame()
    for name in [n for n in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, n))]:
        total_data = total_data.append(load_subject(name, directory_path))
    return total_data

def load_subject(name, root_dir):
    subject_path = os.path.join(root_dir, name)

    prim_cols = ["setup_%d" % i for i in np.arange(0,16)]
    cols = []
    for i in np.arange(0, 16):
        cols.extend([c+"electrode_%d" % i for c in prim_cols])
    cols.append("label")

    subj_data = pd.DataFrame()
    for session_str in os.listdir(subject_path):
        file_data = pd.DataFrame()

        for dfile in os.listdir(os.path.join(subject_path, session_str)):
            f_data = pd.read_csv(
                os.path.join(subject_path, session_str, dfile),
            )

            f_data['filename'] = dfile
            f_data.rename({'gesture_key': 'label'}, axis=1, inplace=True)
            file_data = file_data.append(f_data, ignore_index=True)

        file_data["session"] = session_str
        file_data['session_uid'] = "%s%s" % (name, session_str)

        subj_data = subj_data.append(
            file_data,
            ignore_index=True,
        )

    subj_data['subject'] = name
    return subj_data


def get_interactive_data(path='data', compute_new=False):
    cond = list(map(os.path.isfile, [
        os.path.join(path, 'data_interactive_raw.csv'),
        os.path.join(path, 'data_interactive_global.csv'),
        os.path.join(path, 'data_interactive_local.csv')
    ]))

    if compute_new or not np.all(cond):
        print("Computing data for interactive view")
        calibs_available = list(map(os.path.isfile, [
            os.path.join(path, 'data_raw.csv'),
            os.path.join(path, 'data_global.csv'),
            os.path.join(path, 'data_local.csv')
        ]))

        if not calibs_available:
            print("Precomputed calibrations unavailable, loading dataset from '%s'" % (path))
            dataset = load_complete_dataset(path)

            # Add filerow and iteration information
            rows = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            rows = np.concatenate([rows for i in np.arange(dataset.shape[0] / 10)])
            dataset['filerow'] = rows

            # Identify iterations
            tmp_set = dataset.copy()
            tmp_set['iteration'] = dataset.apply(lambda x: 0 if not x.filename[-5].isnumeric() else int(x.filename[-5]), axis=1)
            tmp_set['filename'] = dataset.apply(lambda x: x.filename[:-4] if not x.filename[-5].isnumeric() else x.filename[:-5], axis=1)
            dataset = tmp_set.copy()
            del tmp_set

            dataset_scaled, global_calib_set, local_calib_set, _ = compute_calibrations(
                dataset, root_dir
            )
            del dataset
        else:
            print('loading precomputed calibration sets')
            dataset_scaled = pd.read_csv(os.path.join(path, "data_raw.csv"), header=0)
            local_calib_set = pd.read_csv(os.path.join(path, "data_local.csv"), header=0)
            global_calib_set = pd.read_csv(os.path.join(path, "data_global.csv"), header=0)

        from sklearn.decomposition import PCA
        feat_cols = [i for i in dataset_scaled.columns if 'electrode' in i]

        dataset_scaled = dataset_scaled.query("subject!='F'")
        global_calib_set = global_calib_set.query("subject!='F'")
        local_calib_set = local_calib_set.query("subject!='F'")

        print("Running PCA...")
        pca_ = PCA(n_components=2)
        raw_pca = pca_.fit_transform(dataset_scaled[feat_cols])
        dataset_scaled['pca_x'] = raw_pca[:, 0]
        dataset_scaled['pca_y'] = raw_pca[:, 1]

        pca_local_calib = pca_.fit_transform(local_calib_set[feat_cols])
        local_calib_set['pca_x'] = pca_local_calib[:, 0]
        local_calib_set['pca_y'] = pca_local_calib[:, 1]

        pca_global_calib = pca_.fit_transform(global_calib_set[feat_cols])
        global_calib_set['pca_x'] = pca_global_calib[:, 0]
        global_calib_set['pca_y'] = pca_global_calib[:, 1]
        print("Done")

        del pca_, raw_pca, pca_local_calib, pca_global_calib

        from sklearn.manifold import TSNE
        import time
        perps = [90]

        print("Running t-SNE calculations")
        for i, perp in enumerate(perps):
            print("\tComputing perplexity %d..." % perp)
            tsne = TSNE(perplexity=perp, n_iter=5000, n_jobs=-1, random_state=42, init='random')
            start = time.time()
            fitted = tsne.fit_transform(dataset_scaled[feat_cols])
            dataset_scaled['tsne_%d_x' % perp] = fitted[:, 0]
            dataset_scaled['tsne_%d_y' % perp] = fitted[:, 1]
            elapsed = time.time() - start
            print("\t   done with raw data (elapsed: %s)" % elapsed)

            start = time.time()
            fitted = tsne.fit_transform(local_calib_set[feat_cols])
            local_calib_set['tsne_%d_x' % perp] = fitted[:, 0]
            local_calib_set['tsne_%d_y' % perp] = fitted[:, 1]
            elapsed = time.time() - start
            print("\t   done with locally calibrated data (elapsed: %s)" % elapsed)

            start = time.time()
            fitted = tsne.fit_transform(global_calib_set[feat_cols])
            global_calib_set['tsne_%d_x' % perp] = fitted[:, 0]
            global_calib_set['tsne_%d_y' % perp] = fitted[:, 1]
            elapsed = time.time() - start
            print("\t   done with globally calibrated data (elapsed: %s)" % elapsed)

        del fitted, tsne, start, elapsed

        df = dataset_scaled.drop(columns=feat_cols)
        df.to_csv(os.path.join(path, "data_interactive_raw.csv"))

        df_global = global_calib_set.drop(columns=feat_cols)
        df_global.to_csv(os.path.join(path, "data_interactive_global.csv"))

        df_local = local_calib_set.drop(columns=feat_cols)
        df_local.to_csv(os.path.join(path, "data_interactive_local.csv"))

        del dataset_scaled, global_calib_set, local_calib_set
    else:
        print("Loading existing interactive sets")
        df = pd.read_csv(os.path.join(path, "data_interactive_raw.csv"))
        df_local = pd.read_csv(os.path.join(path, "data_interactive_local.csv"))
        df_global = pd.read_csv(os.path.join(path, "data_interactive_global.csv"))

    df.label = df.label.astype(float).astype(int).astype(str)
    df_global.label = df_global.label.astype(float).astype(int).astype(str)
    df_local.label = df_local.label.astype(float).astype(int).astype(str)

    df.iteration = df.iteration + 1
    df_global.iteration = df_global.iteration + 1
    df_local.iteration = df_local.iteration + 1

    return df, df_global, df_local
