import argparse
import random,os,sys
import json
import hashlib
import tempfile
import fcntl
import h5py
import keras
import numpy as np
import csv
from scipy import stats
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing
import pandas as pd
import keras.backend as K
from keras.models import Model, Sequential
from keras.models import load_model
from keras.layers import Input,InputLayer,Multiply,ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense,Activation,Dropout,Flatten,Concatenate
from keras.layers import BatchNormalization
from keras.layers import Lambda
from keras import optimizers,utils
from keras.constraints import max_norm
from keras import regularizers
from keras.callbacks import ModelCheckpoint,Callback,EarlyStopping,History
from keras.utils import multi_gpu_model,plot_model
from keras.optimizers import Adam, SGD
from keras.models import model_from_json
import tensorflow as tf
from sklearn.metrics import average_precision_score
from scipy.stats import pearsonr
from model import KerasMultiSourceGCNModel
import hickle as hkl
import scipy.sparse as sp
import argparse
####################################Settings#################################
parser = argparse.ArgumentParser(description='Drug_response_pre')
parser.add_argument('-gpu_id', dest='gpu_id', type=str, default='0', help='GPU devices')
parser.add_argument('-use_mut', dest='use_mut', type=bool, default=True, help='use gene mutation or not')
parser.add_argument('-use_gexp', dest='use_gexp', type=bool, default=True, help='use gene expression or not')
parser.add_argument('-use_methy', dest='use_methy', type=bool, default=True, help='use methylation or not')

parser.add_argument('-israndom', dest='israndom', type=bool, default=False, help='randomlize X and A')
#hyparameters for GCN
parser.add_argument('-unit_list', dest='unit_list', nargs='+', type=int, default=[256,256,256],help='unit list for GCN')
parser.add_argument('-use_bn', dest='use_bn', type=bool, default=True, help='use batchnormalization for GCN')
parser.add_argument('-use_relu', dest='use_relu', type=bool, default=True, help='use relu for GCN')
parser.add_argument('-use_GMP', dest='use_GMP', type=bool, help='use GlobalMaxPooling for GCN')
# TRAINING INFRASTRUCTURE: Choose an output directory; --resume continues that experiment instead of creating a new one.
parser.add_argument('--experiment-dir', help='Output directory (required with --resume)')
parser.add_argument('--resume', action='store_true', help='Resume latest completed epoch in --experiment-dir')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
use_mut,use_gexp,use_methy = args.use_mut,args.use_gexp, args.use_methy
israndom=args.israndom
model_suffix = ('with_mut' if use_mut else 'without_mut')+'_'+('with_gexp' if use_gexp else 'without_gexp')+'_'+('with_methy' if use_methy else 'without_methy')

GCN_deploy = '_'.join(map(str,args.unit_list)) + '_'+('bn' if args.use_bn else 'no_bn')+'_'+('relu' if args.use_relu else 'tanh')+'_'+('GMP' if args.use_GMP else 'GAP')
model_suffix = model_suffix + '_' +GCN_deploy

####################################Constants Settings###########################
TCGA_label_set = ["ALL","BLCA","BRCA","CESC","DLBC","LIHC","LUAD",
                  "ESCA","GBM","HNSC","KIRC","LAML","LCML","LGG",
                  "LUSC","MESO","MM","NB","OV","PAAD","SCLC","SKCM",
                  "STAD","THCA",'COAD/READ']
DPATH = '../data'
Drug_info_file = '%s/GDSC/1.Drug_listMon Jun 24 09_00_55 2019.csv'%DPATH
Cell_line_info_file = '%s/CCLE/Cell_lines_annotations_20181226.txt'%DPATH
Drug_feature_file = '%s/GDSC/drug_graph_feat'%DPATH
Genomic_mutation_file = '%s/CCLE/genomic_mutation_34673_demap_features.csv'%DPATH
Cancer_response_exp_file = '%s/CCLE/GDSC_IC50.csv'%DPATH
Gene_expression_file = '%s/CCLE/genomic_expression_561celllines_697genes_demap_features.csv'%DPATH
Methylation_file = '%s/CCLE/genomic_methylation_561celllines_808genes_demap_features.csv'%DPATH
IC50_thred_file = '%s/CCLE/IC50_thred.txt'%DPATH
Max_atoms = 100


def MetadataGenerate(Drug_info_file,Cell_line_info_file,Genomic_mutation_file,Drug_feature_file,Gene_expression_file,Methylation_file,filtered,use_thred=True):
    #drug_id --> pubchem_id
    reader = csv.reader(open(Drug_info_file,'r'))
    rows = [item for item in reader]
    drugid2pubchemid = {item[0]:item[5] for item in rows if item[5].isdigit()}
    name2pubchemid = {item[1]:item[5] for item in rows if item[5].isdigit()}
    drug_names = [item.strip() for item in open(IC50_thred_file).readlines()[0].split('\t')]
    IC50_threds = [float(item.strip()) for item in open(IC50_thred_file).readlines()[1].split('\t')]
    drug2thred = {name2pubchemid[a]:b for a,b in zip(drug_names,IC50_threds) if a in name2pubchemid.keys()}

    #map cellline --> cancer type
    cellline2cancertype ={}
    for line in open(Cell_line_info_file).readlines()[1:]:
        cellline_id = line.split('\t')[1]
        TCGA_label = line.strip().split('\t')[-1]
        #if TCGA_label in TCGA_label_set:
        cellline2cancertype[cellline_id] = TCGA_label

    #load demap cell lines genomic mutation features
    mutation_feature = pd.read_csv(Genomic_mutation_file,sep=',',header=0,index_col=[0])
    cell_line_id_set = list(mutation_feature.index)

    # load drug features
    drug_pubchem_id_set = []
    drug_feature = {}
    for each in os.listdir(Drug_feature_file):
        drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat,adj_list,degree_list = hkl.load('%s/%s'%(Drug_feature_file,each))
        drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
    assert len(drug_pubchem_id_set)==len(drug_feature.values())
    
    #load gene expression faetures
    gexpr_feature = pd.read_csv(Gene_expression_file,sep=',',header=0,index_col=[0])
    
    #only keep overlapped cell lines
    mutation_feature = mutation_feature.loc[list(gexpr_feature.index)]
    
    #load methylation 
    methylation_feature = pd.read_csv(Methylation_file,sep=',',header=0,index_col=[0])
    assert methylation_feature.shape[0]==gexpr_feature.shape[0]==mutation_feature.shape[0]        
    experiment_data = pd.read_csv(Cancer_response_exp_file,sep=',',header=0,index_col=[0])
    #filter experiment data
    drug_match_list=[item for item in experiment_data.index if item.split(':')[1] in drugid2pubchemid.keys()]
    experiment_data_filtered = experiment_data.loc[drug_match_list]
    
    data_idx = []
    for each_drug in experiment_data_filtered.index:
        for each_cellline in experiment_data_filtered.columns:
            pubchem_id = drugid2pubchemid[each_drug.split(':')[-1]]
            if str(pubchem_id) in drug_pubchem_id_set and each_cellline in mutation_feature.index:
                if not np.isnan(experiment_data_filtered.loc[each_drug,each_cellline]) and each_cellline in cellline2cancertype.keys():
                    ln_IC50 = float(experiment_data_filtered.loc[each_drug,each_cellline])
                    if use_thred:
                        if pubchem_id in drug2thred.keys():
                            binary_IC50 = 1 if ln_IC50 < drug2thred[pubchem_id] else 0
                            data_idx.append((each_cellline,pubchem_id,binary_IC50,cellline2cancertype[each_cellline])) 
                    else:
                        binary_IC50 = 1 if ln_IC50 > -2 else 0
                        data_idx.append((each_cellline,pubchem_id,binary_IC50,cellline2cancertype[each_cellline])) 

    nb_celllines = len(set([item[0] for item in data_idx]))
    nb_drugs = len(set([item[1] for item in data_idx]))
    print('%d instances across %d cell lines and %d drugs were generated.'%(len(data_idx),nb_celllines,nb_drugs))
    return mutation_feature, drug_feature,gexpr_feature,methylation_feature, data_idx

#split into training and test set
def DataSplit(data_idx,ratio = 0.95):
    data_train_idx,data_test_idx = [], []
    for each_type in TCGA_label_set:
        data_subtype_idx = [item for item in data_idx if item[-1]==each_type]
        train_list = random.sample(data_subtype_idx,int(ratio*len(data_subtype_idx)))
        test_list = [item for item in data_subtype_idx if item not in train_list]
        data_train_idx += train_list
        data_test_idx += test_list
    return data_train_idx,data_test_idx

def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm
def random_adjacency_matrix(n):   
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix
def CalculateGraphFeat(feat_mat,adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0])        
    feat[:feat_mat.shape[0],:] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2    
    return [feat,adj_mat]

def FeatureExtract(data_idx,drug_feature,mutation_feature,gexpr_feature,methylation_feature):
    cancer_type_list = []
    nb_instance = len(data_idx)
    nb_mutation_feature = mutation_feature.shape[1]
    nb_gexpr_features = gexpr_feature.shape[1]
    nb_methylation_features = methylation_feature.shape[1]
    drug_data = [[] for item in range(nb_instance)]
    mutation_data = np.zeros((nb_instance,1, nb_mutation_feature,1),dtype='float32')
    gexpr_data = np.zeros((nb_instance,nb_gexpr_features),dtype='float32') 
    methylation_data = np.zeros((nb_instance, nb_methylation_features),dtype='float32') 
    target = np.zeros(nb_instance,dtype='int16')
    for idx in range(nb_instance):
        cell_line_id,pubchem_id,binary_IC50,cancer_type = data_idx[idx]
        #modify
        feat_mat,adj_list,_ = drug_feature[str(pubchem_id)]
        #fill drug data,padding to the same size with zeros
        drug_data[idx] = CalculateGraphFeat(feat_mat,adj_list)
        #randomlize X A
        mutation_data[idx,0,:,0] = mutation_feature.loc[cell_line_id].values
        gexpr_data[idx,:] = gexpr_feature.loc[cell_line_id].values
        methylation_data[idx,:] = methylation_feature.loc[cell_line_id].values
        target[idx] = binary_IC50
        cancer_type_list.append([cancer_type,cell_line_id,pubchem_id])
    return drug_data,mutation_data,gexpr_data,methylation_data,target,cancer_type_list
    
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    recal = recall(y_true, y_pred)
    return 2.0*prec*recal/(prec+recal+K.epsilon())
def average_precision(y_true, y_pred):
    return tf.py_func(average_precision_score, (y_true, y_pred), tf.double) 


# TRAINING INFRASTRUCTURE: Convert NumPy's random-generator position to JSON-compatible lists/numbers.
# Saving its position allows continuation without restarting from a seed.
def numpy_random_state(state=None):
    if state is None:
        state = np.random.get_state()
    return [state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])]


# TRAINING INFRASTRUCTURE: Rebuild the tuple and integer array NumPy expects when restoring its state.
def restore_numpy_random_state(state):
    np.random.set_state((state[0], np.asarray(state[1], dtype='uint32'),
                         state[2], state[3], state[4]))


# TRAINING INFRASTRUCTURE: Hash input shapes, types, values and order to reject changed data on resume.
# Read small blocks to limit memory use; the dataset itself is unchanged.
def input_fingerprint(arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.asarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode('ascii'))
        for start in range(0, len(array), 64):
            digest.update(np.ascontiguousarray(array[start:start + 64]).tobytes())
    return digest.hexdigest()


# TRAINING INFRASTRUCTURE: Read the saved bookkeeping dictionary and reject unknown checkpoint formats.
def read_checkpoint_state(path):
    with h5py.File(path, 'r') as checkpoint:
        state = json.loads(checkpoint.attrs['deepcdr_state'])
        if state.get('format_version') != 1:
            raise ValueError('Unsupported DeepCDR checkpoint format')
        return state


# TRAINING INFRASTRUCTURE: Store numerical arrays in numbered HDF5 datasets so their order is preserved.
def write_arrays(group, values):
    group.attrs['count'] = len(values)
    for index, value in enumerate(values):
        group.create_dataset(str(index), data=value)


# TRAINING INFRASTRUCTURE: Read arrays in numerical order; [()] also handles scalar optimizer counters.
def read_arrays(group):
    return [group[str(index)][()] for index in range(int(group.attrs['count']))]


# TRAINING INFRASTRUCTURE: Write a temporary file on the same filesystem, flush it, then replace the target.
# An interrupted write leaves the previous published checkpoint usable.
# An unfinished epoch must be repeated when resuming.
def atomic_save(path, writer):
    descriptor, temporary = tempfile.mkstemp(prefix='.saving-', suffix='.h5',
                                             dir=os.path.dirname(path))
    os.close(descriptor)
    try:
        writer(temporary)
        with open(temporary, 'rb') as saved:
            os.fsync(saved.fileno())
        os.replace(temporary, path)
        descriptor = os.open(os.path.dirname(path), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# TRAINING INFRASTRUCTURE: Fresh runs get a new directory; explicit resume requires an existing checkpoint.
# Refusing existing directories for fresh runs prevents overwriting another experiment.
def prepare_experiment():
    if args.resume and not args.experiment_dir:
        parser.error('--resume requires --experiment-dir')
    if args.experiment_dir:
        directory = os.path.abspath(args.experiment_dir)
        if args.resume:
            if not os.path.isfile(os.path.join(directory, 'latest_checkpoint.h5')):
                parser.error('No latest_checkpoint.h5 in the experiment directory')
        else:
            os.makedirs(directory, exist_ok=False)
    else:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..',
                                           'checkpoint', 'experiments'))
        os.makedirs(root, exist_ok=True)
        directory = tempfile.mkdtemp(prefix=time.strftime('%Y%m%d-%H%M%S-'), dir=root)
    # TRAINING INFRASTRUCTURE: Keep an exclusive lock open until evaluation finishes to prevent concurrent saves.
    # The operating system releases the lock when the handle closes or the process dies.
    lock = open(os.path.join(directory, '.training.lock'), 'a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        lock.close()
        raise RuntimeError('Another process is using this experiment directory')
    print('Experiment directory: %s' % directory)
    return directory, lock


class MyCallback(Callback):
    def __init__(self, validation_data, patience, experiment_dir, run_config,
                 preprocessing_numpy_state, resume_state=None):
        super(MyCallback, self).__init__()
        self.x_val, self.y_val = validation_data
        self.patience = patience
        self.experiment_dir = experiment_dir
        self.run_config = run_config
        self.preprocessing_numpy_state = preprocessing_numpy_state
        self.best_weight = None
        self.wait = 0
        self.stopped_epoch = None
        self.best = -np.inf
        self.best_epoch = None
        self.completed_epochs = 0
        # TRAINING INFRASTRUCTURE: Restore best AUROC, consecutive non-improving epochs (wait), and epoch counters.
        # This preserves the original early-stopping rule across restarts.
        if resume_state is not None:
            self.wait = resume_state['wait']
            self.stopped_epoch = resume_state['stopped_epoch']
            self.best = resume_state['best_auroc']
            self.best_epoch = resume_state['best_epoch']
            self.completed_epochs = resume_state['completed_epochs']

    # TRAINING INFRASTRUCTURE: Do not reset counters here: fit() calls this hook again after a restart.
    def on_train_begin(self, logs=None):
        pass

    # TRAINING INFRASTRUCTURE: Select best weights for evaluation without changing the last-epoch checkpoint.
    # Earlier best weights must not be paired with the last epoch's Adam state.
    def on_train_end(self, logs=None):
        if self.best_weight is not None:
            self.model.set_weights(self.best_weight)
        if self.stopped_epoch is not None:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    # TRAINING INFRASTRUCTURE: Export the best validation model separately from the checkpoint used to resume.
    def save_best_model(self):
        atomic_save(os.path.join(self.experiment_dir, 'best_model.h5'),
                    lambda path: self.model.save(path))

    # TRAINING INFRASTRUCTURE: Save one completed epoch's weights, optimizer, counters and random states together.
    # Embed best weights too, allowing best_model.h5 to be repaired after an interruption.
    def save_checkpoint(self):
        state = dict(format_version=1, completed_epochs=self.completed_epochs,
                     best_auroc=float(self.best), best_epoch=self.best_epoch,
                     best_model='best_model.h5', wait=self.wait,
                     patience=self.patience, stopped_epoch=self.stopped_epoch,
                     run_config=self.run_config,
                     preprocessing_numpy_state=self.preprocessing_numpy_state,
                     python_random_state=random.getstate(),
                     numpy_random_state=numpy_random_state())
        # TRAINING INFRASTRUCTURE: Adam needs its update counter and gradient-history arrays, not just model weights.
        optimizer_weights = self.model.optimizer.get_weights()
        if not optimizer_weights or self.best_weight is None:
            raise RuntimeError('Cannot save incomplete model/optimizer state')
        # TRAINING INFRASTRUCTURE: Append metadata and optimizer/best arrays to the same model-weights file.
        # atomic_save publishes it only after the complete file has been closed.
        def writer(path):
            self.model.save_weights(path)
            with h5py.File(path, 'a') as checkpoint:
                checkpoint.attrs['deepcdr_state'] = json.dumps(state, allow_nan=False)
                write_arrays(checkpoint.create_group('deepcdr_optimizer'), optimizer_weights)
                write_arrays(checkpoint.create_group('deepcdr_best_weights'), self.best_weight)
        atomic_save(os.path.join(self.experiment_dir, 'latest_checkpoint.h5'), writer)

    def on_epoch_end(self, epoch, logs=None):
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        precision, recall, _ = metrics.precision_recall_curve(self.y_val, y_pred_val)
        pr_val = -np.trapz(precision, recall)
        print('roc-val: %.4f, pr-val:%.4f' % (roc_val, pr_val))
        # TRAINING INFRASTRUCTURE: Reject invalid AUROC values before advancing the checkpoint.
        if not np.isfinite(roc_val):
            raise ValueError('Non-finite validation AUROC; checkpoint not advanced')
        if roc_val > self.best:
            self.best = roc_val
            self.wait = 0
            self.best_epoch = epoch + 1
            self.best_weight = self.model.get_weights()
            # TRAINING INFRASTRUCTURE: Save each AUROC improvement immediately rather than waiting for training to finish.
            self.save_best_model()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
        # TRAINING INFRASTRUCTURE: Keras epoch indices start at 0; save the completed count after updating stopping state.
        self.completed_epochs = epoch + 1
        self.save_checkpoint()


def ModelTraining(model,X_drug_data_train,X_mutation_data_train,X_gexpr_data_train,X_methylation_data_train,Y_train,validation_data,experiment_dir,preprocessing_numpy_state,nb_epoch=100):
    optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer = optimizer,loss='binary_crossentropy',metrics=['accuracy',precision,recall,f1_score,average_precision])
    X_drug_feat_data_train = np.array([item[0] for item in X_drug_data_train])
    X_drug_adj_data_train = np.array([item[1] for item in X_drug_data_train])
    train_inputs = [X_drug_feat_data_train, X_drug_adj_data_train,
                    X_mutation_data_train, X_gexpr_data_train, X_methylation_data_train]
    # TRAINING INFRASTRUCTURE: Record existing settings, library versions and input signatures for resume checks.
    run_config = dict(model_suffix=model_suffix, israndom=israndom,
                      input_fingerprint=input_fingerprint(train_inputs + [Y_train] +
                                                          validation_data[0] + [validation_data[1]]),
                      optimizer=optimizer.get_config(), loss='binary_crossentropy',
                      batch_size=64, epochs=nb_epoch, patience=10,
                      keras=keras.__version__, tensorflow=tf.__version__,
                      python=list(sys.version_info[:3]))
    path = os.path.join(experiment_dir, 'latest_checkpoint.h5')
    state = read_checkpoint_state(path) if args.resume else None
    if state is not None:
        # TRAINING INFRASTRUCTURE: Reject changed experiments and inconsistent epoch/early-stopping records.
        if state['run_config'] != run_config or state['patience'] != 10:
            raise ValueError('Resume configuration or input data differs from checkpoint')
        if not 1 <= state['completed_epochs'] <= nb_epoch:
            raise ValueError('Invalid completed epoch count in checkpoint')
        if (not np.isfinite(state['best_auroc']) or
                not 0 <= state['best_auroc'] <= 1 or
                not 1 <= state['best_epoch'] <= state['completed_epochs'] or
                not 0 <= state['wait'] <= 10 or
                (state['stopped_epoch'] is None) != (state['wait'] < 10) or
                (state['stopped_epoch'] is not None and
                 state['stopped_epoch'] != state['completed_epochs'] - 1)):
            raise ValueError('Invalid early-stopping state in checkpoint')
    callback = MyCallback(validation_data, 10, experiment_dir, run_config,
                          preprocessing_numpy_state, state)
    if state is not None:
        # TRAINING INFRASTRUCTURE: Legacy Keras creates Adam's state variables here without executing training updates.
        # Restore last-epoch model weights and Adam state into this same compiled model.
        model._make_train_function()
        model.load_weights(path)
        with h5py.File(path, 'r') as checkpoint:
            saved_optimizer = read_arrays(checkpoint['deepcdr_optimizer'])
            callback.best_weight = read_arrays(checkpoint['deepcdr_best_weights'])
        # TRAINING INFRASTRUCTURE: Require matching optimizer array counts, shapes and types, then verify restored values.
        # Any failure stops execution instead of silently using a fresh optimizer.
        current_optimizer = model.optimizer.get_weights()
        if (len(current_optimizer) != len(saved_optimizer) or
                any(a.shape != b.shape or a.dtype != b.dtype
                    for a, b in zip(current_optimizer, saved_optimizer))):
            raise ValueError('Optimizer state is incompatible; refusing a fresh optimizer')
        model.optimizer.set_weights(saved_optimizer)
        if not all(np.array_equal(a, b) for a, b in
                   zip(model.optimizer.get_weights(), saved_optimizer)):
            raise RuntimeError('Optimizer restoration verification failed')
        # TRAINING INFRASTRUCTURE: Repair the best export from embedded best weights, then restore last-epoch weights.
        # Omit optimizer state from this export because it belongs to the last epoch.
        latest_weights = model.get_weights()
        model.set_weights(callback.best_weight)
        atomic_save(os.path.join(experiment_dir, 'best_model.h5'),
                    lambda filename: model.save(filename, include_optimizer=False))
        model.set_weights(latest_weights)
        # TRAINING INFRASTRUCTURE: Restore epoch-boundary random states after rebuilding features and the model.
        # JSON lists become tuples for Python's generator. This cannot restore TensorFlow
        # dropout state or guarantee identical legacy Keras shuffling/GPU results.
        py_state = state['python_random_state']
        random.setstate((py_state[0], tuple(py_state[1]), py_state[2]))
        restore_numpy_random_state(state['numpy_random_state'])
        print('Restored epoch %d, Adam iteration %d, best AUROC %.6f, wait %d' %
              (callback.completed_epochs, int(saved_optimizer[0]), callback.best, callback.wait))
        # TRAINING INFRASTRUCTURE: Already stopped/finished runs only select best weights for evaluation.
        if callback.stopped_epoch is not None or callback.completed_epochs >= nb_epoch:
            callback.set_model(model)
            callback.on_train_end()
            return model
    # TRAINING INFRASTRUCTURE: initial_epoch is the completed count: after 20 epochs, continue with displayed Epoch 21.
    # epochs remains the total limit (100), not an extra 100 epochs.
    model.fit(x=train_inputs,y=Y_train,batch_size=64,epochs=nb_epoch,
              initial_epoch=callback.completed_epochs,validation_split=0,callbacks=[callback])
    return model


def ModelEvaluate(model,X_drug_data_test,X_mutation_data_test,X_gexpr_data_test,X_methylation_data_test,Y_test,cancer_type_test_list,file_path):
    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)#nb_instance * Max_stom * Max_stom    
    Y_pred = model.predict([X_drug_feat_data_test,X_drug_adj_data_test,X_mutation_data_test,X_gexpr_data_test,X_methylation_data_test])
    auROC_all = metrics.roc_auc_score(Y_test, Y_pred)
    fpr,tpr,_,= metrics.roc_curve(Y_test,Y_pred)
    precision,recall,_, = metrics.precision_recall_curve(Y_test,Y_pred)
    auPR_all = -np.trapz(precision,recall)
    print("The overall AUC and auPR is %.4f and %.4f."%(auROC_all,auPR_all))

    

# TRAINING INFRASTRUCTURE: Hold the experiment lock through training/evaluation and release it even on errors.
def main():
    experiment_dir, experiment_lock = prepare_experiment()
    try:
        return run_experiment(experiment_dir)
    finally:
        experiment_lock.close()


def run_experiment(experiment_dir):
    random.seed(0)
    # TRAINING INFRASTRUCTURE: Keep the pre-feature random state separately from the epoch-boundary state.
    # On resume, replay randomized graph features while leaving the original seed/split logic intact.
    preprocessing_numpy_state = numpy_random_state()
    if args.resume:
        state = read_checkpoint_state(os.path.join(experiment_dir, 'latest_checkpoint.h5'))
        preprocessing_numpy_state = state['preprocessing_numpy_state']
        restore_numpy_random_state(preprocessing_numpy_state)
    mutation_feature, drug_feature,gexpr_feature,methylation_feature, data_idx = MetadataGenerate(Drug_info_file,Cell_line_info_file,Genomic_mutation_file,Drug_feature_file,Gene_expression_file,Methylation_file,False)
    data_train_idx,data_test_idx = DataSplit(data_idx)
    #Extract features for training and test 
    X_drug_data_train,X_mutation_data_train,X_gexpr_data_train,X_methylation_data_train,Y_train,cancer_type_train_list = FeatureExtract(data_train_idx,drug_feature,mutation_feature,gexpr_feature,methylation_feature)
    X_drug_data_test,X_mutation_data_test,X_gexpr_data_test,X_methylation_data_test,Y_test,cancer_type_test_list = FeatureExtract(data_test_idx,drug_feature,mutation_feature,gexpr_feature,methylation_feature)

    X_drug_feat_data_test = [item[0] for item in X_drug_data_test]
    X_drug_adj_data_test = [item[1] for item in X_drug_data_test]
    X_drug_feat_data_test = np.array(X_drug_feat_data_test)#nb_instance * Max_stom * feat_dim
    X_drug_adj_data_test = np.array(X_drug_adj_data_test)#nb_instance * Max_stom * Max_stom  
    
    validation_data = [[X_drug_feat_data_test,X_drug_adj_data_test,X_mutation_data_test,X_gexpr_data_test,X_methylation_data_test],Y_test]
    model = KerasMultiSourceGCNModel(use_mut,use_gexp,use_methy,regr=False).createMaster(X_drug_data_train[0][0].shape[-1],X_mutation_data_train.shape[-2],X_gexpr_data_train.shape[-1],X_methylation_data_train.shape[-1],args.unit_list,args.use_relu,args.use_bn,args.use_GMP)
    print('Begin training...')
    model = ModelTraining(model,X_drug_data_train,X_mutation_data_train,X_gexpr_data_train,X_methylation_data_train,Y_train,validation_data,experiment_dir,preprocessing_numpy_state,nb_epoch=100)
    ModelEvaluate(model,X_drug_data_test,X_mutation_data_test,X_gexpr_data_test,X_methylation_data_test,Y_test,cancer_type_test_list,os.path.join(experiment_dir, 'evaluation.log'))

if __name__=='__main__':
    main()
    
