import argparse
import gc
import keras
import os
import pandas as pd
from keras import Sequential, Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers import *
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from tqdm import tqdm
import tensorflow as tf
from keras.models import Sequential, Model, Input
from keras.layers import Input, LSTM, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D, AveragePooling2D, \
    concatenate, \
    Activation, ZeroPadding2D, Lambda, Embedding, Permute, Concatenate
from keras.layers import add, Flatten
from sklearn.externals.joblib import dump, load

pd.set_option('display.max_columns', None)
import warnings

warnings.filterwarnings("ignore")


#####################################CNN add  particle_category  ######################

def reduce_mem_usage(df, df_col=None, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        if df_col and col != df_col:
            continue
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.float16)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float16)
                else:
                    df[col] = df[col].astype(np.float16)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def Conv2d_BN(x, nb_filter, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


def Net(l=10, cols_len=10):
    emb_size = 6
    input_1 = Input(shape=(l, cols_len, 1), name='input_1')
    input_1_1 = Lambda(lambda x: x[:, :, 1:, :])(input_1)
    input_1_2 = Lambda(lambda x: x[:, :, 0, :])(input_1)
    cate_emb = Embedding(output_dim=emb_size, input_dim=15)(input_1_2)
    cate_emb = Permute((1, 3, 2))(cate_emb)
    X = Concatenate(axis=2)([input_1_1, cate_emb])
    X = Conv2d_BN(X, nb_filter=64)
    shortcut = Conv2d_BN(X, nb_filter=64, kernel_size=(1, cols_len + emb_size - 1))
    X = Conv2d_BN(X, nb_filter=64, kernel_size=(1, cols_len + emb_size - 1))
    X = Conv2d_BN(X, nb_filter=128)
    X = Conv2d_BN(X, nb_filter=512)
    X = Concatenate(axis=-1)([X, shortcut])
    X = Conv2d_BN(X, nb_filter=64)
    X = Conv2d_BN(X, nb_filter=128)
    X = Conv2d_BN(X, nb_filter=512)
    # X = Conv2d_BN(X, nb_filter=128, kernel_size=(8, 1), padding='same')
    X = AveragePooling2D(pool_size=(l, 1))(X)
    X = BatchNormalization()(Dropout(0.2)(Dense(128, activation='relu')(Flatten()(X))))
    X = Dense(4, activation='softmax')(X)
    model = Model([input_1], X)

    return model


parser = argparse.ArgumentParser(description='nn')
parser.add_argument('--bs', type=int, default=512)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--split', type=int, default=5)
parser.add_argument('--maxlen', type=int, default=280)
parser.add_argument('--order', type=str, default='jet_energy')

args = parser.parse_args()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

data_jet = pd.read_csv('../data/jet_complex_data/complex_test_R04_jet.csv')
data_particle = pd.read_csv('../data/jet_complex_data/complex_test_R04_particle.csv')
data = pd.merge(data_particle, data_jet, on='jet_id', how='left')

cate_lb = LabelEncoder()
cate_lb.classes_ = np.load('cateEncoder.npy')
data['particle_category'] = cate_lb.transform(data['particle_category'])
data['particle_category'] = data['particle_category'].astype(np.int8)
del data_particle, data_jet
gc.collect()

cols = [i for i in data.columns if i not in ['event_id', 'label', 'particle_category', 'jet_id']]
data[cols] = data[cols].astype(np.float32)
data = reduce_mem_usage(data, 'jet_id')
data = reduce_mem_usage(data, 'event_id')
print('loaded data')

data['pT'] = np.sqrt(data['particle_px'] ** 2 + data['particle_py'] ** 2)
data['pT2'] = data['particle_px'] ** 2 + data['particle_py'] ** 2
##############particle info
data['etot'] = np.sqrt(data['particle_px'] ** 2 + data['particle_py'] ** 2 + data['particle_pz'] ** 2)
data['eta'] = np.arccos(data['particle_px'] / data['pT'])
data['theta'] = 0.5 * np.log((data['etot'] + data['particle_pz']) / (data['etot'] - data['particle_pz']))
###########jet info
data['PT'] = np.sqrt(data['jet_px'] ** 2 + data['jet_py'] ** 2)
data['Etot'] = np.sqrt(data['jet_px'] ** 2 + data['jet_py'] ** 2 + data['jet_pz'] ** 2)
data['Eta'] = np.arccos(data['jet_px'] / data['PT'])
data['Theta'] = 0.5 * np.log((data['Etot'] + data['jet_pz']) / (data['Etot'] - data['jet_pz']))
######2020/1/6
data['PT/E'] = data['pT'] / data['particle_energy']
data['mass/pT'] = data['particle_mass'] / data['pT']
data['PZ/energy'] = data['particle_pz'] / data['particle_energy']

data['Ra'] = np.sqrt((data['Theta'] - data['theta']) ** 2 + (data['Eta'] - data['eta']) ** 2)
data['Lesub'] = data['Ra'] * data['pT']

groupByjet = data.groupby('jet_id')
data['pT2_sum'] = groupByjet['pT2'].transform(sum)
data['pT_sum'] = groupByjet['pT'].transform(sum)
data['LeadPT/E'] = groupByjet['PT/E'].transform(max)
data['Leadmass/pT'] = groupByjet['mass/pT'].transform(max)
data['leadPZ/energy'] = groupByjet['PZ/energy'].transform(max)

############2020/1/7 FROM ALICE ref Small radius jet shapes in pp and Pb-Pb collisions at ALICE
data['Lesub-sum'] = groupByjet['Lesub'].transform(sum)
data['g'] = data['Lesub-sum'] / data['PT']
data['ptd'] = np.sqrt(data['pT2_sum']) / data['pT_sum']
del groupByjet

groupByparticle = data.groupby(['jet_id', 'particle_category'])
data['pt2-par_sum'] = groupByparticle['pT2'].transform(sum)
data['pt-par_sum'] = groupByparticle['pT'].transform(sum)
data['ptd-par'] = np.sqrt(data['pt2-par_sum']) / data['pt-par_sum']
del groupByparticle

### 2020/3/12 
data['particle_cos(x)'] = data['particle_px'] / data['etot']
data['particle_cos(y)'] = data['particle_py'] / data['etot']
data['particle_cos(z)'] = data['particle_pz'] / data['etot']
data['particle_angle(x)'] = np.arccos(data['particle_cos(x)'])
data['particle_angle(y)'] = np.arccos(data['particle_cos(y)'])
data['particle_angle(z)'] = np.arccos(data['particle_cos(z)'])

data['jet_cos(x)'] = data['jet_px'] / data['Etot']
data['jet_cos(y)'] = data['jet_py'] / data['Etot']
data['jet_cos(z)'] = data['jet_pz'] / data['Etot']
data['jet_angle(x)'] = np.arccos(data['jet_cos(x)'])
data['jet_angle(y)'] = np.arccos(data['jet_cos(y)'])
data['jet_angle(z)'] = np.arccos(data['jet_cos(z)'])

### 2020/3/14
# data['jet_yz'] = np.sqrt(data['jet_py']**2 + data['jet_pz']**2)
# data['jet_xz'] = np.sqrt(data['jet_px']**2 + data['jet_pz']**2)
# data['jet_sita_xy'] = np.arctan(data['jet_py']/data['jet_px'])
# data['jet_sita_z'] = np.arcsin(data['jet_pz']/data['Etot']) 

# data['particle_yz'] = np.sqrt(data['particle_py']**2 + data['particle_pz']**2)
# data['particle_xz'] = np.sqrt(data['particle_px']**2 + data['particle_pz']**2)
# data['particle_sita_xy'] = np.arctan(data['particle_py']/data['particle_px'])
# data['particle_sita_z'] = np.arcsin(data['particle_pz']/data['etot']) 

cols = [i for i in data.columns if i not in ['event_id', 'label', 'particle_category', 'jet_id']]

print('Scaler....')
for col in cols:
    scaler = load('scaler/%s.bin'%col.replace('/', '_'))
    data[[col]] = scaler.transform(data[[col]])
    data = reduce_mem_usage(data, col)

test = data.copy()#data[data['label'].isnull()].reset_index()
del data
gc.collect()
cols = ['particle_category'] + cols
print('-------------------')
print(cols)
print(len(cols))

test_data_use = test.groupby('event_id')
test_event = np.array(test_data_use['event_id'].max())
# train_label = np.array(test_data_use['label'].mean())
# encoder = LabelEncoder()
# train_label = encoder.fit_transform(train_label)
test_data_use = pad_sequences(np.array(test_data_use.apply(lambda row: (
    np.array(row[cols])))), dtype=np.float16, value=0, padding='post', maxlen=args.maxlen)
l = test_data_use.shape[1]
print(f'maxlen:{l}')
del test
gc.collect()

test_data_use = test_data_use.reshape(list(test_data_use.shape)[:3] + [1])
print(f"test_data_use:{test_data_use.shape}")

kfold = StratifiedKFold(n_splits=args.split, shuffle=True, random_state=1024)
prediction_proba = np.zeros([test_data_use.shape[0], 4])

# for fold, (tr_idx, val_idx) in enumerate(kfold.split(test_data_use, train_label)):
for fold in range(5):
    print(f"fold {fold}")
    # train_labels = to_categorical(train_label, num_classes=4)
    # X_train, X_validate, label_train, label_validate = test_data_use[tr_idx], test_data_use[val_idx], \
    #                                                   train_labels[tr_idx], train_labels[val_idx]
    model = Net(test_data_use.shape[1], test_data_use.shape[2])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
    model.load_weights(f'./model_lstm/lstm_{args.order}_{args.bs}_{args.patience}_{fold}.hdf5')
    prediction_proba += model.predict(test_data_use, verbose=0, batch_size=1024)
    # print(accuracy_score(label_validate.argmax(axis=-1), prediction_proba[val_idx].argmax(axis=-1)))
    # del X_train
    # del X_validate
    # del label_train
    # del label_validate
    del model
    K.clear_session()

prediction_proba /= kfold.n_splits
prediction = prediction_proba.argmax(axis=-1)
pred = pd.DataFrame(prediction, columns=['label_pre'])
pred['event_id'] = test_event
test_data = pd.read_csv('../data/jet_complex_data/complex_test_R04_jet.csv')[['event_id', 'jet_id']]
# test_data['event_id'] = event_lb.transform(test_data['event_id'])
pred = pd.merge(test_data, pred, on='event_id', how='left')
sub = pred[['jet_id', 'label_pre']]
sub.columns = ['id', 'label']
encoder = LabelEncoder().fit([1,4,5,21])
sub['label'] = encoder.inverse_transform(sub['label'])
sub.to_csv(f'sub_{args.order}_{args.bs}_{args.patience}_{args.split}_maxlen{args.maxlen}_5.csv', index=False)

preds = pd.DataFrame(prediction_proba, columns=['p1', 'p4', 'p5', 'p21'])
preds['event_id'] = test_event
preds.to_csv(f'test_preds_maxlen{args.maxlen}_5.csv', index=False)

prediction_proba_new = prediction_proba.copy()
prediction_proba_new[:, 3] = prediction_proba[:, 3] * 1.0
prediction_proba_new[:, 2] = prediction_proba[:, 2] * 1.0
prediction_proba_new[:, 1] = prediction_proba[:, 1] * 1.01
prediction_proba_new[:, 0] = prediction_proba[:, 0] * 1.06
prediction = prediction_proba_new.argmax(axis=-1)
pred = pd.DataFrame(prediction, columns=['label_pre'])
pred['event_id'] = test_event
pred = pd.merge(test_data, pred, on='event_id', how='left')
sub = pred[['jet_id', 'label_pre']]
sub.columns = ['id', 'label']
encoder = LabelEncoder().fit([1,4,5,21])
sub['label'] = encoder.inverse_transform(sub['label'])
sub.to_csv(f'new_sub_{args.order}_{args.bs}_{args.patience}_{args.split}_maxlen{args.maxlen}_5.csv', index=False)

sub_not1 = sub.copy()
sub_not1.loc[sub_not1['label']==1, 'label'] = 0
sub_not1.to_csv('sub_not1.csv', index=False)

sub_all1 = sub.copy()
sub_all1.loc[sub_all1['label']!=1, 'label'] = 0
sub_all1.to_csv('sub_all1.csv', index=False)

print("done!!!!")
