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
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
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
parser.add_argument('--maxlen', type=int, default=200)
parser.add_argument('--order', type=str, default='particle_energy')

args = parser.parse_args()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.set_session(tf.Session(config=config))

if not os.path.exists("model_lstm"):
    os.mkdir("model_lstm")
    
if not os.path.exists("scaler"):
    os.mkdir("scaler")

train_jet = pd.read_csv('../data/jet_complex_data/complex_train_R04_jet.csv')
test_jet = pd.read_csv('../data/jet_complex_data/complex_test_R04_jet.csv')

data_jet = pd.concat([train_jet, test_jet], ignore_index=True)
del train_jet, test_jet
gc.collect()

train_particle = pd.read_csv('../data/jet_complex_data/complex_train_R04_particle.csv')
test_particle = pd.read_csv('../data/jet_complex_data/complex_test_R04_particle.csv')
data_particle = pd.concat([train_particle, test_particle], ignore_index=True)
del train_particle, test_particle
gc.collect()

data = pd.merge(data_particle, data_jet, on='jet_id', how='left')

cate_lb = LabelEncoder().fit(list(data['particle_category'].unique()) + [-99999999])
np.save('cateEncoder.npy', cate_lb.classes_)
data['particle_category'] = cate_lb.transform(data['particle_category'])
data['particle_category'] = data['particle_category'].astype(np.int8)

del data_particle, data_jet
gc.collect()
cols = [i for i in data.columns if i not in ['event_id', 'label', 'particle_category', 'jet_id']]
data[cols] = data[cols].astype(np.float32)
# data = reduce_mem_usage(data)
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
# data['Leadmass'] = data.groupby('jet_id')['particle_mass'].transform(max)
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
data['jet_yz'] = np.sqrt(data['jet_py']**2 + data['jet_pz']**2)
data['jet_xz'] = np.sqrt(data['jet_px']**2 + data['jet_pz']**2)
data['jet_sita_xy'] = np.arctan(data['jet_py']/data['jet_px'])
data['jet_sita_z'] = np.arcsin(data['jet_pz']/data['Etot']) 

data['particle_yz'] = np.sqrt(data['particle_py']**2 + data['particle_pz']**2)
data['particle_xz'] = np.sqrt(data['particle_px']**2 + data['particle_pz']**2)
data['particle_sita_xy'] = np.arctan(data['particle_py']/data['particle_px'])
data['particle_sita_z'] = np.arcsin(data['particle_pz']/data['etot'])

cols = [i for i in data.columns if i not in ['event_id', 'label', 'particle_category', 'jet_id']]

#高阶流 (粒子集体运动效应)       每个粒子的运动效应
data['v1']  = np.cos(data['theta'])
data['v2']  = np.cos(2*data['theta'])
data['v3']  = np.cos(3*data['theta'])
#data['v4'], data['v5'].....基本上v5以后的流就很小
#每个jet的运动效应
groupByjet = data.groupby('jet_id')
data['V1'] =  groupByjet['v1'].transform(sum)
data['V2'] =  groupByjet['v2'].transform(sum)
data['V3'] =  groupByjet['v3'].transform(sum)

print('Scaler....')
for col in cols:
    scaler = MinMaxScaler().fit(data[[col]])
    data[[col]] = scaler.transform(data[[col]])
    dump(scaler, 'scaler/%s.bin'%col.replace('/', '_'), compress=True)
    data = reduce_mem_usage(data, col)

train = data[-data['label'].isnull()].reset_index()
test = data[data['label'].isnull()].reset_index()

del data
gc.collect()

cols = ['particle_category'] + cols
print('-------------------')
print(cols)
print(len(cols))

train_data_use = train.groupby('event_id')
train_event = np.array(train_data_use['event_id'].max())
train_label = np.array(train_data_use['label'].mean())
encoder = LabelEncoder()
train_label = encoder.fit_transform(train_label)
train_data_use = pad_sequences(np.array(train_data_use.apply(lambda row: (
    np.array(row[cols])))), dtype=np.float16, value=0, padding='post', maxlen=args.maxlen)
l = train_data_use.shape[1]
print(f'maxlen:{l}')

test_data_use = test.groupby('event_id')
test_event = np.array(test_data_use['event_id'].max())
test_data_use = pad_sequences(np.array(test_data_use.apply(lambda row: (
    np.array(row[cols])))), dtype=np.float16, value=0, padding='post', maxlen=args.maxlen)

del train, test
gc.collect()

train_data_use = train_data_use.reshape(list(train_data_use.shape)[:3] + [1])
print(f"train_data_use:{train_data_use.shape}")
test_data_use = test_data_use.reshape(list(test_data_use.shape)[:3] + [1])
print(f"test_data_use:{test_data_use.shape}")
del test_data_use
del test_event
gc.collect()

kfold = StratifiedKFold(n_splits=args.split, shuffle=True, random_state=1024)
oof_proba = np.zeros([train_data_use.shape[0], 4])

for fold, (tr_idx, val_idx) in enumerate(kfold.split(train_data_use, train_label)):
    print(f"fold {fold}")
    train_labels = to_categorical(train_label, num_classes=4)
    X_train, X_validate, label_train, label_validate = train_data_use[tr_idx], train_data_use[val_idx], \
                                                       train_labels[tr_idx], train_labels[val_idx]
    model = Net(train_data_use.shape[1], train_data_use.shape[2])

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['acc'])
    if fold == 0:
        print(model.summary())

    plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.1, patience=6)
    early_stopping = EarlyStopping(monitor='val_acc', patience=args.patience, mode='max', verbose=0)
    checkpoint = ModelCheckpoint(f'./model_lstm/lstm_{args.order}_{args.bs}_{args.patience}_{fold}.hdf5',
                                 monitor='val_acc',
                                 verbose=1, save_best_only=True, mode='max', save_weights_only=True)
    history = model.fit(X_train, label_train, epochs=100, batch_size=args.bs,
                    verbose=2, shuffle=True,
                    validation_data=(X_validate, label_validate), callbacks=[early_stopping, checkpoint, plateau])

    model.load_weights(f'./model_lstm/lstm_{args.order}_{args.bs}_{args.patience}_{fold}.hdf5')
    
    oof_proba[val_idx] += model.predict(X_validate, verbose=0, batch_size=1024)
    print(accuracy_score(label_validate.argmax(axis=-1), oof_proba[val_idx].argmax(axis=-1)))
    del model
    del X_train
    del X_validate
    del label_train
    del label_validate
    del train_labels
    gc.collect()
    K.clear_session()

oof = oof_proba.argmax(axis=-1)
print(f'oof acc:{accuracy_score(train_label, oof)}')
val = pd.DataFrame(oof, columns=['label'])
val['event_id'] = train_event
train_data = pd.read_csv('../data/jet_complex_data/complex_train_R04_jet.csv')[['event_id', 'label']]
# train_data['event_id'] = event_lb.transform(train_data['event_id'])
val = pd.merge(train_data, val, on='event_id', how='left')
auc = roc_auc_score(pd.get_dummies(val['label_x']), pd.get_dummies(val['label_y']))
print(f'oof auc:{auc}')

oof = pd.DataFrame(oof_proba, columns=['p1', 'p4', 'p5', 'p21'])
oof['event_id'] = train_event
oof['label'] = train_label
oof.to_csv(f'oof_maxlen{args.maxlen}_5.csv', index=False)

oof_proba_new = oof_proba.copy()
oof_proba_new[:, 3] = oof_proba[:, 3] * 1.0
oof_proba_new[:, 2] = oof_proba[:, 2] * 1.0
oof_proba_new[:, 1] = oof_proba[:, 1] * 1.01
oof_proba_new[:, 0] = oof_proba[:, 0] * 1.06
oof = oof_proba_new.argmax(axis=-1)
print(f'new oof acc:{accuracy_score(train_label, oof)}')
val = pd.DataFrame(oof, columns=['label'])
val['event_id'] = train_event
val = pd.merge(train_data, val, on='event_id', how='left')
auc = roc_auc_score(pd.get_dummies(val['label_x']), pd.get_dummies(val['label_y']))
print(f'new oof auc:{auc}')
print("done!!!!")
