
"""
This is an example in the IEEE 39-bus system to illustrate how the algorithm proposed in the paper 
"Cascading Failure Analysis Based on a Physics-Informed Graph Neural Network" is implemented.

It depends on the following module packages: 
    1)numpy == 1.20.3 ;
    2)pypower, see "https://github.com/rwl/PYPOWER";
    3) sklearn == 0.24.2 ;
    4) tensorflow == 2.5.3 .

As the paper is still under review, we only publish an abbreviated version. 

It is an abbreviated version of the model proposed in the paper, the differences include the followings:
    1) What we constructed in the paper is a cascaded graph neural network, and only the 
        multilayer perceptron is used here;
    2) In the paper, we emphasize that the datasets used for pre-training and fine-tuning are different,
        but here we share the same dataset (X_train and Y_train are used for pre-training, and only 
        X_train is used for fine-tuning).
    3) Here we define five physics-informed losses (loss1-loss5), and we use three (D1-D3) in the paper. 
        D1 = loss1 + loss2
        D2 = loss3 + loss4
        D3 = loss5

The overall steps can be summarized as: 
    1) define DNN and PINN;
    2) define PI Losses;
    3) pre-train DNN;
    4) fine-tune PINN;
    5) compare PI loss, see list variables "PINN_error" & "DNN_error"

Results:
    DNN_error = [0.6731054889662209,
                 0.5069528405504167,
                 0.0005143502082133058,
                 6.087896532216012e-05,
                 2.8260873433930847e-05]
    PINN_error = [0.2352637357203292,
                  0.33192708307416074,
                  0.0004686915234972803,
                  3.6197895329672766e-05,
                  7.226042189181615e-05]

See more comparion results in our paper:
    "Cascading Failure Analysis Based on a Physics-Informed Graph Neural Network"
    
"""
import pypower
import copy
import random
import numpy as np
from pypower.bustypes import bustypes
from pypower.ext2int import ext2int
from pypower.loadcase import loadcase
from pypower.ppoption import ppoption
from pypower.ppver import ppver
from pypower.makeBdc import makeBdc
from pypower.makeSbus import makeSbus
from pypower.dcpf import dcpf
from pypower.makeYbus import makeYbus
from pypower.newtonpf import newtonpf
from pypower.fdpf import fdpf
from pypower.gausspf import gausspf
from pypower.makeB import makeB
from pypower.pfsoln import pfsoln
from pypower.printpf import printpf
from pypower.savecase import savecase
from pypower.int2ext import int2ext
from pypower.idx_bus import PD, QD, VM, VA, GS, BUS_TYPE, PV, PQ, REF
from pypower.idx_brch import PF, PT, QF, QT
from pypower.idx_gen import PG, QG, VG, QMAX, QMIN, GEN_BUS, GEN_STATUS
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def makeVbus(case):
    bus_num = case['bus'].shape[0]
    V = case['bus'][:,7]
    A = case['bus'][:,8]
    V_real = V * np.cos(np.deg2rad(A))
    V_imag = V * np.sin(np.deg2rad(A))
    return V_real,V_imag

def Ybus(case39):
    ppc = ext2int(case39[0])
    baseMVA, bus, gen, branch = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)
    Ybus_view = np.zeros(Ybus.shape,dtype='complex')
    for i in range(Ybus.shape[0]):
        for j in range(Ybus.shape[1]):
            Ybus_view[i,j] = Ybus[i,j]
    return Ybus_view

def cal_Sbus(case_dic,Sbus_dic):
    sample_num = len(case_dic)
    for i in range(sample_num):
        temp_case = case_dic[i+1][0]
        ppc = loadcase(temp_case)
        ppc = ext2int(ppc)
        baseMVA, bus, gen, branch = \
            ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"]
        Sbus = makeSbus(baseMVA, bus, gen)
        Sbus_dic[i+1] = Sbus
        
        print('\r%s/%s'%(i,sample_num),end='\r')
    return Sbus_dic

def cal_Vbus(case_dic,Vbus_dic):
    sample_num = len(case_dic)
    for i in range(sample_num):
        temp_case = case_dic[i+1][0]
        V_r,V_i = makeVbus(temp_case)
        Vbus_dic[i+1] = {}
        Vbus_dic[i+1]['V_real'],Vbus_dic[i+1]['V_imag'] = V_r,V_i
        print('\r%s/%s'%(i,sample_num),end='\r')
    return Vbus_dic



def make_PINN():
    "model0.get_layer('dense1')"
    inputs1 = tf.keras.Input(shape=(39,))
    inputs2 = tf.keras.Input(shape=(39,))
    inputs = tf.keras.layers.Concatenate(axis=1)([inputs1, inputs2])
    #x = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1,)(inputs)
    x = DNN.get_layer('normal1')(inputs)
    x = DNN.get_layer('dense1')(x)
    x = DNN.get_layer('dense2')(x)
    x = DNN.get_layer('dense3')(x)
    output1 = DNN.get_layer('dense4')(x) 
    output2 = DNN.get_layer('dense5')(x) 
    PINN = tf.keras.Model([inputs1,inputs2], [output1,output2])
    global V_hat,S_hat,I_hat
    V_hat = tf.keras.layers.Reshape((39,1))(tf.cast(tf.complex(output1,output2),dtype='complex128'))
    V_imag = tf.math.imag(V_hat)
    I_hat = tf.matmul(Ybus,V_hat)
    S_hat = V_hat * tf.math.conj(I_hat)
    S_real_hat = tf.math.real(S_hat)
    S_imag_hat = tf.math.imag(S_hat)
    inputs1 = tf.keras.layers.Reshape((39,1))(inputs1)
    inputs2 = tf.keras.layers.Reshape((39,1))(inputs2)
    inputs1,inputs2 = tf.cast(inputs1,dtype='float64'),tf.cast(inputs2,dtype='float64')
    loss1 = tf.reduce_mean(tf.math.abs(S_real_hat-inputs1))
    loss2 = tf.reduce_mean(tf.math.abs(S_imag_hat-inputs2))
    V_a = tf.math.abs(V_hat)
    loss3 = tf.reduce_mean(tf.math.abs(V_a*isPV.reshape(1,39,1)-PV_V.reshape(1,39,1)))
    loss4 = tf.reduce_mean(tf.math.abs(V_a*isPt.reshape(1,39,1)-Pt_V.reshape(1,39,1)))
    loss5 = tf.reduce_mean(tf.math.abs(V_imag*isPt.reshape(1,39,1)))
    
    
    loss = loss1+loss2+loss2+100*loss3+100*loss4+100*loss5
    PINN.add_metric(loss1,name='loss1')
    PINN.add_metric(loss2,name='loss2')
    PINN.add_metric(loss3,name='loss3')
    PINN.add_metric(loss4,name='loss4')
    PINN.add_metric(loss5,name='loss5')
    PINN.add_loss(loss)
    
    PINN.get_layer('normal1').trainable = False
    PINN.get_layer('dense1').trainable = False
    #PINN.get_layer('dense2').trainable = False
    #PINN.get_layer('dense3').trainable = False
    #PINN.get_layer('dense4').trainable = False
    PINN.compile(optimizer='Adamax')
    PINN.fit(x=[X_train[:,0:39],X_train[:,39:]],batch_size=128,epochs=500)
    
    return PINN

def PQ():
    isPQ = np.zeros((39,))
    wherePQ = np.where(case39[0]['bus'][:,1]==1)[0]
    for i in range(wherePQ.shape[0]):
        isPQ[wherePQ[i]] = 1
    return isPQ

def PV():
    isPV = np.zeros((39,))
    wherePV = np.where(case39[0]['bus'][:,1]==2)[0]
    for i in range(wherePV.shape[0]):
        isPV[wherePV[i]] = 1
    return isPV

def Pt():
    isPt= np.zeros((39,))
    wherePt = np.where(case39[0]['bus'][:,1]==3)[0]
    for i in range(wherePt.shape[0]):
        isPt[wherePt[i]] = 1
    return isPt



def make_DNN():
    inputs1 = tf.keras.Input(shape=39)
    inputs2 = tf.keras.Input(shape=39)
    inputs = tf.keras.layers.Concatenate(axis=1)([inputs1, inputs2])
    x = tf.keras.layers.experimental.preprocessing.Normalization(axis=-1,name='normal1')(inputs)
    x = tf.keras.layers.Dense(256,name='dense1',activation='gelu')(x)
    x = tf.keras.layers.Dense(128,name='dense2',activation='gelu')(x)
    x = tf.keras.layers.Dense(128,name='dense3',activation='gelu')(x)
    output1 = tf.keras.layers.Dense(39,name='dense4')(x) 
    output2 = tf.keras.layers.Dense(39,name='dense5')(x) 
    DNN = tf.keras.Model([inputs1,inputs2], [output1,output2])
    global V_hat,S_hat,I_hat
    V_hat = tf.keras.layers.Reshape((39,1))(tf.cast(tf.complex(output1,output2),dtype='complex128'))
    V_imag = tf.math.imag(V_hat)
    I_hat = tf.matmul(Ybus,V_hat)
    S_hat = V_hat * tf.math.conj(I_hat)
    S_real_hat = tf.math.real(S_hat)
    S_imag_hat = tf.math.imag(S_hat)
    inputs1 = tf.keras.layers.Reshape((39,1))(inputs1)
    inputs2 = tf.keras.layers.Reshape((39,1))(inputs2)
    inputs1,inputs2 = tf.cast(inputs1,dtype='float64'),tf.cast(inputs2,dtype='float64')
    loss1 = tf.reduce_mean(tf.math.abs(S_real_hat-inputs1))
    loss2 = tf.reduce_mean(tf.math.abs(S_imag_hat-inputs2))
    V_a = tf.math.abs(V_hat)
    loss3 = tf.reduce_mean(tf.math.abs(V_a*isPV.reshape(1,39,1)-PV_V.reshape(1,39,1)))
    loss4 = tf.reduce_mean(tf.math.abs(V_a*isPt.reshape(1,39,1)-Pt_V.reshape(1,39,1)))
    loss5 = tf.reduce_mean(tf.math.abs(V_imag*isPt.reshape(1,39,1)))
    
    
    loss = loss1+loss2+loss2+10*loss3+10*loss4+10*loss5
    DNN.add_metric(loss1,name='loss1')
    DNN.add_metric(loss2,name='loss2')
    DNN.add_metric(loss3,name='loss3')
    DNN.add_metric(loss4,name='loss4')
    DNN.add_metric(loss5,name='loss5')
    #DNN.add_loss(loss)
    DNN.compile(optimizer='Adamax',loss='MSE')
    DNN.fit(x=[X_train[:,0:39],X_train[:,39:]],y=[Y_train[:,0:39],Y_train[:,39:]],batch_size=128,epochs=100)
    #tf.keras.models.save_model(model=DNN,filepath='./dnn')
    return DNN

def error_check(DNN):
    Y_hat = DNN.predict([X_train[:,0:39],X_train[:,39:]])
    global V_hat
    V_hat = tf.cast(tf.complex(Y_hat[0],Y_hat[1]),dtype='complex128').numpy().reshape(10000,39,1)
    global V_imag,temp1,temp2,temp3,temp4,temp5,S_real_hat,S_hat,input1,V_a
    V_real = np.real(V_hat)
    V_imag = np.imag(V_hat)
    I_hat = np.matmul(Ybus,V_hat)
    S_hat = V_hat * np.conj(I_hat)
    S_real_hat = np.real(S_hat)
    S_imag_hat = np.imag(S_hat)
    input1 = X_train[:,0:39].reshape(10000,39,1)
    input2 = X_train[:,39:].reshape(10000,39,1)
    temp1 = S_real_hat-input1
    loss1 = tf.reduce_mean(tf.math.abs(S_real_hat-input1)).numpy()
    temp2 = S_imag_hat-input2
    loss2 = tf.reduce_mean(tf.math.abs(S_imag_hat-input2)).numpy()
    V_a = tf.math.abs(V_hat).numpy().reshape(10000,39,1)
    temp3 = V_a*isPV.reshape(1,39,1)-PV_V.reshape(1,39,1)
    loss3 = tf.reduce_mean(tf.math.abs(V_a*isPV.reshape(1,39,1)-PV_V.reshape(1,39,1))).numpy()
    temp4 = V_a*isPt.reshape(1,39,1)-Pt_V.reshape(1,39,1)
    loss4 = tf.reduce_mean(tf.math.abs(V_a*isPt.reshape(1,39,1)-Pt_V.reshape(1,39,1))).numpy()
    temp5 = V_imag*isPt.reshape(1,39,1)
    loss5 = tf.reduce_mean(tf.math.abs(V_imag*isPt.reshape(1,39,1))).numpy()
    return [loss1,loss2,loss3,loss4,loss5]

def predict(DNN):
    Y_hat = DNN.predict([X_train[:,0:39],X_train[:,39:]])
    return Y_hat

if __name__ == '__main__':
    
    case39 = pypower.runpf.runpf(pypower.case39.case39())
    Ybus = Ybus(case39)
    X_train = np.load('X_train.npy')
    Y_train = np.load('Y_train.npy')
    isPQ = PQ()
    isPV = PV()
    isPt = Pt()
    PV_V = isPV * case39[0]['bus'][:,7]
    Pt_V = isPt * case39[0]['bus'][:,7]
    DNN = make_DNN()
    DNN_error = error_check(DNN)
    Y_hat  =  predict(DNN)
    normal1 =  DNN.get_layer('normal1').get_weights()
    dense1 = DNN.get_layer('dense1').get_weights()
    PINN = make_PINN()
    
    
    
    PINN.evaluate([X_train[:,0:39],X_train[:,39:]])
    normal1_b =  PINN.get_layer('normal1').get_weights()
    dense1_b = PINN.get_layer('dense1').get_weights()
    #PINN.fit(x=[X_train[:,0:39],X_train[:,39:]],batch_size=128,epochs=300)
    normal1_a =  PINN.get_layer('normal1').get_weights()
    dense1_a = PINN.get_layer('dense1').get_weights()
    Y_hat2 = predict(PINN)
    PINN_error = error_check(PINN)