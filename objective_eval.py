import argparse
import os
import numpy as np
import tensorflow as tf
import scipy.io as scio
import pylab
import argparse
import scipy.stats as scistat

import preprocess as preproc

from model import EncDecGen as edg
from model_proper_lkl import EncDecGen as edg_prop_lkl
from model_no_enc_loss import EncDecGen as edg_no_enc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

SAMPLING_RATE = 16000
NUM_MCEP = 23
FRAME_PERIOD = 5.0

def conversion(input_mfc_A, input_mfc_B, \
               input_pitch_A, input_pitch_B, \
               input_momenta_A2B, model_name, 
               no_enc=False, lkl=False):
    
    tf.reset_default_graph()

    test_dec_loss_MAE = list()
    test_dec_loss_COR = list()
    test_gen_loss_MAE = list()
    
    if lkl:
        model = edg_prop_lkl(num_mfc_features=NUM_MCEP, pre_train=None)
    elif no_enc:
        model = edg_no_enc(num_mfc_features=NUM_MCEP, pre_train=None)
    else:
        model = edg(num_mfc_features=NUM_MCEP, pre_train=None)

    model.load(filepath=model_name)
    
    for i in range(len(input_mfc_A)):
        
        if lkl:
            gen_momenta, gen_pitch, gen_mfc, \
            enc_loss, dec_loss, gen_loss, \
                = model.compute_test_loss(input_mfc_A=input_mfc_A[i:i+1], \
                    input_pitch_A=input_pitch_A[i:i+1], \
                    input_momenta_A2B=input_momenta_A2B[i:i+1], \
                    input_mfc_B=input_mfc_B[i:i+1], \
                    input_pitch_B=input_pitch_B[i:i+1])
    
            gen_pitch = np.reshape(gen_pitch, (-1,))
            target_pitch = np.reshape(input_pitch_B[i:i+1], (-1,))
            corr = np.corrcoef(list(gen_pitch), list(target_pitch))
        elif no_enc:
            gen_pitch, gen_mfc, \
            dec_loss, gen_loss, \
                = model.compute_test_loss(input_mfc_A=input_mfc_A[i:i+1], \
                    input_pitch_A=input_pitch_A[i:i+1], \
                    input_mfc_B=input_mfc_B[i:i+1], \
                    input_pitch_B=input_pitch_B[i:i+1])
    
            gen_pitch = np.reshape(gen_pitch, (-1,))
            target_pitch = np.reshape(input_pitch_B[i:i+1], (-1,))
            corr = np.corrcoef(list(gen_pitch), list(target_pitch))
        else:
            gen_momenta, gen_pitch, gen_mfc, \
            enc_loss, dec_loss, gen_loss, \
                = model.compute_test_loss(input_mfc_A=input_mfc_A[i:i+1], \
                    input_pitch_A=input_pitch_A[i:i+1], \
                    input_momenta_A2B=input_momenta_A2B[i:i+1], \
                    input_mfc_B=input_mfc_B[i:i+1], \
                    input_pitch_B=input_pitch_B[i:i+1])
    
            gen_pitch = np.reshape(gen_pitch, (-1,))
            target_pitch = np.reshape(input_pitch_B[i:i+1], (-1,))
            corr = np.corrcoef(list(gen_pitch), list(target_pitch))

        test_dec_loss_MAE.append(dec_loss)
        test_dec_loss_COR.append(corr[0,1])
        test_gen_loss_MAE.append(gen_loss)
#        print("Processed {}th data".format(i))

    return test_dec_loss_MAE, test_dec_loss_COR, test_gen_loss_MAE

if __name__ == '__main__':

    args = argparse.ArgumentParser()
    args.add_argument('--emo_pair', type=str, help='Emotion pair', \
                        default='neu-ang', choices=['neu-ang', \
                        'neu-hap', 'neu-sad'])
    argv = args.parse_args()
    emo_pair = argv.emo_pair

    data_test = scio.loadmat(os.path.join('./data/'+emo_pair\
                                +'/test_mod_dtw_harvest.mat'))

    pitch_A_test = np.expand_dims(data_test['src_f0_feat'], axis=-1)
    pitch_B_test = np.expand_dims(data_test['tar_f0_feat'], axis=-1)
    mfc_A_test = data_test['src_mfc_feat']
    mfc_B_test = data_test['tar_mfc_feat']
    momenta_A2B_test = np.expand_dims(data_test['momenta_f0'], axis=-1)
    
    stacked_mfc_A, stacked_f0_A, stacked_mfc_B, \
        stacked_f0_B, stacked_momenta_A2B \
            = preproc.sample_data(mfc_A=mfc_A_test, pitch_A=pitch_A_test, \
                              mfc_B=mfc_B_test, pitch_B=pitch_B_test, \
                              momenta_A2B=momenta_A2B_test)

    stacked_mfc_A = np.transpose(np.vstack(mfc_A_test), axes=(0,2,1))
    stacked_mfc_B = np.transpose(np.vstack(mfc_B_test), axes=(0,2,1))

    stacked_f0_A = np.transpose(np.vstack(pitch_A_test), axes=(0,2,1))
    stacked_f0_B = np.transpose(np.vstack(pitch_B_test), axes=(0,2,1))

    stacked_momenta_A2B = np.transpose(np.vstack(momenta_A2B_test), axes=(0,2,1))
    
    model_name = './model/'+emo_pair+'.ckpt'
    reg_dec_mae, reg_dec_cor, reg_gen_mae = conversion(input_mfc_A=stacked_mfc_A, \
                input_mfc_B=stacked_mfc_B, input_pitch_A=stacked_f0_A, \
                input_pitch_B=stacked_f0_B, \
                input_momenta_A2B=stacked_momenta_A2B , model_name=model_name)
    
    print("Regularized: ", np.mean(reg_dec_mae), np.mean(reg_dec_cor), np.mean(reg_gen_mae))
    
    model_name = './model/'+emo_pair+'_no_enc.ckpt'
    unreg_dec_mae, unreg_dec_cor, unreg_gen_mae = conversion(input_mfc_A=stacked_mfc_A, \
                input_mfc_B=stacked_mfc_B, input_pitch_A=stacked_f0_A, \
                input_pitch_B=stacked_f0_B, \
                input_momenta_A2B=stacked_momenta_A2B, \
                model_name=model_name, no_enc=True)
    
    print("Unregularized: ", np.mean(unreg_dec_mae), np.mean(unreg_dec_cor), np.mean(unreg_gen_mae))
    
    print("2-sample ttest: ", scistat.ttest_ind(reg_dec_mae, unreg_dec_mae))
    
#    model_name = './model/'+emo_pair+'_le_0.01_ld_0.0001_lg_0.0001_prop_lkl_neu_ang.ckpt'
#    unreg_dec_mae, unreg_dec_cor, unreg_gen_mae = conversion(input_mfc_A=stacked_mfc_A, \
#                input_mfc_B=stacked_mfc_B, input_pitch_A=stacked_f0_A, \
#                input_pitch_B=stacked_f0_B, \
#                input_momenta_A2B=stacked_momenta_A2B, \
#                model_name=model_name, lkl=True)
#    
#    print("Proper lkl: ", np.mean(unreg_dec_mae), np.mean(unreg_dec_cor), np.mean(unreg_gen_mae))
    





#    barWidth=0.5
#    r1 = np.arange(2)
#    r2 = [x + barWidth for x in r1]
#
#    pylab.figure()
#    pylab.bar(r1, [np.mean(reg_dec_mae), np.mean(unreg_dec_mae)], \
#                   width = barWidth, color = 'blue', \
#                   edgecolor = 'black', \
#                   yerr=[np.std(reg_dec_mae), np.std(unreg_dec_mae)], \
#                   capsize=7)
#    
#    pylab.xticks([r for r in range(2)], \
#                  ['Regularized', 'Unregularized'])
#    pylab.ylabel('MAE')
#    pylab.title('F0 MAE: '+emo_pair)
#    pylab.savefig('/home/ravi/Desktop/reg_vs_unreg_F0_'+emo_pair+'.png')
#    pylab.close()
        
#    pylab.figure()
#    pylab.bar(r2, [np.mean(reg_gen_mae), np.mean(unreg_gen_mae)], \
#                   width = barWidth, color = 'blue', \
#                   edgecolor = 'black', \
#                   yerr=[np.std(reg_gen_mae), np.std(unreg_gen_mae)], \
#                   capsize=7)
#
#    pylab.xticks([r + barWidth for r in range(2)], \
#                  ['Regularized', 'Unregularized'])
#    pylab.ylabel('MAE')
#    pylab.title('Spect MAE: '+emo_pair)
#    pylab.savefig('/home/ravi/Desktop/reg_vs_unreg_Spect_'+emo_pair+'.png')
#    pylab.close()
#    scio.savemat('/home/ravi/Desktop/reg_vs_unreg_'+emo_pair+'.mat', 
#                 {'reg_f0':np.asarray(reg_dec_mae),
#                  'reg_spect':np.asarray(reg_gen_mae),
#                  'unreg_f0':np.asarray(unreg_dec_mae),
#                  'unreg_spect':np.asarray(unreg_gen_mae)})


