            #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 10 13:39:45 2019

@author: ravi
"""

from glob import glob
import os
import scipy.io.wavfile as scwav
import numpy as np
import librosa
import scipy.io as scio
import pylab
import scipy.signal as scisig
import pyworld as pw

import warnings
warnings.filterwarnings('ignore')

from dtw import constrained_dtw
from feat_utils import smooth, \
    smooth_contour, generate_interpolation, \
    generate_context, pre_emp, highpass_filter

def get_feats(FILE_LIST, weights, sample_rate, window_len, \
                                window_stride, n_feats=128, n_mfc=23):
    """ 
    FILE_LIST: A list containing the source (first) and target (second) utterances location
    sample_rate: Sampling frequency of the speech
    window_len: Length of the analysis window for getting features (in ms)
    """
    FILE_LIST_src = FILE_LIST[0]
    FILE_LIST_tar = FILE_LIST[1]

    f0_feat_src = []
    f0_feat_tar = []
    
    log_f0_feat_src = []
    log_f0_feat_tar = []
    
    ec_feat_src = []
    ec_feat_tar = []
    
    mfc_feat_src = []
    mfc_feat_tar = []

    file_list   = []

    for s,t in zip(FILE_LIST_src, FILE_LIST_tar):
        print(t)
        
        """
        Utterance level features for context expansion
        """
        utt_log_f0_src      = list()
        utt_log_f0_tar      = list()
        utt_f0_src          = list()
        utt_f0_tar          = list()
        utt_ec_src          = list()
        utt_ec_tar          = list()
        utt_mfc_src         = list()
        utt_mfc_tar         = list()
        
        file_id = int(s.split('/')[-1][:-4])
        weight = weights[np.where(weights[:,0]==file_id)[0],1][0]
        if weight<=0.5:
            continue
        else:
            try:
                src_wav = scwav.read(s)
                src = np.asarray(src_wav[1], np.float64)
                src = np.copy(highpass_filter(src, 20, 16000), order='C')

                tar_wav = scwav.read(t)
                tar = np.asarray(tar_wav[1], np.float64)
                tar = np.copy(highpass_filter(tar, 20, 16000), order='C')
                
#                src = ((src - np.min(src)) / (np.max(src) - np.min(src)))
#                tar = ((tar - np.min(tar)) / (np.max(tar) - np.min(tar)))

                f0_src, t_src   = pw.harvest(src, sample_rate, frame_period=int(1000*window_len))
                straight_src    = pw.cheaptrick(src, f0_src, t_src, sample_rate)
#                f0_src          = pw.stonemask(src, f0_src, t_src, sample_rate)
#                ap_src          = pw.d4c(src, f0_src, t_src, sample_rate)

                f0_tar, t_tar   = pw.harvest(tar, sample_rate,frame_period=int(1000*window_len))
                straight_tar    = pw.cheaptrick(tar, f0_tar, t_tar, sample_rate)
#                f0_tar          = pw.stonemask(tar, f0_tar, t_tar, sample_rate)
#                ap_tar          = pw.d4c(tar, f0_tar, t_tar, sample_rate)

                f0_src = scisig.medfilt(f0_src, kernel_size=3)
                f0_tar = scisig.medfilt(f0_tar, kernel_size=3)
                f0_src = np.asarray(f0_src, np.float32)
                f0_tar = np.asarray(f0_tar, np.float32)

                ec_src = np.sqrt(np.sum(np.square(straight_src), axis=1))
                ec_tar = np.sqrt(np.sum(np.square(straight_tar), axis=1))
                ec_src = scisig.medfilt(ec_src, kernel_size=3)
                ec_tar = scisig.medfilt(ec_tar, kernel_size=3)
                ec_src = np.asarray(ec_src, np.float32)
                ec_tar = np.asarray(ec_tar, np.float32)

                f0_src = np.asarray(generate_interpolation(f0_src), np.float32)
                f0_tar = np.asarray(generate_interpolation(f0_tar), np.float32)
                ec_src = np.asarray(generate_interpolation(ec_src), np.float32)
                ec_tar = np.asarray(generate_interpolation(ec_tar), np.float32)
                
                f0_src = smooth(f0_src, window_len=13)
                f0_tar = smooth(f0_tar, window_len=13)
                ec_src = smooth(ec_src, window_len=13)
                ec_tar = smooth(ec_tar, window_len=13)

                src_mfc = pw.code_spectral_envelope(straight_src, sample_rate, n_mfc)
                tar_mfc = pw.code_spectral_envelope(straight_tar, sample_rate, n_mfc)

                src_mfcc = librosa.feature.mfcc(y=src, sr=sample_rate, \
                                                hop_length=int(sample_rate*window_len), \
                                                win_length=int(sample_rate*window_len), \
                                                n_fft=1024, n_mels=128)
                
                tar_mfcc = librosa.feature.mfcc(y=tar, sr=sample_rate, \
                                                hop_length=int(sample_rate*window_len), \
                                                win_length=int(sample_rate*window_len), \
                                                n_fft=1024, n_mels=128)

#                _, cords = librosa.sequence.dtw(X=src_mfcc, Y=tar_mfcc, metric='cosine')
                _, _, cords = constrained_dtw(x=src_mfcc.T, y=tar_mfcc.T)

                del src_mfcc, tar_mfcc

#                mean_f0_src = np.mean(f0_src[np.where(f0_src>1.0)[0]])
#                mean_f0_tar = np.mean(f0_tar[np.where(f0_tar>1.0)[0]])
#                
#                std_f0_src = np.std(f0_src[np.where(f0_src>1.0)[0]])
#                std_f0_tar = np.std(f0_tar[np.where(f0_tar>1.0)[0]])
                
                f0_src = f0_src.reshape(-1,1)
                f0_tar = f0_tar.reshape(-1,1)
                
                ec_src = ec_src.reshape(-1,1)
                ec_tar = ec_tar.reshape(-1,1)

#                nmz_f0_src = (f0_src - mean_f0_src) / std_f0_src
#                nmz_f0_tar = (f0_tar - mean_f0_tar) / std_f0_tar
                
                ext_src_f0 = []
                ext_tar_f0 = []
                ext_src_ec = []
                ext_tar_ec = []
                ext_src_mfc = []
                ext_tar_mfc = []
                
                for i in range(len(cords)-1, -1, -1):
                    ext_src_f0.append(f0_src[cords[i,0],0])
                    ext_tar_f0.append(f0_tar[cords[i,1],0])
                    ext_src_ec.append(ec_src[cords[i,0],0])
                    ext_tar_ec.append(ec_tar[cords[i,1],0])
                    ext_src_mfc.append(src_mfc[cords[i,0],:])
                    ext_tar_mfc.append(tar_mfc[cords[i,1],:])
                
                ext_src_f0 = np.reshape(np.asarray(ext_src_f0), (-1,1))
                ext_tar_f0 = np.reshape(np.asarray(ext_tar_f0), (-1,1))
                ext_src_ec = np.reshape(np.asarray(ext_src_ec), (-1,1))
                ext_tar_ec = np.reshape(np.asarray(ext_tar_ec), (-1,1))
                ext_log_src_f0 = np.reshape(np.log(np.asarray(ext_src_f0)), (-1,1))
                ext_log_tar_f0 = np.reshape(np.log(np.asarray(ext_tar_f0)), (-1,1))
                ext_src_mfc = np.asarray(ext_src_mfc)
                ext_tar_mfc = np.asarray(ext_tar_mfc)

                if cords.shape[0]<n_feats:
                    continue
                else:
                    for sample in range(20):
                        start = np.random.randint(0, cords.shape[0]-n_feats+1)
                        end = start + n_feats
                        
                        utt_f0_src.append(ext_src_f0[start:end,:])
                        utt_f0_tar.append(ext_tar_f0[start:end,:])
                        
                        utt_log_f0_src.append(ext_log_src_f0[start:end,:])
                        utt_log_f0_tar.append(ext_log_tar_f0[start:end,:])
                        
                        utt_ec_src.append(ext_src_ec[start:end,:])
                        utt_ec_tar.append(ext_tar_ec[start:end,:])
                        
                        utt_mfc_src.append(ext_src_mfc[start:end,:])
                        utt_mfc_tar.append(ext_tar_mfc[start:end,:])
                    
                    f0_feat_src.append(utt_f0_src)
                    f0_feat_tar.append(utt_f0_tar)
                    
                    log_f0_feat_src.append(utt_log_f0_src)
                    log_f0_feat_tar.append(utt_log_f0_tar)
                    
                    ec_feat_src.append(utt_ec_src)
                    ec_feat_tar.append(utt_ec_tar)
                    
                    mfc_feat_src.append(utt_mfc_src)
                    mfc_feat_tar.append(utt_mfc_tar)
                        
                    file_list.append(file_id)

            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)

    file_list = np.asarray(file_list).reshape(-1,1)
    return file_list, (f0_feat_src, log_f0_feat_src, ec_feat_src, \
                     mfc_feat_src, f0_feat_tar, log_f0_feat_tar, \
                     ec_feat_tar, mfc_feat_tar)


##----------------------------------generate all features---------------------------------
if __name__=='__main__':
   file_name_dict = {}
   target_emo = 'sad'
   emo_dict = {'neutral-angry':'neu-ang', 'neutral-happy':'neu-hap', \
               'neutral-sad':'neu-sad'}
   
   for i in ['test', 'valid', 'train']:
   
       FILE_LIST_src = sorted(glob(os.path.join('/home/ravi/Downloads/Emo-Conv/', \
                           'neutral-'+target_emo+'/'+i+'/neutral/', '*.wav')))
       FILE_LIST_tar = sorted(glob(os.path.join('/home/ravi/Downloads/Emo-Conv/', \
                           'neutral-'+target_emo+'/'+i+'/'+target_emo+'/', '*.wav')))
       weights = scio.loadmat('/home/ravi/Downloads/Emo-Conv/neutral-' \
                            +target_emo+'/emo_weight.mat')
       
       sample_rate = 16000.0
       window_len = 0.005
       window_stride = 0.005
       
       FILE_LIST = [FILE_LIST_src, FILE_LIST_tar]
       
       file_names, (src_f0_feat, src_log_f0_feat, src_ec_feat, src_mfc_feat, \
                 tar_f0_feat, tar_log_f0_feat, tar_ec_feat, tar_mfc_feat) \
                 = get_feats(FILE_LIST, weights['weight'], sample_rate, \
                                                       window_len, window_stride, \
                                                       n_feats=128, n_mfc=23)

       scio.savemat('./data/'+emo_dict['neutral-'+target_emo]+'/'+i+'_mod_dtw_harvest.mat', \
                    { \
                         'src_mfc_feat':           src_mfc_feat, \
                         'tar_mfc_feat':           tar_mfc_feat, \
                         'src_f0_feat':            src_f0_feat, \
                         'tar_f0_feat':            tar_f0_feat, \
                         'src_log_f0_feat':        src_log_f0_feat, \
                         'tar_log_f0_feat':        tar_log_f0_feat, \
                         'src_ec_feat':            src_ec_feat, \
                         'tar_ec_feat':            tar_ec_feat, \
                         'file_names':             file_names
                     })

       file_name_dict[i] = file_names

       del file_names, src_mfc_feat, src_f0_feat, src_log_f0_feat, src_ec_feat, \
           tar_mfc_feat, tar_f0_feat, tar_log_f0_feat, tar_ec_feat


##-----------------------generate features for fine-tuning-------------------------------
# if __name__=='__main__':
#     file_name_dict = {}
#     target_emo = 'sad'
#     emo_dict = {'neutral-angry':'neu-ang', 'neutral-happy':'neu-hap', \
#                 'neutral-sad':'neu-sad'}
    
#     FILE_LIST_src = sorted(glob(os.path.join('/home/ravi/Desktop/Pitch-Energy/Wavenet-tts-samples/speech_US/fine-tune-'+target_emo+'/neutral', '*.wav')))
#     FILE_LIST_tar = sorted(glob(os.path.join('/home/ravi/Desktop/Pitch-Energy/Wavenet-tts-samples/speech_US/fine-tune-'+target_emo+'/'+target_emo, '*.wav')))
    
#     weights = np.concatenate((np.arange(1,253).reshape(-1,1), \
#                               0.9*np.ones((252,1))), axis=1)
    
#     sample_rate = 16000.0
#     window_len = 0.005
#     window_stride = 0.005
    
#     FILE_LIST = [FILE_LIST_src, FILE_LIST_tar]
    
#     file_names, (src_f0_feat, src_log_f0_feat, src_ec_feat, src_mfc_feat, \
#               tar_f0_feat, tar_log_f0_feat, tar_ec_feat, tar_mfc_feat) \
#               = get_feats(FILE_LIST, weights, sample_rate, \
#                                                     window_len, window_stride, \
#                                                     n_feats=128, n_mfc=23)

#     scio.savemat('./data/'+emo_dict['neutral-'+target_emo]+'/fine_tune.mat', \
#                  { \
#                       'src_mfc_feat':           src_mfc_feat, \
#                       'tar_mfc_feat':           tar_mfc_feat, \
#                       'src_f0_feat':            src_f0_feat, \
#                       'tar_f0_feat':            tar_f0_feat, \
#                       'src_log_f0_feat':        src_log_f0_feat, \
#                       'tar_log_f0_feat':        tar_log_f0_feat, \
#                       'src_ec_feat':            src_ec_feat, \
#                       'tar_ec_feat':            tar_ec_feat, \
#                       'file_names':             file_names
#                   })

#     del file_names, src_mfc_feat, src_f0_feat, src_log_f0_feat, src_ec_feat, \
#         tar_mfc_feat, tar_f0_feat, tar_log_f0_feat, tar_ec_feat
