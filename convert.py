import argparse
import os
import librosa
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as scwav
import scipy.signal as scisig

import preprocess as preproc

from model import EncDecGen
from helper import smooth, generate_interpolation

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

def conversion(model_path, data_dir, output_dir, no_spec=False):

    sampling_rate = 16000
    num_mcep = 23
    frame_period = 5.0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    model = EncDecGen(num_mfc_features=23, pre_train=None)
    model.load(filepath=model_path)

    for file in os.listdir(data_dir):

        try:

            wav = scwav.read(os.path.join(data_dir, file))
            wav = wav[1].astype(np.float64)
            wav = preproc.wav_padding(wav=wav, sr=sampling_rate, \
                    frame_period=frame_period, multiple=4)
            f0, sp, ap = preproc.world_decompose(wav=wav, \
                    fs=sampling_rate, frame_period=frame_period)
            code_sp = preproc.world_encode_spectral_envelop(sp, \
                                    sampling_rate, dim=num_mcep)

            z_idx = np.where(f0<10.0)[0]
            f0 = scisig.medfilt(f0, kernel_size=3)
            f0 = generate_interpolation(f0)
            f0 = smooth(f0, window_len=13)
            f0 = np.reshape(f0, (1,-1,1))
            code_sp = np.reshape(code_sp, (1,-1,num_mcep))

            code_sp = np.transpose(code_sp, axes=(0,2,1))
            f0 = np.transpose(f0, axes=(0,2,1))

            # Prediction
            _, f0_conv, code_sp_conv = model.test(input_mfc=code_sp, \
                                                  input_pitch=f0)
            
            code_sp_conv = np.transpose(code_sp_conv, axes=(0,2,1))

            f0_conv = np.asarray(np.reshape(f0_conv,(-1,)), np.float64)
            code_sp_conv = np.asarray(np.squeeze(code_sp_conv), np.float64)
            code_sp_conv = np.copy(code_sp_conv, order='C')
            sp_conv = preproc.world_decode_spectral_envelop(code_sp_conv, \
                                                            sampling_rate)
            f0_conv[z_idx] = 0.0
            
            if no_spec == True:
                ec = np.reshape(np.sqrt(np.sum(np.square(sp), axis=1)), (-1,1))
                ec_conv = np.reshape(np.sqrt(np.sum(np.square(sp_conv), axis=1)), \
                                     (-1,1))

                # Making sure silence remains silence
                sil_zone = np.where(ec<1e-10)[0]
                ec_conv[sil_zone] = 1e-10
                
                sp = np.divide(np.multiply(sp, ec_conv), ec)
                sp = np.copy(sp, order='C')
                
                wav_transformed = preproc.world_speech_synthesis(f0=f0_conv, \
                                    decoded_sp=sp, \
                                    ap=ap, fs=sampling_rate, \
                                    frame_period=frame_period)
            else:
                wav_transformed = preproc.world_speech_synthesis(f0=f0_conv, \
                                    decoded_sp=sp_conv, \
                                    ap=ap, fs=sampling_rate, \
                                    frame_period=frame_period)
            
            librosa.output.write_wav(os.path.join(output_dir, \
                    os.path.basename(file)), wav_transformed, sampling_rate)
            print("Reconstructed file "+os.path.basename(file))
        except Exception as ex:
            print(ex)


if __name__ == '__main__':
    
    tf.reset_default_graph()
    parser = argparse.ArgumentParser(description = 'EncDecGen Model.')
    parser.add_argument('--emo_pair', type=str, 
                        help='Emotion pair', default='neu-ang', 
                        choices=['neu-ang', 'neu-hap', 'neu-sad'])
    parser.add_argument('--model_path', type=str, 
                        help='Full path to the ckpt model', 
                        default='./model/neu-ang.ckpt')
    parser.add_argument('--data_dir', type=str, 
                        help='Directory of wav files for conversion')
    parser.add_argument('--output_dir', type=str, 
                        help='Directory to store converted samples')
    
    argv = parser.parse_args()

    model_path = argv.model_path
    data_dir = argv.data_dir
    output_dir = argv.output_dir

    conversion(model_path=model_path, \
               data_dir=data_dir, output_dir=output_dir, no_spec=False)
