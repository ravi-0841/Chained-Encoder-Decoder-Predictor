import os
import numpy as np
import argparse
import time
import librosa
import sys
import scipy.io.wavfile as scwav
import joblib
import scipy.io as scio
import scipy.signal as scisig
import pylab
import logging

import preprocess as preproc

from joblib import Parallel, delayed
from model import EncDecGen
from sklearn.preprocessing import StandardScaler

from helper import smooth, generate_interpolation

def train(emo_pair, train_dir, model_dir, model_name, \
            random_seed, validation_dir, output_dir, \
            tensorboard_log_dir, pre_train=None, \
            lambda_encoder=1, lambda_decoder=1, \
            lambda_generator=1):

    np.random.seed(random_seed)

    num_epochs = 1000
    mini_batch_size = 1 
    encoder_learning_rate = 0.0001
    decoder_learning_rate = 0.0001
    generator_learning_rate = 0.0001
    
    sampling_rate = 16000
    num_mcep = 23
    frame_period = 5.0
    n_frames = 128 

    lambda_encoder = lambda_encoder
    lambda_decoder = lambda_decoder
    lambda_generator = lambda_generator

    le_ld_lg = "le_"+str(lambda_encoder)+"_ld_"+str(lambda_decoder) \
                +"_lg_"+str(lambda_generator)+'_'+emo_pair

    logger_file = './log/'+le_ld_lg+'.log'
    if os.path.exists(logger_file):
        os.remove(logger_file)

    logging.basicConfig(filename="./log/logger_"+le_ld_lg+".log", \
                            level=logging.DEBUG)

    logging.info("encoder_loss - L1")
    logging.info("decoder_loss - L1")
    logging.info("generator_loss - L1")

    logging.info("lambda_encoder - {}".format(lambda_encoder))
    logging.info("lambda_decoder - {}".format(lambda_decoder))
    logging.info("lambda_generator - {}".format(lambda_generator))

    if not os.path.isdir("./generated_pitch_spect/"+le_ld_lg):
        os.mkdir("./generated_pitch_spect/" + le_ld_lg)
    
    logging.info('Loading Data...')

    start_time = time.time()

    data_train = scio.loadmat(os.path.join(train_dir, 'train.mat'))
    data_valid = scio.loadmat(os.path.join(train_dir, 'valid.mat'))
    
    pitch_A_train = np.expand_dims(data_train['src_f0_feat'], axis=-1)
    pitch_B_train = np.expand_dims(data_train['tar_f0_feat'], axis=-1)
    mfc_A_train = data_train['src_mfc_feat']
    mfc_B_train = data_train['tar_mfc_feat']
    momenta_A2B_train = np.expand_dims(data_train['momenta_f0'], axis=-1)
    
    pitch_A_valid = np.expand_dims(data_valid['src_f0_feat'], axis=-1)
    pitch_B_valid = np.expand_dims(data_valid['tar_f0_feat'], axis=-1)
    mfc_A_valid = data_valid['src_mfc_feat']
    mfc_B_valid = data_valid['tar_mfc_feat']
    momenta_A2B_valid = np.expand_dims(data_valid['momenta_f0'], axis=-1)
    
    mfc_A_valid, pitch_A_valid, mfc_B_valid, pitch_B_valid, momenta_A2B_valid \
        = preproc.sample_data(mfc_A=mfc_A_valid, pitch_A=pitch_A_valid, \
                              mfc_B=mfc_B_valid, pitch_B=pitch_B_valid, \
                              momenta_A2B=momenta_A2B_valid)

    if validation_dir is not None:
        validation_output_dir = os.path.join(output_dir, le_ld_lg)
        if not os.path.exists(validation_output_dir):
            os.makedirs(validation_output_dir)

    end_time = time.time()
    time_elapsed = end_time - start_time

    logging.info('Loading Done.')

    logging.info('Time Elapsed for Data Preprocessing: %02d:%02d:%02d' % (time_elapsed // 3600, \
                                                                   (time_elapsed % 3600 // 60), \
                                                                   (time_elapsed % 60 // 1)))

    model = EncDecGen(num_mfc_features=23, pre_train=pre_train) #use pre_train arg to provide trained model
    
    for epoch in range(1,num_epochs+1):

        logging.info('Epoch: %d' % epoch)

        start_time_epoch = time.time()

        mfc_A, pitch_A, mfc_B, \
                pitch_B, momenta_A2B = preproc.sample_data(mfc_A=mfc_A_train, \
                                        pitch_A=pitch_A_train, mfc_B=mfc_B_train, \
                                        pitch_B=pitch_B_train, momenta_A2B=momenta_A2B_train)
        
        n_samples = mfc_A.shape[0]
       
        batch_enc_loss = list()
        batch_dec_loss = list()
        batch_gen_loss = list()
        batch_tot_loss = list()

        for i in range(n_samples // mini_batch_size):

            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            encoder_loss, decoder_loss, generator_loss, \
            gen_momenta, gen_pitch, gen_mfc \
                = model.train(input_mfc_A=mfc_A[start:end], \
                            input_mfc_B=mfc_B[start:end], \
                            input_pitch_A=pitch_A[start:end], \
                            input_pitch_B=pitch_B[start:end], \
                            input_momenta_A2B=momenta_A2B[start:end], \
                            lambda_encoder=lambda_encoder, \
                            lambda_decoder=lambda_decoder, \
                            lambda_generator=lambda_generator, \
                            encoder_learning_rate=encoder_learning_rate, \
                            decoder_learning_rate=decoder_learning_rate, \
                            generator_learning_rate = generator_learning_rate)
            
            batch_enc_loss.append(encoder_loss)
            batch_dec_loss.append(decoder_loss)
            batch_gen_loss.append(generator_loss)
            batch_tot_loss.append(lambda_encoder*encoder_loss \
                    + lambda_decoder*decoder_loss + lambda_generator*generator_loss)

        model.save(directory=model_dir, filename=model_name)

        logging.info("Train Encoder Loss- {}".format(np.mean(batch_enc_loss)))
        logging.info("Train Decoder Loss- {}".format(np.mean(batch_dec_loss)))
        logging.info("Train Generator Loss- {}".format(np.mean(batch_gen_loss)))
        logging.info("Train Total Loss- {}".format(np.mean(batch_tot_loss)))

        # Getting results on validation set
        
        valid_enc_loss = list()
        valid_dec_loss = list()
        valid_gen_loss = list()
        valid_tot_loss = list()

        for i in range(mfc_A_valid.shape[0]):
            
            gen_momenta, gen_pitch, gen_mfc, \
            enc_loss, dec_loss, gen_loss, \
                = model.compute_test_loss(input_mfc_A=mfc_A_valid[i:i+1], \
                             input_pitch_A=pitch_A_valid[i:i+1], \
                             input_momenta_A2B=momenta_A2B_valid[i:i+1], \
                             input_mfc_B=mfc_B_valid[i:i+1], \
                             input_pitch_B=pitch_B_valid[i:i+1])

            valid_enc_loss.append(enc_loss)
            valid_dec_loss.append(dec_loss)
            valid_gen_loss.append(gen_loss)
            valid_tot_loss.append(lambda_encoder*enc_loss \
                    + lambda_decoder*dec_loss + lambda_generator*gen_loss)

            if epoch % 100 == 0:
                pylab.figure(figsize=(12,12))
                pylab.plot(pitch_A_valid[i].reshape(-1,), label="Input Neutral")
                pylab.plot(pitch_B_valid[i].reshape(-1,), label="Target Angry")
                pylab.plot(gen_pitch.reshape(-1,), label="Generated Angry")
                pylab.plot(momenta_A2B_valid[i].reshape(-1,), label="Target Momentum")
                pylab.plot(gen_momenta.reshape(-1,), label="Generated Momentum")
                pylab.legend(loc=1)
                pylab.title("Epoch "+str(epoch)+" example "+str(i+1))
                pylab.savefig("./generated_pitch_spect/"+le_ld_lg+'/'+str(epoch)\
                                + "_"+str(i+1)+".png")
                pylab.close()

        logging.info("Valid Encoder Loss- {}".format(np.mean(valid_enc_loss)))
        logging.info("Valid Decoder Loss- {}".format(np.mean(valid_dec_loss)))
        logging.info("Valid Generator Loss- {}".format(np.mean(valid_gen_loss)))
        logging.info("Valid Total Loss- {}".format(np.mean(valid_tot_loss)))

        end_time_epoch = time.time()
        time_elapsed_epoch = end_time_epoch - start_time_epoch

        logging.info('Time Elapsed for This Epoch: %02d:%02d:%02d' % (time_elapsed_epoch // 3600, \
                (time_elapsed_epoch % 3600 // 60), (time_elapsed_epoch % 60 // 1)))

        if validation_dir is not None:
            if epoch % 100 == 0:
                logging.info('Generating Validation Data B from A...')
                sys.stdout.flush()
                for file in sorted(os.listdir(validation_dir)):
                    try:
                        filepath = os.path.join(validation_dir, file)
                        wav = scwav.read(filepath)
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
                        wav_transformed = preproc.world_speech_synthesis(f0=f0_conv, \
                                            decoded_sp=sp_conv, \
                                            ap=ap, fs=sampling_rate, \
                                            frame_period=frame_period)
                        librosa.output.write_wav(os.path.join(validation_output_dir, \
                                os.path.basename(file)), wav_transformed, sampling_rate)
                        logging.info("Reconstructed file "+os.path.basename(file))
                    except Exception as ex:
                        logging.info(ex)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train CycleGAN model for datasets.')

    emo_dict = {"neu-ang":['neutral', 'angry'], "neu-sad":['neutral', 'sad'], "neu-hap":['neutral', 'happy']}

    emo_pair = "neu-ang"
    train_dir_default = "./data/"+emo_pair
    model_dir_default = "./model"
    model_name_default = emo_pair+".ckpt"
    validation_dir_default = './data/evaluation/'+emo_pair+'/'+emo_dict[emo_pair][0]
    output_dir_default = './validation_output/'+emo_pair
    tensorboard_log_dir_default = './log/'+emo_pair
    random_seed_default = 0

    parser.add_argument('--train_dir', type = str, help = 'Directory for training data.', \
                        default = train_dir_default)
    parser.add_argument('--model_dir', type = str, help = 'Directory for saving models.', \
                        default = model_dir_default)
    parser.add_argument('--model_name', type = str, help = 'File name for saving model.', \
                        default = model_name_default)
    parser.add_argument('--random_seed', type = int, help = 'Random seed for model training.', \
                        default = random_seed_default)
    parser.add_argument('--validation_dir', type = str, \
                        help = 'Convert validation A after each training epoch. Set None for no conversion', \
                        default = validation_dir_default)
    parser.add_argument('--output_dir', type = str, \
                        help = 'Output directory for converted validation voices.', default = output_dir_default)
    parser.add_argument('--tensorboard_log_dir', type = str, help = 'TensorBoard log directory.', \
                    default = tensorboard_log_dir_default)
    parser.add_argument('--current_iter', type = int, \
                    help = "Current iteration of the model (Fine tuning)", default = 1)
    parser.add_argument("--lambda_encoder", type=float, help="hyperparam for encoder loss", \
                    default=0.01)#0.0001
    parser.add_argument("--lambda_decoder", type=float, help="hyperparam for decoder loss", \
                    default=0.0001)#0.0001
    parser.add_argument("--lambda_generator", type=float, help="hyperparam for generator loss", \
                    default=0.0001)#0.1

    argv = parser.parse_args()

    train_dir = argv.train_dir
    model_dir = argv.model_dir
    model_name = argv.model_name
    random_seed = argv.random_seed
    validation_dir = None if argv.validation_dir == 'None' or argv.validation_dir == 'none' \
                        else argv.validation_dir
    output_dir = argv.output_dir
    tensorboard_log_dir = argv.tensorboard_log_dir

    lambda_encoder = argv.lambda_encoder
    lambda_decoder = argv.lambda_decoder
    lambda_generator = argv.lambda_generator

    train(emo_pair=emo_pair, train_dir=train_dir, model_dir=model_dir, \
            model_name=model_name+'-'+str(argv.current_iter)+".ckpt", \
            random_seed=random_seed, validation_dir=validation_dir, \
            output_dir=output_dir, tensorboard_log_dir=tensorboard_log_dir, \
            pre_train=None, lambda_encoder=lambda_encoder, \
            lambda_decoder=lambda_decoder, lambda_generator=lambda_generator)
