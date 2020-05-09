import os
import tensorflow as tf
from nn_models import encoder, decoder, generator
from utils import l1_loss
from datetime import datetime

class EncDecGen(object):

    def __init__(self, num_mfc_features=23, encoder=encoder, \
                 decoder=decoder, generator=generator, \
                 mode='train', log_dir='./log', pre_train=None):

        self.num_mfc_features = num_mfc_features
        self.input_pitch_shape = [None, 1, None] # [batch_size, num_features, num_frames]
        self.input_mfc_shape = [None, num_mfc_features, None]

        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.mode = mode

        self.build_model()
        self.optimizer_initializer()

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        if pre_train is not None:
            self.saver.restore(self.sess, pre_train)
        else:
            self.sess.run(tf.global_variables_initializer())

        if self.mode == 'train':
            self.train_step = 0
            now = datetime.now()
            self.log_dir = os.path.join(log_dir, now.strftime('%Y%m%d-%H%M%S'))

    def build_model(self):

        # Placeholders for training samples
        self.input_pitch_A = tf.placeholder(tf.float32, shape=self.input_pitch_shape, name='input_pitch_A')
        self.input_pitch_B = tf.placeholder(tf.float32, shape=self.input_pitch_shape, name='input_pitch_B')
        self.input_momenta_A2B = tf.placeholder(tf.float32, shape=self.input_pitch_shape, name='input_moment_A2B')
        self.input_mfc_A = tf.placeholder(tf.float32, shape=self.input_mfc_shape, name='input_mfc_A')
        self.input_mfc_B = tf.placeholder(tf.float32, shape=self.input_mfc_shape, name='input_mfc_B')

        # Placeholders for test samples
        self.input_mfc_test = tf.placeholder(tf.float32, shape=self.input_mfc_shape, name='input_mfc_test')
        self.input_pitch_test = tf.placeholder(tf.float32, shape=self.input_pitch_shape, name='input_pitch_test')
        
        # Generate momenta and pitch B
        self.generation_momenta_A2B = self.encoder(input_mfc=self.input_mfc_A, \
                                        input_pitch=self.input_pitch_A, reuse=False, \
                                        scope_name='encoder')
        self.generation_pitch_B = self.decoder(input_momenta=self.generation_momenta_A2B, \
                                    input_pitch=self.input_pitch_A, reuse=False, \
                                    scope_name='decoder')
        self.generation_mfc_B = self.generator(input_mfc=self.input_mfc_A, \
                                    input_pitch=self.generation_pitch_B, \
                                    num_mfc=self.num_mfc_features, \
                                    training=True, reuse=False, \
                                    scope_name='generator')

        # Encoder loss
        self.encoder_loss = l1_loss(y=self.input_momenta_A2B, y_hat=self.generation_momenta_A2B)

        # Decoder loss
        self.decoder_loss = l1_loss(y=self.input_pitch_B, y_hat=self.generation_pitch_B)

        # Generator loss
        self.generator_loss = l1_loss(y=self.input_mfc_B, y_hat=self.generation_mfc_B)

        # Place holder for lambda_encoder and lambda_decoder
        self.lambda_encoder = tf.placeholder(tf.float32, None, name='lambda_encoder')
        self.lambda_decoder = tf.placeholder(tf.float32, None, name='lambda_decoder')
        self.lambda_generator = tf.placeholder(tf.float32, None, name='lambda_generator')
        
        # Merge the encoder-decoder-generator
        self.encoder_decoder_loss = self.lambda_encoder * self.encoder_loss \
                                + self.lambda_decoder * self.decoder_loss \
                                + self.lambda_generator * self.generator_loss

        # Categorize variables because we have to optimize the two sets of the variables separately
        trainable_variables = tf.trainable_variables()
        self.encoder_vars = [var for var in trainable_variables if 'encoder' in var.name]
        self.decoder_vars = [var for var in trainable_variables if 'decoder' in var.name]
        self.generator_vars = [var for var in trainable_variables if 'generator' in var.name]
        #for var in t_vars: print(var.name)

        # Reserved for test
        self.momenta_A2B_test = self.encoder(input_mfc=self.input_mfc_test, \
                                input_pitch=self.input_pitch_test, \
                                reuse=True, scope_name='encoder')
        self.pitch_B_test = self.decoder(input_momenta=self.momenta_A2B_test, \
                                input_pitch=self.input_pitch_test, reuse=True, \
                                scope_name='decoder')
        self.mfc_B_test = self.generator(input_mfc=self.input_mfc_test, \
                                input_pitch=self.pitch_B_test, \
                                num_mfc=self.num_mfc_features, training=False, \
                                reuse=True, scope_name='generator')

    def optimizer_initializer(self):

        self.generator_learning_rate = tf.placeholder(tf.float32, None, name='generator_learning_rate')
        self.encoder_learning_rate = tf.placeholder(tf.float32, None, name='encoder_learning_rate')
        self.decoder_learning_rate = tf.placeholder(tf.float32, None, name='decoder_learning_rate')

        self.encoder_optimizer = tf.train.AdamOptimizer(learning_rate=self.encoder_learning_rate, \
                                beta1=0.5).minimize(self.encoder_decoder_loss, \
                                var_list = self.encoder_vars)
        self.decoder_optimizer = tf.train.AdamOptimizer(learning_rate=self.decoder_learning_rate, \
                                beta1=0.5).minimize(self.encoder_decoder_loss, \
                                var_list = self.decoder_vars)
        self.generator_optimizer = tf.train.AdamOptimizer(learning_rate=self.generator_learning_rate, \
                                beta1=0.5).minimize(self.encoder_decoder_loss, \
                                var_list = self.generator_vars) 

    def train(self, input_mfc_A, input_mfc_B, input_pitch_A, \
                input_pitch_B, input_momenta_A2B, lambda_encoder, lambda_decoder, \
                lambda_generator, encoder_learning_rate, decoder_learning_rate, \
                generator_learning_rate):

        generation_momenta, generation_pitch, generation_mfc, \
                encoder_loss, decoder_loss, generator_loss, _, _, _ \
                        = self.sess.run([self.generation_momenta_A2B, \
                            self.generation_pitch_B, self.generation_mfc_B, \
                            self.encoder_loss, self.decoder_loss, self.generator_loss, \
                            self.encoder_optimizer, self.decoder_optimizer, self.generator_optimizer], \
                                feed_dict = {self.lambda_encoder:lambda_encoder, \
                                        self.lambda_decoder:lambda_decoder, \
                                        self.lambda_generator:lambda_generator, \
                                        self.input_pitch_A:input_pitch_A, \
                                        self.input_mfc_A:input_mfc_A, \
                                        self.input_momenta_A2B:input_momenta_A2B, \
                                        self.input_pitch_B:input_pitch_B, \
                                        self.input_mfc_B:input_mfc_B, \
                                        self.encoder_learning_rate:encoder_learning_rate, \
                                        self.decoder_learning_rate:decoder_learning_rate, \
                                        self.generator_learning_rate:generator_learning_rate})

        self.train_step += 1

        return encoder_loss, decoder_loss, generator_loss, \
                generation_momenta, generation_pitch, generation_mfc

    def test(self, input_mfc, input_pitch):

        generation_momenta = self.sess.run(self.momenta_A2B_test, \
                            feed_dict={self.input_mfc_test:input_mfc, \
                                        self.input_pitch_test:input_pitch})

        generation_pitch = self.sess.run(self.pitch_B_test, \
                            feed_dict={self.input_pitch_test:input_pitch, \
                                        self.input_mfc_test:input_mfc})

        generation_mfc = self.sess.run(self.mfc_B_test, \
                            feed_dict={self.input_pitch_test:input_pitch, \
                                        self.input_mfc_test:input_mfc})
        return generation_momenta, generation_pitch, generation_mfc

    def compute_test_loss(self, input_mfc_A, input_pitch_A, \
            input_momenta_A2B, input_mfc_B, input_pitch_B):

        gen_momenta, gen_pitch, gen_mfc = self.test(input_mfc=input_mfc_A, \
                                            input_pitch=input_pitch_A)
        enc_loss, dec_loss, gen_loss = self.sess.run([self.encoder_loss, \
                                        self.decoder_loss, self.generator_loss], \
                                        feed_dict={self.input_pitch_B:input_pitch_B, \
                                        self.input_mfc_B:input_mfc_B, \
                                        self.input_momenta_A2B:input_momenta_A2B, \
                                        self.generation_pitch_B:gen_pitch, \
                                        self.generation_momenta_A2B:gen_momenta, \
                                        self.generation_mfc_B:gen_mfc})
        
        return gen_momenta, gen_pitch, gen_mfc, enc_loss, dec_loss, gen_loss

    def save(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        self.saver.save(self.sess, os.path.join(directory, filename))

    def load(self, filepath):

        self.saver.restore(self.sess, filepath)

