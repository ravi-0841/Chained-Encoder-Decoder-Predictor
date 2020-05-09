import tensorflow as tf
from modules import *

def encoder(input_mfc, input_pitch, final_filters=4, \
        reuse=False, scope_name='encoder'):

    inputs = tf.concat([input_mfc, input_pitch], axis=1, 
            name='encoder_input')

    # inputs has shape [batch_size, num_features, time]
    # we need to convert it to [batch_size, time, num_features] for 1D convolution
    inputs_transposed = tf.transpose(inputs, perm=[0, 2, 1], 
            name='encoder_input_transpose')

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv1d_layer(inputs=inputs_transposed, filters=64, kernel_size=15, \
                strides=1, activation=None, name='h1_conv')
        h1_gates = conv1d_layer(inputs=inputs_transposed, filters=64, kernel_size=15, \
                strides=1, activation=None, name='h1_conv_gates')
        h1_glu = gated_linear_layer(inputs=h1, gates=h1_gates, name='h1_glu')

        # Downsample
        d1 = downsample1d_block(inputs=h1_glu, filters=128, kernel_size=5, \
                strides=2, name_prefix='downsample1d_block1_')
        d2 = downsample1d_block(inputs=d1, filters=256, kernel_size=5, \
                strides=2, name_prefix='downsample1d_block2_')

        # Residual blocks
        r1 = residual1d_block(inputs=d2, filters=512, kernel_size=3, \
                strides=1, name_prefix='residual1d_block1_')
        r2 = residual1d_block(inputs=r1, filters=512, kernel_size=3, \
                strides=1, name_prefix='residual1d_block2_')

        # Upsample
        u1 = upsample1d_block(inputs=r2, filters=512, kernel_size=5, \
                strides=1, shuffle_size=2, name_prefix='upsample1d_block1_')
        u2 = upsample1d_block(inputs=u1, filters=256, kernel_size=5, \
                strides=1, shuffle_size=2, name_prefix='upsample1d_block2_')

        # Output
        o1 = conv1d_layer(inputs=u2, filters=final_filters, kernel_size=15, \
                strides=1, activation=None, name='o1_conv')
        o2 = tf.transpose(o1, perm=[0, 2, 1], name='output_transpose')
        o2 = tf.reduce_mean(o2, axis=1, keepdims=True)
        
    return o2
    
def decoder(input_momenta, input_pitch, final_filters=4, \
        reuse=False, scope_name='decoder'):

    inputs = tf.concat([input_pitch, input_momenta], 
            axis=1, name='decoder_input')

    # inputs has shape [batch_size, num_features, time]
    # we need to convert it to [batch_size, time, num_features] for 1D convolution
    inputs_transposed = tf.transpose(inputs, perm=[0, 2, 1], 
            name='decoder_input_transpose')

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv1d_layer(inputs=inputs_transposed, filters=64, \
                kernel_size=15, strides=1, activation=None, name='h1_conv')
        h1_gates = conv1d_layer(inputs=inputs_transposed, filters=64, \
                kernel_size=15, strides=1, activation=None, name='h1_conv_gates')
        h1_glu = gated_linear_layer(inputs=h1, gates=h1_gates, name='h1_glu')

        # Downsample
        d1 = downsample1d_block(inputs=h1_glu, filters=128, kernel_size=5, \
                strides=2, name_prefix='downsample1d_block1_')
        d2 = downsample1d_block(inputs=d1, filters=256, kernel_size=5, \
                strides=2, name_prefix='downsample1d_block2_')

        # Residual blocks
        r1 = residual1d_block(inputs=d2, filters=512, kernel_size=3, \
                strides=1, name_prefix='residual1d_block1_')
        r2 = residual1d_block(inputs=r1, filters=512, kernel_size=3, \
                strides=1, name_prefix='residual1d_block2_')

        # Upsample
        u1 = upsample1d_block(inputs=r2, filters=512, kernel_size=5, \
                strides=1, shuffle_size=2, name_prefix='upsample1d_block1_')
        u2 = upsample1d_block(inputs=u1, filters=256, kernel_size=5, \
                strides=1, shuffle_size=2, name_prefix='upsample1d_block2_')

        # Output
        o1 = conv1d_layer(inputs=u2, filters=final_filters, kernel_size=15, \
                strides=1, activation=None, name='o1_conv')
        o2 = tf.transpose(o1, perm=[0, 2, 1], name='output_transpose')
        o2 = tf.reduce_mean(o2, axis=1, keepdims=True)
        
    return o2


def generator(input_mfc, input_pitch, num_mfc=23, training=True, \
                reuse=False, scope_name='generator'):

    inputs = tf.concat([input_mfc, input_pitch], axis=1, 
            name='generator_input')

    # inputs has shape [batch_size, num_features, time]
    # we need to convert it to [batch_size, time, num_features] for 1D convolution
    inputs_transposed = tf.transpose(inputs, perm=[0, 2, 1], 
            name='generator_input_transpose')

    with tf.variable_scope(scope_name) as scope:
        # Discriminator would be reused in CycleGAN
        if reuse:
            scope.reuse_variables()
        else:
            assert scope.reuse is False

        h1 = conv1d_layer(inputs=inputs_transposed, filters=64, kernel_size=15, \
                strides=1, activation=None, name='h1_conv')
        h1_gates = conv1d_layer(inputs=inputs_transposed, filters=64, kernel_size=15, \
                strides=1, activation=None, name='h1_conv_gates')
        h1_glu = gated_linear_layer(inputs=h1, gates=h1_gates, name='h1_glu')

        # Downsample
        d1 = downsample1d_block(inputs=h1_glu, filters=128, kernel_size=5, \
                strides=2, name_prefix='downsample1d_block1_')
        d2 = downsample1d_block(inputs=d1, filters=256, kernel_size=5, \
                strides=2, name_prefix='downsample1d_block2_')

        # Residual blocks
        r1 = residual1d_block(inputs=d2, filters=512, kernel_size=3, \
                strides=1, name_prefix='residual1d_block1_')
        r2 = residual1d_block(inputs=r1, filters=512, kernel_size=3, \
                strides=1, name_prefix='residual1d_block2_')
        r3 = residual1d_block(inputs=r2, filters=512, kernel_size=3, \
                strides=1, name_prefix='residual1d_block3_')

        # Upsample
        u1 = upsample1d_block(inputs=r3, filters=512, kernel_size=5, \
                strides=1, shuffle_size=2, name_prefix='upsample1d_block1_')
        u2 = upsample1d_block(inputs=u1, filters=256, kernel_size=5, \
                strides=1, shuffle_size=2, name_prefix='upsample1d_block2_')

        # Output
        o1 = conv1d_layer(inputs=u2, filters=num_mfc, kernel_size=15, \
                strides=1, activation=None, name='o1_conv')
        o2 = tf.transpose(o1, perm=[0, 2, 1], name='output_transpose')
        
    return o2

