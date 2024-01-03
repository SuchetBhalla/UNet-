from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, \
                                    MaxPool2D, Conv2DTranspose, concatenate, \
                                    Input, Cropping2D

from tensorflow.keras.models import Model

def conv_block(inp, num_filters):

    global c
    out = Conv2D( num_filters, (3, 3), padding="same", name= "Conv_" + str(c))(inp)
    out = BatchNormalization(name= "BN_"+ str(c))(out)
    out = Activation("relu", name= "ReLU_"+ str(c))(out)
    c = c+1

    out = Conv2D(num_filters, (3, 3), padding="same", name= "Conv_" + str(c))(out)
    out = BatchNormalization(name= "BN_"+ str(c))(out)
    out = Activation("relu", name= "ReLU_"+ str(c))(out)
    c = c+1

    return out

def encoder_block(inp, num_filters):

    out = conv_block(inp, num_filters)
    pool = MaxPool2D((2,2), name ="MP_" + str(c-1))(out)

    return out, pool

def decoder_block( inp, num_filters, skip_connec):
    global d, s
    out = Conv2DTranspose( num_filters, (2,2), 2, name= "UpConv_"+str(d))(inp)
    out = concatenate( [out, skip_connec], name= "Concat_" + str(d))
    d = d+1
    out = conv_block( out, num_filters)

    return out

def build_unet(input_shape):
    #These variables are used to name the layers
    global c, d, s
    c, d, s = 1, 1, 1

    inputs = Input( input_shape, name= "Input" )

    krnl = 64
    kernels = []
    for i in range(4):
        kernels.append(krnl * 2**i)

    #x00
    sk00, mp00 = encoder_block(inputs, kernels[0])

    #x10
    sk10, mp10 = encoder_block(mp00, kernels[1])

    #x01
    sk01 = decoder_block( sk10, kernels[0], sk00 )

    #x20
    sk20, mp20 = encoder_block(mp10, kernels[2])

    #x11
    sk11 = decoder_block( sk20, kernels[1], sk10 )

    #x02
    sk02 = decoder_block( sk11, kernels[0], sk01 )

    #x30
    sk30, mp30 = encoder_block(mp20, kernels[3])

    #x21
    sk21 = decoder_block( sk30, kernels[2], sk20 )

    #x12
    sk12 = decoder_block( sk21, kernels[1], sk11 )

    #x03
    sk03 = decoder_block( sk12, kernels[0], sk02 )

    #x40
    sk40 = conv_block(mp30, 512)

    #x31
    sk31 = decoder_block( sk40, kernels[3], sk30 )

    #x22
    sk22 = decoder_block( sk31, kernels[2], sk21 )

    #x13
    sk13 = decoder_block( sk22, kernels[1], sk12 )

    #x04
    sk04 = decoder_block( sk13, kernels[0], sk03 )

    outputs = Conv2D( filters= 1, kernel_size= (1, 1), strides=1,\
                     activation= "sigmoid", padding= "same", name= "Output")(sk04)

    model = Model(inputs, outputs, name= "U_net_pp")
    return model
