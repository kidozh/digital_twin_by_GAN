from keras.layers.merge import add,concatenate
from keras.layers.convolutional import Conv2D,MaxPooling2D,ZeroPadding2D,AveragePooling2D,Conv1D,MaxPooling1D
from keras.layers.core import Dense,Activation,Flatten,Dropout,Masking
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input,TimeDistributed
from keras_contrib.layers.normalization import GroupNormalization

def first_block(tensor_input,filters,kernel_size=3,pooling_size=1,dropout=0.5):
    k1,k2 = filters

    out = Conv1D(k1,1,padding='same')(tensor_input)
    out = GroupNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(k2,kernel_size,strides=2,padding='same')(out)


    pooling = MaxPooling1D(pooling_size,strides=2,padding='same')(tensor_input)


    # out = merge([out,pooling],mode='sum')
    out = add([out,pooling])
    return out

def repeated_block(x,filters,kernel_size=3,pooling_size=1,dropout=0.5):

    k1,k2 = filters


    out = GroupNormalization()(x)
    out = Activation('relu')(out)
    out = Conv1D(k1,kernel_size,padding='same')(out)
    out = GroupNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(dropout)(out)
    out = Conv1D(k2,kernel_size,strides=2,padding='same')(out)


    pooling = MaxPooling1D(pooling_size,strides=2,padding='same')(x)

    out = add([out, pooling])

    #out = merge([out,pooling])
    return out

def build_multi_input_main_residual_network(batch_size,
                                time_length,
                                input_dim,
                                output_dim,
                                loop_depth=15,
                                dropout=0.5):
    inp = Input(shape=(time_length,input_dim),name='a2')
    out = Conv1D(128,5)(inp)
    out = GroupNormalization()(out)
    out = Activation('relu')(out)

    out = first_block(out,(64,128),dropout=dropout)

    for _ in range(loop_depth):
        out = repeated_block(out,(64,128),dropout=dropout)

    # add flatten
    out = Flatten()(out)

    out = GroupNormalization()(out)
    out = Activation('relu')(out)
    out = Dense(output_dim)(out)

    model = Model(inputs=[inp],outputs=[out])

    model.compile(loss='logcosh',optimizer='adam',metrics=['mse','mae'])
    return model