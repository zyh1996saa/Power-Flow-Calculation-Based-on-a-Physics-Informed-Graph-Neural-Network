import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    def __init__(self, units, activation=tf.identity, l2=0.0):
        super(Attention, self).__init__()
        #print('units:',units)
        self.l2 = l2
        self.activation = activation
        self.units = units

    def build(self, input_shape):
        #print(input_shape)
        H_shape, A_shape = input_shape
        
        #self.W 维度为[4,7]
        self.W = self.add_weight(
          shape=(H_shape[2], self.units),
          initializer='glorot_uniform',
          dtype=tf.float32,
          #regularizer=tf.keras.regularizers.l2(self.l2)
        )
        #print('self.W.shape:',self.W.shape)
        #1.self.W.shape:(4, 4)
        #2.self.W.shape:(32, 4)
        
        self.a_1 = self.add_weight(
          shape=(self.units, 1),
          initializer='glorot_uniform',
          dtype=tf.float32,
          #regularizer=tf.keras.regularizers.l2(self.l2)
        )
        #1.self.a1.shape:(4, 1)
        #self.a1 shape [7,1]
        self.a_2 = self.add_weight(
          shape=(self.units, 1),
          initializer='glorot_uniform',
          dtype=tf.float32,
          #regularizer=tf.keras.regularizers.l2(self.l2)
        )
        #1.self.a2.shape:(4, 1)
        
    def call(self, inputs):
        # H :[None,1433] A:[None,2708]
        H, A = inputs
        #print(H.shape,A.shape)
        #H.shape:(None, 39, 4) A.shape:(None, 39, 39)
        # X:[None,7]
        X = tf.matmul(H, tf.cast(self.W,dtype=tf.complex64))
        #print(X.shape)
        #X.shape:(None, 39, 4)
        
        attn_self = tf.matmul(X, tf.cast(self.a_1,dtype=tf.complex64))
        #print(attn_self.shape)
        #attn_self.shape:(None, 39, 1)
        
        attn_neighbours = tf.matmul(X, tf.cast(self.a_2,dtype=tf.complex64))
        #print(attn_neighbours.shape)
        #attn_neighbours.shape:(None, 39, 1)
        
        
        attention = attn_self + tf.transpose(attn_neighbours,perm=[0, 2, 1])
        #print(tf.transpose(attn_neighbours,perm=[0, 2, 1]).shape)
        #print(attention.shape)
        
        
        E1 = tf.nn.leaky_relu(tf.math.real(attention))
        E2 = tf.nn.leaky_relu(tf.math.imag(attention))
        E = tf.complex(E1,E2)
        #print('---------')
        #print(E.shape)
        mask = mask = -10e9 * (1.0 - A)
        #print(mask.shape)
        masked_E = E + mask

        # A = tf.cast(tf.math.greater(A, 0.0), dtype=tf.float32)
        alpha1 = tf.nn.softmax(tf.math.real(masked_E))
        alpha2 = tf.nn.softmax(tf.math.imag(masked_E))
        alpha = tf.complex(alpha1,alpha2)
        H_cap = alpha @ X
        out1 = self.activation(tf.math.real(H_cap))
        out2 = self.activation(tf.math.imag(H_cap))
        out = tf.complex(out1,out2)
        return out



class GraphAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, units, num_heads, output_layer=False, activation=tf.identity, l2=0.0):
        super(GraphAttentionLayer, self).__init__()

        self.activation = activation
        self.num_heads = num_heads
        self.output_layer = output_layer

        self.attn_layers = [Attention(units, l2=l2) for x in range(num_heads)]

    def call(self, inputs):

        H, A = inputs
        #print('H.shape:',H.shape)
        #print('A.shape:',A.shape)
        #H.shape: (None, 39, 4)
        #A.shape: (None, 39, 39)
        H_out = [self.attn_layers[i]([H, A]) for i in range(self.num_heads)]

        if self.output_layer:
            multi_head_attn = tf.reduce_mean(tf.stack(H_out), axis=0)
            out1 = self.activation(tf.math.real(multi_head_attn))
            out2 = self.activation(tf.math.imag(multi_head_attn))
            out = tf.complex(out1,out2)
        else:
            multi_head_attn = tf.concat(H_out, axis=-1)
            out1 = self.activation(tf.math.real(multi_head_attn))
            out2 = self.activation(tf.math.imag(multi_head_attn))
            out = tf.complex(out1,out2)
        return out
