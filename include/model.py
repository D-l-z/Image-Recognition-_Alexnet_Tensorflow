import tensorflow as tf

def model():
    _IMAGE_SIZE = 227
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10
    
    # 参数丢弃率,置0则取消
    dropout_rate_full = 0.5
    ###权重正则项，scale置0即取消
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
    
    with tf.name_scope('main_params'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')  
        
    with tf.variable_scope('conv1') as scope:
        conv1 = tf.layers.conv2d(
            inputs=x_image,
            filters=96,
            kernel_size=[11, 11],
            strides = 4,
            padding='VALID',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer
        )
        lrn1 = tf.nn.lrn(conv1,4,bias=1,alpha=1e-3/9,beta=0.75,name="lrn1")
        pool1 = tf.layers.max_pooling2d(lrn1, pool_size=[3, 3], strides=2, padding='VALID')
        

    with tf.variable_scope('conv2') as scope:
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=256,
            kernel_size=[5, 5],
            padding='SAME',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer
        )
        lrn2 = tf.nn.lrn(conv2,4,bias=1,alpha=1e-3/9,beta=0.75,name="lrn2")
        pool2 = tf.layers.max_pooling2d(lrn2, pool_size=[3, 3], strides=2, padding='VALID')
        
        
		
    with tf.variable_scope('conv3') as scope:
        conv3 = tf.layers.conv2d(
            inputs=pool2,
            filters=384,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer
        )
#        pool3 = tf.layers.max_pooling2d(conv3, pool_size=[2, 2], strides=2, padding='SAME')
    
    with tf.variable_scope('conv4') as scope:
        conv4 = tf.layers.conv2d(
            inputs=conv3,
            filters=384,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer
        )

    with tf.variable_scope('conv5') as scope:
        conv5 = tf.layers.conv2d(
            inputs=conv4,
            filters=256,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu,
            kernel_regularizer=regularizer
        )
        pool5 = tf.layers.max_pooling2d(conv5, pool_size=[3, 3], strides=2, padding='VALID')
        
        
    with tf.variable_scope('fc1') as scope:
        flat = tf.reshape(pool5, [-1, 6*6*256])
        fc1 = tf.layers.dense(
                inputs=flat, 
                units=4096, 
                activation=tf.nn.relu, 
                kernel_regularizer=regularizer)
        drop_fc1 = tf.layers.dropout(fc1, rate=dropout_rate_full)
        
    with tf.variable_scope('fc2') as scope:
        fc2 = tf.layers.dense(
                inputs=drop_fc1, 
                units=4096, 
                activation=tf.nn.relu, 
                kernel_regularizer=regularizer)
        drop_fc2 = tf.layers.dropout(fc2, rate=dropout_rate_full)
        
    with tf.variable_scope('fc3') as scope:
        softmax = tf.layers.dense(
                inputs=drop_fc2, 
                units=_NUM_CLASSES, 
                name=scope.name, 
                kernel_regularizer=regularizer)
        
    y_pred_cls = tf.argmax(softmax, axis=1)

    return x, y, softmax, y_pred_cls, global_step, learning_rate


def lr(epoch):
    learning_rate = 1e-3
    if epoch > 80:
        learning_rate *= 0.5e-3
    elif epoch > 60:
        learning_rate *= 1e-3
    elif epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 20:
        learning_rate *= 1e-1
    return learning_rate
        
