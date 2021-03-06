from __future__ import print_function
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from tensorflow.python.client import timeline as tf_timeline
import numpy as np
import cv2
import datetime

import TensorflowUtils as utils
import read_LadybugData as scene_parsing
import datetime
import BatchDatsetReader as dataset
import pdb
from six.moves import xrange

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/model_0612_newloss/", "path to logs directory")
tf.flags.DEFINE_string("vis_dir", "logs/vis/test_newloss/", "path to save results of visualization")
tf.flags.DEFINE_string("data_dir", "Data_zoo/ladybug/", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-5", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_bool('timeline', "False", "Timeline mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(7e4 + 1)
NUM_OF_CLASSESS = 9
IMAGE_SIZE = 224
IMAGE_HEIGHT = 144
IMAGE_WIDTH = 1080
R = 6
w1 = tf.constant(0.6)
w2 = tf.constant(0.4)
# w3 = tf.constant(1)
# M = tf.constant((2*R+1)*(2*R+1)-1)
w3M = tf.constant(1.0/float((2*R+1)*(2*R+1)-1))

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    return net


def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    with tf.variable_scope("inference"):
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]

        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)

        annotation_pred_no0 = tf.argmax(conv_t3[:,:,:,1:], dimension=3, name="prediction_no0") + 1
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), tf.expand_dims(annotation_pred_no0, dim=3), conv_t3


def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)

def FixLogitsWithIgnoreClass(logits, labels):
    # Make zeros everywhere except for the ignored label
    ignoreLabel = 0
    tmpLabel0 = tf.expand_dims(labels, -1)
    tmpLabel1 = tf.ones_like(logits[:,:,:,1:], dtype=tf.int32)
    ignoreMatrix = tf.concat([tmpLabel0, tmpLabel1], 3)
    # Make corresponding logits big enough
    logitsFixed = tf.where(tf.equal(ignoreMatrix, 0), 1e20 * tf.ones_like(logits), logits)
    return logitsFixed

def cond(i, consLoss, indexList, neighbourList, logits, image):
    return i < indexListlen

def body(i, consLoss, indexList, neighbourList, logits, image):
    # tmp = tf.reduce_sum(tf.multiply(logits[:,indexList[i][0],indexList[i][1],:], logits[:,neighbourList[i][0],neighbourList[i][1],:]))
    tmp = tf.cast(indexList[i][0], tf.float32)
    # tmp = tf.cast(logits[0,1,1,0], tf.float32)
    consLoss = tf.add(consLoss, tf.negative(tf.multiply(w3M, tmp)))
    return i + 1, consLoss, indexList, neighbourList, logits, image

def GetConstraintLoss(indexList, neighbourList, logits, image):
    i, consLoss, indexList, neighbourList, logits, image = tf.while_loop(cond, body, (0, tf.zeros(FLAGS.batch_size), indexList, neighbourList, logits, image))
    return consLoss

def PrepareNeighbour():
    index = []
    neighbours = []
    for i in range(0, IMAGE_HEIGHT, R):
        for j in range(0, IMAGE_WIDTH, R):
            for di in range(-R/2, R/2):
                for dj in range(-R/2, R/2):
                    x = i + di
                    y = j + dj
                    if ((di != 0 or dj != 0) and (x >= 0) and (y >= 0) and (x < IMAGE_HEIGHT) and(y < IMAGE_WIDTH)):
                        index.append([i,j])
                        neighbours.append([x, y])
    return index, neighbours

index, neighbour = PrepareNeighbour()
indexListlen = len(index)

def main(argv=None):
    indexList = tf.constant(index)
    neighbourList = tf.constant(neighbour)

    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="input_image")
    annotation = tf.placeholder(tf.int32,shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, 1], name="annotation")

    pred_annotation, pred_annotation_no0, logits = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)
    labels = tf.squeeze(annotation, squeeze_dims=[3])
    fixedLogits = FixLogitsWithIgnoreClass(logits, labels)
    # Calculate loss
    class_weights = tf.constant([0.1, 1., 1., 0.1, 20., 0.1, 8., 8., 0.1])
    onehot_labels = tf.one_hot(labels, depth=9)
    weights = tf.reduce_sum(class_weights * onehot_labels, axis=3)
    unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fixedLogits, labels=labels, name="entropy")
    weighted_loss = unweighted_loss * weights
    loss = tf.reduce_mean(weighted_loss)
        loss_constraint = GetConstraintLoss(indexList, neighbourList, logits, image)
    loss = tf.add(loss, tf.reduce_mean(loss_constraint))

    # loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels, name="entropy")))
    tf.summary.scalar("entropy", loss)
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var)

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader...")
    train_records, valid_records, test_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print(len(train_records))
    print(len(valid_records))
    print(len(test_records))

    print("Setting up dataset reader")
    image_options = {'resize': False, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)
    if FLAGS.mode == 'test':
        test_dataset_reader = dataset.BatchDatset(test_records, image_options)

    sess = tf.Session()
    print("Setting up Saver...")
    saver = tf.train.Saver(max_to_keep = 20)
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    if FLAGS.timeline == True:
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

    if FLAGS.mode == "train":
        lastTime = datetime.datetime.now()
        for itr in xrange(MAX_ITERATION):
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}

            if FLAGS.timeline == True:
                sess.run(train_op, options=options, run_metadata=run_metadata, feed_dict=feed_dict)
                fetched_timeline = tf_timeline.Timeline(run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open('timeline/timeline_step_%d.json' % itr, 'w') as f:
                    f.write(chrome_trace)
            else:
                sess.run(train_op, feed_dict=feed_dict)

            # Debug
            nowTime = datetime.datetime.now()
            print((nowTime - lastTime).seconds,'sec, ', itr)
            lastTime = nowTime
            # output = sess.run(indexList, feed_dict=feed_dict)
            # print("logits shape : ", output.shape)
            # output = sess.run(loss_constraint, feed_dict=feed_dict)
            # print("loss shape: ", output.shape)
            # print("logits : ", logits[0][0][0])
            # output = sess.run(weights, feed_dict=feed_dict)
            # print("weight shape : ", output.shape)
            # print("weight : ", output[0])
            # output = sess.run(weighted_loss, feed_dict=feed_dict)
            # print("weighted_loss shape: ", output.shape)
            # print("weighted_loss : ", output)
            # output = sess.run(loss, feed_dict=feed_dict)
            # print("loss shape: ", output.shape)
            # print("loss : ", output)
            # Debug

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if (itr % 2000 == 0):
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", itr)

    elif FLAGS.mode == "visualize":
        valid_images, valid_annotations, valid_files = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.vis_dir, name="inp_" + valid_files[itr])
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.vis_dir, name="gt_" + valid_files[itr])
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.vis_dir, name="pred_" + valid_files[itr])
            print("Saved image: %d" % itr)

    elif FLAGS.mode == "test":
        # videoWriter = cv2.VideoWriter('test_newloss.avi', cv2.cv.CV_FOURCC('M', 'J', 'P', 'G'), 5, (IMAGE_WIDTH, IMAGE_HEIGHT), False)
        for itr in range(len(test_records)):
        # for itr in range(len(train_records)):
            test_images, test_annotations = test_dataset_reader.next_batch(1)
            pred = sess.run(pred_annotation, feed_dict={image: test_images, annotation: test_annotations,
                                                        keep_probability: 1.0})
            test_annotations = np.squeeze(test_annotations, axis=3)
            pred = np.squeeze(pred, axis=3)
            print(itr)
            # videoWriter.write(pred[0].astype(np.uint8))
            # utils.save_image(test_images[0].astype(np.uint8), FLAGS.vis_dir, name="inp_" + train_records[itr]['filename'])
            # utils.save_image(test_annotations[0].astype(np.uint8), FLAGS.vis_dir, name="gt_" + train_records[itr]['filename'])
            # utils.save_image(pred[0].astype(np.uint8), FLAGS.vis_dir, name="pred_" + train_records[itr]['filename'])
            utils.save_image(test_images[0].astype(np.uint8), FLAGS.vis_dir, name="inp_" + test_records[itr]['filename'])
            utils.save_image(test_annotations[0].astype(np.uint8), FLAGS.vis_dir, name="gt_" + test_records[itr]['filename'])
            utils.save_image(pred[0].astype(np.uint8), FLAGS.vis_dir, name="pred_" + test_records[itr]['filename'])
        # videoWriter.release()

if __name__ == "__main__":
    tf.app.run()
