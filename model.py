import tensorflow as tf
import os
import data_preprocessing as dp
import evaluation as ev
import tqdm
from architecture import Xception, TCDCN
import numpy as np
import plot_generator as pg

class MultiTaskRecognizer:

    def __init__(self, tasks):
        self.__BATCH_SIZE = 32
        self.__DATA_DIR = 'MTFL/'
        self.__TMP_DIR = 'tmp/'

        self.__position = {'landmarks': 0, 'gender': 1, 'smile': 2, 'glasses': 3, 'head_pose': 4}
        self.__tasks = ['' for _ in self.__position]
        for task in tasks:
            self.__tasks[self.__position[task]] = task

        self.__len_landmarks = 10
        self.__len_gender = 2
        self.__len_smile = 2
        self.__len_glasses = 2
        self.__len_head_pose = 5

        self.__xtrain = 10000

        self.__shape = (40, 40)

        if not os.path.exists(self.__TMP_DIR):
            os.makedirs(self.__TMP_DIR)

        self.__GLOBAL_EPOCH = dp.global_epoch(self.__TMP_DIR + 'epoch.txt')

        self.__inp = {}
        self.__out = {}

        self.__graph = tf.Graph()
        with self.__graph.as_default():
            self.__images = tf.placeholder(tf.float32, shape=[None, self.__shape[0], self.__shape[1], 1])
            if 'landmarks' in self.__tasks:
                self.__inp['landmarks'] = tf.placeholder(tf.float32, shape=[None, self.__len_landmarks])
            if 'gender' in self.__tasks:
                self.__inp['gender'] = tf.placeholder(tf.int32, shape=[None])
            if 'smile' in self.__tasks:
                self.__inp['smile'] = tf.placeholder(tf.int32, shape=[None])
            if 'glasses' in self.__tasks:
                self.__inp['glasses'] = tf.placeholder(tf.int32, shape=[None])
            if 'head_pose' in self.__tasks:
                self.__inp['head_pose'] = tf.placeholder(tf.int32, shape=[None])

            output_landmarks, output_gender, output_smile, output_glasses, output_head_pose = TCDCN(self.__images, self.__tasks)

            if 'landmarks' in self.__tasks:
                with tf.name_scope('loss_landmarks'):
                    #Root Mean Squared Error (RMSE)
                    self.__loss_landmarks = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(output_landmarks, self.__inp['landmarks']))))
            else:
                self.__loss_landmarks = 0

            if 'gender' in self.__tasks:
                with tf.name_scope('loss_gender'):
                    self.__losses_gender = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_gender,
                                                                                          labels=self.__inp['gender'])
                    self.__loss_gender = tf.reduce_mean(self.__losses_gender)
            else:
                self.__loss_gender = 0

            if 'smile' in self.__tasks:
                with tf.name_scope('loss_smile'):
                    self.__losses_smile = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_smile,
                                                                                         labels=self.__inp['smile'])
                    self.__loss_smile = tf.reduce_mean(self.__losses_smile)
            else:
                self.__loss_smile = 0

            if 'glasses' in self.__tasks:
                with tf.name_scope('loss_glasses'):
                    self.__losses_glasses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_glasses,
                                                                                           labels=self.__inp['glasses'])
                    self.__loss_glasses = tf.reduce_mean(self.__losses_glasses)
            else:
                self.__loss_glasses = 0

            if 'head_pose' in self.__tasks:
                with tf.name_scope('loss_head_pose'):
                    self.__losses_head_pose = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output_head_pose,
                                                                                             labels=self.__inp['head_pose'])
                    self.__loss_head_pose = tf.reduce_mean(self.__losses_head_pose)
            else:
                self.__loss_head_pose = 0

            with tf.name_scope('optimizer'):
                self.__global_step = tf.Variable(0, trainable=False)
                self.__loss = self.__loss_landmarks + self.__loss_gender + self.__loss_smile + self.__loss_glasses + self.__loss_head_pose
                self.__optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
                self.__train_op = self.__optimizer.minimize(self.__loss, global_step=self.__global_step)

            with tf.name_scope('predictions'):
                if 'landmarks' in self.__tasks:
                    self.__out['landmarks'] = output_landmarks
                if 'gender' in self.__tasks:
                    self.__out['gender'] = tf.argmax(output_gender, axis=1)
                if 'smile' in self.__tasks:
                    self.__out['smile'] = tf.argmax(output_smile, axis=1)
                if 'glasses' in self.__tasks:
                    self.__out['glasses'] = tf.argmax(output_glasses, axis=1)
                if 'head_pose' in self.__tasks:
                    self.__out['head_pose'] = tf.argmax(output_head_pose, axis=1)

            self.__saver = tf.train.Saver()

            # Add variable initializer.
            init = tf.global_variables_initializer()

            ### SESSION ###
            self.__session = tf.Session(graph=self.__graph)

            # We must initialize all variables before we use them.
            init.run(session=self.__session)

            # reload the model if it exists and continue to train
            try:
                self.__saver.restore(self.__session, os.path.join(self.__TMP_DIR, 'model.ckpt'))
                print('Model restored')
            except:
                print('Model initialized')


    def train(self, epochs=1):
        num_steps = 10000 // self.__BATCH_SIZE

        # Open a writer to write summaries.
        self.__writer = tf.summary.FileWriter(self.__TMP_DIR, self.__session.graph)

        for epoch in range(epochs):

            average_loss = 0
            f_train = open(self.__DATA_DIR + 'training.txt', 'r')

            for step in tqdm.tqdm(range(num_steps), desc='Epoch ' + str(epoch + 1 + self.__GLOBAL_EPOCH) + '/' + str(epochs + self.__GLOBAL_EPOCH)):

                images = []
                landmarks = []
                gender = []
                smile = []
                glasses = []
                head_pose = []

                for _ in range(self.__BATCH_SIZE):
                    try:
                        row = f_train.readline()
                    except:
                        break
                    row = row.strip().split()
                    directory = row[0]
                    img, size = dp.load_image(self.__DATA_DIR + directory, self.__shape)
                    x = np.asarray([float(number) for number in row[1: 6]]) * self.__shape[0] / size[0]
                    y = np.asarray([float(number) for number in row[6: 11]]) * self.__shape[1] / size[1]
                    land = np.concatenate([x, y])
                    gend = int(row[11]) - 1
                    sm = int(row[12]) - 1
                    glass = int(row[13]) - 1
                    hp = int(row[14]) - 1

                    images.append(img)
                    landmarks.append(land)
                    gender.append(gend)
                    smile.append(sm)
                    glasses.append(glass)
                    head_pose.append(hp)

                run_metadata = tf.RunMetadata()
                _, l = self.__session.run([self.__train_op, self.__loss],
                                          feed_dict=self.__create_feed_dict(images, [landmarks, gender, smile, glasses, head_pose]),
                                          run_metadata=run_metadata)

                #average_loss_landmarks += l_landmarks
                average_loss += l

                # print loss and accuracy on test set every 3 steps
                if step > 0 and step % 3 == 0:
                    #evaluation on test set
                    y_true_landmarks = []
                    y_pred_landmarks = []
                    y_true_gender = []
                    y_pred_gender = []
                    y_true_smile = []
                    y_pred_smile = []
                    y_true_glasses = []
                    y_pred_glasses = []
                    y_true_head_pose = []
                    y_pred_head_pose = []

                    f_test = open(self.__DATA_DIR + 'testing.txt', 'r')
                    test_steps = 2995

                    for _ in range(test_steps):
                        row = f_test.readline()
                        row = row.strip().split()
                        directory = row[0]
                        img, size = dp.load_image(self.__DATA_DIR + directory, self.__shape)
                        x = np.asarray([float(number) for number in row[1: 6]]) * self.__shape[0] / size[0]
                        y = np.asarray([float(number) for number in row[6: 11]]) * self.__shape[1] / size[1]
                        land = np.concatenate([x, y])
                        gend = int(row[11]) - 1
                        sm = int(row[12]) - 1
                        glass = int(row[13]) - 1
                        hp = int(row[14]) - 1

                        pred = self.__session.run([self.__out[k] for k in self.__tasks if k],
                                                    feed_dict=self.__create_feed_dict([img], [[land], [gend], [sm], [glass], [hp]]),
                                                    run_metadata=run_metadata)

                        k = 0

                        if 'landmarks' in self.__tasks:
                            y_true_landmarks.append(land)
                            y_pred_landmarks.append(pred[k][0])
                            k += 1

                        if 'gender' in self.__tasks:
                            y_true_gender.append(gend)
                            y_pred_gender.append(pred[k][0])
                            k += 1

                        if 'smile' in self.__tasks:
                            y_true_smile.append(sm)
                            y_pred_smile.append(pred[k])
                            k += 1

                        if 'glasses' in self.__tasks:
                            y_true_glasses.append(glass)
                            y_pred_glasses.append(pred[k][0])
                            k += 1

                        if 'head_pose' in self.__tasks:
                            y_true_head_pose.append(hp)
                            y_pred_head_pose.append(pred[k][0])

                    print('Total Loss:', average_loss / step)
                    with open(self.__TMP_DIR + '/log_loss.txt', 'a', encoding='utf8') as f:
                        f.write(str(average_loss / step) + '\n')

                    if 'landmarks' in self.__tasks:
                        rmse_landmarks = ev.rmse(np.asarray(y_true_landmarks), np.asarray(y_pred_landmarks))
                        print('RMSE landmarks:', rmse_landmarks)
                        with open(self.__TMP_DIR + '/log_landmarks.txt', 'a', encoding='utf8') as f:
                            f.write(str(rmse_landmarks) + '\n')

                    if 'gender' in self.__tasks:
                        accuracy_gender = ev.accuracy(y_true_gender, y_pred_gender)
                        print('Accuracy gender:', accuracy_gender)
                        with open(self.__TMP_DIR + '/log_gender.txt', 'a', encoding='utf8') as f:
                            f.write(str(accuracy_gender) + '\n')

                    if 'smile' in self.__tasks:
                        accuracy_smile = ev.accuracy(y_true_smile, y_pred_smile)
                        print('Accuracy smile:', accuracy_smile)
                        with open(self.__TMP_DIR + '/log_smile.txt', 'a', encoding='utf8') as f:
                            f.write(str(accuracy_smile) + '\n')

                    if 'glasses' in self.__tasks:
                        accuracy_glasses = ev.accuracy(y_true_glasses, y_pred_glasses)
                        print('Accuracy glasses:', accuracy_glasses)
                        with open(self.__TMP_DIR + '/log_glasses.txt', 'a', encoding='utf8') as f:
                            f.write(str(accuracy_glasses) + '\n')

                    if 'head_pose' in self.__tasks:
                        accuracy_head_pose = ev.accuracy(y_true_head_pose, y_pred_head_pose)
                        print('Accuracy head pose:', accuracy_head_pose)
                        with open(self.__TMP_DIR + '/log_head_pose.txt', 'a', encoding='utf8') as f:
                            f.write(str(accuracy_head_pose) + '\n')

                    f_test.close()

                if step == (num_steps - 1) and epoch + 1 == epochs:
                    s = self.__session.run(self.__global_step)
                    self.__writer.add_run_metadata(run_metadata, 'step%d' % step, global_step=s)

            f_train.close()

        self.__saver.save(self.__session, os.path.join(self.__TMP_DIR, 'model.ckpt'))
        dp.global_epoch(self.__TMP_DIR + 'epoch.txt', update=self.__GLOBAL_EPOCH + epochs)

        self.__writer.close()

        for task in self.__tasks:
            if task == 'landmarks':
                pg.generate_plot(data_dir=self.__TMP_DIR, mode='rmse', label=task, batch_size=self.__BATCH_SIZE)
            else:
                pg.generate_plot(data_dir=self.__TMP_DIR, label=task, batch_size=self.__BATCH_SIZE)

        pg.generate_loss_plot(self.__TMP_DIR, batch_size=self.__BATCH_SIZE)

    def __create_feed_dict(self, img, inputs):
        feed_dict = {}
        feed_dict[self.__images] = img
        for key in self.__position:
            if key in self.__tasks:
                feed_dict[self.__inp[key]] = inputs[self.__position[key]]
        return feed_dict

a = MultiTaskRecognizer(tasks=['landmarks', 'glasses', 'smile', 'head_pose', 'gender'])
a.train()
