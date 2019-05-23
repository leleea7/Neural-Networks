import pandas as pd
import numpy as np
import os
import data_preprocessing as dp
import tensorflow as tf
from architecture import TCDCN
import tqdm
import evaluation as ev
import plot_generator as pg


class MultiTaskRecognizer:

    def __init__(self):
        self.__BATCH_SIZE = 32
        self.__DATA_DIR = 'MAFL/'
        self.__TMP_DIR = 'tmp/'

        self.__tasks = ['Landmarks', 'Eyeglasses', 'Male', 'No_Beard', 'Smiling', 'Young']

        self.__xtrain = 19000
        self.__ytrain = 1000

        self.__len_landmarks = 10

        self.__shape = (40, 40)

        if not os.path.exists(self.__TMP_DIR):
            os.makedirs(self.__TMP_DIR)

        self.__GLOBAL_EPOCH = dp.global_epoch(self.__TMP_DIR + 'epoch.txt')

        self.__inp = {}
        self.__loss = {}
        self.__out = {}

        self.__graph = tf.Graph()
        with self.__graph.as_default():

            self.__images = tf.placeholder(tf.float32, shape=[None, self.__shape[0], self.__shape[1], 1])

            for task in self.__tasks:
                if task == 'Landmarks':
                    self.__inp[task] = tf.placeholder(tf.float32, shape=[None, self.__len_landmarks])
                else:
                    self.__inp[task] = tf.placeholder(tf.int32, shape=[None])

            output = TCDCN(self.__images, self.__tasks)

            for task in self.__tasks:
                if task == 'Landmarks':
                    # Root Mean Squared Error (RMSE)
                    self.__loss[task] = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.__inp[task], output[task]))))
                else:
                    self.__loss[task] = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output[task], labels=self.__inp[task]))

            self.__total_loss = 0
            for task in self.__tasks:
                self.__total_loss += self.__loss[task]

            self.__global_step = tf.Variable(0, trainable=False)
            self.__optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            self.__train_op = self.__optimizer.minimize(self.__total_loss, global_step=self.__global_step)

            for task in self.__tasks:
                if task == 'Landmarks':
                    self.__out[task] = output[task]
                else:
                    self.__out[task] = tf.argmax(output[task], axis=1)

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
        num_steps = self.__xtrain // self.__BATCH_SIZE

        # Open a writer to write summaries.
        self.__writer = tf.summary.FileWriter(self.__TMP_DIR, self.__session.graph)

        attributes = pd.read_csv(self.__DATA_DIR + 'list_attr_celeba.csv')
        landmarks = pd.read_csv(self.__DATA_DIR + 'list_landmarks_align_celeba.csv')

        for epoch in range(epochs):

            average_loss = 0
            f_train = open(self.__DATA_DIR + 'training.txt', 'r')

            for step in tqdm.tqdm(range(num_steps), desc='Epoch ' + str(epoch + 1 + self.__GLOBAL_EPOCH) + '/' + str(epochs + self.__GLOBAL_EPOCH)):

                images = []
                lands = []
                data = {task: [] for task in self.__tasks if task != 'Landmarks'}

                for _ in range(self.__BATCH_SIZE):
                    try:
                        row = f_train.readline()
                    except:
                        break

                    row = row.strip()
                    img, size = dp.load_image(self.__DATA_DIR + 'img_align_celeba/' + row, self.__shape)
                    attr = np.array(attributes[attributes['image_id'] == row][self.__tasks[1:]])[0]
                    attr = np.where(attr == -1, 0, attr)
                    l = np.array(landmarks[landmarks['image_id'] == row])[0][1:]
                    x = np.array([l[i] for i in range(0, len(l), 2)]) * self.__shape[0] / size[0]
                    y = np.array([l[i] for i in range(1, len(l), 2)]) * self.__shape[1] / size[1]
                    land = np.concatenate([x, y])

                    images.append(img)
                    lands.append(land)

                    for i in range(len(self.__tasks[1:])):
                        data[self.__tasks[i + 1]].append(attr[i])

                run_metadata = tf.RunMetadata()
                _, l = self.__session.run([self.__train_op, self.__total_loss],
                                          feed_dict=self.__create_feed_dict(images, lands, data),
                                          run_metadata=run_metadata)

                average_loss += l

                if (step % 100 == 0 and step > 0) or step == num_steps - 1:
                    y_true = {task: [] for task in self.__tasks}
                    y_pred = {task: [] for task in self.__tasks}
                    data = {}
                    f_test = open(self.__DATA_DIR + 'testing.txt', 'r')

                    for _ in range(self.__ytrain):
                        row = f_test.readline().strip()
                        img, size = dp.load_image(self.__DATA_DIR + 'img_align_celeba/' + row, self.__shape)
                        attr = np.array(attributes[attributes['image_id'] == row][self.__tasks[1:]])[0]
                        attr = np.where(attr == -1, 0, attr)
                        l = np.array(landmarks[landmarks['image_id'] == row])[0][1:]
                        x = np.array([l[i] for i in range(0, len(l), 2)]) * self.__shape[0] / size[0]
                        y = np.array([l[i] for i in range(1, len(l), 2)]) * self.__shape[1] / size[1]
                        land = np.concatenate([x, y])

                        pred = self.__session.run([self.__out[task] for task in self.__tasks],
                                                  feed_dict={self.__images: [img]},
                                                  run_metadata=run_metadata)

                        for i in range(len(self.__tasks[1:])):
                            try:
                                data[self.__tasks[i + 1]].append(attr[i])
                            except:
                                data[self.__tasks[i + 1]] = [attr[i]]

                        k = 0

                        for task in self.__tasks:
                            if task == 'Landmarks':
                                y_true[task].append(land)
                                y_pred[task].append(pred[k][0])
                                k += 1
                            else:
                                y_true[task].append(data[task][0])
                                y_pred[task].append(pred[k][0])
                                k += 1

                    print('Total Loss:', average_loss / step)
                    with open(self.__TMP_DIR + '/log_loss.txt', 'a', encoding='utf8') as f:
                        f.write(str(average_loss / step) + '\n')

                    for task in self.__tasks:
                        with open(self.__TMP_DIR + '/log_' + task + '.txt', 'a', encoding='utf8') as f:
                            if task == 'Landmarks':
                                rmse_landmarks = ev.rmse(np.asarray(y_true[task]), np.asarray(y_pred[task]))
                                print('RMSE landmarks:', rmse_landmarks)
                                f.write(str(rmse_landmarks) + '\n')
                            else:
                                acc = ev.accuracy(y_true[task], y_pred[task])
                                print('Accuracy ' + task + ': ' + str(acc))
                                f.write(str(acc) + '\n')

                    f_test.close()

                if step == num_steps - 1 and epoch + 1 == epochs:
                    self.__writer.add_run_metadata(run_metadata, 'step%d' % step)

            f_train.close()

        self.__saver.save(self.__session, os.path.join(self.__TMP_DIR, 'model.ckpt'))
        dp.global_epoch(self.__TMP_DIR + 'epoch.txt', update=self.__GLOBAL_EPOCH + epochs)

        self.__writer.close()

        for task in self.__tasks:
            if task == 'Landmarks':
                pg.generate_plot(data_dir=self.__TMP_DIR, mode='rmse', label=task, step=100)
            else:
                pg.generate_plot(data_dir=self.__TMP_DIR, label=task, step=100)

        pg.generate_loss_plot(self.__TMP_DIR, step=self.__BATCH_SIZE)


    def __create_feed_dict(self, img, landmarks, inputs):
        feed_dict = {}
        feed_dict[self.__images] = img
        for task in self.__tasks:
            if task == 'Landmarks':
                feed_dict[self.__inp[task]] = landmarks
            else:
                feed_dict[self.__inp[task]] = inputs[task]
        return feed_dict

    def predict(self, img):
        run_metadata = tf.RunMetadata()
        pred = self.__session.run([self.__out[k] for k in self.__tasks],
                                  feed_dict={self.__images: [img]},
                                  run_metadata=run_metadata)
        output = {}
        k = 0
        for task in self.__tasks:
            output[task] = pred[k]
            k += 1
        return output

'''a = MultiTaskRecognizer()
a.train()'''








