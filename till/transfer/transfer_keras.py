from functools import reduce
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras import models, layers, optimizers
from sklearn.metrics import classification_report, confusion_matrix

from datetime import datetime
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from till.utils import (create_multiple_folder)


class tl_k(object):

    """asdfasfd"""

    def __init__(self, args: {}):
        self.pretrained_model = args["pretrained_model"]
        self.image_size = args["image_size"]
        self.freeze_layers = args["freeze_layers"]
        self.experiment = args["experiment"]
        create_multiple_folder(
            ["experiments", "experiments/" + self.experiment,
             "experiments/"+self.experiment+"/roc",
             "experiments/"+self.experiment+"/errors"])

    def freeze_layers_model(self):
        """dfasdf"""
        for layer in self.pre_model.layers[:-self.freeze_layers]:
            layer.trainable = False
        return True

    def select_model(self):
        """dfsad"""
        print(self.pretrained_model)
        model = self.pretrained_model
        if model == "VGG16":
            return VGG16(weights="imagenet", include_top=False,
                         input_shape=(self.image_size, self.image_size, 3))
        if model == "VGG19":
            return VGG19(weights="imagenet", include_top=False,
                         input_shape=(self.image_size, self.image_size, 3))
        if model == "InceptionV3":
            return InceptionV3(weights="imagenet", include_top=False,
                               input_shape=(self.image_size, self.image_size, 3))
        if model == "ResNet50":
            return ResNet50(weights="imagenet", include_top=False,
                            input_shape=(self.image_size, self.image_size, 3))

    def pretrain_model(self):
        """dafasdf"""
        try:
            self.pre_model = self.select_model()
            self.freeze_layers_model()
            print("Layers Freezed OK!")
        except:
            print("An error occured in the pretrain_model.")

    def add_dense(self, config_layer):
        """asfasdf"""
        number_neurons = config_layer[1]
        func_activation = config_layer[2]
        self.classifier.add(layers.Dense(
            number_neurons, activation=func_activation))
        print("Added Dense Layer")

    def add_dropout(self, config_layer):
        """dfasd"""
        rate = config_layer[1]
        self.classifier.add(layers.Dropout(float(rate)))
        print("Added Dropout Layer")

    def add_layers(self, layer_list):

        print("This is the layer:", layer_list)
        """dfsd"""
        {
            "Dense": lambda layer: self.add_dense(layer),
            "Dropout": lambda layer: self.add_dropout(layer)
        }[layer_list[0]](layer_list)
        print("Layers Added")

    def create_classifier(self, layer_list: list=list()):
        """asfsdf"""
        self.classifier = models.Sequential()
        try:
            self.pretrain_model()
        except:
            print("An error occured in the pretrained.")
        self.classifier.add(self.pre_model)
        self.classifier.add(layers.Flatten())

        for layer in layer_list:
            self.add_layers(layer)
        self.classifier.compile(loss="categorical_crossentropy",
                                optimizer=optimizers.RMSprop(1e-4), metrics=["acc"])

    def train(self, batch_train, batch_valid, batch_test, epochs):
        self.train_datagen = ImageDataGenerator(
            rescale=1./255, shear_range=0.2, zoom_range=0.4, horizontal_flip=True)
        self.test_datagen = ImageDataGenerator(rescale=1./255)

        self.train_generator = self.train_datagen.flow_from_directory(
            "dataset/train",
            target_size=(self.image_size, self.image_size),
            class_mode="categorical",
            shuffle=True,
            batch_size=batch_train)

        self.validation_generator = self.test_datagen.flow_from_directory(
            "dataset/val",
            batch_size=batch_valid,
            class_mode="categorical",
            shuffle=True,
            target_size=(self.image_size, self.image_size))

        self.test_generator = self.test_datagen.flow_from_directory(
            "dataset/test",
            batch_size=batch_valid,
            class_mode="categorical",
            shuffle=False,
            target_size=(self.image_size, self.image_size))

        self.model = self.classifier.fit_generator(self.train_generator,
                                                   steps_per_epoch=self.train_generator.samples/batch_train,
                                                   epochs=epochs,
                                                   validation_data=self.validation_generator,
                                                   validation_steps=self.validation_generator.samples /
                                                   self.validation_generator.batch_size
                                                   )
        print("Done!")

    def test(self, display=False):
        self.file_names = self.test_generator.filenames
        self.ground_truth = self.test_generator.classes
        self.label2index = self.test_generator.class_indices
        self.idx2label = dict((v, k) for k, v in self.label2index.items())
        self.predictions = self.classifier.predict_generator(
            self.test_generator, steps=self.test_generator.samples/self.test_generator.batch_size, verbose=1)
        self.predicted_classes = np.argmax(self.predictions, axis=1)
        self.errors = np.where(
            self.predicted_classes != self.ground_truth)[0]
        print("No of errors = {}/{}".format(len(self.errors),
                                            self.test_generator.samples))
        if display:
            for i in range(len(self.errors)):
                pred_class = np.argmax(self.predictions[self.errors[i]])
                pred_label = self.idx2label[pred_class]

                title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
                    self.file_names[self.errors[i]].split('/')[0],
                    pred_label,
                    self.predictions[self.errors[i]][pred_class])

                original = load_img(
                    '{}/{}'.format("dataset/test", self.file_names[self.errors[i]]))
                plt.clf()
                plt.figure(figsize=[7, 7])
                plt.axis('off')
                plt.title(title)
                plt.imshow(original)
                plt.show()

    def compute_roc(self, zoom=False):

        y_test = label_binarize(
            self.ground_truth, classes=np.arange(self.predictions.shape[1]))
        line_width = 2
        false_positive_recall = dict()
        true_positive_recall = dict()
        roc_auc = dict()
        for i in range(self.predictions.shape[1]):
            false_positive_recall[i], true_positive_recall[i], _ = roc_curve(
                y_test[:, i], self.predictions[:, i])
            roc_auc[i] = auc(false_positive_recall[i],
                             true_positive_recall[i])

        false_positive_recall["micro"], true_positive_recall["micro"], _ = roc_curve(
            y_test.ravel(), self.predictions.ravel())
        roc_auc["micro"] = auc(
            false_positive_recall["micro"], true_positive_recall["micro"])

        all_false_positive_recall = np.unique(
            np.concatenate([false_positive_recall[i] for i in range(self.predictions.shape[1])]))
        mean_true_positive_recall = np.zeros_like(
            all_false_positive_recall)
        for i in range(self.predictions.shape[1]):
            mean_true_positive_recall += interp(
                all_false_positive_recall, false_positive_recall[i], true_positive_recall[i])
        mean_true_positive_recall /= self.predictions.shape[1]
        false_positive_recall["macro"] = all_false_positive_recall
        true_positive_recall["macro"] = mean_true_positive_recall
        roc_auc["macro"] = auc(
            false_positive_recall["macro"], true_positive_recall["macro"])

        # Plot all ROC curves
        plt.clf()
        plt.figure(1)
        plt.plot(false_positive_recall["micro"], true_positive_recall["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

        plt.plot(false_positive_recall["macro"], true_positive_recall["macro"],
                 label='macro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)

        colors = cycle(
            ['aqua', 'darkorange', 'cornflowerblue', "tomato", "darkcyan", "navy"])
        for i, color in zip(range(self.predictions.shape[1]), colors):
            plt.plot(false_positive_recall[i], true_positive_recall[i], color=color, lw=line_width,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(self.idx2label[i], roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=line_width)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(
            'Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.show()
        if zoom:
            plt.figure(2)
            plt.xlim(0, 0.2)
            plt.ylim(0.8, 1)
            plt.plot(false_positive_recall["micro"], true_positive_recall["micro"],
                     label='micro-average ROC curve (area = {0:0.2f})'
                     ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)

            plt.plot(false_positive_recall["macro"], true_positive_recall["macro"],
                     label='macro-average ROC curve (area = {0:0.2f})'
                     ''.format(roc_auc["macro"]),
                     color='navy', linestyle=':', linewidth=4)

            colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
            for i, color in zip(range(self.predictions.shape[1]), colors):
                plt.plot(false_positive_recall[i], true_positive_recall[i], color=color, lw=line_width,
                         label='ROC curve of class {0} (area = {1:0.2f})'
                         ''.format(i, roc_auc[i]))

            plt.plot([0, 1], [0, 1], 'k--', lw=line_width)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(
                'Some extension of Receiver operating characteristic to multi-class')
            plt.legend(loc="lower right")
            plt.show()

    def compute_c_matrix(self):
        metrics_model = confusion_matrix(self.test_generator.classes,
                                         self.predicted_classes)
        print("Metricas")
        print(metrics_model)
        TruePositive = np.diag(metrics_model)

        FalsePositive = []
        for i in range(self.predictions.shape[1]):
            FalsePositive.append(
                sum(metrics_model[:, i]) - metrics_model[i, i])

        FalseNegative = []
        for i in range(self.predictions.shape[1]):
            FalseNegative.append(
                sum(metrics_model[i, :]) - metrics_model[i, i])

        TrueNegative = []
        for i in range(self.predictions.shape[1]):
            temp = np.delete(metrics_model, i, 0)   # delete ith row
            temp = np.delete(temp, i, 1)  # delete ith column
            TrueNegative.append(sum(sum(temp)))

        print("TruePositive", TruePositive)
        print("FalsePositive", FalsePositive)
        print("FalseNegative", FalseNegative)
        print("TrueNegative", TrueNegative)

        precision = (TruePositive)/(TruePositive+FalsePositive)
        pre_prom = np.sum(precision)/self.predictions.shape[1]
        print("Presicion Promedio", pre_prom)
        return pre_prom
