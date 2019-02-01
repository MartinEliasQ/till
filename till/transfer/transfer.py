"""Transfer Learning"""

import torch
from till.utils import (get_from_dict, metrics)
from .data_augmentation import ImgAugTransform
from torch import (optim, nn)
from torchvision import (datasets, transforms, models)
import PIL


class tl(object):
    def __init__(self, premodel):
        self.model = self.select_premodel(premodel)
        self.train_losses, self.test_losses = [], []

    @staticmethod
    def check_gpu():
        return (True, 'cuda') if torch.cuda.is_available() else (False, 'cpu')

    @staticmethod
    def create_loaders(train, val, test, batch_size=50):
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=batch_size)

        return (train_loader, val_loader, test_loader)

    @staticmethod
    def load_dataset(dataset_path="dataset", data_transform_config=""):
        data_transform = tl.transforms_data()
        data_transform_train = tl.transforms_data()
        if data_transform_config != "":
            print("Next : Data augmentation")
            data_transform_train = tl.transforms_data(data_transform_config)

        train_data = datasets.ImageFolder(
            dataset_path+"/train", transform=data_transform_train)

        val_data = datasets.ImageFolder(
            dataset_path+"/val", transform=data_transform)

        test_data = datasets.ImageFolder(
            dataset_path + "/test", transform=data_transform)

        print("Train Image : ", len(train_data))
        print("Train Image : ", len(val_data))
        print("Train Image : ", len(test_data))
        return (train_data, val_data, test_data)

    @staticmethod
    def transforms_data(config_transform_config=None):

        resize = transforms.RandomResizedCrop(224)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        config_transform = []

        to_tensor = transforms.ToTensor()
        config_transform.append(resize)
        config_transform.append(to_tensor)
        config_transform.append(normalize)
        print("config_transform", config_transform)
        return transforms.Compose(config_transform)

    @staticmethod
    def select_premodel(pre_model):
        return {
            "resnet18": models.resnet18(),
            "alexnet": models.alexnet(),
            "vgg16": models.vgg16(),
            "vgg19": models.vgg19(),
            "densenet": models.densenet161(),
            "inception": models.inception_v3()
        }[pre_model]

    @staticmethod
    def freeze_layers(model, n_layers):
        for params in model.parameters():
            params.requires_grad = False
        return model

    def set_classifier(self, new_classifier):
        self.model.classifier = new_classifier

    def select_optimizer(self, optimizer):
        return {
            "Adam": optim.Adam(self.model.parameters())
        }[optimizer]

    def select_criterion(self, criterion):
        return {
            "NLLLoss": nn.NLLLoss()
        }[criterion]

    def train(self, config, params, transform):
        torch.cuda.empty_cache()
        trigger = True
        self.metrics_list = []
        device = tl.check_gpu()[1]
        devide = "cuda"
        epochs, steps = get_from_dict(params, ["epochs", "steps"])
        criterion_config, classifier_config, optimizer_config = get_from_dict(
            config, ["criterion", "classifier", "optimizer"])
        self.set_classifier(classifier_config)
        # optimizer = self.select_optimizer(optimizer_config)
        optimizer = optim.Adam(self.model.parameters())
        criterion = self.select_criterion(criterion_config)
        self.model.to(tl.check_gpu()[1])
        (trainf, valf, testf) = "", "", ""
        (train_loader, val_loader, test_loader) = "", "", ""
        (trainf, valf, testf) = tl.load_dataset(
            data_transform_config=transform)
        (train_loader, val_loader, test_loader) = tl.create_loaders(
            trainf, valf, testf)
        print(train_loader)
        for e in range(epochs):
            running_loss = 0
            print("Trainning....")

            for images, labels in train_loader:
                print(images.size())
                images, labels = images.to(
                    device), labels.to(device)
                optimizer.zero_grad()

                log_ps = self.model(images)
                loss = criterion(log_ps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            else:
                val_loss = 0
                accuracy = 0

                with torch.no_grad():
                    self.model.eval()
                    for images, labels in val_loader:
                        images, labels = images.to(
                            device), labels.to(device)

                        log_ps = self.model(images)
                        val_loss += criterion(log_ps, labels)

                        ps = torch.exp(log_ps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

#                        self.metrics_list.append(
#                            metrics(labels.view(*top_class.shape).cpu().numpy(), top_class.cpu().numpy()))
                self.model.train()
                self.train_losses.append(running_loss/len(train_loader))
                self.test_losses.append(val_loss/len(val_loader))

                print("Epoch", str(e + 1), "/", str(epochs))
                print("Training Loss", str(self.train_losses[-1]))
                print("accuracy", self.test_losses[-1])
