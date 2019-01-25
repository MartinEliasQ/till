"""Transfer Learning"""

import torch
from till.utils import get_from_dict
from torchvision import (datasets, transforms, models)
from torch import optim


class tl(object):
    def __init__(self, premodel):
        self.model = self.select_premodel(premodel)

    @staticmethod
    def check_gpu():
        return (True, 'cuda') if torch.cuda.is_available() else (False, 'cpu')

    @staticmethod
    def create_loaders(train, val, test, batch_size=20):
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=batch_size)
        return (train_loader, val_loader, test_loader)

    @staticmethod
    def load_dataset(dataset_path="dataset", data_transform=""):
        if data_transform == "":
            data_transform = tl.transforms_data()

        train_data = datasets.ImageFolder(
            dataset_path+"/train", transform=data_transform)

        val_data = datasets.ImageFolder(
            dataset_path+"/val", transform=data_transform)

        test_data = datasets.ImageFolder(
            dataset_path + "/test", transform=data_transform)
        return (train_data, val_data, test_data)

    @staticmethod
    def transforms_data(config_transform: [] = []):
        resize = transforms.RandomResizedCrop(224)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()
        config_transform.append(resize)
        config_transform.append(normalize)
        config_transform.append(to_tensor)
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

    def set_optimizer(self):
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.003)

    def train(self, config, params):
        criterion, classifier = get_from_dict(
            config, ["criterion", "classifier"])
        self.set_classifier(classifier)
        self.set_optimizer()
        self.model.to(tl.check_gpu()[1])

        epochs, steps = get_from_dict(params, ["epochs", "steps"])
        running_loss = 0
        print_every = 5
        (trainf, valf, testf) = tl.load_dataset()

        (train_loader, val_loader, train_loader) = tl.create_loaders(
            trainf, valf, testf)

        for epoch in range(epochs):
            for inputs, labels in train_loader:
                steps += 1
                inputs, labels = inputs.to(
                    tl.check_gpu()[1]), labels.to(tl.check_gpu()[1])

                self.optimizer.zero_grad()
                logps = self.model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in val_loader:
                            inputs, labels = inputs.to(
                                tl.check_gpu()[1]), labels.to(tl.check_gpu()[1])
                            logps = self.model.forward(inputs)
                            batch_loss = criterion(logps, labels)

                            test_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(
                                *top_class.shape)
                            accuracy += torch.mean(
                                equals.type(torch.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Test loss: {test_loss/len(val_loader):.3f}.. "
                      f"Test accuracy: {accuracy/len(val_loader):.3f}")
                running_loss = 0
                self.model.train()
