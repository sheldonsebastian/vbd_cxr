# %% --------------------START HERE
import torch.nn as nn
from torchvision import models


# %% --------------------
def get_param_to_optimize(model, feature_extracting):
    # feature_extract_param = True means all layers frozen except the last user added layers
    # feature_extract_param = False means all layers unfrozen and entire network learns new weights
    # and biases
    print(f"Params to learn, when feature extract = {feature_extracting}:")
    params_to_update = model.parameters()

    if feature_extracting:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
        return params_to_update
    else:
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                print("\t", name)
        return params_to_update


# %% --------------------
def set_parameter_requires_grad(model, feature_extracting):
    # feature_extract_param = True means all layers frozen except the last user added layers
    # feature_extract_param = False means all layers unfrozen and entire network learns new weights
    # and biases
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True


# %% --------------------initialize pretrained model & return input size desired by pretrained model
def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        params_to_update = get_param_to_optimize(model_ft, feature_extract)

    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        params_to_update = get_param_to_optimize(model_ft, feature_extract)

    elif model_name == "resnet101":
        """ Resnet101
        """
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        params_to_update = get_param_to_optimize(model_ft, feature_extract)

    elif model_name == "resnet152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        params_to_update = get_param_to_optimize(model_ft, feature_extract)

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        params_to_update = get_param_to_optimize(model_ft, feature_extract)

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        params_to_update = get_param_to_optimize(model_ft, feature_extract)

    elif model_name == "vgg19":
        """ VGG19_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        params_to_update = get_param_to_optimize(model_ft, feature_extract)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, params_to_update
