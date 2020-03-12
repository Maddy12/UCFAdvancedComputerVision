from main import run_experiment
import argparse
from torchvision import transforms

def conv_n3():  # Running locally
    num_conv = 3
    run_experiment('imagenet', '../data/tiny-imagenet-200', num_conv, perc_class=100, batch_size=128, epochs=10, num_classes=200, train=True)


def conv_n4():  # Running on cluster
    num_conv = 4
    run_experiment('imagenet', '../data/tiny-imagenet-200', num_conv, perc_class=100, batch_size=128, epochs=10, num_classes=200, train=True)


def conv_n5():
    num_conv = 5
    run_experiment( 'imagenet', '../data/tiny-imagenet-200', num_conv,perc_class=100, batch_size=128, epochs=10, num_classes=200, train=True)


def perc_class_50():
    perc_class = 50
    num_conv = 3
    run_experiment('imagenet', '../data/tiny-imagenet-200', num_conv, perc_class=perc_class, batch_size=128, epochs=10, num_classes=200, train=True)


def perc_class_75():
    perc_class = 75
    num_conv = 3
    run_experiment( 'imagenet', '../data/tiny-imagenet-200', num_conv, perc_class=perc_class, batch_size=128, epochs=10, num_classes=200, train=True)


def perc_class_25():
    perc_class = 25
    num_conv = 3
    run_experiment('imagenet', '../data/tiny-imagenet-200', num_conv, perc_class=perc_class, batch_size=128, epochs=10, num_classes=200, train=True)


def finetune_alexnet_svhn():
    perc_class = 100
    resize = (64, 64)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(resize), transforms.ToTensor(), normalize]) 
    run_experiment('svhn', '../data/svhn', None, perc_class=perc_class, batch_size=128, epochs=50, num_classes=10, train=True, fine_tune=True, transform=transform)

def finetune_alexnet_imagenet():
    perc_class = 100
    
    run_experiment('imagenet', '../data/tiny-imagenet-200', None, perc_class=perc_class, batch_size=128, epochs=50, num_classes=200, train=True, fine_tune=True, transform=None)

def run_while_I_sleep():
    conv_n4()
    conv_n5()
    perc_class_25()
    perc_class_50()
    perc_class_75()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run experiments for project 1.')

    parser.add_argument('exp_name', metavar='Experiment name', type=str,
                        help='One of: conv_n3, conv_n4, conv_5, perc_class_25, perc_class_50, perc_class_75')

    # Parse the script's arguments
    args = parser.parse_args()
    exp_name = args.exp_name
    if exp_name == 'conv_n3':
        conv_n3()
    elif exp_name == 'conv_n4':
        conv_n4()
    elif exp_name == 'conv_n5':
        conv_n5()
    elif exp_name == 'perc_class_25':
        perc_class_25()
    elif exp_name == 'perc_class_50':
        perc_class_50()
    elif exp_name == 'perc_class_75':
        perc_class_75()
    else:
        print("Incorrect experiment name passed.")