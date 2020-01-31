from main import run_experiment
import argparse

def conv_n3():
    num_conv = 3
    run_experiment(num_conv, 'imagenet', 'data/tiny-imagenet-200', perc_class=100, batch_size=128, epochs=10, num_classes=200, train=True)

def conv_n4():
    num_conv = 4
    run_experiment(num_conv, 'imagenet', 'data/tiny-imagenet-200', perc_class=100, batch_size=128, epochs=10, num_classes=200, train=True)


def conv_n5():
    num_conv = 5
    run_experiment(num_conv, 'imagenet', 'data/tiny-imagenet-200', perc_class=100, batch_size=128, epochs=10, num_classes=200, train=True)


def perc_class_50():
    perc_class = 50
    num_conv = 3
    run_experiment(num_conv, 'imagenet', 'data/tiny-imagenet-200', perc_class=perc_class, batch_size=128, epochs=10, num_classes=200, train=True)

def perc_class_75():
    perc_class = 75
    num_conv = 3
    run_experiment(num_conv, 'imagenet', 'data/tiny-imagenet-200', perc_class=perc_class, batch_size=128, epochs=10, num_classes=200, train=True)

def perc_class_25():
    perc_class = 25
    num_conv = 3
    run_experiment(num_conv, 'imagenet', 'data/tiny-imagenet-200', perc_class=perc_class, batch_size=128, epochs=10, num_classes=200, train=True)


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
        perc_class_25()()
    elif exp_name == 'perc_class_50':
        perc_class_50
    elif exp_name == 'perc_class_75':
        perc_class_75()
    else:
        print("Incorrect experiment name passed.")