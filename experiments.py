from main import run_experiment


def conv_n3():
    num_conv = 3
    run_experiment(num_conv, 'imagenet', 'data/tiny-imagenet-200', perc_class=100, batch_size=128, epochs=100, num_classes=200, train=True)

def conv_n4():
    num_conv = 4
    run_experiment(num_conv, 'imagenet', 'data/tiny-imagenet-200', perc_class=100, batch_size=128, epochs=100, num_classes=200, train=True)


def conv_n5():
    num_conv = 5
    run_experiment(num_conv, 'imagenet', 'data/tiny-imagenet-200', perc_class=100, batch_size=128, epochs=100, num_classes=200, train=True)


def perc_class_50():
    perc_class = 50
    num_conv = 3
    run_experiment(num_conv, 'imagenet', 'data/tiny-imagenet-200', perc_class=perc_class, batch_size=128, epochs=100, num_classes=200, train=True)

def perc_class_75():
    perc_class = 75
    num_conv = 3
    run_experiment(num_conv, 'imagenet', 'data/tiny-imagenet-200', perc_class=perc_class, batch_size=128, epochs=100, num_classes=200, train=True)

def perc_class_25():
    perc_class = 25
    num_conv = 3
    run_experiment(num_conv, 'imagenet', 'data/tiny-imagenet-200', perc_class=perc_class, batch_size=128, epochs=100, num_classes=200, train=True)


