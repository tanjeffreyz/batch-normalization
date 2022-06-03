import os
import numpy as np
from matplotlib import pyplot as plt


TARGET_NO_BN = 'models/mnist/no_batch_norm/06_02_2022/19_53_48'
TARGET_WITH_BN = 'models/mnist/with_batch_norm/06_02_2022/20_08_33'


def plot_errors():
    num_epochs = 15
    test_errors_with_bn = np.load(os.path.join(TARGET_WITH_BN, 'test_errors.npy'))
    test_errors_no_bn = np.load(os.path.join(TARGET_NO_BN, 'test_errors.npy'))

    plt.figure(figsize=(8, 4))
    plt.plot(test_errors_with_bn[0, :num_epochs], 1 - test_errors_with_bn[1, :num_epochs], 'g-', label='With batch normalization')
    plt.plot(test_errors_no_bn[0, :num_epochs], 1 - test_errors_no_bn[1, :num_epochs], 'r-', label='Without batch normalization')
    plt.ylim(0.7, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_percentiles():
    percentiles_with_bn = np.load(os.path.join(TARGET_WITH_BN, 'input_dist_percentiles.npy'))
    percentiles_no_bn = np.load(os.path.join(TARGET_NO_BN, 'input_dist_percentiles.npy'))

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    no_bn, with_bn = axs
    for ax, data in ((no_bn, percentiles_no_bn),
                     (with_bn, percentiles_with_bn)):
        for i in range(3):
            ax.plot(data[0], data[i+1], 'b-', linewidth=1)
    no_bn.set_title('Without batch normalization')
    no_bn.set_ylabel('Activation')
    no_bn.set_xlabel('Iteration')
    with_bn.set_title('With batch normalization')
    with_bn.set_xlabel('Iteration')
    plt.show()


if __name__ == '__main__':
    plot_errors()
    plot_percentiles()
