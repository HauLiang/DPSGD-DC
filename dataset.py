from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np
import os

def load_dataset(args):

    # Data transformation
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    os.makedirs(args.data_path, exist_ok=True)

    full_train_data = datasets.CIFAR10(root=args.data_path, train=True,  download=True, transform=transform)
    test_data  = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)

    # target training data
    print('-' * 10 + 'Generating data for target model...' + '-' * 10 + '\n')
    target_subset_indices = list(range(args.target_data_size))
    target_train_subset = Subset(full_train_data, target_subset_indices)

    
    # shadow training data
    print('-' * 10 + 'Generating data for shadow models...' + '-' * 10 + '\n')
    shadow_subset_indices = list(range(args.target_data_size, len(full_train_data)))
    shadow_data_subset = Subset(full_train_data, shadow_subset_indices)
    shadow_indices = np.arange(len(shadow_subset_indices))

    shadow_subset_train = list(range(args.target_data_size))
    shadow_subset_test = list(range(args.target_data_size, 2*args.target_data_size))

    shadow_train_tests = []
    for i in range(args.n_shadow):
        print('Generating data for shadow model {}...'.format(i+1))
        shadow_i_indices = np.random.choice(shadow_indices, 2 * args.target_data_size, replace=False)
        shadow_i_data = Subset(shadow_data_subset, shadow_i_indices)

        # shadow train and test
        shadow_i_train = Subset(shadow_i_data, shadow_subset_train)
        shadow_i_test = Subset(shadow_i_data, shadow_subset_test)

        shadow_train_tests.append({
            'train': shadow_i_train,
            'test': shadow_i_test
        })

    return {
        'target_train': target_train_subset,
        'shadows': shadow_train_tests,
        'test_data': test_data
    }


