from trainer import train_models, train_classifier, get_attack_data
from dataset import load_dataset
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import argparse
import torch
import random
import warnings
import os
import gc


random.seed(21312)
np.random.seed(21312)
np.random.seed(21312)
torch.manual_seed(21312)
if torch.cuda.is_available():
    torch.cuda.manual_seed(21312)


# target model setting
def train_target_model(train_data, test_data, epochs=None, batch_size=None, learning_rate=None,
                       scheme=None, dp_noise_sigma=None, dp_clip_norm=None, device=None):
    
    # target model training
    net = train_models(train_data=train_data, test_data=test_data, 
                       epochs=epochs, learning_rate=learning_rate,
                       batch_size=batch_size, scheme=scheme,
                       dp_noise_sigma=dp_noise_sigma, dp_clip_norm=dp_clip_norm, device=device)
    
    # genertate attack data (for test)
    attack_x, attack_y, attack_x_classes = get_attack_data(train_data=train_data, test_data=test_data, 
                                                           net=net, batch_size=batch_size, device=device)

    # target results for attack (as test dataset)
    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    classes = np.concatenate(attack_x_classes)

    return attack_x, attack_y, classes


# shadow model setting
def train_shadow_models(shadow_dataset, epochs=None, batch_size=None, learning_rate=None, 
                       scheme=None, dp_noise_sigma=None, dp_clip_norm=None, device=None):
    
    attack_x, attack_y = [], []
    classes = []

    # Train multiple shadow models
    for i, shadow_data in enumerate(shadow_dataset):
        print('-' * 10 + 'Training Shadow Moodel {}'.format(i+1) + '-' * 10 + '\n')
        
        # shadow model training
        net = train_models(train_data = shadow_data['train'], test_data = shadow_data['test'], 
                           epochs=epochs, learning_rate=learning_rate,
                           batch_size=batch_size, scheme=scheme,
                           dp_noise_sigma=dp_noise_sigma, dp_clip_norm=dp_clip_norm, device=device)
        
        print('Generate training data for attack model')
        
        # generate attack data (for training)
        attack_i_x, attack_i_y, attack_i_x_classes = get_attack_data(train_data=shadow_data['train'], test_data=shadow_data['test'],
                                                                     net=net, batch_size=batch_size, device=device)
        
        attack_x += attack_i_x
        attack_y += attack_i_y
        classes.append(np.concatenate(attack_i_x_classes))

        # To avoid CUDA out of memory
        del net
        torch.cuda.empty_cache()
        gc.collect()

        print('Finish generate training data for attack model')

        
    # shadow results for attack (as train dataset)
    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    classes = np.concatenate(classes)

    return attack_x, attack_y, classes


# Training and evaluating the attack model
def train_attack_model(dataset, classes, epochs=None, batch_size=None, learning_rate=None, device=None):
    
    # load dataset
    train_x, train_y, test_x, test_y = dataset

    # Extract unique class for train
    train_classes, test_classes = classes
    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes)

    # For each class, train an attack model
    true_y, pred_y = [], []
    for c in unique_classes:
        print('-' * 10 + 'Training attack model for class {}...'.format(c) + '-' * 10 + '\n')
        # contruct dataset
        c_train_indices = train_indices[train_classes == c]
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
        c_test_indices = test_indices[test_classes == c]
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]
        c_dataset = (c_train_x, c_train_y, c_test_x, c_test_y)

        c_pred_y = train_classifier(c_dataset, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, device=device)
        
        true_y.append(c_test_y)
        pred_y.append(c_pred_y)

    # Evaluate attack model
    print('-' * 10 + 'Final Evaluation' + '-' * 10 + '\n')
    true_y = np.concatenate(true_y)
    pred_y = np.concatenate(pred_y)
    print('Testing Accuracy: {}'.format(accuracy_score(true_y, pred_y)))

    # Calculate the confusion matrix
    tn, fp, fn, tp = confusion_matrix(true_y, pred_y).ravel()

    # FPR and FNR
    fpr = fp / (fp + tn)  
    fnr = fn / (fn + tp)    
    print('FPR: {}'.format(fpr))
    print('FNR: {}'.format(fnr))

    # privacy estimation
    delta = 1e-5   
    epsilon = np.maximum(np.log((1-delta-fpr)/fnr),np.log((1-delta-fnr)/fpr))

    print('Privacy Parameter Epsilon: {}'.format(epsilon))
    
    return epsilon


def parse_option():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--data_path', default='./data/cifar10/', help='Place to load dataset')

    # target model configuration
    parser.add_argument('--target_data_size', type=int, default=10000)   # number of data point used in target model
    parser.add_argument('--target_scheme', type=str, default='dp')
    parser.add_argument('--target_learning_rate', type=float, default=0.1)

    # dp
    parser.add_argument('--target_dp_noise_sigma', type=float, default=2e-3)
    parser.add_argument('--target_dp_clip_norm', type=float, default=20.0)

    # shadow model configuration
    parser.add_argument('--n_shadow', type=int, default=10)

    # attack model configuration
    parser.add_argument('--attack_learning_rate', type=float, default=0.01)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--attack_epochs', type=int, default=50)

    # other paras.
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--mc_runs', type=int, default=10)

    # parse configuration
    args = parser.parse_args()
    
    return args



def main(target_batch_size, target_epochs, seed=None):
    # parameter Setting
    args = parse_option()

    warnings.filterwarnings('ignore')

    # load dataset
    print('-' * 10 + 'Loading Dataset' + '-' * 10 + '\n')
    datasets = load_dataset(args)

    # target model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('-' * 10 + 'Train Target Model' + '-' * 10 + '\n')
    attack_test_x, attack_test_y, test_classes = train_target_model(
        train_data = datasets['target_train'], test_data = datasets['test_data'], device = device,
        epochs=target_epochs,
        batch_size=target_batch_size,
        learning_rate=args.target_learning_rate,
        scheme=args.target_scheme, dp_noise_sigma=args.target_dp_noise_sigma, dp_clip_norm=args.target_dp_clip_norm)
    
    # To avoid CUDA out of memory
    torch.cuda.empty_cache()
    gc.collect()

    # shadow models
    print('-' * 10 + 'Train Shadow Models' + '-' * 10 + '\n')
    attack_train_x, attack_train_y, train_classes = train_shadow_models(
        shadow_dataset = datasets['shadows'], device = device,
        epochs=target_epochs,
        batch_size=target_batch_size,
        learning_rate=args.target_learning_rate,
        scheme=args.target_scheme, dp_noise_sigma=args.target_dp_noise_sigma, dp_clip_norm=args.target_dp_clip_norm)
    
    # To avoid CUDA out of memory
    torch.cuda.empty_cache()
    gc.collect()

    # attack model
    print('-' * 10 + 'Train Attack Model' + '-' * 10 + '\n')    
    dataset = (attack_train_x, attack_train_y, attack_test_x, attack_test_y)
    epsilon = train_attack_model(
            dataset=dataset, classes=(train_classes, test_classes), device = device,
            epochs=args.attack_epochs,
            batch_size=args.attack_batch_size,
            learning_rate=args.attack_learning_rate)

    return epsilon
    
    
if __name__ == '__main__':
    
    # parameter Setting
    args = parse_option()

    # record results for all monte carlo runs
    epsilon_results = [0 for _ in range(args.mc_runs)]
    
    for i in range(args.mc_runs):
        epsilon_result = main(target_batch_size = args.batch_size, target_epochs = args.epochs)
        epsilon_results[i] = float(epsilon_result)

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    output_file_path = os.path.join(output_dir, 'epsilon_results_{}.txt'.format(args.batch_size))
    
    with open(output_file_path, 'w') as f:
        f.write('All Monte Carlo Results:\n')
        for i, eps in enumerate(epsilon_results):
            f.write(f'Monte Carlo Run {i + 1}: {eps}\n')
