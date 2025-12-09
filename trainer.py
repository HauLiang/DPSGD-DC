from sklearn.metrics import accuracy_score
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import random
import torchvision.models as models
from tqdm import tqdm
from dataset import load_dataset
import warnings
import gc


random.seed(21312)
np.random.seed(21312)
np.random.seed(21312)
torch.manual_seed(21312)
if torch.cuda.is_available():
    torch.cuda.manual_seed(21312)


class NeuralNetwork(nn.Module):
    def __init__(self, n_in, n_out):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(n_in, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, n_out)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_one_epoch(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    y_true = []
    y_predict = []

    for _, (batch_x, batch_y) in enumerate(tqdm(data_loader)):

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        batch_y_predict = model(batch_x)

        # loss update
        loss = criterion(batch_y_predict, batch_y)

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # record train acc
        predict_label = torch.argmax(batch_y_predict, dim=1)
        y_true.append(batch_y.flatten())
        y_predict.append(predict_label.flatten())

    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)

    return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": epoch_loss / len(data_loader),
            }


def test(data_loader, model, criterion, device):

    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    loss_sum = []

    for (batch_x, batch_y) in tqdm(data_loader):

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        # test acc
        batch_y_predict = model(batch_x)
        
        loss = criterion(batch_y_predict, batch_y)

        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_true.append(batch_y.flatten())
        y_predict.append(batch_y_predict.flatten())
        loss_sum.append(loss.item())

    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)
    loss = sum(loss_sum) / len(loss_sum)

    return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": loss,
            }


# Train target models
def train_models(train_data, test_data, batch_size=None, epochs=None, learning_rate=None, 
          scheme=None, dp_noise_sigma=None, dp_clip_norm=None, device=None):
    
    print('-' * 10 + 'Building model with {} training data, {} classes...'.format(len(train_data), 10) + '-' * 10 + '\n')
    
    # ResNet18 Setup
    net = models.resnet18(pretrained=False) 
    net.fc = torch.nn.Linear(net.fc.in_features, 10)
    net = ModuleValidator.fix(net)
    ModuleValidator.validate(net, strict=False)
    net = net.to(device)
    
    # sgd update
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    # data loader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
                             
    print('Training...')
       
    if scheme == 'dp':
        # DP
        print('Using Differential Privacy!')
        privacy_engine = PrivacyEngine()
        net, optimizer, train_loader = privacy_engine.make_private(
            module=net,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=dp_noise_sigma*batch_size/dp_clip_norm,
            max_grad_norm=dp_clip_norm,
            poisson_sampling=False
        )
    
    for epoch in range(epochs):
        # training
        train_stats = train_one_epoch(train_loader, net, criterion, optimizer, device)
        test_stats = test(test_loader, net, criterion, device)

        print(f"# Epoch {epoch+1}: Train Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['acc']* 100:.2f}%, Test Acc: {test_stats['acc']* 100:.2f}%\n")

    return net
    

# Train classifier for attack model
def train_classifier(dataset, batch_size=None, epochs=None, learning_rate=None, device=None):
    
    # load dataset
    train_x, train_y, test_x, test_y = dataset
    n_in = train_x.shape[1]
    n_out = len(np.unique(train_y))

    print('-' * 10 + 'Building model with {} training data, {} classes...'.format(len(train_x), n_out) + '-' * 10 + '\n')
    
    net = NeuralNetwork(n_in, n_out).to(device)

    # sgd
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    # training data
    train_x_tensor = torch.tensor(train_x, dtype=torch.float32, requires_grad=False).to(device)
    train_y_tensor = torch.tensor(train_y, dtype=torch.long, requires_grad=False).to(device)
    train_dataset = TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # test data
    test_x_tensor = torch.tensor(test_x, dtype=torch.float32, requires_grad=False).to(device)
    test_y_tensor = torch.tensor(test_y, dtype=torch.long, requires_grad=False).to(device)
    test_dataset = TensorDataset(test_x_tensor, test_y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print('Training...')

    net.train()
    for epoch in range(epochs):
        train_stats = train_one_epoch(train_loader, net, criterion, optimizer, device)

        print(f"# Epoch {epoch+1}: Train Loss: {train_stats['loss']:.4f}, Train Acc: {train_stats['acc']* 100:.2f}%\n")


    # Obtain attack results
    print('Testing...')
    pred_y = []

    net.eval()
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(device)
            pred = net(batch_x).detach().cpu().numpy()
            pred_y.append(np.argmax(pred, axis=1))
        
        pred_y = np.concatenate(pred_y)

        print('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))

    return pred_y



def get_attack_data(train_data, test_data, net, batch_size = None, device = None):
    # Initialization
    attack_x, attack_y = [], []
    attack_x_classes = []

    # train loader
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    net.eval()        
    with torch.no_grad():
        # Label data used for training model, label as 1
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            prob = net(batch_x).cpu().numpy()  # obtain network inference results
            attack_x.append(prob)
            attack_x_classes.append(batch_y)
            attack_y.append(np.ones(len(batch_x)))

        # Label data not used for training model, label as 0
        for (batch_x, batch_y) in test_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            prob = net(batch_x).cpu().numpy()  # obtain network inference results
            attack_x.append(prob)
            attack_x_classes.append(batch_y)
            attack_y.append(np.zeros(len(batch_x)))
    
    return attack_x, attack_y, attack_x_classes


def parse_option():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--data_path', default='./data/cifar10/', help='Place to load dataset')

    # target model configuration
    parser.add_argument('--target_data_size', type=int, default=10000)   # number of data point used in target model
    parser.add_argument('--target_scheme', type=str, default='dp')
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--target_batch_size', type=int, default=1000)
    parser.add_argument('--target_epochs', type=int, default=50)

    # dp para.
    parser.add_argument('--target_dp_noise_sigma', type=float, default=5e-5)
    parser.add_argument('--target_dp_clip_norm', type=float, default=50.0)

    # shadow model configuration
    parser.add_argument('--n_shadow', type=int, default=10)

    args = parser.parse_args()

    return args


def main():
    # Parameter Setting
    args = parse_option()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    warnings.filterwarnings('ignore')

    # load dataset
    print('-' * 10 + 'Loading Dataset' + '-' * 10 + '\n')
    datasets = load_dataset(args)
    
    # train target model
    print('-' * 10 + 'Train Target Model' + '-' * 10 + '\n')
    train_models(train_data = datasets['target_train'], test_data = datasets['test_data'], 
                 scheme = args.target_scheme, dp_noise_sigma=args.target_dp_noise_sigma, dp_clip_norm=args.target_dp_clip_norm,
                 learning_rate=args.target_learning_rate,
                 batch_size=args.target_batch_size,
                 epochs=args.target_epochs, device=device)

    torch.cuda.empty_cache()
    gc.collect()
    
        
if __name__ == '__main__':
    main()
