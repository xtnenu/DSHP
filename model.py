import torch
import torch.nn as nn
import numpy as np
from Pyfeat_v2 import loadcsv
from sklearn.model_selection import StratifiedKFold

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        x_cnn = self.data[0][index]
        x_lstm = self.data[1][index]
        x_fc = self.data[2][index]
        y = self.labels[index]
        return x_cnn, x_lstm, x_fc, y

    def __len__(self):
        return len(self.labels)

class AttentionNet(nn.Module):
    def __init__(self):
        super(AttentionNet, self).__init__()
        self.dense1=nn.Sequential(nn.Linear(24444,10))
        # CNN subnet
        self.cnn_subnet = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        # LSTM subnet
        self.lstm_subnet = nn.LSTM(input_size=10, hidden_size=20, num_layers=2, batch_first=True)

        # Attention subnet
        self.att_subnet = nn.Sequential(
            nn.Linear(64 + 20, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x_cnn, x_lstm, x_fc):
        x_cnn = self.cnn_subnet(x_cnn)
        x_lstm, _ = self.lstm_subnet(x_lstm)
        x_lstm = x_lstm[:, -1, :]
        x = torch.cat((x_cnn, x_lstm), dim=1)
        x_att = self.att_subnet(x)
        x_att = x_att.view(x_att.size(0), -1, 1)
        x = x * x_att
        x = x.sum(dim=1)
        x_fc = x_fc.view(x_fc.size(0), -1)
        x = torch.cat((x, x_fc), dim=1)
        x = self.fc_subnet(x)
        return x

def main():
    b = loadcsv(['Data.csv'], ",")
    b = np.array(b)
    b = b.astype("float32")

    np.random.seed(0)
    np.random.shuffle(b)
    X = b[:, :-1]
    Y = b[:, -1]
    #kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    # Define hyperparameters
    batch_size = 64
    lr = 0.001
    num_epochs = 10
    train_dataset=MyDataset(X,Y)
    # Initialize model and optimizer
    model = AttentionNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Initialize data loader
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train model
    for epoch in range(num_epochs):
        for batch_idx, (x_cnn, x_lstm, x_fc, y) in enumerate(train_loader):
            # Forward pass
            y_pred = model(x_cnn, x_lstm, x_fc)
            loss = nn.functional.binary_cross_entropy_with_logits(y_pred, y.unsqueeze(1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss
            if batch_idx % 10 == 0:
                print('Epoch {} Batch {} Loss: {:.4f}'.format(epoch, batch_idx, loss.item()))

if __name__=="__main__":
    main()