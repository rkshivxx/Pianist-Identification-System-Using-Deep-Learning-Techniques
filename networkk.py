import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, num_of_feature):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(num_of_feature, 64, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(64),

            nn.Conv1d(64, 64, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.AvgPool1d(5, stride=5)
            nn.Conv1d(64, 64, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.MaxPool1d(5, stride=5), nn.Dropout(0.5),
            nn.Conv1d(64, 128, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.MaxPool1d(5, stride=5), nn.Dropout(0.5),
            nn.Conv1d(128, 128, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.AvgPool1d(5, stride=5)
            nn.Conv1d(128, 128, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.MaxPool1d(5, stride=5), nn.Dropout(0.5)
        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(128, 128, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 128, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.Upsample(scale_factor=5),
            nn.Conv1d(128, 64, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Upsample(scale_factor=5),
            nn.Conv1d(64, num_of_feature, 10, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class PID_CNN1D_with_Autoencoder(nn.Module):
    def __init__(self, n_classes, input_size, num_of_feature):
        super(PID_CNN1D_with_Autoencoder, self).__init__()
        self.autoencoder = Autoencoder(num_of_feature)

        self.convnet = nn.Sequential(
            nn.Conv1d(num_of_feature, 64, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.MaxPool1d(5, stride=5), nn.Dropout(0.5),
            nn.Conv1d(64, 64, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.MaxPool1d(5, stride=5), nn.Dropout(0.5),
            nn.Conv1d(64, 128, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.MaxPool1d(5, stride=5), nn.Dropout(0.5),
            nn.Conv1d(128, 128, 10, padding=1), nn.ReLU(), nn.BatchNorm1d(128),
            nn.MaxPool1d(5, stride=5), nn.Dropout(0.5),
            nn.Conv1d(128, 128, 10, padding=1), nn.RReLU(), nn.BatchNorm1d(128),
            nn.MaxPool1d(5, stride=5), nn.Dropout(0.5),
            nn.Conv1d(128, 128, 10, padding=1), nn.RReLU(), nn.BatchNorm1d(128),
            nn.AvgPool1d(5, stride=5)
        )

        self.fc = nn.Sequential(nn.Linear(128 * 7, 1024),
                                nn.Dropout(0.5),
                                nn.ReLU(),
                                nn.Linear(1024, n_classes),
                                nn.Dropout(0.5),
                                nn.Softmax(dim=1)
                                )
        self.input_size = input_size

    def forward(self, input_layer):
        encoded, decoded = self.autoencoder(input_layer)
        output = self.convnet(input_layer)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output, encoded, decoded

    def _class_name(self):
        return "PIDCNN1D-large"