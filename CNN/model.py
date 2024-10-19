
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
    super(CNNResidualBlock, self).__init__()
    self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
    self.bn1 = nn.BatchNorm1d(out_channels)
    self.relu = nn.ReLU()
    self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
    self.bn2 = nn.BatchNorm1d(out_channels)
    self.shortcut = nn.Sequential()
    if stride != 1 or in_channels != out_channels:
      self.shortcut = nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride*2, bias=False),
        nn.BatchNorm1d(out_channels)
      )

  def forward(self, x):
    residual = self.shortcut(x)
    # print(f"Residual shape: {residual.shape}")
    out = self.conv1(x)
    # print(f"Conv1 shape: {out.shape}")
    out = self.bn1(out)
    # print(f"BN1 shape: {out.shape}")
    out = F.relu(out)
    # print(f"ReLU shape: {out.shape}")
    out = self.conv2(out)
    # print(f"Conv2 shape: {out.shape}")
    out = self.bn2(out)
    # print(f"BN2 shape: {out.shape}")
    out += residual
    # print(f"Residual added shape: {out.shape}")
    out = F.relu(out)
    # print(f"Final ReLU shape: {out.shape}")
    return out



class BacteriaNN(nn.Module):
    def __init__(self, embedding_dim, output_dim, kernel_size, padding, avg_poolsize, embedding_reduction_factor, resblock_num, stride, dropout_rate = 0.2):
        super(BacteriaNN, self).__init__()
        self.conv1 = nn.Conv1d(embedding_dim, embedding_dim//embedding_reduction_factor, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm1d(embedding_dim//embedding_reduction_factor)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size, stride=stride, padding=padding)
        for i in range(resblock_num):
            setattr(self, f'resblock{i+1}', CNNResidualBlock(embedding_dim//embedding_reduction_factor, embedding_dim//embedding_reduction_factor, kernel_size=kernel_size, padding=padding, stride=stride))
        self.avgpool = nn.AdaptiveAvgPool1d(avg_poolsize)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(avg_poolsize*embedding_dim//embedding_reduction_factor, output_dim)
        self.resblock_num = resblock_num

    def forward(self, x):
        x = x.permute(0, 2, 1)
        # print(f"Input shape: {x.shape}")
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)
        # print(f"After maxpool: {x.shape}")
        for i in range(self.resblock_num):
            x = getattr(self, f'resblock{i+1}')(x)
            # print(f"After resblock{i+1}: {x.shape}")
        # print(f"After resblock4: {x.shape}")
        x = self.avgpool(x)
        # print(f"After avgpool: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"After view: {x.shape}")
        x = self.dropout(x)
        # print(f"After dropout: {x.shape}")
        x = self.fc(x)
        return x
    
class ViralNN(nn.Module):
    def __init__(self, embedding_dim, length, output_dim, hidden_dims=[128, 64], dropout_rate = 0.2):
        super(ViralNN, self).__init__()

        fc_input_dim = embedding_dim * length
        layers = []
        current_dim = fc_input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
class PhINN(nn.Module):
    def __init__(self, bacteria_nn, viral_nn, embedding_dim, output_dim, skip_connection=False, skip_only = False):
        super(PhINN, self).__init__()
        self.bacteria_nn = bacteria_nn
        self.viral_nn = viral_nn
        self.fc = nn.Linear(output_dim*2, output_dim)
        self.fc_skip_only = nn.Linear(output_dim*2, output_dim)
        self.fc_skip0 = nn.Linear(embedding_dim*2, output_dim*2)
        self.fc_skip = nn.Linear(output_dim*4, output_dim)
        self.fc2 = nn.Linear(output_dim, 1)
        self.skip_connection = skip_connection
        self.bn = nn.BatchNorm1d(output_dim)
        self.skip_only = skip_only

    def forward(self, bacteria_input, viral_input):
        bacteria_output = self.bacteria_nn(bacteria_input)
        viral_output = self.viral_nn(viral_input)

        combined_output = torch.cat((bacteria_output, viral_output), dim=1)
        if self.skip_only:
            bacteria_output_skip = bacteria_input.sum(dim=1).squeeze()  # Shape: (batch_size, output_dim)
            viral_output_skip = bacteria_input.sum(dim=1).squeeze()
            combined_output_skip = torch.cat((bacteria_output_skip, viral_output_skip), dim=1)
            combined_output_skip = self.fc_skip0(combined_output_skip)
            output = self.fc_skip_only(combined_output)

        elif self.skip_connection:
            bacteria_output_skip = bacteria_input.sum(dim=1).squeeze()  # Shape: (batch_size, output_dim)
            viral_output_skip = bacteria_input.sum(dim=1).squeeze()
            combined_output_skip = torch.cat((bacteria_output_skip, viral_output_skip), dim=1)
            combined_output_skip = self.fc_skip0(combined_output_skip)
            combined_output = torch.cat((combined_output_skip, combined_output), dim = 1)
            output = self.fc_skip(combined_output)
        else:
            output = self.fc(combined_output)
        
        output = self.bn(output)
        output = nn.ReLU()(output)
        logits = self.fc2(output)
        probs = torch.sigmoid(logits)
        return probs
