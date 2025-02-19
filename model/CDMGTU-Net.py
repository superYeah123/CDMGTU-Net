import torch.nn as nn
import torch.nn.functional as F
import torch

class GTU(nn.Module):
    def __init__(self, in_channels, frequency_strides, kernel_size):
        super(GTU, self).__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.kernel_size = kernel_size
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, frequency_strides))

    def forward(self, x):
        x = F.pad(x, (self.kernel_size - 1, 0))
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, : self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = self.tanh(x_p) * self.sigmoid(x_q)
        return x_gtu

class Multi_GTU(nn.Module):
    def __init__(self,  in_channels, frequency_strides, kernel_size, pool=False):
        super(Multi_GTU, self).__init__()
        self.gtu0 = GTU(in_channels, frequency_strides, kernel_size[0])
        self.gtu1 = GTU(in_channels, frequency_strides, kernel_size[1])
        self.gtu2 = GTU(in_channels, frequency_strides, kernel_size[2])
        self.pool = pool
        self.fcmy = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 3), stride=(1, 3))

    def forward(self, X):
        B, C, F, T = X.shape
        x_gtu = []
        x_gtu.append(self.gtu0(X))
        x_gtu.append(self.gtu1(X))
        x_gtu.append(self.gtu2(X))
        stacked = torch.stack((x_gtu[0],x_gtu[1],x_gtu[2]), dim=-1)
        stacked_reshaped = stacked.view(B, C, F, T * 3)
        time_conv = stacked_reshaped
        if self.pool:
            time_conv = self.pooling(time_conv)
            time_conv = self.fcmy(time_conv)
        else:
            time_conv = self.fcmy(time_conv)
        time_conv_output = torch.nn.functional.relu(X + time_conv)
        return time_conv_output

class MSTF(nn.Module):
    def __init__(self, in_channels):
        super(MSTF, self).__init__()
        out_channels = in_channels

        # First project layer with Conv2d, BatchNorm, and ReLU
        self.project1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        # Second project layer with Conv2d, BatchNorm, and ReLU
        self.project2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x0, x1):

        B, T, F, C = x0.shape  # B: Batch size, T: Time steps, N: Number of features, C: Channels

        # Concatenate the two inputs along the batch dimension
        x_concat = torch.cat([x0, x1], 0)

        # Reshape for processing in Conv2D
        x_reshaped = x_concat.reshape(-1, T, F * B, C)

        x_projected = self.project1(x_reshaped.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)

        # Calculate attention weights with softmax
        attention_weights = torch.nn.functional.softmax(x_projected, dim=0)

        # Reshape the concatenated inputs again for summation
        x_reshaped = x_concat.reshape(-1, T, F * B, C)

        # Weighted sum of inputs based on attention weights
        weighted_sum = (attention_weights * x_reshaped).sum(0)

        # Reshape the output to match the input dimensions (B, T, N, C)
        output = weighted_sum.reshape(B, T, F, C)

        # Permute dimensions and apply the second project layer
        output = output.permute(0, 3, 2, 1)
        return self.project2(output).permute(0, 3, 2, 1)

class LCFTUNIT(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_layers: int = 2, bidirectional: bool = True):
        super(LCFTUNIT, self).__init__()

        # LSTM layer to process sequential data
        self.sequence_model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        # Flatten parameters to improve performance
        self.sequence_model.flatten_parameters()

        # Output layer based on bidirectionality
        if bidirectional:
            self.fc_output_layer = nn.Conv2d(
                in_channels=hidden_size * 2,
                out_channels=output_size,
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0),
                dilation=1,
                groups=1,
                bias=True
            )
        else:
            self.fc_output_layer = nn.Conv2d(
                in_channels=hidden_size,
                out_channels=output_size,
                kernel_size=(3, 1),
                stride=1,
                padding=(1, 0),
                dilation=1,
                groups=1,
                bias=True
            )

        # ReLU activation function
        self.relu = nn.ReLU()

    def forward(self, x, B, T, F):

        # Get LSTM output
        o, _ = self.sequence_model(x)

        # Handle the case where the output shape is [B*T, F, -1] (B*T is the batch size)
        if B * T == o.shape[0]:
            o = o.reshape(B, T, F, -1)
            o = o.permute(0, 3, 2, 1)  # Change the order for Conv2D processing
            o = self.fc_output_layer(o)  # Apply Conv2D
            o = o.permute(0, 3, 2, 1)  # Revert the permutation
            o = o.reshape(B * T, F, -1)  # Reshape to [B*T, F, -1]
        # Handle the case where the output shape is [B*F, T, -1] (B*F is the batch size)
        elif B * F == o.shape[0]:
            o = o.reshape(B, F, T, -1)
            o = o.permute(0, 3, 1, 2)  # Change the order for Conv2D processing
            o = self.fc_output_layer(o)  # Apply Conv2D
            o = o.permute(0, 2, 3, 1)  # Revert the permutation
            o = o.reshape(B * F, T, -1)  # Reshape to [B*F, T, -1]

        # Return the output after applying ReLU activation
        return self.relu(o)



class CDMGTUNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_processor = LCFTUNIT(12, 64, 128, 1, True)
        self.temporal_narrow = LCFTUNIT(76, 64, 256, 1, False)
        self.spectral = LCFTUNIT(37, 64, 128, 1, True)
        self.processor = LCFTUNIT(70, 64, 256,1, False)
        self.multi_scale_gtu_spatial = Multi_GTU(64, 1, (3, 5, 10))
        self.multi_scale_gtu_spectral = Multi_GTU(64, 1, (3, 5, 10))
        self.feature_fusion = MSTF(64)
        self.output_processor = LCFTUNIT(64, 2, 128, 1, True)

    def forward(self, spatial_input, spectral_mag):
        batch_size, time_steps, freq_bins, _ = spatial_input.shape
        original_input = spatial_input
        spatial_feat = spatial_input.reshape(batch_size * time_steps, freq_bins, -1)
        spatial_feat = self.freq_processor(spatial_feat, batch_size, time_steps, freq_bins)
        spatial_feat = spatial_feat.reshape(batch_size, time_steps, freq_bins, -1)
        spatial_feat = torch.cat([spatial_feat, original_input], dim=-1)
        spatial_feat = spatial_feat.permute(0, 2, 1, 3).reshape(batch_size * freq_bins, time_steps, -1)
        spatial_feat = self.temporal_narrow(spatial_feat, batch_size, time_steps, freq_bins)
        spatial_feat = spatial_feat.reshape(batch_size, freq_bins, time_steps, -1).permute(0, 2, 1, 3)
        spatial_feat = spatial_feat.permute(0, 3, 1, 2)  # [B, C, T, F]
        spatial_feat = self.multi_scale_gtu_spatial(spatial_feat)
        spatial_output = spatial_feat.permute(0, 2, 3, 1)  # 恢复[B, T, F, C]格式

        spectral_input = spectral_mag[:, :, :, -1].unsqueeze(-1).permute(0, 1, 3, 2)
        spectral_input = spectral_input.reshape(batch_size * time_steps, -1, freq_bins, 1)

        padded_spectral = torch.cat([spectral_input[:, :, :3, :], spectral_input,
                                     spectral_input[:, :, -3:, :]], dim=2)
        spectral_sub = torch.nn.functional.unfold(padded_spectral, kernel_size=(7, 1))
        spectral_feat = spectral_mag.permute(0, 1, 3, 2).reshape(batch_size * time_steps, -1, freq_bins, 1)
        padded_feat = torch.cat([spectral_feat[:, :, :2, :], spectral_feat,
                                 spectral_feat[:, :, -2:, :]], dim=2)
        unfolded_feat = torch.nn.functional.unfold(padded_feat, kernel_size=(5, 1))
        combined_spectral = torch.cat([unfolded_feat, spectral_sub], dim=1)
        processed_spectral = combined_spectral.reshape(batch_size, time_steps, -1, freq_bins).permute(0,1,3,2)
        processed_spectral = processed_spectral.reshape(batch_size * time_steps, freq_bins, -1)
        processed_spectral = self.spectral(processed_spectral, batch_size, time_steps, freq_bins).reshape(batch_size, time_steps, freq_bins, -1)
        spectral_feat = processed_spectral.permute(0,2,1,3)
        spectral_feat = spectral_feat.reshape(batch_size * freq_bins, time_steps, -1)
        padded_real_mag = spectral_mag[:, :, :, -1].unsqueeze(-1).permute(0, 2, 3, 1)
        padded_real_mag = torch.nn.functional.pad(padded_real_mag, pad=(5, 0), mode='constant', value=0)
        unfolded_real_mag = torch.nn.functional.unfold(
            padded_real_mag.reshape(batch_size * freq_bins, 1, -1, 1),
            kernel_size=(6, 1)
        )
        unfolded_real_mag = unfolded_real_mag.reshape(batch_size, freq_bins, -1, time_steps)
        unfolded_real_mag = unfolded_real_mag.permute(0, 1,3,2).reshape(batch_size * freq_bins, time_steps, -1)
        final_spectral = torch.cat([spectral_feat, unfolded_real_mag], dim=-1)
        final_spectral = self.processor(final_spectral, batch_size, time_steps, freq_bins)
        final_spectral = final_spectral.reshape(batch_size, time_steps, freq_bins, -1)
        final_spectral = final_spectral.permute(0, 3, 1, 2)
        final_spectral = self.multi_scale_gtu_spectral(final_spectral)
        spectral_output = final_spectral.permute(0, 2, 3, 1)
        fused_features = self.feature_fusion(spatial_output, spectral_output)
        fused_features = fused_features.reshape(batch_size * time_steps, freq_bins, -1)
        network_output = self.output_processor(fused_features, batch_size, time_steps, freq_bins).reshape(batch_size, time_steps, freq_bins, -1)
        return network_output

