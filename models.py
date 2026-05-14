"""
Neural network architectures for flood depth prediction.

- ConvLSTM: Baseline recurrent convolutional model
- UNetConvLSTM: Proposed model with U-Net spatial encoding + ConvLSTM temporal
"""

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell."""

    def __init__(self, input_channels, hidden_channels, kernel_size):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
            bias=True,
        )

    def forward(self, x, h, c):
        """
        Args:
            x: (B, C_in, H, W)
            h: (B, C_hidden, H, W) hidden state
            c: (B, C_hidden, H, W) cell state
        
        Returns:
            h_new, c_new
        """
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)
        
        i, f, g, o = torch.split(conv_out, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        
        return h_new, c_new


class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM encoder-decoder."""

    def __init__(
        self,
        input_channels,
        hidden_channels,
        kernel_size,
        num_layers,
        output_channels,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.output_channels = output_channels
        
        # Build cells
        self.cells = nn.ModuleList()
        for layer_idx in range(num_layers):
            in_ch = input_channels if layer_idx == 0 else hidden_channels[layer_idx - 1]
            out_ch = hidden_channels[layer_idx]
            self.cells.append(ConvLSTMCell(in_ch, out_ch, kernel_size))
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_channels[-1], output_channels, 1)

    def forward(self, x):
        """
        Args:
            x: (B, T, C_in, H, W) input sequence
        
        Returns:
            y: (B, 1, C_out, H, W) single output frame
        """
        B, T, C, H, W = x.shape
        
        # Initialize hidden and cell states
        h = [torch.zeros(B, ch, H, W, device=x.device) for ch in self.hidden_channels]
        c = [torch.zeros(B, ch, H, W, device=x.device) for ch in self.hidden_channels]
        
        # Process sequence
        for t in range(T):
            x_t = x[:, t, :, :, :]  # (B, C_in, H, W)
            
            for layer_idx in range(self.num_layers):
                h_t, c_t = self.cells[layer_idx](x_t, h[layer_idx], c[layer_idx])
                h[layer_idx] = h_t
                c[layer_idx] = c_t
                x_t = h_t
        
        # Project to output
        y = self.output_proj(h[-1])  # (B, C_out, H, W)
        return y.unsqueeze(1)  # (B, 1, C_out, H, W)


class UNetConvLSTM(nn.Module):
    """U-Net encoder + ConvLSTM temporal + decoder."""

    def __init__(
        self,
        input_channels,
        unet_channels,
        convlstm_hidden,
        convlstm_layers,
        output_channels,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.unet_channels = unet_channels
        self.output_channels = output_channels
        
        # U-Net encoder (downsample)
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = input_channels
        for out_ch in unet_channels:
            self.enc_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            self.pools.append(nn.MaxPool2d(2))
            in_ch = out_ch
        
        # ConvLSTM at bottleneck
        self.convlstm = ConvLSTM(
            input_channels=unet_channels[-1],
            hidden_channels=[convlstm_hidden] * convlstm_layers,
            kernel_size=3,
            num_layers=convlstm_layers,
            output_channels=unet_channels[-1],
        )
        
        # U-Net decoder (upsample + skip connections)
        self.dec_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        
        # Build decoder: one block per encoder level
        # Bottleneck comes in with channels unet_channels[-1]
        prev_channels = unet_channels[-1]
        for i, out_ch in enumerate(reversed(unet_channels)):
            # Skip connection index: counting from the end
            # For reversed order at iteration i, the skip is at index -(i+1) in original
            skip_idx = -(i + 1)
            skip_channels = unet_channels[skip_idx]
            
            self.upsamples.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.dec_blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_channels + skip_channels, out_ch, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.ReLU(inplace=True),
                )
            )
            prev_channels = out_ch
        
        # Final output
        self.final_conv = nn.Conv2d(unet_channels[0], output_channels, 1)

    def forward(self, x):
        """
        Args:
            x: (B, T, C_in, H, W) input sequence
        
        Returns:
            y: (B, 1, C_out, H, W) prediction
        """
        B, T, C, H, W = x.shape
        
        # Process each timestep through encoder separately
        encoder_outputs = []
        for t in range(T):
            x_t = x[:, t, :, :, :]  # (B, C_in, H, W)
            skip_connections = []
            
            # Encode
            for enc_block, pool in zip(self.enc_blocks, self.pools):
                x_t = enc_block(x_t)
                skip_connections.append(x_t)
                x_t = pool(x_t)
            
            encoder_outputs.append((x_t, skip_connections))
        
        # Extract bottleneck features -> ConvLSTM
        bottlenecks = torch.stack([e[0] for e in encoder_outputs], dim=1)  # (B, T, C, H, W)
        convlstm_out = self.convlstm(bottlenecks)  # (B, 1, C, H, W)
        x_t = convlstm_out[:, 0, :, :, :]  # (B, C, H, W)
        
        # Decoder (use last timestep's skip connections in reverse order)
        last_skips = encoder_outputs[-1][1]
        # Reverse skips to match decoder order: we collected [res_256, res_128, res_64]
        # For decoder, we need [res_64, res_128, res_256]
        reversed_skips = last_skips[::-1]
        
        for upsample, dec_block, skip in zip(self.upsamples, self.dec_blocks, reversed_skips):
            x_t = upsample(x_t)
            x_t = torch.cat([x_t, skip], dim=1)
            x_t = dec_block(x_t)
        
        # Project to output
        y = self.final_conv(x_t)  # (B, C_out, H, W)
        return y.unsqueeze(1)  # (B, 1, C_out, H, W)
