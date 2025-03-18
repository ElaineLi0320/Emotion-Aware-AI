"""
    Implementation of EmoNeXt for FER introduced in a 2025 paper.
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.ops import StochasticDepth

# A list of ConvNeXt models with different channel and block combinations
model_urls = {
    "convnext_tiny_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth",
    "convnext_small_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth",
    "convnext_base_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth",
    "convnext_large_1k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth",
    "convnext_tiny_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    "convnext_small_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth",
    "convnext_base_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth",
    "convnext_large_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth",
    "convnext_xlarge_22k": "https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth",
}

class SELayer(nn.Module):
    """
        Sequeeze and excitation block for learning channel relationship
    """
    def __init__(self, channel, r=16):
        """
            channel: number of channels in input
            r: reduction ratio 
        """
        super(SELayer, self).__init__()
        # Reduce spatical dimensions H, W to 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation stage
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
            x: data input of shape (Batch, Channel, Height, Width)
        """
        b, c, _, _ = x.shape
        # Output shape: (b, c)
        y = self.avg_pool(x).view(b, c)
        # Output shape: (b, c, 1, 1)
        y = self.fc(y).view(b, c, 1, 1)
        # Final output: (b, c, h, w)
        return x * y.expand_as(x)


class Attention(nn.Module):
    """
        Standard self attention mechanism
    """
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.input_dim = input_dim
        # Layer normalization to reduce covariate shift
        self.norm = nn.LayerNorm(input_dim)
        # Derived query vectors from word embeddings
        self.query = nn.Linear(input_dim, input_dim)
        # Derived key vectors from word embeddings
        self.key = nn.Linear(input_dim, input_dim)
        # Derived value vectors from word embeddings
        self.value = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        """
            x: data input of shape (Batch, Sentence_Length, Embedding_Dim)
        """
        x = self.norm(x)
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Mathmatical stablizer 
        scale = 1 / math.sqrt(self.input_dim)
        # Compute pairwise attention weight
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        # Apply softmax along the column and multiply with value matrix
        weights = torch.softmax(scores, dim=-1)

        # Attention weights
        attended_values = torch.matmul(weights, value)
        output = x + attended_values

        return output, weights
    
class LayerNorm(nn.Module):
    """
        Custom implementation of a layer normalization function. It handles "channel first" 
        data format (Batch, Channel, Height, Width) and "channel last" data format
        (Batch, Height, Width, Channel).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channel_last"):
        super().__init__()
        # Learnable paramaters for normalization
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = (normalized_shape, )
        # Small value for numerical stability
        self.eps = eps
        if data_format not in ["channel_first", "channel_last"]:
            raise NotImplementedError("Data format should be either 'channel_first' or 'channel_last'.")
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == "channel_last":
            return F.layer_norm(x, self.normalized_shape, 
                                self.weight, self.bias, self.eps)
        elif self.data_format == "channel_first":
            # Compute channel wise mean
            u = x.mean(1, keepdim=True)
            # Compute channel wise standard deviation
            std = (x - u).pow(2).mean(1, keepdim=True)
            # Normalize input
            x = (x - u) / torch.sqrt(std + self.eps)
            # Apply learnt weight and bias
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    """
        ConvNeXt block. Two equivalent implementations:
        1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all data in (N, C, H, W)
        2) DwConv -> Permute to (N, H, W, C) -> LayerNorm -> Linear -> GELU -> Linear -> Permute back 
        The second one is slightly faster in Pytorch
    """
    def __init__(self, dim, drop_path=0.0, layer_scale=1e-6):
        """
            dim: input dimension
            drop_path: probability in stochastic depth
            layer_scale: init value for layer scale
        """
        super().__init__()
        # Depth-wise convolution to mimic weighted sum in self-attention
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim)
        # Point-wise convolution projecting up
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        # Activation function
        self.act = nn.GELU()
        # Point-wise convolution projecting down
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # A leanrable scaling factor
        self.gamma = (
            nn.Parameter(layer_scale * torch.ones((dim)), requires_grad=True)
            if layer_scale > 0
            else None
        )
        self.stochastic_depth = StochasticDepth(drop_path, "row")

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x 
        # (N, H, W, C) -> (N, C, H, W)
        x = x.permute(0, 3, 1, 2)

        # Residual connection
        x = input + self.stochastic_depth(x)
        return x
    
class EmoNeXt(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, depths=None, dims=None, drop_path_probs=0.0, layer_scale=1e-6):
        """
            in_chans = number of input channels
            num_classes: ???
            depths: stage compute ratio
            dims: a list of dimensions for each ConvNeXt block
            drop_path_probs: probability in stochastic depth
            layer_scale: init value for layer scale
        """
        super().__init__()

        if dims is None:
            dims = [96, 192, 384, 768]
        if depths is None:
            depths = [3, 3, 9, 3]
        
        # Spatial transformation network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),

            nn.Conv2d(8, 10, kernel_size=5),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Spatial affine matrix of shape 3 * 2
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 32), 
            nn.ReLU(True), 
            nn.Linear(32, 3 * 2)
        )

        # Stem layers and three intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], data_format="channel_first")
        )

        self.downsample_layers = (nn.ModuleList())

        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], data_format="channel_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                SELayer(dims[i+1]),
            )
            self.downsample_layers.append(downsample_layer)

        # Four feature resolution stages, each consisting of multiple residual blocks
        self.stages = (
            nn.ModuleList()
        )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_probs, sum(depths))]
        cur = 0
        for i in range(4):
            # Each stage consists of 4 sequential layers
            stage = nn.Sequential(
                *[
                    Block(
                        dim=dims[i],
                        drop_path=dp_rates[cur+j],
                        layer_scale=layer_scale
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        # Final normalization layer
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.attention = Attention(dims[-1])
        # Final classification layer
        self.head = nn.Linear(dims[-1], num_classes)

        # Interate over modules in the model
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # Conv and linear layer are initialized with truncated normal distribution
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize all weights to 0
        self.fc_loc[2].weight.data.zero_()
        # Initialize bias to an identity transformation
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def stn(self, x):
        """
            Spatial transformation network
        """
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 52 * 52)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)

        return x
    
    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # Apply final global average pooling (B, C, H, W) -> (B, C)
        return self.norm(x.mean([-2, -1]))
    
    def forward(self, x, labels=None):
        x = self.stn(x)
        x = self.forward_features(x)
        _, weights = self.attention(x)
        logits = self.head(x)

        if labels is not None:
            mean_attn_weight = torch.mean(weights)
            # Self attention regularization term
            attn_loss = torch.mean((weights - mean_attn_weight) ** 2)

            loss = F.cross_entropy(logits, labels, label_smoothing=0.2) + attn_loss
            return torch.argmax(logits, dim=1), logits, loss
        
        return torch.argmax(logits, dim=1), logits

def get_model(num_classes, model_size="tiny", in_22k=False):
    """
        Load emonext with pretrained convnext model weights and prepare it for use
    """
    if model_size == "tiny":
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]
        url = (
            model_urls["convnext_tiny_22k"]
            if in_22k
            else model_urls["convnext_tiny_1k"]
        )
    elif model_size == "small":
        depths = [3, 3, 27, 3]
        dims = [96, 192, 384, 768]
        url = (
            model_urls["convnext_small_22k"]
            if in_22k
            else model_urls["convnext_small_1k"]
        )
    elif model_size == "base":
        depths = [3, 3, 27, 3]
        dims = [128, 256, 512, 1024]
        url = (
            model_urls["convnext_base_22k"]
            if in_22k
            else model_urls["convnext_base_1k"]
        )
    elif model_size == "large":
        depths = [3, 3, 27, 3]
        dims = [192, 384, 768, 1536]
        url = (
            model_urls["convnext_large_22k"]
            if in_22k
            else model_urls["convnext_large_1k"]
        )
    else:
        depths = [3, 3, 27, 3]
        dims = [256, 512, 1024, 2048]
        url = model_urls["convnext_xlarge_22k"]
    
    default_num_classes = 1000
    if in_22k:
        default_num_classes = 21841

    # Initialize a neural network
    net = EmoNeXt(
        depths=depths, dims=dims, num_classes=default_num_classes, drop_path_probs=0.1
    )

    # Load weights of pre-trained ConvNeXt by url
    state_dict = load_state_dict_from_url(url)
    # Load weights into our neural network
    net.load_state_dict(state_dict["model"], strict=False)
    net.head = nn.Linear(dims[-1], num_classes)

    return net
    

        





