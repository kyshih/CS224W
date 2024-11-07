from bpnetlite import BPNet
import torch
import torch.nn as nn

class BPExtractor(nn.Module):
    def __init__(self, original_model):
        super(BPExtractor, self).__init__()
        self.original_model = original_model

    def forward(self, X, X_ctl=None, return_intermediate=True):
        start, end = self.original_model.trimming, X.shape[2] - self.original_model.trimming

        # Initial Convolution Block
        X = self.original_model.irelu(self.original_model.iconv(X))

        # Residual Convolutions
        for i in range(self.original_model.n_layers):
            X_conv = self.original_model.rrelus[i](self.original_model.rconvs[i](X))
            X = torch.add(X, X_conv)

        # If X_ctl is provided, concatenate
        if X_ctl is None:
            X_w_ctl = X
        else:
            X_w_ctl = torch.cat([X, X_ctl], dim=1)

        # Profile prediction (y_profile)
        y_profile = self.original_model.fconv(X_w_ctl)[:, :, start:end]

        # Counts prediction (X before linear)
        X_before_linear = torch.mean(X[:, :, start-37:end+37], dim=2)

        if X_ctl is not None:
            X_ctl = torch.sum(X_ctl[:, :, start-37:end+37], dim=(1, 2))
            X_ctl = X_ctl.unsqueeze(-1)
            X_before_linear = torch.cat([X_before_linear, torch.log(X_ctl + 1)], dim=-1)

        # If return_intermediate is True, return X_before_linear
        if return_intermediate:
            return X_before_linear

        # Pass X_before_linear to the linear layer for y_counts
        y_counts = self.original_model.linear(X_before_linear).reshape(X_before_linear.shape[0], 1)

        # Return the original model outputs
        return y_profile, y_counts