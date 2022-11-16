import torch
from torch import nn

class CrossEntropyForMulticlassLoss(torch.nn.CrossEntropyLoss):
    # This loss applies cross entropy after reducing the number of prediction
    # dimensions to the number of classes in the target

    # TODO: loss.item() doesn't work so the displayed losses are Nans
    def __init__(self, num_classes, weight=None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0) -> None:
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction, ignore_index=ignore_index)
        self.num_classes = num_classes

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.zeros_like(input[:, :, 0])
        for b in range(target.shape[1]):
            l = super().forward(input[:, b, 0:len(torch.unique(target[:, b]))], target[:, b])
            loss[:, b] += l
        return loss.flatten()

def JointBCELossWithLogits(output, target):
    # output shape: (S, B, NS) with NS = Number of sequences
    # target shape: (S, B, SL)
    # Loss = -log(mean_NS(prod_SL(p(target_SL, output_NS))))
    # Here at the moment NS = SL
    output = output.unsqueeze(-1).repeat(1, 1, 1, target.shape[-1]) # (S, B, NS, SL)
    output = output.permute(2, 0, 1, 3) # (NS, S, B, SL)
    print(target.shape, output.shape)
    loss = (target * torch.sigmoid(output)) + ((1-target) * (1-torch.sigmoid(output)))
    loss = loss.prod(-1)
    loss = loss.mean(0)
    loss = -torch.log(loss)
    loss = loss.mean()
    return loss

class ScaledSoftmaxCE(nn.Module):
    def forward(self, x, label):
        logits = x[..., :-10]
        temp_scales = x[..., -10:]

        logprobs = logits.softmax(-1)