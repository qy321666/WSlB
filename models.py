"""网络结构:ResNet1D 骨干 + 分类头 + 域判别器 + 梯度反转层。"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 梯度反转层(DANN, Ganin et al. 2016)
# ---------------------------------------------------------------------------


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha):
    """前向恒等,反向将梯度乘以 -alpha。alpha=0 时相当于不更新特征。"""
    return GradientReversal.apply(x, alpha)


# ---------------------------------------------------------------------------
# ResNet1D 骨干
# ---------------------------------------------------------------------------


class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(cout)
        self.conv2 = nn.Conv1d(cout, cout, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(cout)
        self.short = None
        if cin != cout or stride != 1:
            self.short = nn.Sequential(
                nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                nn.BatchNorm1d(cout),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.short is not None:
            x = self.short(x)
        return F.relu(out + x)


class Backbone(nn.Module):
    """输入 (B, 1, L) -> 输出 (B, out_dim)。窗口 L=1024 时经 stem 与 3 个 block 下采样。"""

    def __init__(self, in_ch=1, channels=(32, 64, 128), out_dim=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, channels[0], 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        blocks, cin = [], channels[0]
        for i, cout in enumerate(channels):
            blocks.append(ResBlock1D(cin, cout, stride=2 if i > 0 else 1))
            cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(cin, out_dim)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class Classifier(nn.Module):
    def __init__(self, in_dim, n_class):
        super().__init__()
        self.fc = nn.Linear(in_dim, n_class)

    def forward(self, z):
        return self.fc(z)


class Discriminator(nn.Module):
    """域判别器:输入特征 -> 域标量(BCEWithLogits)。"""

    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


def build_model(n_class, device):
    backbone = Backbone().to(device)
    classifier = Classifier(backbone.head.out_features, n_class).to(device)
    disc = Discriminator(backbone.head.out_features).to(device)
    return backbone, classifier, disc
