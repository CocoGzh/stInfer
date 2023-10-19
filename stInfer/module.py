from torch import nn
import timm


class ImageEncoderResnet50(nn.Module):
    """
    Encode images to a fixed size vector
    """

    def __init__(
            self, model_name='resnet50', pretrained=True, trainable=True):
        super().__init__()
        self.model = timm.create_model(
            model_name, pretrained, num_classes=0, global_pool="avg"
        )
        for p in self.model.parameters():
            p.requires_grad = trainable

    def forward(self, x):
        return self.model(x)


class ProjectionHead(nn.Module):
    def __init__(
            self,
            in_dim=2048,
            out_dim=1000,
            dropout=0.5
    ):
        super().__init__()
        self.projection = nn.Linear(in_dim, out_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
