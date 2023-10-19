import torch
import timm
from torch import nn
from stInfer.module import ProjectionHead


class ClassModel(nn.Module):
    def __init__(
            self,
            gene_num=1000,
            model_name='resnet50',
            pretrained=True,
    ):
        super().__init__()
        assert model_name in ['vgg16', 'densenet121', 'resnet18', 'resnet50']
        self.gene_num = gene_num
        self.image_encoder = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="avg")
        print(f'create model {model_name}!')
        if pretrained:
            self.image_encoder.load_state_dict(torch.load(f'./timm_model/{model_name}.pth'), strict=False)
            print(f'load pretrained model!')
        else:
            pass
        if model_name == 'resnet18':
            self.image_header = ProjectionHead(512, 256)
        elif model_name == 'resnet50':
            self.image_header = ProjectionHead(2048, 256)
        elif model_name == 'densenet121':
            self.image_header = ProjectionHead(1024, 256)
        elif model_name == 'vgg16':
            self.image_header = ProjectionHead(4096, 256)
        else:
            raise NotImplementedError
        self.image_classifier = nn.Linear(256, self.gene_num)

    def forward(self, image):
        # Getting Image and spot Features
        # Getting Image and Spot Embeddings (with same dimension) 
        image_features = self.image_encoder(image)
        image_embeddings = self.image_header(image_features)
        y_pred = self.image_classifier(image_embeddings)
        return image_features, y_pred


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()


if __name__ == '__main__':
    images = torch.randn(8, 3, 224, 224)
    expression = torch.randn(8, 1000)
    batch = {
        'image': images,
        'expression': expression,
    }

    model = ClassModel(1000)
    loss = model(batch)
    print("")
