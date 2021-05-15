from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        num_features: int = None,
        n_charts: int = None,
        dim_z: int = None,
        dropout: float = 0.0,
        detach: bool = False,
        norm_layer: Optional[str] = None,
        normalize: bool = False,
        set_bn_eval: bool = False,
        remap: bool = False,
        normalize_weight: bool = False,
    ) -> None:
        assert (num_features is not None) != (
            n_charts is not None and dim_z is not None
        )
        self.is_atlas = num_features is None

        if self.is_atlas:
            assert dropout == 0.0
            assert not normalize_weight
        super(BaseModel, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.detach = detach
        self.normalize = normalize
        self.set_bn_eval = set_bn_eval
        self.normalize_weight = normalize_weight

        self.norm_layer = nn.Identity()
        if norm_layer == "layer":
            self.norm_layer = nn.LayerNorm(
                self.backbone_features, elementwise_affine=False
            )
        if norm_layer == "batch":
            self.norm_layer = nn.BatchNorm1d(self.backbone_features, affine=False)

        self.remap = nn.Identity()
        if not self.is_atlas and (remap or num_features != self.backbone_features):
            self.remap = nn.Linear(self.backbone_features, num_features)
            nn.init.zeros_(self.remap.bias)

        self.dropout = nn.Dropout(p=dropout)

        if self.is_atlas:
            self.coords = nn.ModuleList(
                [nn.Linear(1000, dim_z) for _ in range(n_charts)]
            )
            self.q = nn.Sequential(nn.Linear(1000, n_charts), nn.Softmax(1))
            self.classifiers = nn.ModuleList(
                [nn.Linear(dim_z, num_classes) for _ in range(n_charts)]
            )
            for c in self.classifiers:
                nn.init.zeros_(c.weight)
                nn.init.zeros_(c.bias)
        else:
            self.classifier = nn.Linear(num_features, num_classes)
            nn.init.zeros_(self.classifier.weight)
            nn.init.zeros_(self.classifier.bias)

    def init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.feature_extractor(x)
        features = self.norm_layer(features)
        features = self.remap(features)

        if self.normalize:
            features = nn.functional.normalize(features, p=2, dim=1)

        if self.is_atlas:
            q, coords = self.q(features), torch.stack([c(x) for c in self.coords], 1)
            logit_comps = torch.stack(
                [p(coords.select(1, i)) for i, p in enumerate(self.classifiers)], 1
            )
            logits = (q.unsqueeze(2) * logit_comps).sum(1)

            return logits, q, coords
        else:
            classification_features = self.dropout(
                features.detach() if self.detach else features
            )
            classifier_weight = self.classifier.weight
            if self.normalize_weight:
                classifier_weight = F.normalize(classifier_weight)
            logits = F.linear(
                classification_features, classifier_weight, self.classifier.bias
            )

            return logits, features

    def train(self, mode=True):
        r"""Sets the module in training mode.

        This has any effect only on certain modules. See documentations of
        particular modules for details of their behaviors in training/evaluation
        mode, if they are affected, e.g. :class:`Dropout`, :class:`BatchNorm`,
        etc.

        Args:
            mode (bool): whether to set training mode (``True``) or evaluation
                         mode (``False``). Default: ``True``.

        Returns:
            Module: self
        """
        self.training = mode
        for module in self.children():
            module.train(mode)

        if self.set_bn_eval:
            for module in self.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        return self
