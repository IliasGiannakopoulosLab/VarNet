import torch
import numpy as np
from pytorch_grad_cam import GradCAM
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import check_img_size

# -------------------------------------------#
# -------- yolov5 feature extractor -------- #
# -------------------------------------------#
class YOLOv5FeatureExtractor(DetectMultiBackend):
    def __init__(self, cfg, weights, device, imgsz=(640,640), feature_layers=[4,6,9]):
        device = torch.device(device)
        super().__init__(weights, device=device, dnn=False, data=cfg)

        self.imgsz = check_img_size(imgsz, s=self.stride)
        self.feature_layers = feature_layers
        self._features = {}

        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        self._register_hooks()

    def _register_hooks(self):
        def make_hook(idx):
            def hook(module, inp, out):
                self._features[idx] = out
            return hook

        for idx in self.feature_layers:
            self.model.model[idx].register_forward_hook(make_hook(idx))

    def forward(self, x):
        self._features = {}
        _ = super().forward(x, augment=False)
        return [self._features[idx] for idx in self.feature_layers]


# -------------------------------------------#
# ----------- dummy scalar target ---------- #
# -------------------------------------------#
class YoloConfidenceTarget:
    def __call__(self, model_output):
        return model_output[0][:, 4].sum()


# -------------------------------------------#
# ---------------- yolo gradcam ------------ #
# -------------------------------------------#
class YOLOGradCAM:
    def __init__(self, yolo_backend_model):
        self.model = yolo_backend_model.eval()
        self.target_layer = self.model.model.model[21]

    def __call__(self, x):

        device = x.device
        self.model = self.model.to(device)

        for p in self.model.parameters():
            p.requires_grad_(True)

        x = x.float().requires_grad_(True)
        cam = GradCAM(model=self.model,target_layers=[self.target_layer])

        with torch.enable_grad():
            grayscale_cam = cam(input_tensor=x,targets=[YoloConfidenceTarget()])[0]
        grayscale_cam = (grayscale_cam - grayscale_cam.min() ) / (grayscale_cam.max() + 1e-8)

        for p in self.model.parameters():
            p.requires_grad_(False)

        return torch.from_numpy(grayscale_cam).to(x.device)
