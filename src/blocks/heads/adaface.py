import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F

from src.common.utils import get_logger

logger = get_logger()

class AdaFace(nn.Module):
    def __init__(self,
                 embedding_size=512,
                 classnum=10575,
                 m=0.4,
                 h=0.333,
                 s=64.0,
                 t_alpha=1.0):
        """
        AdaFace head with EMA-based margin adjustment and stability handling.

        Args:
            embedding_size (int): Size of input feature vectors (embedding size).
            classnum (int): Number of output classes (number of identities).
            m (float): Base angular margin.
            h (float): Scaling factor for margin adaptation.
            s (float): Scaling factor for the cosine similarities.
            t_alpha (float): Exponential moving average factor.
        """
        super(AdaFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))

        # Initialize the kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.h = h
        self.s = s
        self.eps = 1e-3
        self.t_alpha = t_alpha

        # Register buffers for EMA
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1) * 20)
        self.register_buffer('batch_std', torch.ones(1) * 100)

        logger.info(f"AdaFace with the following properties: m: {self.m}, h: {self.h}, s: {self.s}, t_alpha: {self.t_alpha}")

    def forward(self, embbedings, label, norms):
        """
        Forward pass through AdaFace head.
        Args:
            embbedings (torch.Tensor): Embedding features from the backbone.
            norms (torch.Tensor): Norms of the embeddings.
            label (torch.Tensor): Ground truth class labels.
        """
        kernel_norm = l2_norm(self.kernel,axis=0)
        cosine = torch.mm(embbedings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps)

        safe_norms = torch.clip(norms, min=0.001, max=100)
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps)
        margin_scaler = margin_scaler * self.h
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        # g_angular
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # g_additive
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m


def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output