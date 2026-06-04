import torch
import torch.nn.functional as F


def _to_tuple(v, n):
    if isinstance(v, tuple):
        return v
    return (v,) * n


class OrPooling2d(torch.nn.Module):
    """Logic gate based pooling layer."""

    def __init__(self, kernel_size, stride, padding=0, export_mode=False):
        super(OrPooling2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel_size_tuple = _to_tuple(kernel_size, 2)
        self.stride_tuple = _to_tuple(stride, 2)
        self.padding_tuple = _to_tuple(padding, 2)
        self.export_mode = export_mode

    def forward(self, x):
        """Pool using logical OR (export) or max pooling (normal)."""
        if self.export_mode:
            assert x.dtype == torch.bool, "Export mode requires boolean input"
            return self._torch_or_bool(x)
            
        assert isinstance(x, torch.Tensor), "Input must be a PyTorch tensor in non-export mode."
        assert x.dim() == 4, "Input tensor must be 4d"

        orig_dtype = x.dtype
        x = F.max_pool2d(
            x.float(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        return (x > 0) if orig_dtype == torch.bool else x.to(orig_dtype)


    def _torch_or_bool(self, x):
        kh, kw = self.kernel_size_tuple
        sh, sw = self.stride_tuple
        ph, pw = self.padding_tuple

        x = F.pad(x, (pw, pw, ph, ph), value=False)
        x = x.unfold(2, kh, sh).unfold(3, kw, sw).flatten(-2)  # (n, c, out_h, out_w, kh*kw)

        result = x[..., 0]
        for i in range(1, x.shape[-1]):
            result = result | x[..., i]
        return result 


    def set_export_mode(self, export_mode: bool):
        """Set export mode for the layer."""
        self.eval()
        self.export_mode = export_mode


class OrPooling3d(torch.nn.Module):
    """Logic gate based pooling layer."""

    def __init__(self, kernel_size, stride, padding=0, export_mode=False):
        super(OrPooling3d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel_size_tuple = _to_tuple(kernel_size, 3)
        self.stride_tuple = _to_tuple(stride, 3)
        self.padding_tuple = _to_tuple(padding, 3)
        self.export_mode = export_mode

    def forward(self, x):

        if self.export_mode:
            assert x.dtype == torch.bool, "Export mode requires boolean input"
            return self._torch_or_pool(x)
            
        assert isinstance(x, torch.Tensor), "Expected torch tensor"
        assert x.dim() == 5, "Input tensor must be 5d"

        orig_dtype = x.dtype

        x = F.max_pool3d(
            x.float(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        return (x > 0) if orig_dtype == torch.bool else x.to(orig_dtype)


    def _torch_or_pool(self, x):
        kd, kh, kw = self.kernel_size_tuple
        sd, sh, sw = self.stride_tuple
        pd, ph, pw = self.padding_tuple

        x = F.pad(x, (pw, pw, ph, ph, pd, pd), value=False)
        x = x.unfold(2, kd, sd).unfold(3, kh, sh).unfold(4, kw, sw).flatten(-3)  # (n, c, out_d, out_h, out_w, kd*kh*kw)

        result = x[..., 0]
        for i in range(1, x.shape[-1]):
            result = result | x[..., i]
        return result
    

    def set_export_mode(self, export_mode: bool):
        """Set export mode for the layer."""
        self.eval()
        self.export_mode = export_mode
