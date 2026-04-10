import numpy as np
import torch

class OrPooling2d(torch.nn.Module):
    """Logic gate based pooling layer."""

    def __init__(self, kernel_size, stride, padding=0):
        super(OrPooling2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        """Pool using logical OR (export) or max pooling (normal)."""
        if self.export_mode:
            if isinstance(x, np.ndarray):
                assert x.dtype == np.bool_, "Export mode requires boolean input"
                return self._np_or_bool(x)
            elif isinstance(x, torch.Tensor):
                assert x.dtype == torch.bool, "Export mode requires boolean input"
                return self._torch_or_bool(x)
            else:
                raise TypeError("Unsupported input type for export mode.")
            
        assert isinstance(x, torch.Tensor), "Input must be a PyTorch tensor in non-export mode."
        assert x.dim() == 4, "Input tensor must be 4d"

        orig_dtype = x.dtype
        x = torch.nn.functional.max_pool2d(
            x.float(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        return (x > 0) if orig_dtype == torch.bool else x.to(orig_dtype)

    def _np_or_bool(self, x):
        kernel_size = _to_tuple(self.kernel_size, 2)
        stride = _to_tuple(self.stride, 2)
        padding = _to_tuple(self.padding, 2)

        n, c, h, w = x.shape
        kh, kw = kernel_size
        sh, sw = stride
        ph, pw = padding

        x = np.pad(
            x,
            ((0, 0), (0, 0), (ph, ph), (pw, pw)),
            mode="constant",
            constant_values=False,
        )

        out_h = (h + 2 * ph - kh) // sh + 1
        out_w = (w + 2 * pw - kw) // sw + 1

        shape = (n, c, out_h, out_w, kh, kw)
        strides = (
            x.strides[0],
            x.strides[1],
            sh * x.strides[2],
            sw * x.strides[3],
            x.strides[2],
            x.strides[3],
        )

        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        return np.any(windows, axis=(-2, -1))

    def _torch_or_bool(self, x):
        kh, kw = _to_tuple(self.kernel_size, 2)
        sh, sw = _to_tuple(self.stride, 2)
        ph, pw = _to_tuple(self.padding, 2)

        x = torch.nn.functional.pad(x, (pw, pw, ph, ph), value=False)
        x = x.unfold(2, kh, sh).unfold(3, kw, sw)

        return x.any(dim=(-1, -2))

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
        self.export_mode = export_mode

    def forward(self, x):

        if self.export_mode:
            if isinstance(x, np.ndarray):
                assert x.dtype == np.bool_, "Export mode requires boolean input"
                if self.training:
                    raise TypeError("NumPy input is only supported in eval mode.")
                return self._np_or_pool(x)

            elif isinstance(x, torch.Tensor):
                assert x.dtype == torch.bool, "Export mode requires boolean input"
                return self._torch_or_pool(x)

            else:
                raise TypeError("Unsupported input type in export mode")
            
        assert isinstance(x, torch.Tensor), "Expected torch tensor"
        assert x.dim() == 5, "Input tensor must be 5d"

        orig_dtype = x.dtype

        x = torch.nn.functional.max_pool3d(
            x.float(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )

        return (x > 0) if orig_dtype == torch.bool else x.to(orig_dtype)

    def _np_or_pool(self, x):
        kernel_size = _to_tuple(self.kernel_size, 3)
        stride = _to_tuple(self.stride, 3)
        padding = _to_tuple(self.padding, 3)

        n, c, d, h, w = x.shape
        kd, kh, kw = kernel_size
        sd, sh, sw = stride
        pd, ph, pw = padding

        x = np.pad(
            x,
            ((0, 0), (0, 0), (pd, pd), (ph, ph), (pw, pw)),
            mode="constant",
            constant_values=False,
        )

        out_d = (d + 2 * pd - kd) // sd + 1
        out_h = (h + 2 * ph - kh) // sh + 1
        out_w = (w + 2 * pw - kw) // sw + 1

        shape = (n, c, out_d, out_h, out_w, kd, kh, kw)
        strides = (
            x.strides[0],
            x.strides[1],
            sd * x.strides[2],
            sh * x.strides[3],
            sw * x.strides[4],
            x.strides[2],
            x.strides[3],
            x.strides[4],
        )

        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
        return np.any(windows, axis=(-3, -2, -1))
    
    def _torch_or_pool(self, x):
        kd, kh, kw = _to_tuple(self.kernel_size, 3)
        sd, sh, sw = _to_tuple(self.stride, 3)
        pd, ph, pw = _to_tuple(self.padding, 3)

        x = torch.nn.functional.pad(x, (pw, pw, ph, ph, pd, pd), value=False)
        x = x.unfold(2, kd, sd).unfold(3, kh, sh).unfold(4, kw, sw)

        return x.any(dim=(-1, -2, -3))
    
    def set_export_mode(self, export_mode: bool):
        """Set export mode for the layer."""
        self.eval()
        self.export_mode = export_mode
