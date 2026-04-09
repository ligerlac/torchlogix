import numpy as np
import torch

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
        self.export_mode = export_mode

    def forward(self, x):
        """Pool the max value in the kernel."""
        if isinstance(x, np.ndarray) and self.export_mode:
            
            assert x.ndim == 4, "Input tensor must be 4d"

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
                constant_values=0,
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
            return windows.max(axis=(-2, -1))

        assert x.dim() == 4, "Input tensor must be 4d"
        x = torch.nn.functional.max_pool2d(
            x.float(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        return x
    
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

    def forward(self, x):
        if isinstance(x, np.ndarray) and self.export_mode:
            if self.training:
                raise TypeError("NumPy input is only supported in eval mode.")

            assert x.ndim == 5, "Input tensor must be 5d"

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
                constant_values=0,
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
            return windows.max(axis=(-3, -2, -1))
        
        assert x.dim() == 5, "Input tensor must be 5d"
        """Pool the max value in the kernel."""
        x = torch.nn.functional.max_pool3d(
            x.float(),
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        return x
    
    def set_export_mode(self, export_mode: bool):
        """Set export mode for the layer."""
        self.eval()
        self.export_mode = export_mode
