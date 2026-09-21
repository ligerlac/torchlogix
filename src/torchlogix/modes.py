import torch


class ExportableModule(torch.nn.Module):
    """An ``nn.Module`` that knows about train, eval and export mode.

    Logic layers, their connections and the binarization layers all need the
    same three-way split: a relaxed differentiable path for training, a
    discrete path for evaluation, and an export path whose forward consists
    only of boolean and indexing operations so it can be traced into a
    :class:`~torchlogix.circuit.Circuit`.

    ``train`` vs ``eval`` is plain PyTorch — read ``self.training`` in
    ``forward``. Export mode is this class: setting it implies ``eval()``, and
    subclasses override :meth:`_on_export_mode` to freeze whatever the export
    path needs into buffers (a resolved LUT id, a resolved wiring index, ...)
    so that no ``argmax`` or sampling survives into the traced graph.

    ``torchlogix.utils.set_export_mode`` walks a model and calls
    :meth:`set_export_mode` on every module that has it, so inheriting from
    this class is all a module needs to join in.
    """

    export_mode: bool = False

    def set_export_mode(self, enabled: bool = True):
        """Enable or disable export mode for this module and its children.

        Recursive, like ``nn.Module.eval()``: a layer's connections live in a
        child module, so a non-recursive version would leave them sampling and
        arg-maxing while the parent believes it is exporting.

        Args:
            enabled: Whether to switch the module into export mode.
        """
        self.eval()
        self.export_mode = enabled
        self._on_export_mode(enabled)
        for child in self.children():
            if isinstance(child, ExportableModule):
                child.set_export_mode(enabled)

    def _on_export_mode(self, enabled: bool):
        """Hook for subclasses to materialize or release export-only buffers.

        Called by :meth:`set_export_mode` after ``export_mode`` has been set.
        The default is a no-op, which is correct for any module whose forward
        is already pure boolean/indexing operations.
        """
        pass
