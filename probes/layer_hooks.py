# layer_hooks.py
import torch


class LayerCache:
    def __init__(self, layers_to_collect):
        self.layers = set(layers_to_collect or [])
        self.buf = {}

    def clear(self):
        self.buf.clear()

    def hook_maker(self, layer_idx:int):
        def _hook(module, inputs, output):
            # output: [B, T, D]
            if layer_idx in self.layers and isinstance(output, torch.Tensor):
                self.buf[layer_idx] = output[:, -1, :].detach()
        return _hook


def register_lasttoken_hooks(model, layers_to_collect):
    lc = LayerCache(layers_to_collect)
    handles = []
    for i, layer in enumerate(model.model.layers):
        if i in lc.layers:
            h = layer.register_forward_hook(lc.hook_maker(i))
            handles.append(h)
    return lc, handles
