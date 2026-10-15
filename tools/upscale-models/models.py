"""PyTorch definitions of the two bundled upscalers.

* AnimeVideo-v3 (Real-ESRGAN, BSD-3-Clause): https://github.com/xinntao/Real-ESRGAN
* Anime4K CNN (bloc97, MIT): https://github.com/bloc97/Anime4K -- rebuilt from the GLSL shader weights.
"""

import re
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F

ANIMEVIDEO_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
)
ANIME4K_URL = (
    "https://raw.githubusercontent.com/bloc97/Anime4K/v4.0.1/glsl/Upscale/"
    "Anime4K_Upscale_CNN_x2_{variant}.glsl"
)


def download(url: str, dest: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "aidoku-model-tools"})
    with urllib.request.urlopen(request) as response, open(dest, "wb") as out:
        out.write(response.read())
    return dest


# --- Real-ESRGAN AnimeVideo-v3 --------------------------------------------------------------


class SRVGGNetCompact(nn.Module):
    """Same layout as realesrgan.archs.srvgg_arch.SRVGGNetCompact (so the .pth loads as is)."""

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4):
        super().__init__()
        self.upscale = upscale
        body = [nn.Conv2d(num_in_ch, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)]
        for _ in range(num_conv):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(nn.PReLU(num_parameters=num_feat))
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.body = nn.ModuleList(body)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode="nearest")


class AnimeVideoX2(nn.Module):
    """The x4 network with its output area-averaged down to 2x."""

    def __init__(self, net: SRVGGNetCompact):
        super().__init__()
        self.net = net

    def forward(self, x):
        return F.avg_pool2d(self.net(x), 2)


def load_animevideo(weights_path: str) -> AnimeVideoX2:
    net = SRVGGNetCompact()
    state = torch.load(weights_path, map_location="cpu")
    state = state.get("params", state.get("params_ema", state))
    net.load_state_dict(state, strict=True)
    return AnimeVideoX2(net.eval()).eval()


# --- Anime4K ---------------------------------------------------------------------------------

_DEFINE = re.compile(r"#define\s+go_(\d+)\(x_off, y_off\)\s+\((.*)\)\s*$")
_RELU = re.compile(r"max\((-?)\((\w+?)_texOff\(vec2\(x_off, y_off\)\)\), 0\.0\)$")
_PLAIN = re.compile(r"(\w+?)_texOff\(vec2\(x_off, y_off\)\)$")
_TERM = re.compile(r"mat4\(([^)]*)\)\s*\*\s*go_(\d+)\((-?[\d.]+), (-?[\d.]+)\)")
_BIAS = re.compile(r"result \+= vec4\(([^)]*)\);")


def _split_passes(text: str):
    passes, current = [], None
    for line in text.splitlines():
        if line.startswith("//!DESC"):
            current = {"desc": line[7:].strip(), "binds": [], "save": None, "body": []}
            passes.append(current)
        elif current is None:
            continue
        elif line.startswith("//!BIND"):
            current["binds"].append(line.split()[1])
        elif line.startswith("//!SAVE"):
            current["save"] = line.split()[1]
        elif line.startswith("//!"):
            pass  # HOOK / WIDTH / HEIGHT / COMPONENTS / WHEN
        else:
            current["body"].append(line)
    return passes


class _ConvPass(nn.Module):
    def __init__(self, save, inputs, weight, bias):
        super().__init__()
        self.save = save
        self.inputs = inputs  # [(texture, sign)]; sign: 0 identity, +1 relu(x), -1 relu(-x)
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(self, tex):
        parts = []
        for name, sign in self.inputs:
            t = tex[name]
            if sign == 1:
                t = F.relu(t)
            elif sign == -1:
                t = F.relu(-t)
            parts.append(t)
        inp = torch.cat(parts, dim=1) if len(parts) > 1 else parts[0]
        return F.conv2d(inp, self.weight, self.bias, padding=1)


class Anime4K(nn.Module):
    """Stack of 3x3 convs with CReLU activations, then depth-to-space + bilinear residual."""

    def __init__(self, glsl_text: str):
        super().__init__()
        passes = _split_passes(glsl_text)
        convs = []
        self.d2s = None
        for p in passes:
            body = "\n".join(p["body"])
            if "mat4" in body:
                convs.append(self._build_conv(p, body))
            elif "Depth-to-Space" in p["desc"]:
                names = [b for b in p["binds"] if b != "MAIN"]
                if len(names) != 3:
                    raise ValueError(f"unexpected depth-to-space binds: {p['binds']}")
                self.d2s = names
            else:
                raise ValueError(f"unsupported pass: {p['desc']}")
        if self.d2s is None or not convs:
            raise ValueError("shader has no conv passes or no depth-to-space pass")
        self.convs = nn.ModuleList(convs)

    @staticmethod
    def _build_conv(p, body):
        defines = {}
        for line in p["body"]:
            m = _DEFINE.match(line.strip())
            if not m:
                continue
            k, expr = int(m.group(1)), m.group(2)
            r = _RELU.match(expr)
            if r:
                defines[k] = (r.group(2), -1 if r.group(1) else 1)
            else:
                plain = _PLAIN.match(expr)
                if not plain:
                    raise ValueError(f"unsupported go_{k} expression: {expr}")
                defines[k] = (plain.group(1), 0)
        inputs = [defines[k] for k in sorted(defines)]
        widths = [3 if name == "MAIN" else 4 for name, _ in inputs]
        offsets = [sum(widths[:i]) for i in range(len(widths))]
        weight = torch.zeros(4, sum(widths), 3, 3)
        for m in _TERM.finditer(body):
            values = [float(v) for v in m.group(1).split(",")]
            k, dx, dy = int(m.group(2)), int(float(m.group(3))), int(float(m.group(4)))
            if abs(dx) > 1 or abs(dy) > 1:
                raise ValueError("kernel larger than 3x3")
            for j in range(widths[k]):
                for o in range(4):
                    # GLSL mat4 is column-major: out[o] = sum_j values[4*j + o] * in[j]
                    weight[o, offsets[k] + j, dy + 1, dx + 1] = values[4 * j + o]
            if widths[k] == 3 and any(values[12:16]):
                raise ValueError("alpha channel weights are not zero")
        bias_match = _BIAS.search(body)
        bias = torch.tensor([float(v) for v in bias_match.group(1).split(",")])
        return _ConvPass(p["save"], inputs, weight, bias)

    def forward(self, x):
        tex = {"MAIN": x}
        for conv in self.convs:
            tex[conv.save] = conv(tex)
        residual = torch.cat([F.pixel_shuffle(tex[name], 2) for name in self.d2s], dim=1)
        base = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return residual + base


def load_anime4k(glsl_path: str) -> Anime4K:
    with open(glsl_path, "r", encoding="utf-8") as f:
        return Anime4K(f.read()).eval()
