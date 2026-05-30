from plugins.BaseTechnique import BaseTechnique, Enum, Palette

import numpy as np
from PIL import Image

try:
    art_kit  # injected by sandbox at exec time
except NameError:
    art_kit = None

_PRESETS = {
    # f, k, steps -- chosen so the resulting B texture is mature, not noisy.
    # Native grid is 384x384; values picked from the canonical mitchell map
    # and verified to produce the named texture at this scale.
    "spots":  (0.0540, 0.0620, 5000),
    "maze":   (0.0290, 0.0570, 5000),
    "worms":  (0.0780, 0.0610, 5000),
    "coral":  (0.0620, 0.0620, 5000),
    "uskate": (0.0620, 0.0609, 6000),
}

def _laplacian(z, out):
    # 5-point stencil, periodic boundary, in-place into `out`.
    # Replaces 4 x np.roll (each one allocates + copies the whole grid) with
    # eight in-place slice adds. Same math, ~2x faster, zero per-call allocations.
    np.multiply(z, -4.0, out=out)
    out[1:, :]  += z[:-1, :]
    out[:-1, :] += z[1:, :]
    out[0, :]   += z[-1, :]
    out[-1, :]  += z[0, :]
    out[:, 1:]  += z[:, :-1]
    out[:, :-1] += z[:, 1:]
    out[:, 0]   += z[:, -1]
    out[:, -1]  += z[:, 0]
    return out


class GrayScottTechnique(BaseTechnique):
    name = 'Gray-Scott'
    description = 'Reaction-diffusion as Gray-Scott PDE on a 256x256 grid, integrated for ~3500 steps then upscaled. Two species A and B diffuse and react; the steady-state texture depends entirely on the feed rate f and kill rate k. Five named presets explore the parameter landscape: spots, mazes, worms, coral, and U-skate (moving solitons). Final B concentration mapped through palette LUT -- the patterns emerge from chemistry, not drawing. Good for "reaction diffusion", "gray scott", "texture", "spots", "stripes", "coral", "organic", or any biology-flavored algorithmic motif.'
    kind = "background"
    palette = Palette()
    regime = Enum([('spots', 'Spots'), ('maze', 'Maze'), ('worms', 'Worms'), ('coral', 'Coral'), ('uskate', 'U-skate (solitons)')], default='maze')

    def run(self, canvas):
        s = int(canvas.size)
        seed = int(canvas.seed)
        f, k, n_steps = _PRESETS.get(str(self.regime), _PRESETS["coral"])
        rng = np.random.default_rng(seed)

        N = 384
        A = np.ones((N, N), dtype=np.float32)
        B = np.zeros((N, N), dtype=np.float32)
        # Seed many small noisy patches of B distributed across the canvas so the
        # pattern develops from multiple nucleation sites and fills the frame.
        n_seeds = 14
        for _ in range(n_seeds):
            r0 = int(rng.integers(N // 8, N - N // 8))
            c0 = int(rng.integers(N // 8, N - N // 8))
            rad = int(rng.integers(8, 18))
            for dr in range(-rad, rad + 1):
                for dc in range(-rad, rad + 1):
                    if dr * dr + dc * dc <= rad * rad:
                        rr, cc = (r0 + dr) % N, (c0 + dc) % N
                        A[rr, cc] = 0.5
                        B[rr, cc] = 0.25 + 0.4 * float(rng.random())

        Du, Dv = 0.16, 0.08
        dt = 1.0
        kf = k + f

        # Pre-allocate every per-step buffer so the inner loop allocates
        # nothing. Previously each step churned ~12 fresh (N,N) float32
        # arrays; at N=384, 5000 steps, that was ~30 GB of allocator churn.
        La = np.empty_like(A)
        Lb = np.empty_like(B)
        ABB = np.empty_like(A)
        tmp = np.empty_like(A)

        for _ in range(n_steps):
            _laplacian(A, La)
            _laplacian(B, Lb)
            # ABB = A * B * B
            np.multiply(A, B, out=ABB)
            np.multiply(ABB, B, out=ABB)

            # A += dt * (Du*La - ABB + f*(1-A))  →  compute increment in La
            np.multiply(La, Du, out=La)
            np.subtract(La, ABB, out=La)
            np.multiply(A, f, out=tmp)
            np.subtract(La, tmp, out=La)
            La += f                    # scalar add, no allocation
            np.multiply(La, dt, out=La)
            np.add(A, La, out=A)

            # B += dt * (Dv*Lb + ABB - (k+f)*B)  →  compute increment in Lb
            np.multiply(Lb, Dv, out=Lb)
            np.add(Lb, ABB, out=Lb)
            np.multiply(B, kf, out=tmp)
            np.subtract(Lb, tmp, out=Lb)
            np.multiply(Lb, dt, out=Lb)
            np.add(B, Lb, out=B)

            np.clip(A, 0.0, 1.0, out=A)
            np.clip(B, 0.0, 1.0, out=B)

        # Stretch B's actual range to [0,1] so the palette ramp uses the full LUT.
        bmin = float(B.min())
        bmax = float(B.max())
        if bmax - bmin > 1e-6:
            t_field = (B - bmin) / (bmax - bmin)
        else:
            t_field = np.zeros_like(B)
        t_field = t_field ** 0.85  # gentle gamma to lift mid-range

        LUT = 256
        lut = np.array(
            [art_kit.hex_to_rgb(art_kit.palette_color(k_ / (LUT - 1))) for k_ in range(LUT)],
            dtype=np.uint8,
        )
        idx = np.clip((t_field * (LUT - 1)).astype(np.int32), 0, LUT - 1)
        rgb = lut[idx]
        img = Image.fromarray(rgb, "RGB").resize((s, s), Image.LANCZOS).convert("RGBA")
        canvas.commit(img)
