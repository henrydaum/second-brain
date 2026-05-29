from plugins.BaseSkill import BaseSkill, Slider, Palette

import random
from PIL import Image, ImageDraw, ImageFilter

try:
    art_kit
except NameError:
    art_kit = None


class WaveFunctionCollapseSkill(BaseSkill):
    name = 'Wave Function Collapse'
    description = 'Wave function collapse: a constraint-solver lays down edge-matched pipe tiles cell by cell, always collapsing the lowest-entropy cell first and propagating the consequences, so connections flow unbroken across the whole grid into a coherent circuit network. Good for "wave function collapse", "WFC", "procedural tiles", "pipes", "circuit", "maze", or a constraint-based generative pattern.'
    kind = "background"

    palette = Palette()
    grid    = Slider(10, 28, default=18, step=1)

    def run(self, canvas):
        s = int(canvas.size)
        G = int(self.grid)
        rng = random.Random(int(canvas.seed))

        # 16 tiles: bit per edge connection. N=1, E=2, S=4, W=8.
        N, E, S, W = 1, 2, 4, 8
        tiles = list(range(16))

        def conn(t, b):
            return bool(t & b)

        # Favor 2-3 connection tiles so pipes dominate over blanks/stubs.
        pop = [sum(1 for bit in (N, E, S, W) if t & bit) for t in tiles]
        weight = {t: {0: 0.25, 1: 0.6, 2: 1.7, 3: 1.3, 4: 0.9}[pop[t]] for t in tiles}

        def compat_h(a, b):   # a left, b right
            return conn(a, E) == conn(b, W)

        def compat_v(a, b):   # a top, b bottom
            return conn(a, S) == conn(b, N)

        domains = [set(tiles) for _ in range(G * G)]

        def neighbors(i):
            x, y = i % G, i // G
            out = []
            if x > 0:
                out.append((i - 1, "h_left"))
            if x < G - 1:
                out.append((i + 1, "h_right"))
            if y > 0:
                out.append((i - G, "v_up"))
            if y < G - 1:
                out.append((i + G, "v_down"))
            return out

        def allowed(rel, here, there):
            # `there` tile is allowed if some `here` tile is compatible.
            for a in here:
                if rel == "h_right" and compat_h(a, there):
                    return True
                if rel == "h_left" and compat_h(there, a):
                    return True
                if rel == "v_down" and compat_v(a, there):
                    return True
                if rel == "v_up" and compat_v(there, a):
                    return True
            return False

        def propagate(start):
            stack = [start]
            while stack:
                i = stack.pop()
                for j, rel in neighbors(i):
                    keep = {b for b in domains[j] if allowed(rel, domains[i], b)}
                    if not keep:
                        keep = set(domains[j])      # avoid a hard contradiction
                    if keep != domains[j]:
                        domains[j] = keep
                        stack.append(j)

        for _ in range(G * G):
            # Lowest-entropy uncollapsed cell.
            best, best_n = -1, 99
            for i in range(G * G):
                n = len(domains[i])
                if n > 1 and n < best_n:
                    best_n, best = n, i
            if best < 0:
                break
            opts = list(domains[best])
            ws = [weight[t] for t in opts]
            chosen = rng.choices(opts, weights=ws, k=1)[0]
            domains[best] = {chosen}
            propagate(best)

        img = Image.new("RGBA", (s, s), canvas.palette.background)
        draw = ImageDraw.Draw(img, "RGBA")
        cw = s / G
        lw = max(2, int(cw * 0.22))
        node = cw * 0.16
        for i in range(G * G):
            t = list(domains[i])[0]
            gx, gy = i % G, i // G
            cxp = (gx + 0.5) * cw
            cyp = (gy + 0.5) * cw
            ramp = 0.32 + 0.55 * ((gx + gy) / (2.0 * (G - 1)))
            col = art_kit.palette_color(ramp)
            ends = []
            if conn(t, N):
                ends.append((cxp, gy * cw))
            if conn(t, S):
                ends.append((cxp, (gy + 1) * cw))
            if conn(t, W):
                ends.append((gx * cw, cyp))
            if conn(t, E):
                ends.append(((gx + 1) * cw, cyp))
            for ex, ey in ends:
                draw.line((cxp, cyp, ex, ey), fill=col, width=lw)
            if ends:
                draw.ellipse((cxp - node, cyp - node, cxp + node, cyp + node), fill=col)

        glow = img.filter(ImageFilter.GaussianBlur(radius=s * 0.003))
        canvas.commit(Image.alpha_composite(glow, img))
