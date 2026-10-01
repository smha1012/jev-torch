"""Generate the README artwork (light + dark variants). Run: python docs/assets/build.py"""

from pathlib import Path

OUT = Path(__file__).parent
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

THEMES = {
    "dark": dict(bg1="#0d1117", bg2="#161b2e", card="#161b22", card2="#1c2230", line="#30363d",
                 text="#e6edf3", muted="#8b949e", faint="#3d444d", chip="#21262d"),
    "light": dict(bg1="#ffffff", bg2="#f3f4fb", card="#ffffff", card2="#f6f8fa", line="#d0d7de",
                  text="#1f2328", muted="#59636e", faint="#d8dee4", chip="#eef1f5"),
}
ACCENT = ("#ff7a45", "#ee4c2c")  # PyTorch-ish orange gradient
COOL = "#8b5cf6"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def defs(t):
    return f"""<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
    <stop offset="0" stop-color="{t['bg1']}"/><stop offset="1" stop-color="{t['bg2']}"/>
  </linearGradient>
  <linearGradient id="acc" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0" stop-color="{ACCENT[0]}"/><stop offset="1" stop-color="{ACCENT[1]}"/>
  </linearGradient>
  <radialGradient id="glow" cx="0.82" cy="0.25" r="0.6">
    <stop offset="0" stop-color="{ACCENT[1]}" stop-opacity="0.18"/><stop offset="1" stop-color="{ACCENT[1]}" stop-opacity="0"/>
  </radialGradient>
</defs>"""


def bars(x, y, w, rows, t, label_w=150, row_h=34, bar_h=14):
    """Horizontal probability bars: rows = [(label, p, highlight)]."""
    out = []
    for i, (label, p, hi) in enumerate(rows):
        yy = y + i * row_h
        bw = max(3, (w - label_w - 56) * p)
        fill = "url(#acc)" if hi else t["faint"]
        weight = 600 if hi else 400
        out.append(f'<text x="{x}" y="{yy + 11}" font-family="{MONO}" font-size="14" fill="{t["text"] if hi else t["muted"]}" font-weight="{weight}">{esc(label)}</text>')
        out.append(f'<rect x="{x + label_w}" y="{yy}" width="{w - label_w - 56}" height="{bar_h}" rx="7" fill="{t["chip"]}"/>')
        out.append(f'<rect x="{x + label_w}" y="{yy}" width="{bw:.1f}" height="{bar_h}" rx="7" fill="{fill}"/>')
        out.append(f'<text x="{x + w - 44}" y="{yy + 12}" font-family="{MONO}" font-size="14" fill="{t["text"] if hi else t["muted"]}" font-weight="{weight}">{p:.2f}</text>')
    return "\n".join(out)


def banner(t):
    W, H = 1200, 400
    chips = ["one forward pass", "calibrated probabilities", "Qwen3.5 · 0.8B → 27B"]
    chip_svg, cx = [], 64
    for c in chips:
        cw = 16 + len(c) * 8.4
        chip_svg.append(f'<rect x="{cx}" y="252" width="{cw:.0f}" height="34" rx="17" fill="{t["chip"]}" stroke="{t["line"]}"/>')
        chip_svg.append(f'<text x="{cx + cw / 2:.0f}" y="274" text-anchor="middle" font-family="{FONT}" font-size="15" fill="{t["text"]}">{esc(c)}</text>')
        cx += cw + 12
    card_x, card_y, card_w = 690, 64, 446
    rows = [("infrastructure", 0.48, True), ("producer_change", 0.48, True),
            ("expected_variation", 0.03, False), ("schema_drift", 0.01, False)]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
{defs(t)}
<rect width="{W}" height="{H}" rx="24" fill="url(#bg)"/>
<rect width="{W}" height="{H}" rx="24" fill="url(#glow)"/>
<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="24" fill="none" stroke="{t['line']}"/>

<text x="64" y="132" font-family="{FONT}" font-size="76" font-weight="800" fill="{t['text']}" letter-spacing="-2">jev<tspan fill="url(#acc)">-</tspan>torch</text>
<text x="66" y="184" font-family="{FONT}" font-size="24" fill="{t['text']}">Train JEV-style decision models in PyTorch.</text>
<text x="66" y="218" font-family="{FONT}" font-size="18" fill="{t['muted']}">State + question + options in, a calibrated probability per option out.</text>
{chr(10).join(chip_svg)}
<text x="66" y="346" font-family="{FONT}" font-size="14" fill="{t['muted']}">Unofficial re-implementation · Apache-2.0</text>

<rect x="{card_x}" y="{card_y}" width="{card_w}" height="272" rx="18" fill="{t['card']}" stroke="{t['line']}"/>
<circle cx="{card_x + 26}" cy="{card_y + 28}" r="5" fill="#ff5f57"/><circle cx="{card_x + 44}" cy="{card_y + 28}" r="5" fill="#febc2e"/><circle cx="{card_x + 62}" cy="{card_y + 28}" r="5" fill="#28c840"/>
<text x="{card_x + card_w - 24}" y="{card_y + 33}" text-anchor="end" font-family="{MONO}" font-size="13" fill="{t['muted']}">kind: choice</text>
<text x="{card_x + 24}" y="{card_y + 72}" font-family="{FONT}" font-size="15" fill="{t['muted']}">The nightly feed is byte-identical to yesterday</text>
<text x="{card_x + 24}" y="{card_y + 93}" font-family="{FONT}" font-size="15" fill="{t['muted']}">and arrived 3 hours late.</text>
<text x="{card_x + 24}" y="{card_y + 124}" font-family="{FONT}" font-size="17" font-weight="600" fill="{t['text']}">Root cause?</text>
{bars(card_x + 24, card_y + 146, card_w - 48, rows, t, label_w=168, row_h=30, bar_h=12)}
</svg>
"""


def box(x, y, w, h, title, sub, t, step):
    return f"""<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" fill="{t['card']}" stroke="{t['line']}"/>
<circle cx="{x + 26}" cy="{y + 28}" r="12" fill="url(#acc)"/>
<text x="{x + 26}" y="{y + 33}" text-anchor="middle" font-family="{FONT}" font-size="13" font-weight="700" fill="#ffffff">{step}</text>
<text x="{x + 46}" y="{y + 33}" font-family="{FONT}" font-size="17" font-weight="700" fill="{t['text']}">{esc(title)}</text>
<text x="{x + 20}" y="{y + h - 18}" font-family="{FONT}" font-size="13" fill="{t['muted']}">{esc(sub)}</text>"""


def arrow(x1, x2, y, t):
    return (f'<line x1="{x1}" y1="{y}" x2="{x2 - 9}" y2="{y}" stroke="{t["muted"]}" stroke-width="2"/>'
            f'<path d="M{x2 - 10},{y - 6} L{x2},{y} L{x2 - 10},{y + 6} z" fill="{t["muted"]}"/>')


def how(t):
    W, H = 1200, 320
    y, h = 36, 248
    xs = [20, 320, 610, 880]
    ws = [270, 260, 240, 300]
    prompt_lines = ["[kind] choice", "[state] The nightly feed is…", "[question] Root cause?",
                    "[options]", "A. producer_change  B. …", "[decision]:"]
    prompt = "\n".join(
        f'<text x="{xs[0] + 20}" y="{y + 66 + i * 20}" font-family="{MONO}" font-size="12.5" '
        f'fill="{t["text"] if i in (0, 2, 5) else t["muted"]}">{esc(s)}</text>' for i, s in enumerate(prompt_lines))
    # backbone: stacked layers
    layers = []
    for i in range(6):
        ly = y + 60 + i * 18
        kind_lin = i % 4 != 3
        layers.append(f'<rect x="{xs[1] + 24}" y="{ly}" width="{ws[1] - 48}" height="13" rx="4" fill="{t["chip"]}" stroke="{t["line"]}"/>')
        layers.append(f'<rect x="{xs[1] + ws[1] - 60}" y="{ly + 2}" width="30" height="9" rx="3" fill="{COOL}" opacity="0.85"/>')
        layers.append(f'<text x="{xs[1] + 32}" y="{ly + 10.5}" font-family="{MONO}" font-size="9.5" fill="{t["muted"]}">{"linear-attn" if kind_lin else "full-attn"}</text>')
    layers.append(f'<text x="{xs[1] + ws[1] - 45}" y="{y + 186}" text-anchor="middle" font-family="{FONT}" font-size="11" fill="{COOL}">LoRA</text>')
    # head: 24 slots in 3 groups, choice A-D active
    slots, sx0, sy = [], xs[2] + 22, y + 72
    groups = [("noul", 2, False), ("score", 6, False), ("choice", 16, True)]
    gy = sy
    for name, n, active in groups:
        slots.append(f'<text x="{sx0}" y="{gy - 6}" font-family="{MONO}" font-size="11" fill="{t["muted"]}">{name}</text>')
        for j in range(n):
            cxp = sx0 + (j % 8) * 24
            cyp = gy + (j // 8) * 22
            on = active and j < 4
            slots.append(f'<rect x="{cxp}" y="{cyp}" width="18" height="16" rx="4" fill="{"url(#acc)" if on else t["chip"]}" stroke="{t["line"]}"/>')
        gy += 22 * ((n + 7) // 8) + 18
    rows = [("A", 0.48, True), ("B", 0.01, False), ("C", 0.48, True), ("D", 0.03, False)]
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
{defs(t)}
<rect width="{W}" height="{H}" rx="20" fill="{t['card2']}" stroke="{t['line']}"/>
{box(xs[0], y, ws[0], h, "Prompt", "options listed in the input", t, 1)}
{prompt}
{arrow(xs[0] + ws[0] + 6, xs[1] - 6, y + h / 2, t)}
{box(xs[1], y, ws[1], h, "Qwen3.5 + LoRA", "frozen bf16 backbone · one pass", t, 2)}
{chr(10).join(layers)}
{arrow(xs[1] + ws[1] + 6, xs[2] - 6, y + h / 2, t)}
{box(xs[2], y, ws[2], h, "24-slot head", "last token → active slots", t, 3)}
{chr(10).join(slots)}
{arrow(xs[2] + ws[2] + 6, xs[3] - 6, y + h / 2, t)}
{box(xs[3], y, ws[3], h, "Calibrated output", "softmax(logits ÷ T_kind)", t, 4)}
{bars(xs[3] + 22, y + 66, ws[3] - 44, rows, t, label_w=26, row_h=30, bar_h=12)}
</svg>
"""


if __name__ == "__main__":
    for name, t in THEMES.items():
        (OUT / f"banner-{name}.svg").write_text(banner(t))
        (OUT / f"how-it-works-{name}.svg").write_text(how(t))
    print("wrote", sorted(p.name for p in OUT.glob("*.svg")))
