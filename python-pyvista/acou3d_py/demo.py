"""Demo: design a box for the example driver and visualise it.

    python -m acou3d_py.demo                       # interactive, vented
    python -m acou3d_py.demo --box sealed
    python -m acou3d_py.demo --out renders/        # off-screen PNGs
    python -m acou3d_py.demo --bem                 # BEM radiation (first run solves, then cached)
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pyvista as pv

from . import ts
from .directivity import balloon, balloon_from_levels
from .geometry import Enclosure, build_mesh

REGION_COLORS = ["#b08a5a", "#222222", "#3a6ea5"]  # cabinet, driver, port


def design(kind: str, fb: float) -> Enclosure:
    base = dict(width=0.24, height=0.40, depth=0.30,
                driver_radius=ts.EXAMPLE_DRIVER.radius)
    if kind == "sealed":
        return Enclosure(**base)
    probe = Enclosure(**base)
    port_r = 0.025
    return Enclosure(**base, port_radius=port_r,
                     port_length=ts.port_length(probe.net_volume, fb, port_r))


def response(enc: Enclosure, fb: float, f: np.ndarray) -> np.ndarray:
    drv = ts.EXAMPLE_DRIVER
    if enc.port_radius is None:
        h = ts.sealed_response(drv, enc.net_volume, f)
    else:
        h = ts.vented_response(drv, enc.net_volume, fb, f)
    return ts.spl(drv, h)


def plot_response(enc, fb, path: Path):
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = np.geomspace(10, 1000, 400)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogx(f, response(enc, fb, f))
    ax.set(xlabel="Frequency (Hz)", ylabel="SPL @ 1 W / 1 m (dB)",
           title=f"Net volume {enc.net_volume * 1e3:.1f} L")
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(path, dpi=120, bbox_inches="tight")


def response_chart(enc: Enclosure, fb: float, sw=None) -> tuple[pv.Chart2D, object]:
    """SPL chart plus a vertical marker line for the balloon frequency.

    With a BEM sweep ``sw``, also plots the response including the cabinet's
    baffle step (Thiele-Small assumes half-space radiation).
    """
    f = np.geomspace(10, 20000, 400)
    level = response(enc, fb, f)
    chart = pv.Chart2D(x_label="Frequency (Hz)", y_label="SPL 1 W / 1 m (dB)")
    chart.x_axis.log_scale = True
    chart.line(f, level, color="#3a6ea5", width=2, label="Thiele-Small (half space)")
    lows = [level.min()]
    if sw is not None:
        with_baffle = response(enc, fb, sw.freqs) + 20 * np.log10(sw.baffle_gain)
        chart.line(sw.freqs, with_baffle, color="#e08a2c", width=2,
                   label="incl. baffle step (BEM)")
        chart.scatter(sw.freqs, with_baffle, color="#e08a2c", size=6)
        lows.append(with_baffle.min())
    lo, hi = np.floor(min(lows) / 10) * 10, np.ceil(level.max() / 10) * 10 + 5
    chart.y_range = [lo, hi]
    marker = chart.line([1000, 1000], [lo, hi], color="#d9534f", width=1)
    return chart, marker


def scene(enc: Enclosure, fb: float, freq: float, off_screen: bool,
          sw=None) -> pv.Plotter:
    pl = pv.Plotter(shape=(1, 2), window_size=(1600, 700), off_screen=off_screen)
    chart, marker = response_chart(enc, fb, sw)
    pl.subplot(0, 1)
    pl.add_chart(chart)
    pl.subplot(0, 0)

    dx, dy = enc._driver_xy()
    origin = np.array([dx, dy, enc.depth])
    scale = 0.35
    balloon_bar = {"title": "Directivity (dB)", "vertical": True,
                   "position_x": 0.85, "position_y": 0.3}

    if sw is None:
        pl.add_mesh(build_mesh(enc), scalars="region", cmap=REGION_COLORS, clim=(0, 2),
                    show_edges=False, show_scalar_bar=False)
        f_range = (200, 20000)

        def update(f):
            b = balloon(f, ts.EXAMPLE_DRIVER.radius).scale(scale).translate(origin)
            pl.add_mesh(b, name="balloon", scalars="dB", cmap="viridis",
                        clim=(-40, 0), opacity=0.55, scalar_bar_args=balloon_bar)
            pl.add_text(f"Piston directivity @ {f:.0f} Hz", name="label", font_size=10)
            marker.update([f, f], marker.y)
    else:
        surface = sw.surface.copy()
        f_range = (sw.freqs[0], sw.freqs[-1])
        on_axis = np.argmin(sw.theta[:, 0])  # theta = 0 row

        def update(f):
            i = sw.nearest(f)
            f = sw.freqs[i]
            p = np.abs(surface.point_data[f"p_{i}"])
            surface.point_data["surface dB"] = 20 * np.log10(p / p.max())
            pl.add_mesh(surface, name="cabinet", scalars="surface dB", cmap="inferno",
                        clim=(-30, 0),
                        scalar_bar_args={"title": "Surface (dB)",
                                         "vertical": True, "position_x": 0.04,
                                         "position_y": 0.3})
            pat = np.abs(sw.patterns[i])
            db = 20 * np.log10(pat / pat[on_axis].mean())
            b = balloon_from_levels(sw.theta, sw.phi, db).scale(scale).translate(origin)
            pl.add_mesh(b, name="balloon", scalars="dB", cmap="viridis",
                        clim=(-40, 0), opacity=0.45, scalar_bar_args=balloon_bar)
            pl.add_text(f"BEM @ {f:.0f} Hz  (driver only, port rigid)", name="label",
                        font_size=10)
            marker.update([f, f], marker.y)

    if off_screen:
        update(freq)
    else:
        pl.add_slider_widget(lambda v: update(10**v), np.log10(f_range),
                             value=np.log10(np.clip(freq, *f_range)), title="log10 f (Hz)",
                             pointa=(0.1, 0.08), pointb=(0.9, 0.08))
    pl.camera_position = [(1.3, 0.9, 1.4), (enc.width / 2, enc.height / 2, enc.depth / 2),
                          (0, 1, 0)]
    return pl


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--box", choices=["sealed", "vented"], default="vented")
    ap.add_argument("--fb", type=float, default=42.0, help="port tuning, Hz")
    ap.add_argument("--freq", type=float, default=2000.0, help="balloon frequency, Hz")
    ap.add_argument("--out", type=Path, help="write PNGs here instead of opening windows")
    ap.add_argument("--bem", action="store_true", help="solve exterior radiation with bempp-cl")
    ap.add_argument("--bem-fmax", type=float, default=2000.0,
                    help="highest BEM frequency; sets mesh size to 6 elements per wavelength")
    args = ap.parse_args()

    enc = design(args.box, args.fb)
    drv = ts.EXAMPLE_DRIVER
    print(f"net volume  {enc.net_volume * 1e3:.1f} L")
    print(f"sensitivity {drv.spl_1w1m():.1f} dB @ 1 W / 1 m")
    if enc.port_radius is None:
        fc, qtc = ts.sealed_alignment(drv, enc.net_volume)
        print(f"sealed      fc = {fc:.1f} Hz, Qtc = {qtc:.2f}")
    else:
        print(f"vented      fb = {args.fb:.1f} Hz, port {enc.port_radius * 2e3:.0f} mm "
              f"x {enc.port_length * 1e3:.0f} mm")

    sw = None
    if args.bem:
        from . import bem
        mesh_size = ts.C_AIR / (6 * args.bem_fmax)
        sw = bem.sweep(enc, bem.third_octaves(100, args.bem_fmax), mesh_size)

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        plot_response(enc, args.fb, args.out / "response.png")
        scene(enc, args.fb, args.freq, off_screen=True, sw=sw).screenshot(args.out / "enclosure.png")
        print(f"wrote {args.out / 'response.png'}, {args.out / 'enclosure.png'}")
    else:
        scene(enc, args.fb, args.freq, off_screen=False, sw=sw).show()


if __name__ == "__main__":
    main()
