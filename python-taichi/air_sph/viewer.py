"""Real-time Taichi GGUI viewer for the particle field, colored by
pressure perturbation."""
import taichi as ti


class Viewer:
    def __init__(self, window_name="Air Acoustics", res=(960, 720),
                 camera_pos=(1.2, 0.9, 1.2), show_window=True):
        # show_window=False renders off-screen, for save_image() screenshots.
        self.show_window = show_window
        self.window = ti.ui.Window(window_name, res, vsync=True, show_window=show_window)
        self.canvas = self.window.get_canvas()
        self.camera = ti.ui.Camera()
        self.camera.position(*camera_pos)
        self.camera.lookat(0.0, 0.0, 0.0)
        self.camera.up(0.0, 1.0, 0.0)

    @property
    def running(self):
        return self.window.running

    def render(self, pos_field, radius, colors_field, radii_field=None):
        scene = self.window.get_scene()
        if self.show_window:
            self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)
        scene.set_camera(self.camera)
        scene.ambient_light((0.6, 0.6, 0.6))
        scene.point_light(pos=(2, 2, 2), color=(1, 1, 1))
        # per_vertex_radius, when given, takes priority over the flat `radius`
        # (which still sets the size for any caller that doesn't pass one).
        scene.particles(pos_field, radius=radius, per_vertex_color=colors_field,
                         per_vertex_radius=radii_field)
        self.canvas.scene(scene)
        if self.show_window:
            self.window.show()
