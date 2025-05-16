
import trimesh
import pyrender
import numpy as np
import cv2
import os

class HoodieRenderer:
    def __init__(self, obj_path):
        self.obj_path = obj_path
        self.mesh = self.load_mesh()

    def load_mesh(self):
        mesh_or_scene = trimesh.load(self.obj_path)
        if isinstance(mesh_or_scene, trimesh.Scene):
            mesh = trimesh.util.concatenate(mesh_or_scene.dump())
        else:
            mesh = mesh_or_scene
        return mesh

    def render_to_image(self, width=150, height=150):
        scene = pyrender.Scene()
        mesh_node = pyrender.Mesh.from_trimesh(self.mesh)
        scene.add(mesh_node)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        scene.add(light)

        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        cam_pose = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0.25],
            [0, 0, 1, 1.5],
            [0, 0, 0, 1],
        ])
        scene.add(camera, pose=cam_pose)

        renderer = pyrender.OffscreenRenderer(width, height)
        flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.SKIP_CULL_FACES
        color, _ = renderer.render(scene, flags=flags)
        renderer.delete()
        return color  # RGBA image

# Quick test
if __name__ == "__main__":
    obj_path = os.path.join(os.path.dirname(__file__), "../assets/hoodie_model_1.obj")
    renderer = HoodieRenderer(obj_path)
    img = renderer.render_to_image(200, 200)
    cv2.imwrite("hoodie_render.png", cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA))
    print("✅ Rendered hoodie image saved as hoodie_render.png")
