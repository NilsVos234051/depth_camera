"""
hoodie_model.py - 3D Hoodie Model Loading and Preparation for Overlay

This module provides functionality to load and prepare a 3D hoodie model for real-time rendering
and overlay on a person. It uses Panda3D for 3D rendering and supports common 3D model formats.

Functions:
- Load 3D model files (OBJ, FBX, GLTF)
- Scale the model based on detected person's size
- Position the model to overlay on a person
- Rotate the model to match person's orientation
- Render the model with proper lighting and transparency

Note: This implementation uses a placeholder model for testing. Replace with actual hoodie model when available.
"""

import sys
import os
import math
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    Point3, Vec3, Vec4, LVector3, 
    TransparencyAttrib, AmbientLight, DirectionalLight,
    Filename, GeomNode, NodePath, 
    PerspectiveLens, TextNode, WindowProperties
)

class HoodieModel:
    """Class for loading, manipulating and rendering a 3D hoodie model."""
    
    def __init__(self, model_path=None):
        """
        Initialize the hoodie model handler.
        
        Args:
            model_path (str, optional): Path to the 3D model file. If None, a placeholder will be used.
        """
        self.model_path = model_path
        self.model = None
        self.scale_factor = 1.0
        self.position = Point3(0, 0, 0)
        self.rotation = Vec3(0, 0, 0)
        
    def load_model(self, render_node):
        """
        Load the 3D model from file or create a placeholder.
        
        Args:
            render_node: The Panda3D render node to attach the model to.
            
        Returns:
            bool: True if model loaded successfully, False otherwise.
        """
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load the actual model file
                self.model = render_node.attachNewNode("hoodie_model")
                model_node = loader.loadModel(self.model_path)
                model_node.reparentTo(self.model)
                print(f"Loaded 3D model from: {self.model_path}")
            else:
                # Create a placeholder model (a simple box representing a hoodie)
                print("No model file provided or file not found. Creating placeholder model.")
                self.model = render_node.attachNewNode("hoodie_placeholder")
                
                # Create a simple box as placeholder
                from panda3d.core import CardMaker
                
                # Create the main body of the hoodie (torso)
                cm = CardMaker("body_front")
                cm.setFrame(-0.5, 0.5, -1.0, 0.5)  # Width, height
                body_front = self.model.attachNewNode(cm.generate())
                body_front.setPos(0, 0.25, 0)
                body_front.setColor(0.2, 0.4, 0.8, 0.8)  # Blue, semi-transparent
                
                cm = CardMaker("body_back")
                cm.setFrame(-0.5, 0.5, -1.0, 0.5)
                body_back = self.model.attachNewNode(cm.generate())
                body_back.setPos(0, -0.25, 0)
                body_back.setH(180)
                body_back.setColor(0.2, 0.4, 0.8, 0.8)
                
                # Create sides
                cm = CardMaker("body_right")
                cm.setFrame(-0.25, 0.25, -1.0, 0.5)
                body_right = self.model.attachNewNode(cm.generate())
                body_right.setPos(0.5, 0, 0)
                body_right.setH(90)
                body_right.setColor(0.2, 0.3, 0.7, 0.8)
                
                cm = CardMaker("body_left")
                cm.setFrame(-0.25, 0.25, -1.0, 0.5)
                body_left = self.model.attachNewNode(cm.generate())
                body_left.setPos(-0.5, 0, 0)
                body_left.setH(-90)
                body_left.setColor(0.2, 0.3, 0.7, 0.8)
                
                # Create hood
                cm = CardMaker("hood")
                cm.setFrame(-0.3, 0.3, 0, 0.4)
                hood = self.model.attachNewNode(cm.generate())
                hood.setPos(0, 0, 0.5)
                hood.setP(-30)  # Tilt the hood
                hood.setColor(0.2, 0.4, 0.8, 0.8)
                
                # Create sleeves
                cm = CardMaker("sleeve_right")
                cm.setFrame(-0.2, 0.2, -0.6, 0)
                sleeve_right = self.model.attachNewNode(cm.generate())
                sleeve_right.setPos(0.5, 0, 0.3)
                sleeve_right.setR(20)
                sleeve_right.setColor(0.2, 0.3, 0.7, 0.8)
                
                cm = CardMaker("sleeve_left")
                cm.setFrame(-0.2, 0.2, -0.6, 0)
                sleeve_left = self.model.attachNewNode(cm.generate())
                sleeve_left.setPos(-0.5, 0, 0.3)
                sleeve_left.setR(-20)
                sleeve_left.setColor(0.2, 0.3, 0.7, 0.8)
                
                # Enable transparency
                self.model.setTransparency(TransparencyAttrib.MAlpha)
            
            # Set initial transformations
            self.apply_transformations()
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def set_scale(self, scale_factor):
        """
        Set the scale of the model.
        
        Args:
            scale_factor (float): Scale factor to apply to the model.
        """
        self.scale_factor = scale_factor
        if self.model:
            self.apply_transformations()
    
    def set_position(self, x, y, z):
        """
        Set the position of the model.
        
        Args:
            x (float): X coordinate.
            y (float): Y coordinate.
            z (float): Z coordinate.
        """
        self.position = Point3(x, y, z)
        if self.model:
            self.apply_transformations()
    
    def set_rotation(self, h, p, r):
        """
        Set the rotation of the model in Euler angles (heading, pitch, roll).
        
        Args:
            h (float): Heading angle in degrees.
            p (float): Pitch angle in degrees.
            r (float): Roll angle in degrees.
        """
        self.rotation = Vec3(h, p, r)
        if self.model:
            self.apply_transformations()
    
    def apply_transformations(self):
        """Apply all transformations (scale, position, rotation) to the model."""
        if self.model:
            self.model.setScale(self.scale_factor)
            self.model.setPos(self.position)
            self.model.setHpr(self.rotation)
    
    def adjust_for_person(self, person_height, person_position, person_orientation):
        """
        Adjust the model to fit a detected person.
        
        Args:
            person_height (float): Height of the person in scene units.
            person_position (tuple): (x, y, z) position of the person.
            person_orientation (float): Orientation of the person in degrees.
        """
        # Scale model based on person's height (assuming model's default height is 2 units)
        self.set_scale(person_height / 2.0)
        
        # Position model at person's location, adjusting for the center of the hoodie
        x, y, z = person_position
        self.set_position(x, y, z + person_height * 0.25)  # Position hoodie at upper torso
        
        # Rotate model to match person's orientation
        self.set_rotation(person_orientation, 0, 0)
    
    def show(self):
        """Make the model visible."""
        if self.model:
            self.model.show()
    
    def hide(self):
        """Hide the model."""
        if self.model:
            self.model.hide()
    
    def cleanup(self):
        """Remove the model from the scene."""
        if self.model:
            self.model.removeNode()
            self.model = None


class HoodieModelViewer(ShowBase):
    """Test application for viewing and manipulating the 3D hoodie model."""
    
    def __init__(self, model_path=None):
        """
        Initialize the viewer application.
        
        Args:
            model_path (str, optional): Path to the 3D model file.
        """
        ShowBase.__init__(self)
        
        # Set up the window
        props = WindowProperties()
        props.setTitle("3D Hoodie Model Viewer")
        props.setSize(1024, 768)
        self.win.requestProperties(props)
        
        # Set up the camera
        self.cam.setPos(0, -5, 0)
        self.cam.lookAt(0, 0, 0)
        
        # Set up lighting
        self.setup_lighting()
        
        # Create a simple ground plane for reference
        self.create_ground()
        
        # Load the hoodie model
        self.hoodie = HoodieModel(model_path)
        self.hoodie.load_model(self.render)
        
        # Set up key controls
        self.setup_controls()
        
        # Add a task to rotate the model
        self.taskMgr.add(self.spin_model_task, "SpinModelTask")
        
        # Add text instructions
        self.add_instructions()
    
    def setup_lighting(self):
        """Set up basic lighting for the scene."""
        # Ambient light
        ambient_light = AmbientLight("ambient_light")
        ambient_light.setColor(Vec4(0.3, 0.3, 0.3, 1))
        ambient_node = self.render.attachNewNode(ambient_light)
        self.render.setLight(ambient_node)
        
        # Directional light (key light)
        key_light = DirectionalLight("key_light")
        key_light.setColor(Vec4(0.8, 0.8, 0.8, 1))
        key_node = self.render.attachNewNode(key_light)
        key_node.setHpr(45, -45, 0)
        self.render.setLight(key_node)
        
        # Fill light
        fill_light = DirectionalLight("fill_light")
        fill_light.setColor(Vec4(0.4, 0.4, 0.5, 1))
        fill_node = self.render.attachNewNode(fill_light)
        fill_node.setHpr(-45, -45, 0)
        self.render.setLight(fill_node)
    
    def create_ground(self):
        """Create a simple ground plane for reference."""
        from panda3d.core import CardMaker
        cm = CardMaker("ground")
        cm.setFrame(-10, 10, -10, 10)
        ground = self.render.attachNewNode(cm.generate())
        ground.setP(-90)  # Rotate to be horizontal
        ground.setPos(0, 0, -2)
        ground.setColor(0.7, 0.7, 0.7, 1)
    
    def setup_controls(self):
        """Set up keyboard controls for manipulating the model."""
        # Scale controls
        self.accept("=", self.scale_up)
        self.accept("-", self.scale_down)
        
        # Position controls
        self.accept("arrow_up", self.move_forward)
        self.accept("arrow_down", self.move_backward)
        self.accept("arrow_left", self.move_left)
        self.accept("arrow_right", self.move_right)
        self.accept("page_up", self.move_up)
        self.accept("page_down", self.move_down)
        
        # Rotation controls
        self.accept("a", self.rotate_left)
        self.accept("d", self.rotate_right)
        self.accept("w", self.rotate_up)
        self.accept("s", self.rotate_down)
        self.accept("q", self.rotate_clockwise)
        self.accept("e", self.rotate_counterclockwise)
        
        # Reset
        self.accept("r", self.reset_model)
        
        # Toggle auto-rotation
        self.auto_rotate = True
        self.accept("space", self.toggle_auto_rotate)
        
        # Exit
        self.accept("escape", sys.exit)
    
    def add_instructions(self):
        """Add text instructions to the screen."""
        instructions = [
            "3D Hoodie Model Viewer",
            "",
            "Controls:",
            "= / - : Scale up/down",
            "Arrow keys: Move forward/backward/left/right",
            "Page Up/Down: Move up/down",
            "A / D: Rotate left/right",
            "W / S: Rotate up/down",
            "Q / E: Rotate clockwise/counterclockwise",
            "R: Reset model",
            "Space: Toggle auto-rotation",
            "Esc: Exit"
        ]
        
        text_node = TextNode('instructions')
        text_node.setText('\n'.join(instructions))
        text_node.setAlign(TextNode.ALeft)
        
        text_np = aspect2d.attachNewNode(text_node)
        text_np.setScale(0.05)
        text_np.setPos(-1.5, 0, 0.8)
    
    def scale_up(self):
        """Increase the model scale."""
        self.hoodie.set_scale(self.hoodie.scale_factor * 1.1)
    
    def scale_down(self):
        """Decrease the model scale."""
        self.hoodie.set_scale(self.hoodie.scale_factor * 0.9)
    
    def move_forward(self):
        """Move the model forward."""
        pos = self.hoodie.position
        self.hoodie.set_position(pos.x, pos.y + 0.1, pos.z)
    
    def move_backward(self):
        """Move the model backward."""
        pos = self.hoodie.position
        self.hoodie.set_position(pos.x, pos.y - 0.1, pos.z)
    
    def move_left(self):
        """Move the model left."""
        pos = self.hoodie.position
        self.hoodie.set_position(pos.x - 0.1, pos.y, pos.z)
    
    def move_right(self):
        """Move the model right."""
        pos = self.hoodie.position
        self.hoodie.set_position(pos.x + 0.1, pos.y, pos.z)
    
    def move_up(self):
        """Move the model up."""
        pos = self.hoodie.position
        self.hoodie.set_position(pos.x, pos.y, pos.z + 0.1)
    
    def move_down(self):
        """Move the model down."""
        pos = self.hoodie.position
        self.hoodie.set_position(pos.x, pos.y, pos.z - 0.1)
    
    def rotate_left(self):
        """Rotate the model left."""
        rot = self.hoodie.rotation
        self.hoodie.set_rotation(rot.x + 5, rot.y, rot.z)
    
    def rotate_right(self):
        """Rotate the model right."""
        rot = self.hoodie.rotation
        self.hoodie.set_rotation(rot.x - 5, rot.y, rot.z)
    
    def rotate_up(self):
        """Rotate the model up."""
        rot = self.hoodie.rotation
        self.hoodie.set_rotation(rot.x, rot.y + 5, rot.z)
    
    def rotate_down(self):
        """Rotate the model down."""
        rot = self.hoodie.rotation
        self.hoodie.set_rotation(rot.x, rot.y - 5, rot.z)
    
    def rotate_clockwise(self):
        """Rotate the model clockwise."""
        rot = self.hoodie.rotation
        self.hoodie.set_rotation(rot.x, rot.y, rot.z + 5)
    
    def rotate_counterclockwise(self):
        """Rotate the model counterclockwise."""
        rot = self.hoodie.rotation
        self.hoodie.set_rotation(rot.x, rot.y, rot.z - 5)
    
    def reset_model(self):
        """Reset the model to its default position, rotation, and scale."""
        self.hoodie.set_scale(1.0)
        self.hoodie.set_position(0, 0, 0)
        self.hoodie.set_rotation(0, 0, 0)
    
    def toggle_auto_rotate(self):
        """Toggle auto-rotation of the model."""
        self.auto_rotate = not self.auto_rotate
    
    def spin_model_task(self, task):
        """Task to automatically rotate the model."""
        if self.auto_rotate:
            rot = self.hoodie.rotation
            self.hoodie.set_rotation(rot.x + 0.5, rot.y, rot.z)
        return Task.cont


def test_person_overlay():
    """
    Test the hoodie model's ability to overlay on a simulated person.
    
    This function creates a simple test scenario where a hoodie model is
    positioned and scaled to overlay on a simulated person.
    """
    app = ShowBase()
    
    # Set up the camera
    app.cam.setPos(0, -10, 2)
    app.cam.lookAt(0, 0, 0)
    
    # Set up lighting
    ambient_light = AmbientLight("ambient_light")
    ambient_light.setColor(Vec4(0.3, 0.3, 0.3, 1))
    ambient_node = app.render.attachNewNode(ambient_light)
    app.render.setLight(ambient_node)
    
    key_light = DirectionalLight("key_light")
    key_light.setColor(Vec4(0.8, 0.8, 0.8, 1))
    key_node = app.render.attachNewNode(key_light)
    key_node.setHpr(45, -45, 0)
    app.render.setLight(key_node)
    
    # Create a simple person representation (just a cylinder and sphere)
    from panda3d.core import CardMaker
    
    # Body (cylinder)
    cm = CardMaker("person_body")
    cm.setFrame(-0.4, 0.4, -1.5, 0)
    body = app.render.attachNewNode(cm.generate())
    body.setColor(0.8, 0.7, 0.6, 1)  # Skin tone
    
    # Head (sphere approximation)
    cm = CardMaker("person_head")
    cm.setFrame(-0.3, 0.3, -0.3, 0.3)
    head = app.render.attachNewNode(cm.generate())
    head.setPos(0, 0, 0.3)
    head.setColor(0.8, 0.7, 0.6, 1)  # Skin tone
    
    # Create and position the hoodie model
    hoodie = HoodieModel()
    hoodie.load_model(app.render)
    
    # Simulate person detection data
    person_height = 2.0
    person_position = (0, 0, 0)
    person_orientation = 0
    
    # Adjust hoodie to fit the person
    hoodie.adjust_for_person(person_height, person_position, person_orientation)
    
    # Add text explanation
    text_node = TextNode('overlay_info')
    text_node.setText("Hoodie Model Overlay Test\n\n"
                     "The blue semi-transparent hoodie model\n"
                     "is positioned to overlay on the simulated person.")
    text_node.setAlign(TextNode.ALeft)
    
    text_np = aspect2d.attachNewNode(text_node)
    text_np.setScale(0.05)
    text_np.setPos(-1.5, 0, 0.8)
    
    # Run the application
    app.run()


def main():
    """Main function to run the hoodie model viewer."""
    import argparse
    
    parser = argparse.ArgumentParser(description="3D Hoodie Model Viewer")
    parser.add_argument("--model", type=str, help="Path to the 3D model file")
    parser.add_argument("--test-overlay", action="store_true", help="Run the person overlay test")
    args = parser.parse_args()
    
    if args.test_overlay:
        test_person_overlay()
    else:
        app = HoodieModelViewer(args.model)
        app.run()


if __name__ == "__main__":
    main()