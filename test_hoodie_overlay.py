"""
test_hoodie_overlay.py - Test script for demonstrating hoodie model overlay

This script shows how to use the HoodieModel class to load a 3D hoodie model
and overlay it on a simulated person. It provides a simple example of how
the hoodie_model.py module can be integrated into a larger application.
"""

import sys
import os
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import Point3, Vec3, Vec4, TextNode, AmbientLight, DirectionalLight

# Import the HoodieModel class
from hoodie_model import HoodieModel

class HoodieOverlayDemo(ShowBase):
    """Demo application showing hoodie overlay on a simulated person."""
    
    def __init__(self, model_path=None):
        """Initialize the demo application."""
        ShowBase.__init__(self)
        
        # Set window title
        self.windowTitle = "Hoodie Overlay Demo"
        
        # Set up the camera
        self.cam.setPos(0, -5, 0)
        self.cam.lookAt(0, 0, 0)
        
        # Set up lighting
        self.setup_lighting()
        
        # Create a simple ground plane
        self.create_ground()
        
        # Create a simulated person
        self.create_person()
        
        # Load the hoodie model
        self.hoodie = HoodieModel(model_path)
        self.hoodie.load_model(self.render)
        
        # Adjust the hoodie to fit the person
        self.hoodie.adjust_for_person(2.0, (0, 0, 0), 0)
        
        # Add instructions
        self.add_instructions()
        
        # Set up key controls
        self.setup_controls()
        
        # Add a task to simulate person movement
        self.taskMgr.add(self.move_person_task, "MovePersonTask")
    
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
    
    def create_ground(self):
        """Create a simple ground plane."""
        from panda3d.core import CardMaker
        cm = CardMaker("ground")
        cm.setFrame(-10, 10, -10, 10)
        ground = self.render.attachNewNode(cm.generate())
        ground.setP(-90)  # Rotate to be horizontal
        ground.setPos(0, 0, -2)
        ground.setColor(0.7, 0.7, 0.7, 1)
    
    def create_person(self):
        """Create a simple representation of a person."""
        from panda3d.core import CardMaker
        
        # Create a node for the person
        self.person = self.render.attachNewNode("person")
        
        # Body (cylinder approximation)
        cm = CardMaker("person_body")
        cm.setFrame(-0.4, 0.4, -1.5, 0)
        body = self.person.attachNewNode(cm.generate())
        body.setColor(0.8, 0.7, 0.6, 1)  # Skin tone
        
        # Head (sphere approximation)
        cm = CardMaker("person_head")
        cm.setFrame(-0.3, 0.3, -0.3, 0.3)
        head = self.person.attachNewNode(cm.generate())
        head.setPos(0, 0, 0.3)
        head.setColor(0.8, 0.7, 0.6, 1)  # Skin tone
        
        # Arms (simplified)
        cm = CardMaker("person_arm_right")
        cm.setFrame(-0.1, 0.1, -0.8, 0)
        arm_right = self.person.attachNewNode(cm.generate())
        arm_right.setPos(0.5, 0, -0.2)
        arm_right.setColor(0.8, 0.7, 0.6, 1)  # Skin tone
        
        cm = CardMaker("person_arm_left")
        cm.setFrame(-0.1, 0.1, -0.8, 0)
        arm_left = self.person.attachNewNode(cm.generate())
        arm_left.setPos(-0.5, 0, -0.2)
        arm_left.setColor(0.8, 0.7, 0.6, 1)  # Skin tone
        
        # Legs (simplified)
        cm = CardMaker("person_leg_right")
        cm.setFrame(-0.15, 0.15, -1.5, 0)
        leg_right = self.person.attachNewNode(cm.generate())
        leg_right.setPos(0.2, 0, -1.5)
        leg_right.setColor(0.3, 0.3, 0.7, 1)  # Blue jeans
        
        cm = CardMaker("person_leg_left")
        cm.setFrame(-0.15, 0.15, -1.5, 0)
        leg_left = self.person.attachNewNode(cm.generate())
        leg_left.setPos(-0.2, 0, -1.5)
        leg_left.setColor(0.3, 0.3, 0.7, 1)  # Blue jeans
        
        # Initialize movement variables
        self.person_direction = 1  # 1 for right, -1 for left
        self.person_rotation = 0
    
    def add_instructions(self):
        """Add text instructions to the screen."""
        instructions = [
            "Hoodie Overlay Demo",
            "",
            "This demo shows how to overlay a 3D hoodie model",
            "on a simulated person. The hoodie automatically",
            "adjusts to follow the person's movement and rotation.",
            "",
            "Controls:",
            "Space: Toggle person movement",
            "R: Reset scene",
            "Esc: Exit"
        ]
        
        text_node = TextNode('instructions')
        text_node.setText('\n'.join(instructions))
        text_node.setAlign(TextNode.ALeft)
        
        text_np = aspect2d.attachNewNode(text_node)
        text_np.setScale(0.05)
        text_np.setPos(-1.5, 0, 0.8)
    
    def setup_controls(self):
        """Set up keyboard controls."""
        # Toggle person movement
        self.person_moving = True
        self.accept("space", self.toggle_person_movement)
        
        # Reset scene
        self.accept("r", self.reset_scene)
        
        # Exit
        self.accept("escape", sys.exit)
    
    def toggle_person_movement(self):
        """Toggle person movement on/off."""
        self.person_moving = not self.person_moving
    
    def reset_scene(self):
        """Reset the scene to its initial state."""
        self.person.setPos(0, 0, 0)
        self.person.setH(0)
        self.person_direction = 1
        self.person_rotation = 0
        self.hoodie.adjust_for_person(2.0, (0, 0, 0), 0)
    
    def move_person_task(self, task):
        """Task to move the person and update the hoodie position."""
        if self.person_moving:
            # Get current position
            pos = self.person.getPos()
            
            # Move person side to side
            new_x = pos.x + 0.02 * self.person_direction
            
            # Change direction if reaching the boundaries
            if new_x > 2.0:
                self.person_direction = -1
                self.person_rotation = 180
            elif new_x < -2.0:
                self.person_direction = 1
                self.person_rotation = 0
            
            # Update person position and rotation
            self.person.setPos(new_x, pos.y, pos.z)
            self.person.setH(self.person_rotation)
            
            # Update hoodie to match person
            self.hoodie.adjust_for_person(
                2.0,  # height
                (new_x, pos.y, pos.z),  # position
                self.person_rotation  # orientation
            )
        
        return Task.cont


def main():
    """Main function to run the demo."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hoodie Overlay Demo")
    parser.add_argument("--model", type=str, help="Path to the 3D model file")
    args = parser.parse_args()
    
    app = HoodieOverlayDemo(args.model)
    app.run()


if __name__ == "__main__":
    main()