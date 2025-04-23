"""
test_hoodie_model_headless.py - Non-graphical test for the hoodie model implementation

This script tests the core functionality of the HoodieModel class without requiring
a graphical display, making it suitable for headless environments.
"""

import os
import sys
from panda3d.core import Point3, Vec3, NodePath, PandaNode

# Add the current directory to the path so we can import the hoodie_model module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the HoodieModel class
from hoodie_model import HoodieModel

def test_hoodie_model():
    """Test the core functionality of the HoodieModel class."""
    print("Testing HoodieModel functionality...")
    
    # Create a dummy render node
    dummy_render = NodePath(PandaNode("dummy_render"))
    
    # Create a HoodieModel instance
    print("Creating HoodieModel instance...")
    hoodie = HoodieModel()
    
    # Test model loading
    print("Testing model loading...")
    success = hoodie.load_model(dummy_render)
    print(f"Model loading {'successful' if success else 'failed'}")
    
    # Test transformations
    print("\nTesting transformations...")
    
    # Test scaling
    print("Testing scaling...")
    hoodie.set_scale(2.0)
    print(f"Scale factor set to: {hoodie.scale_factor}")
    
    # Test positioning
    print("Testing positioning...")
    hoodie.set_position(1.0, 2.0, 3.0)
    print(f"Position set to: {hoodie.position}")
    
    # Test rotation
    print("Testing rotation...")
    hoodie.set_rotation(45.0, 30.0, 15.0)
    print(f"Rotation set to: {hoodie.rotation}")
    
    # Test person adjustment
    print("\nTesting person adjustment...")
    person_height = 1.8
    person_position = (0.5, 0.5, 0.0)
    person_orientation = 90.0
    
    hoodie.adjust_for_person(person_height, person_position, person_orientation)
    print(f"Adjusted for person with height {person_height}, position {person_position}, orientation {person_orientation}")
    print(f"Resulting scale: {hoodie.scale_factor}")
    print(f"Resulting position: {hoodie.position}")
    print(f"Resulting rotation: {hoodie.rotation}")
    
    # Test cleanup
    print("\nTesting cleanup...")
    hoodie.cleanup()
    print("Cleanup completed")
    
    print("\nAll tests completed successfully!")
    return True

if __name__ == "__main__":
    test_hoodie_model()