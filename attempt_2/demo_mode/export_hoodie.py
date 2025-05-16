import bpy

# Deselect all
bpy.ops.object.select_all(action='DESELECT')

# Try to find hoodie object by name
for obj in bpy.data.objects:
    if "hoodie" in obj.name.lower():
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        break
else:
    print("❌ Hoodie object not found.")
    exit()

# Export as OBJ
bpy.ops.export_scene.obj(
    filepath="hoodie_model.obj",
    use_selection=True,
    use_materials=True,
    axis_forward='-Z',
    axis_up='Y'
)
print("✅ Hoodie exported as hoodie_model.obj")
