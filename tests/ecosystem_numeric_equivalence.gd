extends SceneTree

func _initialize() -> void:
	call_deferred("run_checks")

func run_checks() -> void:
	var settings = JSON.parse_string(FileAccess.get_file_as_string("res://evolution/ecosystem.json"))
	settings.duration = 1.0
	var numeric = load("res://ecosystem.tscn").instantiate()
	numeric.batch_mode = true
	numeric.cfg = settings.duplicate(true)
	root.add_child(numeric)
	numeric.set_physics_process(false)
	var visual = load("res://ecosystem.tscn").instantiate()
	visual.batch_mode = true
	visual.cfg = settings.duplicate(true)
	root.add_child(visual)
	visual.set_physics_process(false)
	# Exercise the normal mesh/HUD state in the same headless engine.
	visual.numeric_mode = false
	visual.hud = Label.new()
	visual.add_child(visual.hud)
	visual.reset_world()
	if not numeric.find_children("*", "MeshInstance3D", true, false).is_empty() or visual.find_children("*", "MeshInstance3D", true, false).is_empty():
		push_error("Numeric mode must skip meshes while visual mode builds them")
		quit(1)
		return
	for frame in range(60):
		numeric.step(1.0 / 60)
		visual.step(1.0 / 60)
	if numeric.last_summary.is_empty() or numeric.last_summary != visual.last_summary:
		push_error("Numeric and visual simulation results must match")
		quit(1)
		return
	print("Numeric/visual equivalence: PASS")
	quit()
