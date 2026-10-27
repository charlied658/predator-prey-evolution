extends Node3D

@onready var predator: Node3D = $Predator
@onready var prey: Node3D = $Prey
@export var capture_distance: float = 1.0
@export var encounter_duration: float = 20.0
var encounters: Array = []
var results: Array = []
var current_encounter := 0
var elapsed_time := 0.0
var batch_mode := false
var encounter_active := false
var output_path := ""
var minimum_distance := 0.0
var predator_boundary_time := 0.0
var prey_boundary_time := 0.0
var predator_path_length := 0.0
var initial_distance := 0.0
var distance_integral := 0.0
var approach_integral := 0.0
var obstacles: Array = []
var obstacle_visuals: Node3D
var trace: Array = []
var physics_steps := 0
var record_trace := false
var overlay: Label
var trails: MeshInstance3D

func _ready() -> void:
	var args = OS.get_cmdline_user_args()
	if args.size() >= 3 and args[0] == "--batch":
		batch_mode = true
		output_path = args[2]
		var file = FileAccess.open(args[1], FileAccess.READ)
		if file == null:
			fail("Cannot open batch input")
			return
		var data = JSON.parse_string(file.get_as_text())
		if not data is Dictionary or not data.get("encounters") is Array:
			fail("Invalid batch JSON")
			return
		encounters = data["encounters"]
	elif args.size() >= 2 and args[0] == "--replay":
		var file = FileAccess.open(args[1], FileAccess.READ)
		if file == null:
			fail("Cannot open replay")
			return
		var data = JSON.parse_string(file.get_as_text())
		if not data is Dictionary:
			fail("Invalid replay JSON")
			return
		encounters = data.get("encounters", [data.get("replay_encounter", {"predator_genome": data.get("genome", []), "prey_genome": data.get("prey_genome", [])})])
		var layer := CanvasLayer.new()
		add_child(layer)
		overlay = Label.new()
		overlay.position = Vector2(18, 18)
		overlay.add_theme_color_override("font_color", Color.WHITE)
		overlay.add_theme_color_override("font_shadow_color", Color.BLACK)
		layer.add_child(overlay)
		trails = MeshInstance3D.new()
		add_child(trails)
	else:
		return
	if encounters.is_empty():
		fail("Batch contains no encounters")
		return
	start_encounter()

func fail(message: String) -> void:
	push_error(message)
	get_tree().quit(1)

func start_encounter() -> void:
	var config = encounters[current_encounter]
	predator.set_genome(config.get("predator_genome", []))
	prey.set_genome(config.get("prey_genome", []))
	predator.controller = "neural"
	predator.sensor_mode = config.get("predator_sensor_mode", "legacy")
	prey.sensor_mode = config.get("prey_sensor_mode", "legacy")
	prey.controller = config.get("prey_controller", "neural")
	if not predator.genome.size() in [66, 218] or (prey.controller == "neural" and not prey.genome.size() in [66, 218]):
		fail("Neural controllers require 66 or 218 weights")
		return
	if not prey.controller in ["neural", "stationary", "straight", "flee"]:
		fail("Unknown prey controller")
		return
	obstacles = config.get("obstacles", [])
	predator.obstacles = obstacles
	prey.obstacles = obstacles
	predator.reset_metrics()
	prey.reset_metrics()
	if obstacle_visuals != null:
		obstacle_visuals.free()
	obstacle_visuals = Node3D.new()
	add_child(obstacle_visuals)
	for obstacle in obstacles:
		var cylinder := CylinderMesh.new()
		cylinder.top_radius = obstacle[2]
		cylinder.bottom_radius = obstacle[2]
		cylinder.height = 2.0
		var mesh := MeshInstance3D.new()
		mesh.mesh = cylinder
		mesh.position = Vector3(obstacle[0], 0.5, obstacle[1])
		var material := StandardMaterial3D.new()
		material.albedo_color = Color(0.3, 0.4, 0.55)
		mesh.material_override = material
		obstacle_visuals.add_child(mesh)
	var p = config.get("predator_start", [-5, 0])
	var q = config.get("prey_start", [5, 0])
	predator.global_position = Vector3(p[0], 0, p[1])
	prey.global_position = Vector3(q[0], 0, q[1])
	predator.velocity = Vector3.ZERO
	prey.velocity = Vector3.ZERO
	trace = []
	physics_steps = 0
	record_trace = config.get("record_trace", false) or not batch_mode
	elapsed_time = 0.0
	minimum_distance = predator.global_position.distance_to(prey.global_position)
	initial_distance = minimum_distance
	distance_integral = 0.0
	approach_integral = 0.0
	predator_boundary_time = 0.0
	prey_boundary_time = 0.0
	predator_path_length = 0.0
	encounter_active = true
	if record_trace:
		sample_trace()

func _physics_process(delta: float) -> void:
	if not encounter_active:
		return
	# Both decisions observe the same positions and previous-step velocities.
	var pv: Vector3 = predator.choose_velocity()
	var qv: Vector3 = prey.choose_velocity()
	predator.velocity = pv
	prey.velocity = qv
	var previous: Vector3 = predator.global_position
	var toward: Vector3 = (prey.global_position - previous).normalized()
	predator.advance(delta)
	prey.advance(delta)
	predator_path_length += previous.distance_to(predator.global_position)
	# Actual predator movement toward the target, excluding prey movement and wall pushing.
	approach_integral += (predator.global_position - previous).dot(toward) / predator.speed
	elapsed_time += delta
	if maxf(absf(predator.position.x), absf(predator.position.z)) >= 13.9:
		predator_boundary_time += delta
	if maxf(absf(prey.position.x), absf(prey.position.z)) >= 13.9:
		prey_boundary_time += delta
	var distance = predator.global_position.distance_to(prey.global_position)
	minimum_distance = minf(minimum_distance, distance)
	distance_integral += distance * delta
	physics_steps += 1
	if record_trace and physics_steps % 12 == 0:
		sample_trace()
	if overlay != null:
		overlay.text = "%s\n%.1fs | red: predator, green: prey | R: restart, Space: next, Esc: quit" % [encounters[current_encounter].get("label", "Replay"), elapsed_time]
	var captured := distance <= capture_distance and clear_capture()
	if captured or elapsed_time + 0.000001 >= encounter_duration:
		end_encounter(captured, distance)

func end_encounter(captured: bool, distance: float) -> void:
	encounter_active = false
	results.append({
		"encounter": current_encounter,
		"captured": captured, "time": elapsed_time,
		"final_distance": distance, "minimum_distance": minimum_distance,
		"initial_distance": initial_distance,
		"mean_distance": distance_integral / elapsed_time,
		"approach_alignment": approach_integral / elapsed_time,
		"predator_boundary_time": predator_boundary_time,
		"prey_boundary_time": prey_boundary_time,
		"predator_path_length": predator_path_length,
		"prey_path_length": prey.path_length,
		"predator_turn_angle": predator.turn_angle, "prey_turn_angle": prey.turn_angle,
		"predator_blocked_time": predator.blocked_time, "prey_blocked_time": prey.blocked_time,
		"predator_obstacle_time": predator.obstacle_time, "prey_obstacle_time": prey.obstacle_time,
		"predator_fitness": 100.0 + encounter_duration - elapsed_time if captured else 0.0,
		"prey_fitness": elapsed_time
	})
	if record_trace:
		sample_trace()
		results[-1]["trajectory"] = trace
	if not batch_mode:
		print(results[-1])
		overlay.text += "\nCaptured" if captured else "\nPrey survived"
		return
	current_encounter += 1
	if current_encounter < encounters.size():
		start_encounter()
	elif batch_mode:
		var file = FileAccess.open(output_path, FileAccess.WRITE)
		if file == null:
			fail("Cannot write results")
			return
		file.store_string(JSON.stringify({"results": results}))
		file.close()
		get_tree().quit()
	else:
		print(results[0])
		get_tree().quit()


func clear_capture() -> bool:
	var origin: Vector3 = predator.global_position
	var segment: Vector3 = prey.global_position - origin
	for obstacle in obstacles:
		var center := Vector3(obstacle[0], 0, obstacle[1])
		var t := clampf((center - origin).dot(segment) / maxf(segment.length_squared(), 0.000001), 0, 1)
		if (origin + segment * t).distance_to(center) < obstacle[2]:
			return false
	return true

func sample_trace() -> void:
	trace.append({"time": elapsed_time, "predator": [predator.position.x, predator.position.z],
		"prey": [prey.position.x, prey.position.z]})
	if trails != null and trace.size() > 1:
		var mesh := ImmediateMesh.new()
		var material := StandardMaterial3D.new()
		material.shading_mode = BaseMaterial3D.SHADING_MODE_UNSHADED
		material.vertex_color_use_as_albedo = true
		mesh.surface_begin(Mesh.PRIMITIVE_LINES, material)
		for i in range(1, trace.size()):
			for role in ["predator", "prey"]:
				mesh.surface_set_color(Color(1, 0.2, 0.2) if role == "predator" else Color(0.2, 1, 0.2))
				var a = trace[i - 1][role]
				var b = trace[i][role]
				mesh.surface_add_vertex(Vector3(a[0], 0.05, a[1]))
				mesh.surface_add_vertex(Vector3(b[0], 0.05, b[1]))
		mesh.surface_end()
		trails.mesh = mesh

func _unhandled_key_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed and not event.echo and not batch_mode:
		if event.keycode == KEY_ESCAPE:
			get_tree().quit()
		elif event.keycode == KEY_R:
			start_encounter()
		elif event.keycode == KEY_SPACE:
			current_encounter = (current_encounter + 1) % encounters.size()
			start_encounter()
