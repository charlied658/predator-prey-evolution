extends Node3D

# Exploratory rule controllers share the original swept-circle locomotion.
const Agent = preload("res://scripts/agent.gd")
var cfg: Dictionary
var rng := RandomNumberGenerator.new()
var agents: Array = []
var foods: Array = []
var rocks: Array = []
var content: Node3D
var hud: Label
var elapsed := 0.0
var paused := false
var finished := false
var time_scale := 1.0
var output_path := ""
var captures := 0
var meals := 0
var starvations := {"predator": 0, "prey": 0}
var births := 0
var energy_integral := {"predator": 0.0, "prey": 0.0}
var living_time := {"predator": 0.0, "prey": 0.0}

func _ready() -> void:
	var path := "res://evolution/ecosystem.json"
	var args := OS.get_cmdline_user_args()
	for i in range(args.size()):
		if args[i] == "--ecosystem-config" and i + 1 < args.size():
			path = args[i + 1]
		if args[i] == "--ecosystem-output" and i + 1 < args.size():
			output_path = args[i + 1]
	var file := FileAccess.open(path, FileAccess.READ)
	if file == null:
		fail("Cannot open ecosystem config: " + path)
		return
	var parsed = JSON.parse_string(file.get_as_text())
	if not parsed is Dictionary:
		fail("Ecosystem config must be a JSON object")
		return
	cfg = parsed
	if not validate_config():
		return
	var camera := Camera3D.new()
	add_child(camera)
	camera.position = Vector3(0, 36, 0.01)
	camera.look_at(Vector3.ZERO)
	camera.projection = Camera3D.PROJECTION_ORTHOGONAL
	camera.size = 34
	var light := DirectionalLight3D.new()
	add_child(light)
	light.rotation_degrees = Vector3(-70, -20, 0)
	light.light_energy = 1.3
	var environment := WorldEnvironment.new()
	environment.environment = Environment.new()
	environment.environment.background_mode = Environment.BG_COLOR
	environment.environment.background_color = Color("17202a")
	environment.environment.ambient_light_source = Environment.AMBIENT_SOURCE_COLOR
	environment.environment.ambient_light_color = Color.WHITE
	environment.environment.ambient_light_energy = 0.55
	add_child(environment)
	var layer := CanvasLayer.new()
	add_child(layer)
	hud = Label.new()
	layer.add_child(hud)
	hud.position = Vector2(12, 10)
	hud.size = Vector2(270, 600)
	hud.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	hud.add_theme_font_size_override("font_size", 16)
	hud.add_theme_color_override("font_shadow_color", Color.BLACK)
	hud.add_theme_constant_override("shadow_offset_x", 1)
	hud.add_theme_constant_override("shadow_offset_y", 1)
	reset_world()

func fail(message: String) -> void:
	push_error(message)
	get_tree().quit(1)

func validate_config() -> bool:
	var positive := ["duration", "food_regrowth", "respawn_delay", "predator_speed", "prey_speed", "acceleration", "turn_degrees_per_second", "predator_vision", "prey_vision", "food_energy", "capture_energy", "home_radius"]
	var nonnegative := ["predators", "prey", "rocks", "food_pellets", "spawn_grace", "predator_hunger_drain", "prey_hunger_drain", "movement_hunger_drain"]
	for key in positive + nonnegative + ["seed", "predator_fov", "prey_fov", "hunger_speed_floor"]:
		if not cfg.has(key) or not (cfg[key] is float or cfg[key] is int):
			fail("Missing or nonnumeric setting: " + key)
			return false
		if not is_finite(float(cfg[key])):
			fail("Setting must be finite: " + key)
			return false
	for key in positive:
		if cfg[key] <= 0:
			fail(key + " must be positive")
			return false
	for key in nonnegative:
		if cfg[key] < 0:
			fail(key + " cannot be negative")
			return false
	for key in ["predators", "prey", "rocks", "food_pellets"]:
		if cfg[key] != int(cfg[key]) or cfg[key] > 200:
			fail(key + " must be an integer between 0 and 200")
			return false
	for key in ["predator_fov", "prey_fov"]:
		if cfg[key] <= 0 or cfg[key] > 360:
			fail(key + " must be in (0, 360]")
			return false
	if cfg.hunger_speed_floor <= 0 or cfg.hunger_speed_floor > 1 or cfg.home_radius > 4:
		fail("hunger_speed_floor must be in (0, 1]; home_radius must be <= 4")
		return false
	for role in ["predator", "prey"]:
		var home = cfg.get(role + "_home")
		if not home is Array or home.size() != 2:
			fail(role + "_home must be [x, z]")
			return false
		for coordinate in home:
			if not (coordinate is float or coordinate is int) or not is_finite(float(coordinate)) or absf(coordinate) + cfg.home_radius > 14:
				fail("Home bases must fit inside the arena")
				return false
	return true

func ball(parent: Node3D, pos: Vector3, radius: float, color: Color) -> MeshInstance3D:
	var node := MeshInstance3D.new()
	var mesh := SphereMesh.new()
	mesh.radius = radius
	mesh.height = radius * 2
	node.mesh = mesh
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	node.material_override = material
	parent.add_child(node)
	node.position = pos
	return node

func box(parent: Node3D, pos: Vector3, size: Vector3, color: Color) -> MeshInstance3D:
	var node := MeshInstance3D.new()
	var mesh := BoxMesh.new()
	mesh.size = size
	node.mesh = mesh
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	node.material_override = material
	parent.add_child(node)
	node.position = pos
	return node

func home_position(role: String) -> Vector3:
	var h: Array = cfg[role + "_home"]
	return Vector3(h[0], 0, h[1])

func clear_point(pos: Vector3, margin: float) -> bool:
	if absf(pos.x) > 14 - margin or absf(pos.z) > 14 - margin:
		return false
	for rock in rocks:
		if pos.distance_to(Vector3(rock[0], 0, rock[1])) < rock[2] + margin:
			return false
	return true

func random_open_point() -> Vector3:
	for attempt in range(2000):
		var pos := Vector3(rng.randf_range(-13, 13), 0, rng.randf_range(-13, 13))
		if clear_point(pos, 0.7):
			return pos
	fail("Could not place food; reduce rock count")
	return Vector3.ZERO

func reset_world() -> void:
	if content != null:
		content.free()
	content = Node3D.new()
	add_child(content)
	agents.clear()
	foods.clear()
	rocks.clear()
	elapsed = 0
	captures = 0
	meals = 0
	births = 0
	starvations = {"predator": 0, "prey": 0}
	energy_integral = {"predator": 0.0, "prey": 0.0}
	living_time = {"predator": 0.0, "prey": 0.0}
	finished = false
	paused = false
	rng.seed = int(cfg.seed)
	box(content, Vector3(0, -0.65, 0), Vector3(29, 0.3, 29), Color("35483d"))
	for role in ["predator", "prey"]:
		var home := home_position(role)
		var disc := MeshInstance3D.new()
		var cylinder := CylinderMesh.new()
		cylinder.top_radius = cfg.home_radius
		cylinder.bottom_radius = cfg.home_radius
		cylinder.height = 0.05
		disc.mesh = cylinder
		var material := StandardMaterial3D.new()
		material.albedo_color = Color("843e43") if role == "predator" else Color("397b76")
		disc.material_override = material
		content.add_child(disc)
		disc.position = home + Vector3(0, -0.45, 0)
	for index in range(int(cfg.rocks)):
		var placed := false
		for attempt in range(1000):
			var pos := Vector3(rng.randf_range(-12.5, 12.5), 0, rng.randf_range(-12.5, 12.5))
			var radius := rng.randf_range(0.6, 1.35)
			if not clear_point(pos, radius + 1.2):
				continue
			if pos.distance_to(home_position("predator")) < cfg.home_radius + radius + 1 or pos.distance_to(home_position("prey")) < cfg.home_radius + radius + 1:
				continue
			rocks.append([pos.x, pos.z, radius])
			var visual := ball(content, pos, radius, Color("7a8493"))
			visual.scale.y = 1.3
			placed = true
			break
		if not placed:
			fail("Rock placement exhausted; reduce rocks or home radius")
			return
	for index in range(int(cfg.food_pellets)):
		var pos := random_open_point()
		foods.append({"pos": pos, "cooldown": 0.0, "visual": ball(content, pos, 0.18, Color("f7d66c"))})
	for role in ["predator", "prey"]:
		for index in range(int(cfg["predators" if role == "predator" else "prey"])):
			var body = Agent.new()
			content.add_child(body)
			body.sensor_mode = "ecology"
			body.obstacles = rocks
			ball(body, Vector3.ZERO, 0.5, Color("ed665e") if role == "predator" else Color("65d6ae"))
			var nose := ball(body, Vector3(0.45, 0.2, 0), 0.14, Color.WHITE)
			var bar := box(body, Vector3(0, 0.8, -0.6), Vector3(1, 0.08, 0.14), Color("f3d45e"))
			var agent := {"body": body, "role": role, "energy": 100.0, "heading": 0.0, "cooldown": 0.0, "age": 0.0, "nose": nose, "bar": bar, "wander": rng.randf_range(-PI, PI), "wander_timer": 0.0}
			agents.append(agent)
			spawn(agent)
	update_hud()

func spawn(agent: Dictionary) -> void:
	var center := home_position(agent.role)
	var angle := rng.randf_range(-PI, PI)
	var radius := sqrt(rng.randf()) * float(cfg.home_radius)
	agent.body.position = center + Vector3(cos(angle), 0, sin(angle)) * radius
	agent.body.velocity = Vector3.ZERO
	agent.energy = 100.0
	agent.age = 0.0
	agent.heading = PI if agent.role == "prey" else 0.0
	agent.body.visible = true
	births += 1

# Rocks occlude both vision and capture. This tests the complete line segment.
func unobstructed(a: Vector3, b: Vector3) -> bool:
	var segment := b - a
	for rock in rocks:
		var offset := Vector3(rock[0], 0, rock[1]) - a
		var t := clampf(offset.dot(segment) / maxf(segment.length_squared(), 0.000001), 0, 1)
		if (offset - segment * t).length() < rock[2]:
			return false
	return true

func sees(agent: Dictionary, pos: Vector3) -> bool:
	var relative: Vector3 = pos - agent.body.position
	if relative.length() > float(cfg[agent.role + "_vision"]):
		return false
	var facing := Vector3(cos(agent.heading), 0, sin(agent.heading))
	if relative.length() > 0.001 and facing.dot(relative.normalized()) < cos(deg_to_rad(float(cfg[agent.role + "_fov"])) * 0.5):
		return false
	return unobstructed(agent.body.position, pos)

func desired_direction(agent: Dictionary, delta: float) -> Vector3:
	var pos: Vector3 = agent.body.position
	agent.wander_timer -= delta
	if agent.wander_timer <= 0:
		agent.wander = agent.heading + rng.randf_range(-1.4, 1.4)
		agent.wander_timer = rng.randf_range(0.7, 2.0)
	var desired := Vector3(cos(agent.wander), 0, sin(agent.wander)) * 0.5
	var nearest: Dictionary = {}
	var distance := INF
	for other in agents:
		if other.role == agent.role or other.cooldown > 0:
			continue
		var d: float = pos.distance_to(other.body.position)
		if d < distance and sees(agent, other.body.position):
			distance = d
			nearest = other
	if not nearest.is_empty():
		desired = (nearest.body.position - pos).normalized()
		if agent.role == "prey":
			desired = -desired * 2.5
	elif agent.role == "prey" and agent.energy < 85:
		var food_distance := INF
		for food in foods:
			var d := pos.distance_to(food.pos)
			if food.cooldown <= 0 and d < food_distance and sees(agent, food.pos):
				food_distance = d
				desired = (food.pos - pos).normalized()
	# Local clearance is tactile obstacle avoidance, independent of visual FOV.
	var avoid := Vector3.ZERO
	for ray in range(16):
		var angle := TAU * ray / 16.0
		var direction := Vector3(cos(angle), 0, sin(angle))
		var clearance: float = agent.body.clearance(direction, 2.0)
		avoid -= direction * pow(1.0 - clearance, 3)
	for other in agents:
		if other == agent or other.cooldown > 0:
			continue
		var offset: Vector3 = pos - other.body.position
		if offset.length() < 1.2:
			avoid += offset.normalized() * (1.2 - offset.length())
	return (desired + avoid * 1.8).normalized()

func deactivate(agent: Dictionary) -> void:
	agent.cooldown = float(cfg.respawn_delay)
	agent.body.visible = false
	agent.body.velocity = Vector3.ZERO

func _physics_process(delta: float) -> void:
	if paused or finished or cfg == null:
		return
	# Fixed substeps keep 1x/2x/4x controls physically equivalent.
	for substep in range(int(time_scale)):
		step(delta)
		if finished:
			break
	update_hud()

func step(delta: float) -> void:
	elapsed += delta
	for food in foods:
		if food.cooldown > 0:
			food.cooldown -= delta
			if food.cooldown <= 0:
				food.pos = random_open_point()
				food.visual.position = food.pos
				food.visual.visible = true
	var active: Array = []
	for agent in agents:
		if agent.cooldown > 0:
			agent.cooldown -= delta
			if agent.cooldown <= 0:
				spawn(agent)
			continue
		active.append(agent)
	# Decisions observe the same positions, then all bodies advance.
	var directions: Array = []
	for agent in active:
		directions.append(desired_direction(agent, delta))
	for i in range(active.size()):
		var agent: Dictionary = active[i]
		var direction: Vector3 = directions[i]
		var target_angle := atan2(direction.z, direction.x)
		var turn := deg_to_rad(float(cfg.turn_degrees_per_second)) * delta
		agent.heading += clampf(wrapf(target_angle - agent.heading, -PI, PI), -turn, turn)
		var facing := Vector3(cos(agent.heading), 0, sin(agent.heading))
		var speed: float = cfg[agent.role + "_speed"] * lerpf(float(cfg.hunger_speed_floor), 1.0, agent.energy / 100.0)
		agent.body.velocity = agent.body.velocity.move_toward(facing * speed, float(cfg.acceleration) * delta)
		agent.body.advance(delta)
		agent.nose.position = facing * 0.45 + Vector3(0, 0.2, 0)
		agent.age += delta
		agent.energy -= delta * (float(cfg[agent.role + "_hunger_drain"]) + float(cfg.movement_hunger_drain) * agent.body.velocity.length() / float(cfg[agent.role + "_speed"]))
	# Resolve capture before feeding; grace prevents captures immediately after spawning.
	for predator in active:
		if predator.role != "predator" or predator.age < cfg.spawn_grace:
			continue
		for prey in active:
			if prey.role != "prey" or prey.cooldown > 0 or prey.age < cfg.spawn_grace:
				continue
			if predator.body.position.distance_to(prey.body.position) <= 1.0 and unobstructed(predator.body.position, prey.body.position):
				captures += 1
				predator.energy = minf(100, predator.energy + float(cfg.capture_energy))
				deactivate(prey)
	for agent in active:
		if agent.cooldown > 0:
			continue
		if agent.role == "prey":
			for food in foods:
				if food.cooldown <= 0 and agent.energy < 95 and agent.body.position.distance_to(food.pos) < 0.7:
					agent.energy = minf(100, agent.energy + float(cfg.food_energy))
					food.cooldown = float(cfg.food_regrowth)
					food.visual.visible = false
					meals += 1
		if agent.energy <= 0:
			starvations[agent.role] += 1
			deactivate(agent)
		else:
			agent.bar.scale.x = maxf(0.01, agent.energy / 100.0)
			energy_integral[agent.role] += agent.energy * delta
			living_time[agent.role] += delta
	if elapsed + 0.000001 >= float(cfg.duration):
		finish()

func finish() -> void:
	finished = true
	var summary := {"settings": cfg, "simulated_seconds": elapsed, "captures": captures, "food_eaten": meals, "starvations": starvations, "spawns_including_initial": births, "mean_energy": {}}
	for role in ["predator", "prey"]:
		summary.mean_energy[role] = energy_integral[role] / living_time[role] if living_time[role] > 0 else null
	if not output_path.is_empty():
		var file := FileAccess.open(output_path, FileAccess.WRITE)
		if file == null:
			fail("Cannot write summary: " + output_path)
			return
		file.store_string(JSON.stringify(summary, "  ") + "\n")
	print(JSON.stringify(summary))
	if DisplayServer.get_name() == "headless":
		get_tree().quit()

func update_hud() -> void:
	var counts := {"predator": 0, "prey": 0}
	for agent in agents:
		if agent.cooldown <= 0:
			counts[agent.role] += 1
	hud.text = "ECOSYSTEM LAB | seed %d | %.1f / %.0fs | %dx\nRed predators %d/%d | Green prey %d/%d\nCaptures %d | Pellets eaten %d | Starved P/Q %d/%d\nYellow bars: energy | White dots: facing | Colored discs: spawn bases\nSpace pause | R repeat seed | N next seed | 1/2/4 speed | Esc quit\nRule controllers: hunt / evade / forage. Bases are spawn points, not safe zones." % [int(cfg.seed), elapsed, cfg.duration, int(time_scale), counts.predator, cfg.predators, counts.prey, cfg.prey, captures, meals, starvations.predator, starvations.prey]
	if finished:
		hud.text += "\nFINISHED — R to repeat, N for a new world"
	elif paused:
		hud.text += "\nPAUSED"

func _unhandled_key_input(event: InputEvent) -> void:
	if not event is InputEventKey or not event.pressed or event.echo:
		return
	match event.keycode:
		KEY_SPACE:
			paused = not paused
		KEY_R:
			reset_world()
		KEY_N:
			cfg.seed = int(cfg.seed) + 1
			reset_world()
		KEY_1:
			time_scale = 1
		KEY_2:
			time_scale = 2
		KEY_4:
			time_scale = 4
		KEY_ESCAPE:
			get_tree().quit()
	update_hud()
