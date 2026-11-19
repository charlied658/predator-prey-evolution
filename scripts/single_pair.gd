extends Node3D

const Agent = preload("res://scripts/polygon_agent.gd")
const Terrain = preload("res://scripts/terrain.gd")
const Duel = preload("res://scripts/foraging_duel.gd")
const CircleAgent = preload("res://scripts/agent.gd")
const ArticulatedAgent = preload("res://scripts/articulated_agent.gd")
var cfg: Dictionary
var numeric_mode := false
var batch_mode := false
var failed := false
var last_summary: Dictionary = {}
var rng := RandomNumberGenerator.new()
var agents: Array = []
var foods: Array = []
var rocks: Array = []
var collision_polygons: Array = []
var placement_polygons: Array = []
var content: Node3D
var hud: Label
var elapsed := 0.0
var paused := false
var finished := false
var time_scale := 1.0
var output_path := ""
var captures := 0
var meals := 0
var food_centers: Array = []
var foraging_duel := false
var food_competition := false
var solo_probe := false
var articulated := false
var swimming := false
var circle_rocks: Array = []
var tag_cooldown := 0.0

func _ready() -> void:
	var path := "res://src/configs/single_pair.json"
	var args := OS.get_cmdline_user_args()
	for i in range(args.size()):
		if args[i] == "--ecosystem-config" and i + 1 < args.size():
			path = args[i + 1]
		if args[i] == "--ecosystem-output" and i + 1 < args.size():
			output_path = args[i + 1]
	if cfg.is_empty():
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
	numeric_mode = DisplayServer.get_name() == "headless"
	if numeric_mode:
		reset_world()
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
	failed = true
	push_error(message)
	get_tree().quit(1)

func validate_config() -> bool:
	food_competition = cfg.get("mode") == "food_competition"
	solo_probe = food_competition and cfg.get("solo_probe", false)
	articulated = food_competition and cfg.has("body_genes")
	swimming = articulated and cfg.get("swimming", false)
	if articulated:
		if not cfg.body_genes is Array or cfg.body_genes.size() != 2:
			fail("Expected two body genomes")
			return false
		for body_genome in cfg.body_genes:
			if not body_genome is Array or body_genome.size() != 4:
				fail("Expected four genes per body")
				return false
			for gene in body_genome:
				if not (gene is float or gene is int) or not is_finite(float(gene)) or gene < 0 or gene > 1:
					fail("Body genes must be in [0, 1]")
					return false
	foraging_duel = food_competition or cfg.get("mode") == "foraging_duel"
	if foraging_duel:
		var genomes = cfg.get("genomes")
		if not genomes is Array or genomes.size() != 2:
			fail("Foraging duel requires two genomes")
			return false
		var gene_count := Duel.SWIMMING_GENES if swimming else (Duel.ARTICULATED_GENES if articulated else Duel.GENES)
		for genome in genomes:
			if not genome is Array or genome.size() != gene_count:
				fail("Invalid foraging genome length")
				return false
			for weight in genome:
				if not (weight is float or weight is int) or not is_finite(float(weight)):
					fail("Foraging genomes must contain finite weights")
					return false
	# Saved legacy configurations are upgraded at the simulation boundary.
	var defaults = JSON.parse_string(FileAccess.get_file_as_string("res://src/configs/single_pair.json"))
	for key in defaults:
		if not cfg.has(key):
			cfg[key] = defaults[key]
	cfg.predators = 1
	cfg.prey = 1
	var positive := ["duration", "food_regrowth", "predator_speed", "prey_speed", "predator_acceleration", "prey_acceleration", "predator_turn_degrees", "prey_turn_degrees", "predator_vision", "prey_vision", "food_cluster_radius", "home_radius"]
	var counts := ["terrain_formations", "food_pellets", "food_clusters"]
	for key in positive + counts + ["seed"]:
		if not (cfg[key] is float or cfg[key] is int) or not is_finite(float(cfg[key])):
			fail("Setting must be a finite number: " + key)
			return false
	for key in positive:
		if cfg[key] <= 0:
			fail("Setting must be positive: " + key)
			return false
	for key in counts:
		if cfg[key] != int(cfg[key]) or cfg[key] < 0 or cfg[key] > 100:
			fail("Count must be an integer in [0, 100]: " + key)
			return false
	if cfg.terrain_formations > 5 or (cfg.food_pellets > 0 and cfg.food_clusters < 1) or cfg.home_radius > 2:
		fail("Use at most five terrain formations, a food cluster when food exists, and home_radius <= 2")
		return false
	for role in ["predator", "prey"]:
		var home = cfg.get(role + "_home")
		if not home is Array or home.size() != 2:
			fail("Home must be [x, z]")
			return false
		for coordinate in home:
			if not (coordinate is float or coordinate is int) or not is_finite(float(coordinate)) or absf(coordinate) + cfg.home_radius > 13:
				fail("Spawn region must fit inside the arena")
				return false
	if home_position("predator").distance_to(home_position("prey")) <= 2 * cfg.home_radius + 2:
		fail("Spawn regions must be separated")
		return false
	var behavior = cfg.get("behavior", {})
	if not behavior is Dictionary:
		fail("behavior must be an object")
		return false
	for role in behavior:
		if not role in ["predator", "prey"] or not behavior[role] is Dictionary:
			fail("Unknown behavior role")
			return false
		for key in behavior[role]:
			var value = behavior[role][key]
			if not key in ["target_weight", "avoid_weight", "wander_weight", "food_weight", "food_threshold"] or not (value is int or value is float) or not is_finite(float(value)) or value < 0 or value > (100 if key == "food_threshold" else 6):
				fail("Invalid behavior gene: " + str(key))
				return false
	return true

func ball(parent: Node3D, pos: Vector3, radius: float, color: Color) -> Node3D:
	if numeric_mode:
		var placeholder := Node3D.new()
		parent.add_child(placeholder)
		placeholder.position = pos
		return placeholder
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

func box(parent: Node3D, pos: Vector3, size: Vector3, color: Color) -> Node3D:
	if numeric_mode:
		var placeholder := Node3D.new()
		parent.add_child(placeholder)
		placeholder.position = pos
		return placeholder
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

func clear_point(pos: Vector3, margin: float = 0.7) -> bool:
	if absf(pos.x) > 14 - margin or absf(pos.z) > 14 - margin:
		return false
	var point := Vector2(pos.x, pos.z)
	for rock in circle_rocks:
		if point.distance_to(Vector2(rock[0], rock[1])) < rock[2] + margin:
			return false
	# Placement uses a larger margin than the animals to keep food approachable.
	for polygon in placement_polygons:
		if Geometry2D.is_point_in_polygon(point, polygon):
			return false
	return true

func random_open_point() -> Vector3:
	for attempt in range(2000):
		var pos := Vector3(rng.randf_range(-12, 12), 0, rng.randf_range(-12, 12))
		if clear_point(pos):
			return pos
	fail("Could not find an open position")
	return Vector3.ZERO

func generate_terrain() -> void:
	# Bounding discs reserve wide connected routes; only the irregular polygons
	# inside those discs are solid. No overlaps or boundary-spanning barriers.
	var reserved: Array = []
	for index in range(int(cfg.terrain_formations)):
		var placed := false
		for attempt in range(2000):
			var radius := rng.randf_range(4.5, 5.5) if index == 0 else rng.randf_range(2.0, 3.0)
			var span := 2.2 if index == 0 else 11 - radius
			var center := Vector2(rng.randf_range(-span, span), rng.randf_range(-span, span))
			var acceptable := true
			for region in reserved:
				if center.distance_to(region.center) < radius + region.radius + 2:
					acceptable = false
			for role in ["predator", "prey"]:
				var home := home_position(role)
				if center.distance_to(Vector2(home.x, home.z)) < radius + cfg.home_radius + 1.5:
					acceptable = false
			if not acceptable:
				continue
			var polygon := PackedVector2Array()
			var phase := rng.randf_range(-PI, PI)
			var stretch := rng.randf_range(0.65, 1.0)
			for vertex in range(20):
				var angle := TAU * vertex / 20
				var r := radius * (0.78 + 0.13 * sin(3 * angle + phase) + 0.07 * cos(2 * angle - phase))
				var local := Vector2(cos(angle), sin(angle) * stretch).rotated(phase) * r
				polygon.append(center + local)
			rocks.append(polygon)
			collision_polygons.append(Terrain.expanded(polygon, 0.51))
			placement_polygons.append(Terrain.expanded(polygon, 0.85))
			reserved.append({"center": center, "radius": radius})
			if not numeric_mode:
				var visual := MeshInstance3D.new()
				visual.mesh = Terrain.mesh(polygon)
				var material := StandardMaterial3D.new()
				material.albedo_color = Color("68777c")
				material.cull_mode = BaseMaterial3D.CULL_DISABLED
				visual.material_override = material
				content.add_child(visual)
			placed = true
			break
		if not placed:
			fail("Terrain placement exhausted; reduce terrain_formations")
			return

func generate_food_competition_map() -> void:
	for index in range(int(cfg.get("rocks", 14))):
		var placed := false
		for attempt in range(1000):
			var pos := Vector3(rng.randf_range(-12.5, 12.5), 0, rng.randf_range(-12.5, 12.5))
			var radius := rng.randf_range(0.6, 1.35)
			if not clear_point(pos, radius + 1.2):
				continue
			var spawn_clearance := 3.4 if articulated else 1.0
			if pos.distance_to(home_position("predator")) < cfg.home_radius + radius + spawn_clearance:
				continue
			if pos.distance_to(home_position("prey")) < cfg.home_radius + radius + spawn_clearance:
				continue
			circle_rocks.append([pos.x, pos.z, radius])
			var visual := ball(content, pos, radius, Color("7a8493"))
			visual.scale.y = 1.3
			placed = true
			break
		if not placed:
			fail("Circular rock placement exhausted; reduce rock count")
			return
	for index in range(int(cfg.food_pellets)):
		var pos := random_open_point()
		if failed:
			return
		foods.append({"pos": pos, "cooldown": 0.0, "visual": ball(content, pos, 0.18, Color("e7ad27"))})

func reset_world() -> void:
	if content != null:
		content.free()
	content = Node3D.new()
	add_child(content)
	agents.clear()
	foods.clear()
	rocks.clear()
	circle_rocks.clear()
	collision_polygons.clear()
	placement_polygons.clear()
	food_centers.clear()
	elapsed = 0
	captures = 0
	meals = 0
	tag_cooldown = 0
	finished = false
	last_summary = {}
	paused = false
	rng.seed = int(cfg.seed)
	if not numeric_mode:
		box(content, Vector3(0, -0.65, 0), Vector3(28, 0.3, 28), Color("d4cbb0"))
	if food_competition:
		generate_food_competition_map()
	else:
		generate_terrain()
		generate_clustered_food()
	if failed:
		return
	for role in ["predator", "prey"]:
		var body = ArticulatedAgent.new() if articulated else (CircleAgent.new() if food_competition else Agent.new())
		content.add_child(body)
		body.obstacles = circle_rocks if food_competition else collision_polygons
		var color := Color("d84f49") if role == "predator" else Color("219b91")
		if not articulated:
			ball(body, Vector3.ZERO, 0.5, color)
		var nose := ball(body, Vector3(0, 0.28, 0) if articulated else Vector3(0.45, 0.2, 0), 0.11 if articulated else 0.14, Color.WHITE)
		var center := home_position(role)
		var angle := rng.randf_range(-PI, PI)
		body.position = center + Vector3(cos(angle), 0, sin(angle)) * sqrt(rng.randf()) * cfg.home_radius
		agents.append({"body": body, "role": role, "heading": PI if role == "prey" else 0.0, "nose": nose, "wander": rng.randf_range(-PI, PI), "wander_timer": 0.0})
	if solo_probe:
		var unused := 0 if cfg.get("swap_starts", false) else 1
		agents[unused].body.free()
		agents.remove_at(unused)
	if foraging_duel:
		for slot in range(agents.size()):
			var lineage := 1 - slot if not solo_probe and cfg.get("swap_starts", false) else slot
			agents[slot].merge({"lineage": lineage, "genome": cfg.genomes[lineage], "balance": 0,
				"collected": 0, "stolen": 0, "lost": 0})
			if articulated:
				var agent: Dictionary = agents[slot]
				var color := Color("d84f49") if agent.role == "predator" else Color("219b91")
				agent.body.configure(cfg.body_genes[lineage], color, numeric_mode)
				agent.body.swimming = swimming
				agent.body.heading = agent.heading
				agent.body.update_visuals()
				if not agent.body.pose_clear(agent.body.position, agent.heading, 0):
					fail("Articulated body does not fit at spawn")
					return
	update_hud()

func generate_clustered_food() -> void:
	for index in range(int(cfg.food_clusters)):
		var center := random_open_point()
		# Keep patches distinct; impossible custom settings fail explicitly.
		var placed := false
		for attempt in range(200):
			var separated := true
			for existing in food_centers:
				if center.distance_to(existing) < 2 * cfg.food_cluster_radius + 1:
					separated = false
			if separated:
				placed = true
				break
			center = random_open_point()
		if not placed:
			fail("Could not separate food patches; reduce food_clusters or food_cluster_radius")
			return
		food_centers.append(center)
	for index in range(int(cfg.food_pellets)):
		var center: Vector3 = food_centers[index % food_centers.size()]
		var pos := center
		var placed := false
		for attempt in range(200):
			var angle := rng.randf_range(-PI, PI)
			var radius := sqrt(rng.randf()) * float(cfg.food_cluster_radius)
			var candidate := center + Vector3(cos(angle), 0, sin(angle)) * radius
			var separated := true
			for food in foods:
				if candidate.distance_to(food.pos) < 0.4:
					separated = false
			if clear_point(candidate) and separated:
				pos = candidate
				placed = true
				break
		if not placed:
			fail("Could not place clustered pellets; reduce food_pellets or increase food_cluster_radius")
			return
		foods.append({"pos": pos, "cooldown": 0.0, "visual": ball(content, pos, 0.18, Color("e7ad27"))})

func unobstructed(a: Vector3, b: Vector3) -> bool:
	return Terrain.hit(Vector2(a.x, a.z), Vector2(b.x, b.z), rocks).normal == Vector2.ZERO

# Omnidirectional range sensing intentionally passes through terrain.
func sees(agent: Dictionary, pos: Vector3) -> bool:
	return agent.body.position.distance_to(pos) <= float(cfg[agent.role + "_vision"])

func desired_direction(agent: Dictionary, delta: float) -> Vector3:
	var behavior: Dictionary = cfg.get("behavior", {}).get(agent.role, {})
	var pos: Vector3 = agent.body.position
	agent.wander_timer -= delta
	if agent.wander_timer <= 0:
		agent.wander = agent.heading + rng.randf_range(-1.4, 1.4)
		agent.wander_timer = rng.randf_range(0.7, 2.0)
	var desired := Vector3(cos(agent.wander), 0, sin(agent.wander)) * float(behavior.get("wander_weight", 0.5))
	var other: Dictionary = agents[1] if agent.role == "predator" else agents[0]
	if sees(agent, other.body.position):
		var offset: Vector3 = other.body.position - pos
		var weight := float(behavior.get("target_weight", 1.0 if agent.role == "predator" else 2.5))
		if agent.role == "prey":
			# Threat urgency increases smoothly at close range; food remains a choice.
			desired -= offset.normalized() * weight * (8.0 / maxf(2, offset.length()))
		else:
			desired += offset.normalized() * weight
	if agent.role == "prey":
		var nearest := INF
		var food_direction := Vector3.ZERO
		for food in foods:
			var distance: float = pos.distance_to(food.pos)
			if food.cooldown <= 0 and distance < nearest and sees(agent, food.pos):
				nearest = distance
				food_direction = (food.pos - pos).normalized()
		desired += food_direction * float(behavior.get("food_weight", 1.5))
	var avoid := Vector3.ZERO
	for ray in range(16):
		var angle := TAU * ray / 16
		var direction := Vector3(cos(angle), 0, sin(angle))
		avoid -= direction * pow(1 - agent.body.clearance(direction, 2.5), 3)
	return (desired + avoid * float(behavior.get("avoid_weight", 1.8))).normalized()

func _physics_process(delta: float) -> void:
	if paused or finished or cfg == null:
		return
	for substep in range(int(time_scale)):
		step(delta)
		if finished:
			break
	update_hud()

func step(delta: float) -> void:
	if finished:
		return
	var dt := minf(delta, maxf(0, float(cfg.duration) - elapsed))
	elapsed += dt
	for food in foods:
		if not food_competition and food.cooldown > 0:
			food.cooldown = maxf(0, food.cooldown - dt)
			food.visual.visible = food.cooldown <= 0
	var directions: Array = []
	var actions: Array = []
	var previous_positions: Array = []
	for i in range(agents.size()):
		var agent: Dictionary = agents[i]
		if foraging_duel:
			actions.append(Duel.action(agent, {} if solo_probe else agents[1 - i], foods, elapsed))
		else:
			directions.append(desired_direction(agent, dt))
		previous_positions.append(agent.body.position)
	for i in range(agents.size()):
		var agent: Dictionary = agents[i]
		var speed := 4.0 if foraging_duel else float(cfg[agent.role + "_speed"])
		var acceleration := 7.0 if foraging_duel else float(cfg[agent.role + "_acceleration"])
		var turn_rate := 195.0 if foraging_duel else float(cfg[agent.role + "_turn_degrees"])
		var current_speed: float = agent.body.velocity.length() / speed
		var turn := deg_to_rad(turn_rate) * lerpf(1, 0.55, current_speed) * dt
		if swimming:
			turn *= clampf(current_speed * 2, 0, 1)
		var throttle := 1.0
		if foraging_duel:
			agent.heading = wrapf(agent.heading + actions[i].x * turn, -PI, PI)
			throttle = actions[i].y
		else:
			var direction: Vector3 = directions[i]
			var target_angle := atan2(direction.z, direction.x)
			agent.heading += clampf(wrapf(target_angle - agent.heading, -PI, PI), -turn, turn)
		var facing := Vector3(cos(agent.heading), 0, sin(agent.heading))
		if not swimming:
			agent.body.velocity = agent.body.velocity.move_toward(facing * speed * throttle, acceleration * dt)
		if articulated:
			agent.body.advance_pose(dt, agent.heading, actions[i].z * agent.body.max_bend)
			agent.heading = agent.body.heading
		else:
			agent.body.advance(dt)
			agent.nose.position = facing * 0.45 + Vector3(0, 0.2, 0)
	if food_competition:
		Duel.collect(agents, foods, INF)
		if elapsed + 0.000001 >= float(cfg.duration):
			finish()
		return
	# Swept relative contact avoids missing a capture when bodies cross in one step.
	var predator: Dictionary = agents[0]
	var prey: Dictionary = agents[1]
	var start: Vector3 = previous_positions[0] - previous_positions[1]
	var end: Vector3 = predator.body.position - prey.body.position
	var travel := end - start
	var t := clampf(-start.dot(travel) / maxf(travel.length_squared(), 0.000000001), 0, 1)
	var predator_contact: Vector3 = previous_positions[0].lerp(predator.body.position, t)
	var prey_contact: Vector3 = previous_positions[1].lerp(prey.body.position, t)
	var contact := (start + travel * t).length() <= 1 and unobstructed(predator_contact, prey_contact)
	if foraging_duel:
		tag_cooldown = maxf(0, tag_cooldown - dt)
		if contact and Duel.tag(agents, previous_positions, tag_cooldown == 0):
			tag_cooldown = Duel.TAG_COOLDOWN
		Duel.collect(agents, foods, float(cfg.food_regrowth))
		if elapsed + 0.000001 >= float(cfg.duration):
			finish()
		return
	if contact:
		captures = 1
		finish()
		return
	for food in foods:
		if food.cooldown <= 0 and prey.body.position.distance_to(food.pos) < 0.7:
			food.cooldown = float(cfg.food_regrowth)
			food.visual.visible = false
			meals += 1
	if elapsed + 0.000001 >= float(cfg.duration):
		finish()

func finish() -> void:
	finished = true
	var fraction := clampf(elapsed / float(cfg.duration), 0, 1)
	var prey_score := meals + 5.0 * fraction
	var predator_score := float(captures) * (1 + (1 - fraction))
	var terrain_data: Array = []
	for polygon in rocks:
		var points: Array = []
		for point in polygon:
			points.append([point.x, point.y])
		terrain_data.append(points)
	var summary := {"schema_version": 2, "settings": cfg.duplicate(true), "simulated_seconds": elapsed, "survival_seconds": elapsed, "capture_time": elapsed if captures else null, "captures": captures, "food_eaten": meals, "fitness": {"prey": prey_score, "predator": predator_score}, "terrain": terrain_data}
	if foraging_duel:
		var outcomes := [{}, {}]
		for agent in agents:
			outcomes[agent.lineage] = {"balance": agent.balance, "collected": agent.collected,
				"stolen": agent.stolen, "lost": agent.lost}
			if solo_probe:
				outcomes[agent.lineage].mean_speed = agent.body.path_length / elapsed
				outcomes[agent.lineage].turn_radians_per_second = agent.body.turn_angle / elapsed
		summary = {"settings": cfg.duplicate(true), "simulated_seconds": elapsed, "lineages": outcomes}
		if solo_probe:
			summary.lineages = [outcomes[0]]
		elif food_competition:
			var margin: int = outcomes[0].collected - outcomes[1].collected
			summary.payoffs = [margin, -margin]
			summary.rocks = circle_rocks.duplicate(true)
			summary.pellets_remaining = foods.size() - outcomes[0].collected - outcomes[1].collected
	if not output_path.is_empty():
		var file := FileAccess.open(output_path, FileAccess.WRITE)
		if file == null:
			fail("Cannot write summary: " + output_path)
			return
		file.store_string(JSON.stringify(summary, "  ") + "\n")
	last_summary = summary
	if batch_mode:
		return
	print(JSON.stringify(summary))
	if numeric_mode:
		get_tree().quit()

func update_hud() -> void:
	if numeric_mode:
		return
	if solo_probe:
		hud.text = "SOLO FORAGING | seed %d\n%.1f / %.0fs\nPellets collected: %d" % [int(cfg.seed), elapsed, cfg.duration, agents[0].collected]
		return
	if foraging_duel:
		var a: Dictionary = agents[1] if cfg.get("swap_starts", false) else agents[0]
		var b: Dictionary = agents[0] if cfg.get("swap_starts", false) else agents[1]
		var colors := "teal A, red B" if cfg.get("swap_starts", false) else "red A, teal B"
		if food_competition:
			hud.text = "FOOD COMPETITION | %s\n%.1f / %.0fs | seed %d\nPellets collected: A %d, B %d\nPayoffs: A %+d, B %+d\nR repeat | N next map | Space pause" % [colors, elapsed, cfg.duration, int(cfg.seed), a.collected, b.collected, a.collected - b.collected, b.collected - a.collected]
		else:
			hud.text = "FORAGING DUEL | %s\n%.1f / %.0fs | seed %d\nA: %d held, %d collected, %d stolen\nB: %d held, %d collected, %d stolen\nR repeat | N next map | Space pause" % [colors, elapsed, cfg.duration, int(cfg.seed), a.balance, a.collected, a.stolen, b.balance, b.collected, b.stolen]
		if finished:
			hud.text += "\nFINISHED"
		return
	var fraction := clampf(elapsed / float(cfg.duration), 0, 1)
	hud.text = "PURSUIT LAB | seed %d | %.1f / %.0fs | %dx\nOne predator (red), one prey (teal)\nPellets %d | Captured %s\nPrey score %.2f | Predator score %.2f\nSpace pause | R repeat | N new terrain\n1/2/4 speed | Esc quit\n360-degree radius sensing through terrain\nCapture ends the episode; no respawning." % [int(cfg.seed), elapsed, cfg.duration, int(time_scale), meals, str(captures > 0), meals + 5 * fraction, captures * (2 - fraction)]
	if cfg.has("evolution_label"):
		hud.text += "\n" + str(cfg.evolution_label)
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
