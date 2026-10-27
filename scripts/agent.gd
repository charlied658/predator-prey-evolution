extends Node3D

@export var opponent: Node3D
@export var speed: float = 3.0
const ARENA_LIMIT: float = 14.0
const BODY_RADIUS: float = 0.5
var genome: Array = []
var velocity: Vector3 = Vector3.ZERO
var controller: String = "neural"
var sensor_mode: String = "legacy"
var obstacles: Array = []
var blocked_time := 0.0
var turn_angle := 0.0
var path_length := 0.0
var obstacle_time := 0.0
var previous_motion := Vector3.ZERO

func set_genome(new_genome: Array) -> void:
	genome = new_genome

func reset_metrics() -> void:
	blocked_time = 0.0
	turn_angle = 0.0
	path_length = 0.0
	obstacle_time = 0.0
	previous_motion = Vector3.ZERO

# Eight radial range sensors measure walls and circular obstacles at body clearance.
func clearance(direction: Vector3, maximum: float = 6.0) -> float:
	var distance := maximum
	for axis in [0, 2]:
		if absf(direction[axis]) > 0.00001:
			var edge := ARENA_LIMIT if direction[axis] > 0 else -ARENA_LIMIT
			distance = minf(distance, (edge - global_position[axis]) / direction[axis])
	for obstacle in obstacles:
		var offset := global_position - Vector3(obstacle[0], 0, obstacle[1])
		var radius: float = obstacle[2] + BODY_RADIUS
		var b := offset.dot(direction)
		var c := offset.length_squared() - radius * radius
		var discriminant := b * b - c
		if discriminant >= 0:
			var hit := -b - sqrt(discriminant)
			if hit >= -0.0001:
				distance = minf(distance, maxf(0, hit))
	return clampf(distance, 0, maximum) / maximum

func choose_velocity() -> Vector3:
	if opponent == null or controller == "stationary":
		return Vector3.ZERO
	if controller == "straight":
		return Vector3(0, 0, speed)
	if controller == "flee":
		return (global_position - opponent.global_position).normalized() * speed
	if genome.size() != 66 and genome.size() != 218:
		return Vector3.ZERO
	var relative_position = opponent.global_position - global_position
	var sensed_position = relative_position / 30.0
	if sensor_mode in ["bearing", "ecology"]:
		sensed_position = relative_position.normalized()
	var inputs = [sensed_position.x, sensed_position.z, opponent.velocity.x / 3.0,
		opponent.velocity.z / 3.0, relative_position.length() / 30.0]
	var hidden_count := 8
	if genome.size() == 218:
		hidden_count = 12
		inputs.append(velocity.x / speed)
		inputs.append(velocity.z / speed)
		for ray in range(8):
			var angle := TAU * ray / 8.0
			inputs.append(clearance(Vector3(cos(angle), 0, sin(angle))))
	var hidden = []
	var index := 0
	for i in range(hidden_count):
		var value := 0.0
		for j in range(inputs.size()):
			value += inputs[j] * genome[index]
			index += 1
		value += genome[index]
		index += 1
		hidden.append(tanh(value))
	var outputs = []
	for i in range(2):
		var value := 0.0
		for j in range(hidden_count):
			value += hidden[j] * genome[index]
			index += 1
		value += genome[index]
		index += 1
		outputs.append(tanh(value))
	var direction := Vector3(outputs[0], 0, outputs[1])
	if direction.length() > 1:
		direction = direction.normalized()
	return direction * speed

func advance(delta: float) -> void:
	var origin := global_position
	var commanded_speed := velocity.length()
	var remaining := velocity * delta
	var position_now := origin
	# Swept-circle contacts, followed by tangential sliding. No engine callback ordering.
	for iteration in range(3):
		if remaining.length_squared() < 0.0000000001:
			break
		var fraction := 1.0
		var normal := Vector3.ZERO
		for obstacle in obstacles:
			var center := Vector3(obstacle[0], 0, obstacle[1])
			var offset := position_now - center
			var radius: float = obstacle[2] + BODY_RADIUS
			var a := remaining.length_squared()
			var b := offset.dot(remaining)
			var c := offset.length_squared() - radius * radius
			var discriminant := b * b - a * c
			if b < 0 and discriminant >= 0:
				var hit := (-b - sqrt(discriminant)) / a
				if hit >= -0.00001 and hit < fraction:
					fraction = maxf(hit, 0)
					normal = (position_now + remaining * fraction - center).normalized()
		position_now += remaining * (fraction if normal == Vector3.ZERO else maxf(0, fraction - 0.00001))
		if normal == Vector3.ZERO:
			break
		remaining = (remaining * (1 - fraction)).slide(normal)
	global_position = Vector3(clampf(position_now.x, -ARENA_LIMIT, ARENA_LIMIT), 0,
		clampf(position_now.z, -ARENA_LIMIT, ARENA_LIMIT))
	var actual := global_position - origin
	# Legacy observations retain commanded velocities; ecology observes actual post-contact motion.
	if sensor_mode == "ecology":
		velocity = actual / delta
	path_length += actual.length()
	if actual.length() > 0.001:
		if previous_motion.length() > 0.001:
			turn_angle += previous_motion.angle_to(actual)
		previous_motion = actual
	elif commanded_speed > 0.5:
		blocked_time += delta
	for obstacle in obstacles:
		if global_position.distance_to(Vector3(obstacle[0], 0, obstacle[1])) < obstacle[2] + 1.5:
			obstacle_time += delta
			break
