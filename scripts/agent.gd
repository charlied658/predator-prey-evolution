extends Node3D

# Locomotion and obstacle sensing only. Behavior lives in ecosystem.gd.

const ARENA_LIMIT: float = 14.0
const BODY_RADIUS: float = 0.5
var velocity: Vector3 = Vector3.ZERO
var obstacles: Array = []
var blocked_time := 0.0
var turn_angle := 0.0
var path_length := 0.0
var obstacle_time := 0.0
var previous_motion := Vector3.ZERO

func reset_metrics() -> void:
	blocked_time = 0.0
	turn_angle = 0.0
	path_length = 0.0
	obstacle_time = 0.0
	previous_motion = Vector3.ZERO

# Normalized ray clearance to arena walls and circular obstacles.
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
	# Observe actual motion after wall/rock contacts.
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
