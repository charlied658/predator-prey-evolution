extends Node3D

const Terrain = preload("res://scripts/terrain.gd")
const ARENA_LIMIT := 13.5
const BODY_RADIUS := 0.5
var velocity := Vector3.ZERO
# Inflated once at world generation to include the body radius.
var obstacles: Array = []
var blocked_time := 0.0
var turn_angle := 0.0
var path_length := 0.0
var obstacle_time := 0.0
var previous_motion := Vector3.ZERO
var boundary := PackedVector2Array([Vector2(-13.5, -13.5), Vector2(13.5, -13.5), Vector2(13.5, 13.5), Vector2(-13.5, 13.5)])

func reset_metrics() -> void:
	blocked_time = 0.0
	turn_angle = 0.0
	path_length = 0.0
	obstacle_time = 0.0
	previous_motion = Vector3.ZERO

func clearance(direction: Vector3, maximum: float = 6.0) -> float:
	var distance := maximum
	for axis in [0, 2]:
		if absf(direction[axis]) > 0.00001:
			var edge := ARENA_LIMIT if direction[axis] > 0 else -ARENA_LIMIT
			distance = minf(distance, (edge - global_position[axis]) / direction[axis])
	var start := Vector2(global_position.x, global_position.z)
	var end := start + Vector2(direction.x, direction.z) * maximum
	distance = minf(distance, float(Terrain.hit(start, end, obstacles).fraction) * maximum)
	return clampf(distance / maximum, 0, 1)

func advance(delta: float) -> void:
	var origin := global_position
	var position_now := Vector2(origin.x, origin.z)
	var remaining := Vector2(velocity.x, velocity.z) * delta
	var colliders := obstacles + [boundary]
	for iteration in range(4):
		if remaining.length_squared() < 0.000000001:
			break
		var contact := Terrain.hit(position_now, position_now + remaining, colliders)
		position_now += remaining * maxf(0, float(contact.fraction) - 0.00001)
		if contact.normal == Vector2.ZERO:
			break
		remaining = (remaining * (1 - float(contact.fraction))).slide(contact.normal)
	global_position = Vector3(clampf(position_now.x, -ARENA_LIMIT, ARENA_LIMIT), 0, clampf(position_now.y, -ARENA_LIMIT, ARENA_LIMIT))
	var actual := global_position - origin
	if actual.length() < 0.001 and velocity.length() > 0.5:
		blocked_time += delta
	velocity = actual / delta
	path_length += actual.length()
	if actual.length() > 0.001:
		if previous_motion.length() > 0.001:
			turn_angle += previous_motion.angle_to(actual)
		previous_motion = actual
