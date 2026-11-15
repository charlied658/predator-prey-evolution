extends "res://scripts/agent.gd"

# Two capsules share a joint. The head is the movement and pellet pickup point.
var front_length: float
var rear_length: float
var front_radius: float
var rear_radius: float
var max_bend: float
var bend := 0.0
var heading := 0.0
var swimming := false
var meshes: Array[MeshInstance3D] = []

func configure(genes: Array, color: Color, numeric: bool) -> void:
	front_length = lerpf(0.55, 1.25, genes[0])
	rear_length = lerpf(0.65, 1.65, genes[1])
	var ratio := lerpf(0.65, 1.45, genes[2])
	# Solve for a fixed sum of capsule areas before overlap at the joint.
	var quadratic := PI * (1 + ratio * ratio)
	var linear := 2 * (front_length + ratio * rear_length)
	front_radius = 3.6 / (linear + sqrt(linear * linear + 4 * quadratic * 1.8))
	rear_radius = ratio * front_radius
	max_bend = deg_to_rad(lerpf(20, 100, genes[3]))
	if numeric:
		return
	var material := StandardMaterial3D.new()
	material.albedo_color = color
	for segment in range(2):
		var radius := front_radius if segment == 0 else rear_radius
		var length := front_length if segment == 0 else rear_length
		var capsule := CapsuleMesh.new()
		capsule.radius = radius
		capsule.height = length + 2 * radius
		var visual := MeshInstance3D.new()
		visual.mesh = capsule
		visual.material_override = material
		add_child(visual)
		meshes.append(visual)

func direction(angle: float) -> Vector3:
	return Vector3(cos(angle), 0, sin(angle))

func update_visuals() -> void:
	var joint := -direction(heading) * front_length
	var tail := -direction(heading + bend) * rear_length
	if not meshes.is_empty():
		meshes[0].position = joint / 2
		meshes[0].quaternion = Quaternion(Vector3.UP, joint.normalized())
		meshes[1].position = joint + tail / 2
		meshes[1].quaternion = Quaternion(Vector3.UP, tail.normalized())

func pose_clear(head: Vector3, angle: float, joint_bend: float) -> bool:
	var joint := head - direction(angle) * front_length
	var tail := joint - direction(angle + joint_bend) * rear_length
	return segment_clear(head, joint, front_radius) and segment_clear(joint, tail, rear_radius)

func segment_clear(start: Vector3, end: Vector3, radius: float) -> bool:
	var limit := ARENA_LIMIT - radius
	if maxf(absf(start.x), absf(end.x)) > limit or maxf(absf(start.z), absf(end.z)) > limit:
		return false
	var travel := end - start
	var length_squared := travel.length_squared()
	for rock in obstacles:
		var offset := Vector3(rock[0], 0, rock[1]) - start
		var nearest := travel * clampf(offset.dot(travel) / length_squared, 0, 1)
		if (offset - nearest).length_squared() < pow(float(rock[2]) + radius, 2):
			return false
	return true

func advance_pose(delta: float, target_heading: float, target_bend: float) -> void:
	var origin := position
	var bend_before := bend
	var bend_change := clampf(target_bend - bend, -2.5 * delta, 2.5 * delta)
	var turn_change := wrapf(target_heading - heading, -PI, PI)
	var travel := velocity * delta
	# Take two small pose steps. A blocked segment stops a pose change,
	# while the other joint or a tangential translation can still move.
	for step_index in range(2):
		var next_heading := heading + turn_change / 2
		if pose_clear(position, next_heading, bend):
			heading = next_heading
		var next_bend := bend + bend_change / 2
		if pose_clear(position, heading, next_bend):
			bend = next_bend
		if pose_clear(position + travel / 2, heading, bend):
			position += travel / 2
		else:
			for axis in [Vector3.RIGHT, Vector3.BACK]:
				var slide: Vector3 = axis * travel.dot(axis) / 2
				if pose_clear(position + slide, heading, bend):
					position += slide
	var actual := position - origin
	velocity = actual / delta
	if swimming:
		# Approximate water resistance and forward thrust from lateral tail sweep.
		# Only bend that actually happened contributes, including when rocks block it.
		var sweep_speed := absf(bend - bend_before) * rear_length / delta
		var thrust := minf(12.0, 2.2 * sweep_speed * sweep_speed)
		velocity = velocity * exp(-1.8 * delta) + direction(heading) * thrust * delta
		velocity = velocity.limit_length(4.0)
	path_length += actual.length()
	if actual.length() > 0.001:
		if previous_motion.length() > 0.001:
			turn_angle += previous_motion.angle_to(actual)
		previous_motion = actual
	update_visuals()
