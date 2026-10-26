extends Node3D

@export var opponent: Node3D
@export var speed: float = 3.0

const ARENA_LIMIT: float = 14.0

var genome: Array = []
var velocity: Vector3 = Vector3.ZERO


func set_genome(new_genome: Array) -> void:
	genome = new_genome


func _physics_process(delta: float) -> void:
	if opponent == null or genome.size() != 66:
		return

	# Sensory inputs
	var relative_position = opponent.global_position - global_position

	var inputs = [
		relative_position.x / 30.0,
		relative_position.z / 30.0,
		opponent.velocity.x / 3.0,
		opponent.velocity.z / 3.0,
		relative_position.length() / 30.0
	]

	# Hidden layer: 5 inputs -> 8 neurons
	var hidden = []
	var index = 0

	for i in range(8):
		var value = 0.0

		for j in range(5):
			value += inputs[j] * genome[index]
			index += 1

		value += genome[index]
		index += 1

		hidden.append(tanh(value))

	# Output layer: 8 neurons -> 2 outputs
	var outputs = []

	for i in range(2):
		var value = 0.0

		for j in range(8):
			value += hidden[j] * genome[index]
			index += 1

		value += genome[index]
		index += 1

		outputs.append(tanh(value))

	# Convert network outputs into movement
	var direction = Vector3(outputs[0], 0, outputs[1])

	if direction.length() > 1.0:
		direction = direction.normalized()

	velocity = direction * speed

	global_position += velocity * delta

	global_position.x = clampf(global_position.x, -ARENA_LIMIT, ARENA_LIMIT)
	global_position.z = clampf(global_position.z, -ARENA_LIMIT, ARENA_LIMIT)
