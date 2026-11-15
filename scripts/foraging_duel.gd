extends RefCounted

const INPUTS := 12
const HIDDEN := 4
const GENES := HIDDEN * (INPUTS + 1) + 2 * (HIDDEN + 1)
const ARTICULATED_GENES := GENES + HIDDEN + 1
const SWIMMING_GENES := GENES + 2 * HIDDEN
const TAG_COOLDOWN := 1.5

static func action(agent: Dictionary, other: Dictionary, foods: Array, elapsed: float) -> Vector3:
	var facing := Vector3(cos(agent.heading), 0, sin(agent.heading))
	var right := Vector3(-facing.z, 0, facing.x)
	var offset := Vector3.ZERO
	var velocity := Vector3.ZERO
	var rival_has_food := false
	if not other.is_empty():
		offset = other.body.position - agent.body.position
		velocity = other.body.velocity - agent.body.velocity
		rival_has_food = other.balance > 0
	var food_offset := Vector3.ZERO
	var nearest := INF
	for food in foods:
		var candidate: Vector3 = food.pos - agent.body.position
		if food.cooldown <= 0 and candidate.length_squared() < nearest:
			nearest = candidate.length_squared()
			food_offset = candidate
	var inputs := PackedFloat64Array([
		offset.dot(facing) / 28, offset.dot(right) / 28,
		velocity.dot(facing) / 8, velocity.dot(right) / 8,
		food_offset.dot(facing) / 28, food_offset.dot(right) / 28,
		agent.body.velocity.length() / 4, float(nearest < INF),
		agent.body.clearance(facing, 3), agent.body.clearance(facing.rotated(Vector3.UP, 0.8), 3),
		agent.body.clearance(facing.rotated(Vector3.UP, -0.8), 3), float(rival_has_food),
	])
	var genes: Array = agent.genome
	if genes.size() == SWIMMING_GENES:
		# A clock gives the controller a way to alternate strokes.
		var phase := TAU * elapsed * 1.2
		inputs.append(sin(phase))
		inputs.append(cos(phase))
	var hidden := PackedFloat64Array()
	var index := 0
	for neuron in range(HIDDEN):
		var value: float = genes[index]
		index += 1
		for input in inputs:
			value += input * float(genes[index])
			index += 1
		hidden.append(tanh(value))
	var outputs := Vector3.ZERO
	for output in range(3 if genes.size() == ARTICULATED_GENES else 2):
		var value: float = genes[index]
		index += 1
		for activation in hidden:
			value += activation * float(genes[index])
			index += 1
		outputs[output] = tanh(value)
	if genes.size() == SWIMMING_GENES:
		return Vector3(outputs.x, 0, outputs.y)
	return Vector3(outputs.x, (outputs.y + 1) / 2, outputs.z)

static func collect(agents: Array, foods: Array, regrowth: float) -> void:
	for food in foods:
		if food.cooldown > 0:
			continue
		var a: float = agents[0].body.position.distance_squared_to(food.pos)
		var b: float = agents[1].body.position.distance_squared_to(food.pos) if agents.size() > 1 else INF
		if minf(a, b) >= 0.49 or is_equal_approx(a, b):
			continue
		var winner: Dictionary = agents[0] if a < b else agents[1]
		winner.balance += 1
		winner.collected += 1
		food.cooldown = regrowth
		food.visual.visible = false

static func tag(agents: Array, starts: Array, can_tag: bool) -> bool:
	if not can_tag:
		return false
	var axis: Vector3 = (starts[1] - starts[0]).normalized()
	var approach_a: float = (agents[0].body.position - starts[0]).dot(axis)
	var approach_b: float = (starts[1] - agents[1].body.position).dot(axis)
	if maxf(approach_a, approach_b) < 0.005 or absf(approach_a - approach_b) < 0.002:
		return false
	var attacker: Dictionary = agents[0] if approach_a > approach_b else agents[1]
	var victim: Dictionary = agents[1] if approach_a > approach_b else agents[0]
	if victim.balance == 0:
		return false
	attacker.balance += 1
	attacker.stolen += 1
	victim.balance -= 1
	victim.lost += 1
	return true
