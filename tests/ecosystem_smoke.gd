extends SceneTree

var world
var failures := 0

func _initialize() -> void:
	call_deferred("run_checks")

func check(condition: bool, message: String) -> void:
	if not condition:
		push_error(message)
		failures += 1

func run_checks() -> void:
	world = load("res://ecosystem.tscn").instantiate()
	root.add_child(world)
	check(world.agents.size() == 14, "Default population")
	check(world.rocks.size() == 14 and world.foods.size() == 28, "Default resources")
	var first_position: Vector3 = world.agents[0].body.position
	var first_rock: Array = world.rocks[0].duplicate()
	world.reset_world()
	check(world.agents[0].body.position == first_position and world.rocks[0] == first_rock, "Seeded reset must repeat")
	var predator: Dictionary = world.agents[0]
	predator.body.position = Vector3.ZERO
	predator.heading = 0.0
	world.rocks = []
	check(world.sees(predator, Vector3(4, 0, 0)), "Visible target in front")
	check(not world.sees(predator, Vector3(-4, 0, 0)), "FOV hides target behind")
	check(not world.sees(predator, Vector3(12, 0, 0)), "Vision has limited range")
	world.rocks = [[2.0, 0.0, 0.8]]
	check(not world.sees(predator, Vector3(4, 0, 0)), "Rock occludes target")
	check(not world.unobstructed(Vector3.ZERO, Vector3(4, 0, 0)), "Rock blocks capture segment")
	world.rocks = []
	for agent in world.agents:
		agent.cooldown = 100.0
	var prey: Dictionary = world.agents[2]
	predator.cooldown = 0.0
	prey.cooldown = 0.0
	predator.age = 10.0
	prey.age = 10.0
	predator.energy = 40.0
	prey.energy = 40.0
	predator.body.position = Vector3.ZERO
	prey.body.position = Vector3(0.8, 0, 0)
	world.step(1.0 / 60)
	check(world.captures == 1 and prey.cooldown > 0, "Single capture deactivates prey")
	check(predator.energy > 99, "Capture replenishes energy")
	predator.body.position = Vector3(-10, 0, -10)
	predator.energy = 0.001
	world.step(1.0 / 60)
	check(world.starvations.predator == 1 and predator.cooldown > 0, "Starvation deactivates predator")
	world.step(float(world.cfg.respawn_delay) + 0.1)
	check(predator.cooldown <= 0 and predator.energy == 100, "Delayed respawn restores energy")
	check(predator.body.position.distance_to(world.home_position("predator")) <= world.cfg.home_radius, "Respawn stays in home base")
	# Isolated foraging and regrowth check.
	for agent in world.agents:
		agent.cooldown = 100.0
	prey.cooldown = 0.0
	prey.body.position = Vector3.ZERO
	prey.energy = 20.0
	world.foods[0].pos = Vector3.ZERO
	world.foods[0].visual.position = Vector3.ZERO
	world.foods[0].cooldown = 0.0
	var eaten: int = world.meals
	world.step(1.0 / 60)
	check(world.meals == eaten + 1 and prey.energy > 50, "Pellet restores energy")
	check(world.foods[0].cooldown > 0 and not world.foods[0].visual.visible, "Consumed pellet disappears")
	world.step(float(world.cfg.food_regrowth) + 0.1)
	check(world.foods[0].cooldown <= 0 and world.foods[0].visual.visible, "Food regrows")
	print("Ecosystem mechanic checks: ", "PASS" if failures == 0 else "FAIL")
	quit(0 if failures == 0 else 1)
