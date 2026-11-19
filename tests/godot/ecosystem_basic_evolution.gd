extends SceneTree

func _initialize() -> void:
	call_deferred("run_check")

func run_check() -> void:
	var world = load("res://ecosystem.tscn").instantiate()
	root.add_child(world)
	world.set_physics_process(false)
	for agent in world.agents:
		agent.cooldown = 100.0
	var prey: Dictionary = world.agents[2]
	prey.cooldown = 0.0
	prey.body.position = Vector3.ZERO
	prey.energy = 80.0
	prey.wander = 0.0
	prey.wander_timer = 100.0
	world.rocks = []
	prey.body.obstacles = []
	for food in world.foods:
		food.cooldown = 100.0
	world.foods[0].cooldown = 0.0
	world.foods[0].pos = Vector3(0, 0, -2)
	prey.heading = -PI / 2
	world.cfg.behavior = {"prey": {"food_threshold": 40.0}}
	var not_hungry: Vector3 = world.desired_direction(prey, 1.0 / 60)
	world.cfg.behavior.prey.food_threshold = 98.0
	var hungry: Vector3 = world.desired_direction(prey, 1.0 / 60)
	if not_hungry.distance_to(Vector3.RIGHT) > 0.00001 or hungry.distance_to(Vector3.FORWARD) > 0.00001:
		push_error("Changing the food-threshold gene must change the actual controller")
		quit(1)
		return
	print("Basic evolution controller check: PASS")
	quit()
