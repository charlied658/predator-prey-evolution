extends Node3D

@onready var predator: Node3D = $Predator
@onready var prey: Node3D = $Prey
@export var capture_distance: float = 1.0
@export var encounter_duration: float = 20.0
var encounters: Array = []
var results: Array = []
var current_encounter := 0
var elapsed_time := 0.0
var batch_mode := false
var encounter_active := false
var output_path := ""
var minimum_distance := 0.0
var predator_boundary_time := 0.0
var prey_boundary_time := 0.0
var predator_path_length := 0.0
var initial_distance := 0.0
var distance_integral := 0.0
var approach_integral := 0.0

func _ready() -> void:
	var args = OS.get_cmdline_user_args()
	if args.size() >= 3 and args[0] == "--batch":
		batch_mode = true
		output_path = args[2]
		var file = FileAccess.open(args[1], FileAccess.READ)
		if file == null:
			fail("Cannot open batch input")
			return
		var data = JSON.parse_string(file.get_as_text())
		if not data is Dictionary or not data.get("encounters") is Array:
			fail("Invalid batch JSON")
			return
		encounters = data["encounters"]
	elif args.size() >= 2 and args[0] == "--replay":
		var file = FileAccess.open(args[1], FileAccess.READ)
		if file == null:
			fail("Cannot open replay")
			return
		var data = JSON.parse_string(file.get_as_text())
		if not data is Dictionary:
			fail("Invalid replay JSON")
			return
		encounters = [data.get("replay_encounter", {"predator_genome": data.get("genome", []), "prey_genome": data.get("prey_genome", [])})]
	else:
		return
	if encounters.is_empty():
		fail("Batch contains no encounters")
		return
	start_encounter()

func fail(message: String) -> void:
	push_error(message)
	get_tree().quit(1)

func start_encounter() -> void:
	var config = encounters[current_encounter]
	predator.set_genome(config.get("predator_genome", []))
	prey.set_genome(config.get("prey_genome", []))
	predator.controller = "neural"
	predator.sensor_mode = config.get("predator_sensor_mode", "legacy")
	prey.sensor_mode = "legacy"
	prey.controller = config.get("prey_controller", "neural")
	if predator.genome.size() != 66 or (prey.controller == "neural" and prey.genome.size() != 66):
		fail("Neural controllers require 66 weights")
		return
	if not prey.controller in ["neural", "stationary", "straight", "flee"]:
		fail("Unknown prey controller")
		return
	var p = config.get("predator_start", [-5, 0])
	var q = config.get("prey_start", [5, 0])
	predator.global_position = Vector3(p[0], 0, p[1])
	prey.global_position = Vector3(q[0], 0, q[1])
	predator.velocity = Vector3.ZERO
	prey.velocity = Vector3.ZERO
	elapsed_time = 0.0
	minimum_distance = predator.global_position.distance_to(prey.global_position)
	initial_distance = minimum_distance
	distance_integral = 0.0
	approach_integral = 0.0
	predator_boundary_time = 0.0
	prey_boundary_time = 0.0
	predator_path_length = 0.0
	encounter_active = true

func _physics_process(delta: float) -> void:
	if not encounter_active:
		return
	# Both decisions observe the same positions and previous-step velocities.
	var pv: Vector3 = predator.choose_velocity()
	var qv: Vector3 = prey.choose_velocity()
	predator.velocity = pv
	prey.velocity = qv
	var previous: Vector3 = predator.global_position
	var toward: Vector3 = (prey.global_position - previous).normalized()
	predator.advance(delta)
	prey.advance(delta)
	predator_path_length += previous.distance_to(predator.global_position)
	# Actual predator movement toward the target, excluding prey movement and wall pushing.
	approach_integral += (predator.global_position - previous).dot(toward) / predator.speed
	elapsed_time += delta
	if maxf(absf(predator.position.x), absf(predator.position.z)) >= 13.9:
		predator_boundary_time += delta
	if maxf(absf(prey.position.x), absf(prey.position.z)) >= 13.9:
		prey_boundary_time += delta
	var distance = predator.global_position.distance_to(prey.global_position)
	minimum_distance = minf(minimum_distance, distance)
	distance_integral += distance * delta
	if distance <= capture_distance or elapsed_time + 0.000001 >= encounter_duration:
		end_encounter(distance <= capture_distance, distance)

func end_encounter(captured: bool, distance: float) -> void:
	encounter_active = false
	results.append({
		"encounter": current_encounter,
		"captured": captured, "time": elapsed_time,
		"final_distance": distance, "minimum_distance": minimum_distance,
		"initial_distance": initial_distance,
		"mean_distance": distance_integral / elapsed_time,
		"approach_alignment": approach_integral / elapsed_time,
		"predator_boundary_time": predator_boundary_time,
		"prey_boundary_time": prey_boundary_time,
		"predator_path_length": predator_path_length,
		"predator_fitness": 100.0 + encounter_duration - elapsed_time if captured else 0.0,
		"prey_fitness": elapsed_time
	})
	current_encounter += 1
	if current_encounter < encounters.size():
		start_encounter()
	elif batch_mode:
		var file = FileAccess.open(output_path, FileAccess.WRITE)
		if file == null:
			fail("Cannot write results")
			return
		file.store_string(JSON.stringify({"results": results}))
		file.close()
		get_tree().quit()
	else:
		print(results[0])
		get_tree().quit()
