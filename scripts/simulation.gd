extends Node3D

@onready var predator: Node3D = $Predator
@onready var prey: Node3D = $Prey

@export var capture_distance: float = 1.0
@export var encounter_duration: float = 20.0

var encounters: Array = []
var results: Array = []

var current_encounter: int = 0
var elapsed_time: float = 0.0
var batch_mode: bool = false
var encounter_active: bool = false

var output_path: String = ""


func _ready() -> void:
	var args = OS.get_cmdline_user_args()

	if args.size() >= 2 and args[0] == "--batch":
		batch_mode = true
		load_batch(args[1], args[2] if args.size() >= 3 else "user://results.json")

	elif args.size() >= 2 and args[0] == "--replay":
		load_replay(args[1])

	else:
		encounter_active = false


func load_replay(file_path: String) -> void:
	var file = FileAccess.open(file_path, FileAccess.READ)

	if file == null:
		push_error("Cannot open replay file: " + file_path)
		return

	var data = JSON.parse_string(file.get_as_text())

	predator.set_genome(data["genome"])
	prey.set_genome(data["prey_genome"])

	predator.global_position = Vector3(-5, 0, 0)
	prey.global_position = Vector3(5, 0, 0)

	predator.velocity = Vector3.ZERO
	prey.velocity = Vector3.ZERO

	elapsed_time = 0.0
	encounter_active = true

func load_batch(input_path: String, result_path: String) -> void:
	output_path = result_path

	var file = FileAccess.open(input_path, FileAccess.READ)

	if file == null:
		push_error("Cannot open batch file: " + input_path)
		get_tree().quit(1)
		return

	var data = JSON.parse_string(file.get_as_text())

	if not data is Dictionary or not data.has("encounters"):
		push_error("Invalid batch JSON.")
		get_tree().quit(1)
		return

	encounters = data["encounters"]

	if encounters.is_empty():
		push_error("Batch contains no encounters.")
		get_tree().quit(1)
		return

	start_encounter()


func start_encounter() -> void:
	var config = encounters[current_encounter]

	predator.set_genome(config["predator_genome"])
	prey.set_genome(config["prey_genome"])

	predator.global_position = Vector3(-5, 0, 0)
	prey.global_position = Vector3(5, 0, 0)

	predator.velocity = Vector3.ZERO
	prey.velocity = Vector3.ZERO

	elapsed_time = 0.0
	encounter_active = true


func _physics_process(delta: float) -> void:
	if not encounter_active:
		return

	elapsed_time += delta

	var distance = predator.global_position.distance_to(
		prey.global_position
	)

	if distance <= capture_distance:
		end_encounter(true, distance)

	elif elapsed_time >= encounter_duration:
		end_encounter(false, distance)


func end_encounter(captured: bool, final_distance: float) -> void:
	encounter_active = false

	var predator_fitness: float = 0.0
	var prey_fitness: float = elapsed_time

	if captured:
		predator_fitness = 100.0 + (encounter_duration - elapsed_time)

	var result = {
		"encounter": current_encounter,
		"captured": captured,
		"time": elapsed_time,
		"final_distance": final_distance,
		"predator_fitness": predator_fitness,
		"prey_fitness": prey_fitness
	}

	results.append(result)

	if batch_mode:
		current_encounter += 1

		if current_encounter < encounters.size():
			start_encounter()
		else:
			save_results()
			get_tree().quit()
	else:
		print(result)
		get_tree().quit()


func save_results() -> void:
	var file = FileAccess.open(output_path, FileAccess.WRITE)

	if file == null:
		push_error("Cannot write results: " + output_path)
		get_tree().quit(1)
		return

	file.store_string(JSON.stringify({"results": results}, "\t"))
	file.close()
