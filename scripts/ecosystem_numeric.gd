extends SceneTree

# The same simulator runs many numeric episodes per engine process, without graphics.
func _initialize() -> void:
	call_deferred("run_jobs")

func stop(message: String) -> void:
	push_error(message)
	quit(1)

func run_jobs() -> void:
	var args := OS.get_cmdline_user_args()
	if args.size() != 2:
		stop("Expected jobs JSON and results JSON paths")
		return
	var jobs = JSON.parse_string(FileAccess.get_file_as_string(args[0]))
	if not jobs is Array or jobs.is_empty():
		stop("Expected a nonempty array of episode settings")
		return
	for job in jobs:
		if not job is Dictionary:
			stop("Each episode must be a settings object")
			return
	var world = load("res://ecosystem.tscn").instantiate()
	world.batch_mode = true
	world.cfg = jobs[0]
	root.add_child(world)
	world.set_physics_process(false)
	if world.failed:
		return
	var rows: Array = []
	for index in range(jobs.size()):
		if index > 0:
			world.cfg = jobs[index]
			if not world.validate_config():
				return
			world.reset_world()
			if world.failed:
				return
		for frame in range(int(ceil(float(world.cfg.duration) * 60)) + 1):
			world.step(1.0 / 60)
			if world.finished:
				break
		if world.last_summary.is_empty():
			stop("Episode did not complete")
			return
		rows.append(world.last_summary.duplicate(true))
	var file := FileAccess.open(args[1], FileAccess.WRITE)
	if file == null:
		stop("Cannot write numeric results")
		return
	file.store_string(JSON.stringify(rows))
	file.close()
	quit()
