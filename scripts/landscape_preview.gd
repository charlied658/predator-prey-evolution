extends Control

const RESOLUTION := 96
const TITLES := ["Broad outcrops", "Broken ridges", "Connected clearings"]
var seed_value := 42
var coverage := 0.30
var feature_scale := 1.0
var panels: Array = []
var generated: Array = []
var status: Label

func _ready() -> void:
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--seed="):
			seed_value = int(arg.trim_prefix("--seed="))
	get_window().size = Vector2i(1440, 800)
	set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	var background := ColorRect.new()
	background.color = Color("#172321")
	background.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	add_child(background)
	var margin := MarginContainer.new()
	margin.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)
	for side in ["left", "top", "right", "bottom"]:
		margin.add_theme_constant_override("margin_" + side, 24)
	add_child(margin)
	var column := VBoxContainer.new()
	column.add_theme_constant_override("separation", 16)
	margin.add_child(column)
	var title := Label.new()
	title.text = "Landscape explorer"
	title.add_theme_font_size_override("font_size", 30)
	column.add_child(title)
	var controls := HBoxContainer.new()
	controls.add_theme_constant_override("separation", 18)
	column.add_child(controls)
	var seed_label := Label.new()
	seed_label.text = "Seed"
	controls.add_child(seed_label)
	var seed_input := SpinBox.new()
	seed_input.max_value = 1000000
	seed_input.value = seed_value
	seed_input.value_changed.connect(func(value): seed_value = int(value); regenerate())
	controls.add_child(seed_input)
	add_slider(controls, "Solid area", 0.12, 0.44, coverage,
		func(value): coverage = value; regenerate())
	add_slider(controls, "Feature scale", 0.6, 1.6, feature_scale,
		func(value): feature_scale = value; regenerate())
	var export_button := Button.new()
	export_button.text = "Export map masks"
	export_button.pressed.connect(export_maps)
	controls.add_child(export_button)
	var maps := HBoxContainer.new()
	maps.add_theme_constant_override("separation", 22)
	column.add_child(maps)
	for i in range(TITLES.size()):
		var card := VBoxContainer.new()
		card.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		maps.add_child(card)
		var name_label := Label.new()
		name_label.text = TITLES[i]
		name_label.add_theme_font_size_override("font_size", 21)
		card.add_child(name_label)
		var image := TextureRect.new()
		image.custom_minimum_size = Vector2(280, 350)
		image.expand_mode = TextureRect.EXPAND_IGNORE_SIZE
		image.stretch_mode = TextureRect.STRETCH_KEEP_ASPECT_CENTERED
		image.texture_filter = CanvasItem.TEXTURE_FILTER_LINEAR
		card.add_child(image)
		var details := Label.new()
		details.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
		details.add_theme_font_size_override("font_size", 15)
		card.add_child(details)
		panels.append({"image": image, "details": details})
	var legend := Label.new()
	legend.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	legend.text = "Cream = traversable   /   Gray = solid terrain   /   Gold = food   /   Coral and teal = example starts"
	column.add_child(legend)
	status = Label.new()
	status.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
	status.text = "Change the seed or sliders to explore. Starts and food are placed in the connected open region."
	column.add_child(status)
	regenerate()
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--snapshot="):
			await RenderingServer.frame_post_draw
			get_viewport().get_texture().get_image().save_png(arg.trim_prefix("--snapshot="))
			get_tree().quit()

func add_slider(parent: Control, text: String, low: float, high: float, value: float, callback: Callable) -> void:
	var label := Label.new()
	label.text = text
	parent.add_child(label)
	var slider := HSlider.new()
	slider.custom_minimum_size.x = 135
	slider.min_value = low
	slider.max_value = high
	slider.step = 0.02
	slider.value = value
	slider.value_changed.connect(callback)
	parent.add_child(slider)

func make_noise(offset: int, frequency: float) -> FastNoiseLite:
	var noise := FastNoiseLite.new()
	noise.seed = seed_value + offset
	noise.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
	noise.frequency = frequency / feature_scale
	noise.fractal_octaves = 3
	noise.fractal_gain = 0.35
	noise.domain_warp_enabled = true
	noise.domain_warp_amplitude = 12
	noise.domain_warp_frequency = 0.025 / feature_scale
	return noise

func generate(kind: int) -> Dictionary:
	var noise := make_noise(0, 0.045)
	var detail := make_noise(173, 0.075)
	var values := PackedFloat32Array()
	for y in range(RESOLUTION):
		for x in range(RESOLUTION):
			var value: float
			if kind == 1:
				value = noise.get_noise_2d(x * 0.42, y * 1.8)
			elif kind == 2:
				value = 0.75 * noise.get_noise_2d(x * 0.65, y * 0.65) + 0.25 * detail.get_noise_2d(x, y)
			else:
				value = noise.get_noise_2d(x, y)
			values.append(value)
	var sorted := values.duplicate()
	sorted.sort()
	var threshold: float = sorted[int(values.size() * (1 - coverage))]
	var solid := PackedByteArray()
	solid.resize(values.size())
	for index in range(values.size()):
		solid[index] = int(values[index] >= threshold)
	# Disconnected pockets become terrain so every marked open cell is reachable.
	var seen := PackedByteArray()
	seen.resize(values.size())
	var largest := PackedInt32Array()
	for index in range(values.size()):
		if solid[index] or seen[index]:
			continue
		var component := PackedInt32Array([index])
		seen[index] = 1
		var cursor := 0
		while cursor < component.size():
			var current: int = component[cursor]
			cursor += 1
			for step in [Vector2i.LEFT, Vector2i.RIGHT, Vector2i.UP, Vector2i.DOWN]:
				var point: Vector2i = Vector2i(current % RESOLUTION, current / RESOLUTION) + step
				if point.x < 0 or point.y < 0 or point.x >= RESOLUTION or point.y >= RESOLUTION:
					continue
				var neighbor: int = point.y * RESOLUTION + point.x
				if not solid[neighbor] and not seen[neighbor]:
					seen[neighbor] = 1
					component.append(neighbor)
		if component.size() > largest.size():
			largest = component
	solid.fill(1)
	for index in largest:
		solid[index] = 0
	# Markers need clearance from terrain, including arena boundaries.
	var open_points: Array = []
	for y in range(5, RESOLUTION - 5):
		for x in range(5, RESOLUTION - 5):
			var clear := true
			for dy in range(-3, 4):
				for dx in range(-3, 4):
					if solid[(y + dy) * RESOLUTION + x + dx]:
						clear = false
			if clear:
				open_points.append(Vector2i(x, y))
	var rng := RandomNumberGenerator.new()
	rng.seed = seed_value + kind
	var markers: Array = []
	for attempt in range(5000):
		if open_points.is_empty() or markers.size() == 14:
			break
		var candidate: Vector2i = open_points[rng.randi_range(0, open_points.size() - 1)]
		var separated := true
		for existing in markers:
			if candidate.distance_to(existing) < (18 if markers.size() < 2 else 7):
				separated = false
		if separated:
			markers.append(candidate)
	var marker_data: Array = []
	for point in markers:
		marker_data.append([point.x, point.y])
	return {"preset": TITLES[kind], "seed": seed_value, "resolution": RESOLUTION,
		"world_size": 8.0, "coordinate_origin": "top_left", "solid_encoding": "1=blocked, 0=open",
		"solid": Array(solid), "markers": marker_data, "requested_solid_fraction": coverage,
		"feature_scale": feature_scale, "open_fraction": float(largest.size()) / values.size()}

func draw_map(map: Dictionary) -> Image:
	var image := Image.create(RESOLUTION * 4, RESOLUTION * 4, false, Image.FORMAT_RGB8)
	for y in range(image.get_height()):
		for x in range(image.get_width()):
			var index := (y / 4) * RESOLUTION + x / 4
			image.set_pixel(x, y, Color("#6c7774") if map.solid[index] else Color("#e8e4ce"))
	for i in range(map.markers.size()):
		var point: Array = map.markers[i]
		var center := Vector2(point[0] * 4 + 2, point[1] * 4 + 2)
		var radius := 6.5 if i < 2 else 3.5
		var color := Color("#e1755e") if i == 0 else (Color("#239b9d") if i == 1 else Color("#bc921c"))
		for y in range(int(center.y - radius - 1), int(center.y + radius + 2)):
			for x in range(int(center.x - radius - 1), int(center.x + radius + 2)):
				if Vector2(x, y).distance_to(center) <= radius:
					image.set_pixel(x, y, color)
	return image

func regenerate() -> void:
	generated.clear()
	for kind in range(3):
		var map := generate(kind)
		generated.append(map)
		panels[kind].image.texture = ImageTexture.create_from_image(draw_map(map))
		panels[kind].details.text = "%.0f%% connected open area | %d food points" % [
			map.open_fraction * 100, maxi(0, map.markers.size() - 2)]

func export_maps() -> void:
	var directory := "res://results/landscape-previews"
	DirAccess.make_dir_recursive_absolute(directory)
	for i in range(generated.size()):
		var file := FileAccess.open(directory + "/map-%d-seed-%d.json" % [i, seed_value], FileAccess.WRITE)
		if file == null:
			status.text = "Could not write map masks."
			return
		file.store_string(JSON.stringify(generated[i]))
	status.text = "Exported three collision masks to results/landscape-previews."
