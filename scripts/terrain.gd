extends RefCounted

# Static geometry shared by collision, capture tests, placement, and rendering.
static func expanded(polygon: PackedVector2Array, margin: float) -> PackedVector2Array:
	var outlines := Geometry2D.offset_polygon(polygon, margin, Geometry2D.JOIN_ROUND)
	return outlines[0] if not outlines.is_empty() else polygon

static func hit(start: Vector2, end: Vector2, polygons: Array) -> Dictionary:
	var best := {"fraction": 1.0, "normal": Vector2.ZERO}
	var travel := end - start
	if travel.length_squared() < 0.000000001:
		return best
	for polygon in polygons:
		for i in range(polygon.size()):
			var a: Vector2 = polygon[i]
			var b: Vector2 = polygon[(i + 1) % polygon.size()]
			var contact = Geometry2D.segment_intersects_segment(start, end, a, b)
			if contact == null:
				continue
			var fraction: float = (contact - start).dot(travel) / travel.length_squared()
			if fraction < best.fraction:
				var edge := b - a
				var normal := Vector2(-edge.y, edge.x).normalized()
				if normal.dot(travel) > 0:
					normal = -normal
				best = {"fraction": maxf(0, fraction), "normal": normal}
	return best

static func mesh(polygon: PackedVector2Array) -> ArrayMesh:
	var vertices := PackedVector3Array()
	for p in polygon:
		vertices.append(Vector3(p.x, 0.1, p.y))
	var arrays := []
	arrays.resize(Mesh.ARRAY_MAX)
	arrays[Mesh.ARRAY_VERTEX] = vertices
	arrays[Mesh.ARRAY_INDEX] = Geometry2D.triangulate_polygon(polygon)
	var result := ArrayMesh.new()
	result.add_surface_from_arrays(Mesh.PRIMITIVE_TRIANGLES, arrays)
	return result
