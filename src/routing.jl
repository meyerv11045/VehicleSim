function populate_heap!(map, heap_handles, road_min_heap, dist_vals)
    for road_segment in map # road_segment is a Pair: [road_id, RoadSegment]
        road_id = road_segment[1]

        handle = push!(road_min_heap, [1e7, road_id])
        heap_handles[road_id] = handle
        dist_vals[road_id] = 1e7
    end
end

"""
Finds the shortest route of road segments from the
current road segment to the target road segment
    start_road_ID (int): road ID of the current road segment
    end_road_ID (int): road ID of the target road segment
    map: map of the environment
"""
function shortest_path(start_road_ID, end_road_ID, map)
    found_shortest_path = false
    heap_handles = Dict{Int, Int}() # key: road ID, value: handle
    dist_vals = Dict{Int, Any}() #key: road ID, value: current distance from start road
    road_min_heap = MutableBinaryMinHeap{Vector{Any}}() # each heap node is [distance value, road ID]
    parent_road = Dict{Int, Int}() # key: road ID, value: parent road

    populate_heap!(map, heap_handles, road_min_heap, dist_vals)
    update!(road_min_heap, heap_handles[start_road_ID], [0, start_road_ID])
    dist_vals[start_road_ID] = 0

    while !found_shortest_path && !isempty(road_min_heap)
        road_length = 0
        src_road = pop!(road_min_heap)[2]
        heap_handles[src_road] = 0 # change road handle to 0 to indicate road not in heap
        adjacent_roads = map[src_road].children
        lane_boundaries = map[src_road].lane_boundaries

        if lane_boundaries[1].curvature != 0 # road length for 90 degree curved road. Road length is calculated using average radius
            road_length = (abs(1 / lane_boundaries[1].curvature - 1 / lane_boundaries[length(lane_boundaries)].curvature) / 2 + min(1 / lane_boundaries[1].curvature, 1 / lane_boundaries[length(lane_boundaries)].curvature)) * pi / 2
        else
            road_length = norm(lane_boundaries[1].pt_a - lane_boundaries[1].pt_b) # road length for straight road
        end

        for road in adjacent_roads # update the distance of adjacent roads
            new_dist_val = dist_vals[src_road] + road_length
            if (heap_handles[road] != 0) && (new_dist_val < dist_vals[road])
                dist_vals[road] = new_dist_val
                update!(road_min_heap, heap_handles[road], [new_dist_val, road])
                parent_road[road] = src_road
            end
        end

        if heap_handles[end_road_ID] == 0
            found_shortest_path = true
        end
    end

    # work backwards to find the route from start road to end road
    route = Vector{Int}()
    current_road = end_road_ID
    pushfirst!(route, current_road)
    while current_road != start_road_ID
        pushfirst!(route, parent_road[current_road])
        current_road = parent_road[current_road]
    end

    return route
end

"""
Finds the map segment id the vehicle is estimated to be in
    positon: [p1, p2] vector of any vehicle's position
    map: map of the environment
"""
function cur_map_segment_of_vehicle(position, map)
    p1, p2 = position

    polygon_edges = [1 2; 2 4; 4 3; 3 1]
    for road in map
        polygon_vertices = Array{Float64}(undef, 0, 2)
        road_id = road[1]
        road_segment = road[2]
        road_boundaries = road_segment.lane_boundaries
        for boundary in road_boundaries
            if boundary.hard_boundary
                polygon_vertices = vcat(polygon_vertices, boundary.pt_a')
                polygon_vertices = vcat(polygon_vertices, boundary.pt_b')
            end
        end
        if road_boundaries[1].curvature == 0 &&
            inpoly2(position, polygon_vertices, polygon_edges)[1] == 1 # for straight roads
            return road_id
        elseif road_boundaries[1].curvature != 0 # for curved roads
            pt_a1 = Point{2}(polygon_vertices[1,:])
            pt_b1 = Point{2}(polygon_vertices[2,:])
            pt_a2 = Point{2}(polygon_vertices[3,:])
            pt_b2 = Point{2}(polygon_vertices[4,:])
            # l1 = GeometryBasics.Line(pt_a1, pt_a2)
            # l2 = GeometryBasics.Line(pt_b1, pt_b2)

            slope1 = (pt_a1[2] - pt_a2[2]) / (pt_a1[1] - pt_a2[1])
            center_point = [0,0]

            if isinf(slope1)
                center_point = [pt_a1[1], pt_b1[2]]
            else
                center_point = [pt_b1[1], pt_a1[2]]
            end

            min_x = minimum([pt_a1[1], pt_a2[1], pt_b1[1], pt_b2[1]])
            max_x = maximum([pt_a1[1], pt_a2[1], pt_b1[1], pt_b2[1]])
            min_y = minimum([pt_a1[2], pt_a2[2], pt_b1[2], pt_b2[2]])
            max_y = maximum([pt_a1[2], pt_a2[2], pt_b1[2], pt_b2[2]])
            max_radius = 0
            min_radius = 1e7
            dist_from_center = norm(position - center_point)

            for boundary in road_boundaries
                max_radius = maximum([max_radius, abs(1 / boundary.curvature)])
                min_radius = minimum([min_radius, abs(1 / boundary.curvature)])
            end

            if (p1 < max_x) &&
                (p1 > min_x) &&
                (p2 < max_y) &&
                (p2 > min_y) &&
                (dist_from_center > min_radius) &&
                (dist_from_center < max_radius)
                return road_id
            end
        end
    end
    @warn "vehicle at $position is not on the road"
    return nothing # vehicle off the road
end

function find_side_of_road(position, current_road_id, map)
    current_road_segment = map[current_road_id]
    road_boundaries = current_road_segment.lane_boundaries

    polygon_edges = [1 2; 2 4; 4 3; 3 1]
    whole_road_vertices = Array{Float64}(undef, 0, 2)
    left_road_vertices = Array{Float64}(undef, 0, 2)
    right_road_vertices = Array{Float64}(undef, 0, 2)

    for boundary in road_boundaries[1:2]
        whole_road_vertices = vcat(whole_road_vertices, boundary.pt_a')
        whole_road_vertices = vcat(whole_road_vertices, boundary.pt_b')
    end

    pt_a1 = Point{2}(whole_road_vertices[1,:])
    pt_b1 = Point{2}(whole_road_vertices[2,:])
    pt_a2 = Point{2}(whole_road_vertices[3,:])
    pt_b2 = Point{2}(whole_road_vertices[4,:])

    midpoint_a = (pt_a1 + pt_a2) / 2
    midpoint_b = (pt_b1 + pt_b2) / 2

    left_road_vertices = vcat(left_road_vertices, pt_a1')
    left_road_vertices = vcat(left_road_vertices, pt_b1')
    left_road_vertices = vcat(left_road_vertices, midpoint_a')
    left_road_vertices = vcat(left_road_vertices, midpoint_b')

    right_road_vertices = vcat(right_road_vertices, midpoint_a')
    right_road_vertices = vcat(right_road_vertices, midpoint_b')
    right_road_vertices = vcat(right_road_vertices, pt_a2')
    right_road_vertices = vcat(right_road_vertices, pt_b2')

    if road_boundaries[1].curvature == 0 # for straight roads
        if inpoly2(position, right_road_vertices, polygon_edges)[2] == 1 && inpoly2(position, left_road_vertices, polygon_edges)[2] == 1
            return "middle"
        elseif inpoly2(position, left_road_vertices, polygon_edges)[1] == 1
            return "left"
        elseif inpoly2(position, right_road_vertices, polygon_edges)[1] == 1
            return "right"
        else
            return "error"
        end
    else # for curved roads
        min_radius = minimum([abs(1 / road_boundaries[1].curvature), abs(1 / road_boundaries[2].curvature)])
        max_radius = maximum([abs(1 / road_boundaries[1].curvature), abs(1 / road_boundaries[2].curvature)])
        mid_radius = (min_radius + max_radius) / 2
        slope1 = (pt_a1[2] - pt_a2[2]) / (pt_a1[1] - pt_a2[1])
        center_point = [0,0]

        if isinf(slope1)
            center_point = [pt_a1[1], pt_b1[2]]
        else
            center_point = [pt_b1[1], pt_a1[2]]
        end

        dist_from_center = norm(position - center_point)

        if abs(dist_from_center - mid_radius) < 0.00001
            return "middle"
        elseif (dist_from_center > min_radius) && (dist_from_center < max_radius)
            if ((dist_from_center > mid_radius) && (abs(1 / road_boundaries[1].curvature) > abs(1 / road_boundaries[2].curvature))) || (((dist_from_center < mid_radius) && (abs(1 / road_boundaries[1].curvature) < abs(1 / road_boundaries[2].curvature))))
                return "left"
            elseif ((dist_from_center > mid_radius) && (abs(1 / road_boundaries[1].curvature) < abs(1 / road_boundaries[2].curvature))) || (((dist_from_center < mid_radius) && (abs(1 / road_boundaries[1].curvature) > abs(1 / road_boundaries[2].curvature))))
                return "right"
            end
        else
            return "error"
        end
    end
end

function find_stop_sign_location_from_route(route, map)
    """
    ...
    Arguments
    - 'route::Vect': shortest route of road segments from the
                     current road segment to the target road segment.
    Returns
    - 'stop::Vect': Roadsegment id corresponding to 
                    lanetype stopsign. 
    ...
    """
    stop = Vect{Int}()
    for id in route  
        if map[id].lane_types == VehicleSim.stop_sign
            push(stop, id)
        end
    end
    return stop
end

function check_current_position_to_stop_sign(curr_road_segment_id, stop_sign_id)
    """
    ...
    Arguments
    - 'curr_road_segment_id::Int : the map segment id the vehicle is estimated to be in
    - 'stop_sign_id::Vect{Int}(): Roadsegment id corresponding to lanetype stopsign. 
    Returns 
    - stop: boolean: to check if current position of the car is at the stop sign.
    ...
    """
    stop = false
    for stopid ∈ stop_sign_id
        if curr_road_segment_id == stopid
            stop = true
        end
    end
    return stop
end