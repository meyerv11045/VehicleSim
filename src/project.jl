# for estimating the other vehicles
struct SimpleVehicleState
	p1::Float64
	p2::Float64
	θ::Float64
	l::Float64
	w::Float64
	h::Float64
end

struct OtherVehicleStates
	p1::Float64
	p2::Float64
	θ::Float64
	v::Float64
	l::Float64
	w::Float64
	h::Float64
end

# for use with the ego vehicles
struct LocalizationEstimate
	time::Float64
	position::SVector{3, Float64}
	quaternion::SVector{4, Float64}
	linear_vel::SVector{3, Float64}
	angular_vel::SVector{3, Float64}
end

struct Percept
	last_update::Float64
	x::Vector{OtherVehicleStates}
end

function localize(gps_channel, imu_channel, localization_state_channel, shutdown_channel)
	@info "Starting localization task..."

	# TODO: Tune values to improve performance
	# especially the velocity and angular velocity in the process covariance
	# since our dynamics model does not take into account controls

	# Initial state and covariance p(x₀) ~ N(x₀, P₀)
	x₀ = zeros(13)
	x₀[1:3] = [-91, -6, 2.6]
	x₀[7] = 1 # quaternion corresponding to no rotation

	P₀ = Diagonal(ones(13))  # Initial covariance (uncertainty)

	# Process noise covariance
	Q = Diagonal(0.01 * ones(13))

	# Measurement noise covariance (taken from the generator functions)
	imu_variance = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01] .^ 2)
	gps_variance = Diagonal([1.0, 1.0] .^ 2)

	# Initialize the estimate of the state
	x̂ = x₀
	P̂ = P₀

	while true
		sleep(0.001) # prevent thread from hogging resources & freezing other threads
		isready(shutdown_channel) && break

		fresh_gps_meas = []
		while isready(gps_channel)
			meas = take!(gps_channel)
			push!(fresh_gps_meas, meas)
		end
		fresh_imu_meas = []
		while isready(imu_channel)
			meas = take!(imu_channel)
			push!(fresh_imu_meas, meas)
		end

		while length(fresh_gps_meas) > 0 || length(fresh_imu_meas) > 0
			if length(fresh_gps_meas) ≥ length(fresh_imu_meas)
				meas = popfirst!(fresh_gps_meas)
				z = [meas.lat; meas.long]
				R = gps_variance
			else
				meas = popfirst!(fresh_imu_meas)
				z = [meas.linear_vel; meas.angular_vel]
				R = imu_variance
			end

			x̂, P̂ = ekf_step(z, x̂, P̂, Q, R)

			localization_state = LocalizationEstimate(time(), x̂[1:3], x̂[4:7], x̂[8:10], x̂[11:13])
			if isready(localization_state_channel)
				take!(localization_state_channel)
			end
			put!(localization_state_channel, localization_state)
		end
	end
	@info "Terminated localization task."
end

function perception(cam_meas_channel, localization_state_channel, perception_state_channel, shutdown_channel)
	# h(x): Everything here is in pixel space
	# Step 1: Process Model (Loop through perception_state_channel and predict where cars will be in future)
	# Step 2: Transform Resulting future states to image frame
	# Step 3: Run the get_3d_bbox_corners thing to get the bounding box corners for each one
	# Step 4: Pull measurement bounding boxes by getting most recent neasurement from cam1 and cam 2
	# Step 5: Go through each measurement bounding box, and see which future state car is closest: each measurement has a car
	# Step 6: Check to make sure those errors from previous step are reasonable (don't wanna have an instance where we had a car leave frame and enter frame in same timestep and that the mapping is super off)
	# Step 7: If we have extra cars in vehicle state, pop them off because tha car has left the frame
	# Step 8: If we have extra BB Measurements, then we will calculate vehicle state as the following: Find center of bounding box measurments, and apply transform from image to body frame, and store that as new vehicle state
	# Step 9: Run EKF for the rest of them to calculate their vehicle states

	# Easy perception
	# Clear current perception list
	# Pull most recent camera measurements from cam1 and cam2
	# Loop through each bounding box, find center of vehicle in image frame and multiply by body transform
	# Store vehicle in perception state list, but have vehicle be 1.5x
	@info "Starting perception task..."
	vehicle_size = SVector(13.2, 5.7, 5.3)
	num_vehicles = 0
	prev_bbs_and_states = Dict()
	prev_bbs_and_covars = Dict()
	initial_perception_state = Percept(time(), [])
	if isready(perception_state_channel)
		take!(perception_state_channel)
	end
	put!(perception_state_channel, initial_perception_state)
	# set up stuff
	while true
		sleep(0.001) # prevent thread from hogging resources & freezing other threads
		isready(shutdown_channel) && break

		fresh_cam_meas = []
		while isready(cam_meas_channel)
			meas = take!(cam_meas_channel)
			push!(fresh_cam_meas, meas)
		end

		latest_localization_state = fetch(localization_state_channel)
		#@info "Localization"
		#@info latest_localization_state
		vehicle_2_state = LocalizationEstimate(
			1.68188650652e9,
			[-91.6639496981265, -5.001254676640403, 2.6455622444987403],
			[0.7070991651229994, 0.0038137895043521652, -0.003038500205406931, 0.7070975839362454],
			[6.04446209702472e-18, 5.44780420863507e-13, -1.5646262355470607e-13],
			[-1.950424201033533e-13, 2.7176003305590663e-15, 1.4592923169556237e-15],
		)

		vector_localization = [latest_localization_state.position[1:3]; latest_localization_state.quaternion[1:4]]
		vehicle2_1 = vehicle_2_state.position[1]
		vehicle2_2 = vehicle_2_state.position[2]
		vehicle2_3 = atan(
			2 * (vehicle_2_state.quaternion[1] * vehicle_2_state.quaternion[4] + vehicle_2_state.quaternion[2] * vehicle_2_state.quaternion[3]),
			1 - 2 * (vehicle_2_state.quaternion[3]^2 + vehicle_2_state.quaternion[4]^2),
		)
		vehicle2_4 = vehicle_2_state.linear_vel[1]
		vehicle2_5 = vehicle_size[1]
		vehicle2_6 = vehicle_size[2]
		vehicle2_7 = vehicle_size[3]
		vehicle2_meas = h1([vehicle2_1, vehicle2_2, vehicle2_3, vehicle2_4, vehicle2_5, vehicle2_6, vehicle2_7], vector_localization, 640, 480, 0.01, 0.001)
		#@info "Vehicle 2 meas"
		#@info vehicle2_meas
		priors = []
		if (num_vehicles > 0)
			priors = fetch(perception_state_channel).x
		end
		#@info priors
		#@info size(fresh_cam_meas)
		latest_relevant_meas = []
		got_latest_meas = false
		bb_and_prior_map = Dict()
		bb_and_prior_covar_map = Dict()
		image_width = 0
		image_height = 0
		focal_length = 0
		pixel_len = 0
		# Test cases: No vehicles and 2 measurements
		# 2 prior vehicles and 2 measurements
		# 1 prior vehicle and 3 measurements
		# 3 prior vehicles and 1 measurement
		for meas in fresh_cam_meas
			num_meas = size(meas.bounding_boxes)[1]
			#@info size(meas.bounding_boxes)
			if (num_meas % 2 == 0 && num_meas > 0 && meas.camera_id == 2 && !got_latest_meas)
				# Got 2 camera measurements
				latest_relevant_meas = meas
				got_latest_meas = true
				new_vehicles = max(0, num_meas / 2 - num_vehicles)
				# Maps the min bounding box distance to that bounding box idnex
				bb_distances = Dict()
				new_bbs = []
				# Map each measurement to a prior
				for meas_index ∈ 1:2:num_meas
					camera1_bb = latest_relevant_meas.bounding_boxes[meas_index]
					camera2_bb = latest_relevant_meas.bounding_boxes[meas_index+1]
					image_width = latest_relevant_meas.image_width
					image_height = latest_relevant_meas.image_height
					focal_length = latest_relevant_meas.focal_length
					pixel_len = latest_relevant_meas.pixel_length
					full_meas = [camera1_bb; camera2_bb]
					min_distance = 1000000
					min_prev_bb = []
					got_prev = false
					for (prev_bb, prev_state) in prev_bbs_and_states
						got_prev = true
						diff = norm(full_meas - prev_bb)
						if (diff < min_distance)
							min_distance = diff
							min_prev_bb = prev_bb
						end
					end
					if (got_prev)
						bb_distances[min_distance] = full_meas
						bb_and_prior_map[full_meas] = OtherVehicleStates(prev_bbs_and_states[min_prev_bb][1], prev_bbs_and_states[min_prev_bb][2], prev_bbs_and_states[min_prev_bb][3],
							prev_bbs_and_states[min_prev_bb][4], prev_bbs_and_states[min_prev_bb][5], prev_bbs_and_states[min_prev_bb][6], prev_bbs_and_states[min_prev_bb][7])
						bb_and_prior_covar_map[full_meas] = prev_bbs_and_covars[min_prev_bb]
					else
						push!(new_bbs, full_meas)
					end

				end

				sorted_bb_distances_keys = sort(collect(keys(bb_distances)), rev = true)
				# Removes bad box prior pairs
				for repeat_index ∈ 1:trunc(Int, num_meas / 2 - new_vehicles)-1
					push!(new_bbs, bb_distances[sorted_bb_distances_keys[repeat_index]])
					delete!(bb_and_prior_map, bb_distances[sorted_bb_distances_keys[repeat_index]])
				end

				for new_vehicle_index ∈ 1:trunc(Int, new_vehicles)
					the_new_meas_for_new_vehicle = new_bbs[new_vehicle_index]
					the_pixel_width = the_new_meas_for_new_vehicle[4] - the_new_meas_for_new_vehicle[2]
					fake_distance = -3.7 * the_pixel_width + 47.4
					prior_theta = atan(
						2 * (latest_localization_state.quaternion[1] * latest_localization_state.quaternion[4] + latest_localization_state.quaternion[2] * latest_localization_state.quaternion[3]),
						1 - 2 * (latest_localization_state.quaternion[3]^2 + latest_localization_state.quaternion[4]^2),
					)
					prior_p1 = latest_localization_state.position[1] + fake_distance * cos(prior_theta)#vehicle_size[1] * 1.5
					prior_p2 = latest_localization_state.position[2] + fake_distance * sin(prior_theta)#vehicle_size[2] * 1.5

					prior_velocity = latest_localization_state.linear_vel[1]
					prior_l = vehicle_size[1]
					prior_w = vehicle_size[2]
					prior_h = vehicle_size[3]
					prior = OtherVehicleStates(prior_p1, prior_p2, prior_theta, prior_velocity, prior_l, prior_w, prior_h)
					push!(priors, prior)
					bb_and_prior_map[new_bbs[new_vehicle_index]] = prior
					bb_and_prior_covar_map[new_bbs[new_vehicle_index]] = Diagonal([vehicle_size[1]^2; vehicle_size[1]^2; 0.2^2; 0.5^2; 0.0001^2; 0.0001^2; 0.0001^2])
				end
				current_bbs_and_states = Dict()
				current_bbs_and_covars = Dict()
				num_vehicles = length(bb_and_prior_map)
				vehicle_states = []
				for (bb, prior) in bb_and_prior_map
					vector_prior = [prior.p1; prior.p2; prior.θ; prior.v; prior.l; prior.w; prior.h]
					state = vector_prior
					covar = bb_and_prior_covar_map[bb]
					Q = Diagonal([1, 1, 1, 1, 0.0001, 0.0001, 0.0001])
					R = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
					new_x, new_covar = perception_ekf_step(bb, state, covar, Q, R, vector_localization, image_width, image_height, focal_length, pixel_len)
					current_bbs_and_states[bb] = new_x
					current_bbs_and_covars[bb] = new_covar
					vehicle_state = OtherVehicleStates(new_x[1], new_x[2], new_x[3], new_x[4], new_x[5], new_x[6], new_x[7])
					push!(vehicle_states, vehicle_state)
				end
				perception_state = Percept(time(), vehicle_states)
				if isready(perception_state_channel)
					take!(perception_state_channel)
				end
				put!(perception_state_channel, perception_state)
				prev_bbs_and_states = current_bbs_and_states
				prev_bbs_and_covars = current_bbs_and_covars
				#@info meas.bounding_boxes
				#= camera_1_bb = meas.bounding_boxes[1]
				camera_2_bb = meas.bounding_boxes[2]
				image_width = meas.image_width
				image_height = meas.image_height
				focal_length = meas.focal_length
				pixel_len = meas.pixel_length
				full_meas = [camera_1_bb; camera_2_bb]
				if (num_vehicles == 0)
					prior_p1 = latest_localization_state.position[1] + vehicle_size[1] * 1.5
					prior_p2 = latest_localization_state.position[2] + vehicle_size[2] * 1.5
					prior_theta = atan(
						2 * (latest_localization_state.quaternion[1] * latest_localization_state.quaternion[4] + latest_localization_state.quaternion[2] * latest_localization_state.quaternion[3]),
						1 - 2 * (latest_localization_state.quaternion[3]^2 + latest_localization_state.quaternion[4]^2),
					)
					prior_velocity = 0
					prior_l = vehicle_size[1]
					prior_w = vehicle_size[2]
					prior_h = vehicle_size[3]
					prior = OtherVehicleStates(prior_p1, prior_p2, prior_theta, prior_velocity, prior_l, prior_w, prior_h)
					push!(priors, prior)
					bb_and_prior_map[full_meas] = prior
					bb_and_prior_covar_map[full_meas] = Diagonal([vehicle_size[1]^2; vehicle_size[1]^2; 0.2^2; 0.5^2; 0.0001^2; 0.0001^2; 0.0001^2])
				else
					min_distance = 1000000
					min_prev_bb = []
					for (prev_bb, prev_state) in prev_bbs_and_states
						diff = norm(full_meas - prev_bb)
						if (diff < min_distance)
							min_distance = diff
							min_prev_bb = prev_bb
						end
						# Matching algo here
					end
					bb_and_prior_map[min_prev_bb] = OtherVehicleStates(prev_bbs_and_states[min_prev_bb][1], prev_bbs_and_states[min_prev_bb][2], prev_bbs_and_states[min_prev_bb][3],
						prev_bbs_and_states[min_prev_bb][4], prev_bbs_and_states[min_prev_bb][5], prev_bbs_and_states[min_prev_bb][6], prev_bbs_and_states[min_prev_bb][7])
					bb_and_prior_covar_map[min_prev_bb] = prev_bbs_and_covars[min_prev_bb]

				end
				empty!(prev_bbs_and_states)
				empty!(prev_bbs_and_covars)
				num_vehicles = 0
				vehicle_states = []
				for (bb, prior) in bb_and_prior_map
					vector_prior = [prior.p1; prior.p2; prior.θ; prior.v; prior.l; prior.w; prior.h]
					state = vector_prior
					covar = bb_and_prior_covar_map[bb]
					Q = Diagonal([1, 1, 1, 1, 0.0001, 0.0001, 0.0001])
					R = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
					new_x, new_covar = perception_ekf_step(bb, state, covar, Q, R, vector_localization, image_width, image_height, focal_length, pixel_len)
					prev_bbs_and_states[bb] = new_x
					prev_bbs_and_covars[bb] = new_covar
					vehicle_state = OtherVehicleStates(new_x[1], new_x[2], new_x[3], new_x[4], new_x[5], new_x[6], new_x[7])
					push!(vehicle_states, vehicle_state)
					num_vehicles = num_vehicles + 1
				end
				perception_state = Percept(time(), vehicle_states)
				if isready(perception_state_channel)
					take!(perception_state_channel)
				end
				put!(perception_state_channel, perception_state) =#

			end

		end

		#= for (prev_bb,prev_state) in prev_bbs_and_states
			@info "State"
			@info prev_bbs_and_states[prev_bb]
			@info "Covar"
			@info prev_bbs_and_covars[prev_bb]
		end =#
		#@info "Perception attempt start"
		#@info latest_localization_state
		# process bounding boxes / run ekf / do what you think is good
		#@info "Perception attempt end"

	end
	@info "Terminated perception task."
end


function decision_making(localization_state_channel,
    perception_state_channel,
    target_road_segment_channel,
    shutdown_channel,
    map,
    socket)
    try
        # targets = [79, 94] #test target road segments manually
        @info "Starting decision making task..."
        while true
            sleep(0.001) # prevent thread from hogging resources & freezing other threads
            isready(shutdown_channel) && break

            target_road_segment_id = take!(target_road_segment_channel) #popfirst!(targets)
            @info "Target road segment: $target_road_segment_id"

            localization_state = fetch(localization_state_channel)
            position = localization_state.position[1:2] # ground truth for testing
            # position = localization_state.x.position[1:2] # localization estimate
            yaw = extract_yaw_from_quaternion(localization_state.orientation)
            position = [position[1] + 7*cos(yaw), position[2] + 7*sin(yaw)]

            cur_road_segment_id = cur_map_segment_of_vehicle(position, map)
            start_road_id = cur_road_segment_id
            @info "Current road segment: $cur_road_segment_id"
            position_on_road = find_side_of_road(position, cur_road_segment_id, map)
            route = shortest_path(cur_road_segment_id, target_road_segment_id, map)

            @info "Route from $cur_road_segment_id to $target_road_segment_id calculated."
            @info "Following the calculated route $route"

            cur_road_segment_id = popfirst!(route)

            steering_angle = 0.0
            target_velocity = 0.0
            controlled = true

            while !isempty(route)
                # slower checking since we aren't changing road segments that often
                sleep(0.1) # prevent thread from hogging resources & freezing other threads

                println("Current road seg id: $cur_road_segment_id")
                localization_state = fetch(localization_state_channel)
                position = localization_state.position[1:2] # ground truth for testing
                yaw = extract_yaw_from_quaternion(localization_state.orientation)
                println("Yaw: $yaw")

                position = [position[1] + 7*cos(yaw), position[2] + 7*sin(yaw)]
                println("Position: $position")
                # position = localization_state.x.position[1:2] # localization estimate
                if position_on_road[1] == "error"
                    cur_road_segment_id = popfirst!(route)
                end

                if cur_road_segment_id == target_road_segment_id && map[cur_road_segment_id].lane_types[2] == VehicleSim.loading_zone #handles unloading zone
                    position_on_road = find_side_of_load_zone(position, cur_road_segment_id, map)
                    back_car_position_on_road = position_on_road
                    while back_car_position_on_road[1] != "middle"
                        localization_state = fetch(localization_state_channel)
                        position = localization_state.position[1:2]
                        yaw = extract_yaw_from_quaternion(localization_state.orientation)
                        back_position = [position[1] + 4.25*cos(yaw), position[2] + 4.25*sin(yaw)]
                        back_car_position_on_road = find_side_of_load_zone(back_position, cur_road_segment_id, map)
                        position = [position[1] + 7*cos(yaw), position[2] + 7*sin(yaw)]
                        position_on_road = find_side_of_load_zone(position, cur_road_segment_id, map)
                        println("Current road seg id: $cur_road_segment_id")
                        println("Yaw: $yaw")
                        println("Position: $position")
                        println("Center Position: $back_car_position_on_road")
                        println("Side of road: $position_on_road")
                        target_velocity = 3.0
                        steering_angle = find_steering_angle(cur_road_segment_id, map, position_on_road[1], position_on_road[2], yaw)
                        println("Steering Angle: $steering_angle")
                        cmd = VehicleCommand(steering_angle, target_velocity, controlled)
                        serialize(socket, cmd)
                    end
                    target_velocity = 0.0
                    cmd = VehicleCommand(steering_angle, target_velocity, controlled)
                    serialize(socket, cmd)
                else
                    position_on_road = find_side_of_road(position, cur_road_segment_id, map)
                    if cur_road_segment_id == start_road_id #condition for start road
                        target_velocity = 3.5
                        steering_angle = find_steering_angle(cur_road_segment_id, map, position_on_road[1], position_on_road[2], yaw)
                    elseif map[cur_road_segment_id].lane_boundaries[1].curvature != 0 #condition for curved roads
                        target_velocity = 5.0
                        steering_angle = find_steering_angle(cur_road_segment_id, map, position_on_road[1], position_on_road[2], yaw)
                    elseif map[cur_road_segment_id].lane_types[1] == VehicleSim.stop_sign && target_velocity > 0.5
                        target_velocity = get_stop_target_velocity(position, 10.0, cur_road_segment_id, map)
                        steering_angle = find_steering_angle(cur_road_segment_id, map, position_on_road[1], position_on_road[2], yaw)
                        println("Target Velocity: $target_velocity")
                    else #condition for all other roads
                        target_velocity = 10.0 
                        steering_angle = find_steering_angle(cur_road_segment_id, map, position_on_road[1], position_on_road[2], yaw)
                    end
                end
                println("Side of road: $position_on_road")
                println("Steering Angle: $steering_angle")
                println()

                cmd = VehicleCommand(steering_angle, target_velocity, controlled)
                serialize(socket, cmd)
            end
            if (cur_road_segment_id == target_road_segment_id) #FIX ME: This will always return true since we pop the whole stack of road segments in the route plan
                @info "Reached target: $target_road_segment_id."
                steering_angle = 0.0
                target_velocity = 0.0
            else
                @info "Target not reached!"
            end
            cmd = VehicleCommand(steering_angle, target_velocity, controlled)
            serialize(socket, cmd)
        end
        @info "Terminated decision making task."
    catch e
        @info "Decision making task shutdown"
    end
end

function isfull(ch::Channel)
	length(ch.data) ≥ ch.sz_max
end

function test_algorithms(gt_channel,
	localization_state_channel,
	perception_state_channel,
	shutdown_channel,
	gt_localization_state_channel,
	ego_vehicle_id)
	@info "Starting testing task..."

	# estimated_vehicle_states = Dict{Int, Tuple{Float64, Union{SimpleVehicleState, LocalizationEstimate}}}()
	gt_vehicle_states = Dict{Int, GroundTruthMeasurement}()

	# only start testing after we have received the first ground truth measurement
	wait(gt_channel)

	t = time()
	while true
		sleep(0.001) # prevent thread from hogging resources & freezing other threads
		isready(shutdown_channel) && break

		while isready(gt_channel)
			meas = take!(gt_channel)
			id = meas.vehicle_id

			# add measurement to gt state dictionary if vehicle id doesn't
			# exist or measurement is more recent than existing measurement
			if !haskey(gt_vehicle_states, id) || meas.time > gt_vehicle_states[id].time
				gt_vehicle_states[id] = meas
			end
		end
		latest_est_ego_state = fetch(localization_state_channel)

		latest_gt_ego_state = gt_vehicle_states[ego_vehicle_id]

		gt_localization_state = LocalizationEstimate(latest_gt_ego_state.time, latest_gt_ego_state.position, latest_gt_ego_state.orientation, latest_gt_ego_state.velocity, latest_gt_ego_state.angular_velocity)
		if isready(gt_localization_state_channel)
			take!(gt_localization_state_channel)
		end
		put!(gt_localization_state_channel, gt_localization_state)
		if latest_est_ego_state.time < latest_gt_ego_state.time - 0.5
			@warn "Localization algorithm stale."
		else
			estimated_xyz = latest_est_ego_state.position
			true_xyz = latest_gt_ego_state.position
			position_error = norm(estimated_xyz - true_xyz)
			t2 = time()
			if t2 - t > 5.0
				@info "Localization position error: $position_error"
				@info "estimated: $estimated_xyz | true: $true_xyz"
				t = t2
			end
		end

		latest_perception_state = fetch(perception_state_channel)
		#@info "Perception update"
		#@info latest_perception_state.x
		last_perception_update = latest_perception_state.last_update
		vehicles = latest_perception_state.x
		thing = 0
		for vehicle in vehicles
			thing = thing + 1
			xy_position = [vehicle.p1, vehicle.p2]
			closest_id = 0
			closest_dist = Inf
			for (id, gt_vehicle) in gt_vehicle_states
				if id == ego_vehicle_id
					continue
				else
					gt_xy_position = gt_vehicle.position[1:2]
					dist = norm(gt_xy_position - xy_position)
					if dist < closest_dist
						closest_id = id
						closest_dist = dist
					end
				end
			end
			paired_gt_vehicle = gt_vehicle_states[closest_id]

			#     # compare estimated to GT

			if last_perception_update < paired_gt_vehicle.time - 0.5
				@info "Perception upate stale"
			else
				# compare estimated to true size
				estimated_size = [vehicle.l, vehicle.w, vehicle.h]
				actual_size = paired_gt_vehicle.size
				estimated_position = [vehicle.p1, vehicle.p2]
				actual_position = paired_gt_vehicle.position[1:2]
				#@info "Estimated size error: $(norm(actual_size-estimated_size))"
				@info "Estimated position error $thing :  $(norm(actual_position-estimated_position))"
			end
		end
	end
	@info "Terminated testing task."
end

function publish_socket_data_to_channels(socket, gps_channel, imu_channel, cam_channel, gt_channel, target_road_segment_channel, shutdown_channel)
	@info "Starting socket data publishing task..."

	while true
		sleep(0.001) # prevent thread from hogging resources & freezing other threads
		isready(shutdown_channel) && break

		# stand in until the server publishes target road segments
		!isfull(target_road_segment_channel) && put!(target_road_segment_channel, 55)

		# read to end of the socket stream to ensure you
		# are looking at the latest messages
		local measurement_msg
		received = false
		while true
			@async eof(socket)
			if bytesavailable(socket) > 0
				measurement_msg = deserialize(socket)
				received = true
			else
				break
			end
		end
		!received && continue

		# publish measurements from socket to measurement channels
		# so they can be used in the worker threads
		for meas in measurement_msg.measurements
			if meas isa GPSMeasurement
				!isfull(gps_channel) && put!(gps_channel, meas)
			elseif meas isa IMUMeasurement
				!isfull(imu_channel) && put!(imu_channel, meas)
			elseif meas isa CameraMeasurement
				!isfull(cam_channel) && put!(cam_channel, meas)
			elseif meas isa GroundTruthMeasurement
				!isfull(gt_channel) && put!(gt_channel, meas)
			end
		end
	end
	@info "Terminated socket data publishing task."
end

function test_client(host::IPAddr = IPv4(0), port = 4444; v_step = 1.0, s_step = π / 10, use_keyboard = true)
	socket = Sockets.connect(host, port)
	(peer_host, peer_port) = getpeername(socket)
	msg = deserialize(socket) # Visualization info
	@info msg

	map_segments = training_map()
	ego_vehicle_id = 1

	gps_channel = Channel{GPSMeasurement}(32)
	imu_channel = Channel{IMUMeasurement}(32)
	cam_channel = Channel{CameraMeasurement}(32)
	gt_channel = Channel{GroundTruthMeasurement}(32)
	target_road_segment_channel = Channel{Int}(1)
	shutdown_channel = Channel{Bool}(1)

	localization_state_channel = Channel{LocalizationEstimate}(1)
	perception_state_channel = Channel{Percept}(1)
	gt_localization_state_channel = Channel{LocalizationEstimate}(1)

	# wrap worker threads in error monitor to print any errors
	# does not raise an error though
	# so we have to press q to quit and code below will handle killing worker threads
	errormonitor(@async publish_socket_data_to_channels(socket, gps_channel, imu_channel, cam_channel, gt_channel, target_road_segment_channel, shutdown_channel))
	errormonitor(@async localize(gps_channel, imu_channel, localization_state_channel, shutdown_channel))
	errormonitor(@async perception(cam_channel, gt_localization_state_channel, perception_state_channel, shutdown_channel))
	if !use_keyboard
		# use gt channel instead of localization channel for testing purposes
		errormonitor(@async decision_making(gt_channel, perception_state_channel, target_road_segment_channel, shutdown_channel, map_segments, socket))
	end
	errormonitor(@async test_algorithms(gt_channel, localization_state_channel, perception_state_channel, shutdown_channel, gt_localization_state_channel, ego_vehicle_id))

	target_velocity = 0.0
	steering_angle = 0.0
	controlled = true
	@info "Press 'q' at any time to terminate vehicle."
	while controlled && isopen(socket)
		key = get_c()
		if key == 'q'
			# terminate vehicle
			controlled = false
			target_velocity = 0.0
			steering_angle = 0.0
			@info "Terminating test client and its worker threads"
			# sends msg to workers threads to throw error so they die
			# this lets us use Revise to handle code changes
			put!(shutdown_channel, true)
		elseif key == 'i'
			# increase target velocity
			target_velocity += v_step
			@info "Target velocity: $target_velocity"
		elseif key == 'k'
			# decrease forward force
			target_velocity -= v_step
			@info "Target velocity: $target_velocity"
		elseif key == 'j'
			# increase steering angle
			steering_angle += s_step
			@info "Target steering angle: $steering_angle"
		elseif key == 'l'
			# decrease steering angle
			steering_angle -= s_step
			@info "Target steering angle: $steering_angle"
		end
		if use_keyboard
			cmd = VehicleCommand(steering_angle, target_velocity, controlled)
			serialize(socket, cmd)
		end
	end
end
