# for estimating the other vehicles
struct SimpleVehicleState
    p1::Float64
    p2::Float64
    θ::Float64
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
    x::Vector{SimpleVehicleState}
end


function matrix_to_quaternion(R)
    w = sqrt(1 + R[1,1] + R[2,2] + R[3,3]) / 2
    x = (R[3,2] - R[2,3]) / (4 * w)
    y = (R[1,3] - R[3,1]) / (4 * w)
    z = (R[2,1] - R[1,2]) / (4 * w)
    q = [w, x, y, z]
    return q
end

function get_driving_dir_of_road_seg(road_segment_id, map)
    # tested and verified with map segment 101 which should have driving dir
    # of [-1, 0] in map frame which corresponds to θ = π

    # also tested and verified with vehicle's initial position in sim with 1
    # vehicle. θ = π/2 which corresponds to driving dir of [0, 1] in map frame
    cur_segment = map[road_segment_id]

    lane_boundary = cur_segment.lane_boundaries[1]
    b = lane_boundary.pt_b
    a = lane_boundary.pt_a
    dir = (b-a) / norm(b-a)
    θ = atan(dir[2], dir[1])
    return θ
end


function localize(map, gps_channel, imu_channel, localization_state_channel, shutdown_channel)
    @info "Starting localization task..."

    # Initial state and covariance p(x₀) ~ N(x₀, P₀)
    x₀ = zeros(13)

    # use first GPS measurement to get a storng prior on initial position + orientation
    init_gps_meas = take!(gps_channel)

    t = [-3.0, 1, 2.6] # gps sensor in reference to the body frame
    xy_gps_in_map = [init_gps_meas.lat, init_gps_meas.long]

    θ = init_gps_meas.heading

    # rotate from map frame to estimate of body's orientation in map frame
    R_3D = [cos(θ) -sin(θ) 0;
            sin(θ) cos(θ) 0;
            0 0 1]
    q  = matrix_to_quaternion(R_3D)
    x₀[4:7] = q   # estimate of initial orientation

    R_2D = [cos(θ) -sin(θ);
            sin(θ) cos(θ)]

    # need to apply the rotation to the translation, not the point in map frame
    xy_body_in_map = xy_gps_in_map - R_2D*t[1:2]

    x₀[1:2] = xy_body_in_map

    init = ones(13)
    init[4:7] .= 0.001
    P₀ = Diagonal(init)  # Initial covariance (uncertainty)

    # Process noise covariance
    # trust the process model a lot!!
    Q = Diagonal(0.00001*ones(13))

    # Measurement noise covariance (taken from the generator functions)
    imu_variance = Diagonal([0.1, 0.1, 0.1, 0.1, 0.1, 0.1].^2)
    gps_variance = Diagonal([10.0, 10.0, 1.0].^2)

    # Initialize the estimate of the state
    x̂ = x₀
    P̂ = P₀

    last_update = time()
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
                z = [meas.lat; meas.long; meas.heading]
                R = gps_variance
            else
                meas = popfirst!(fresh_imu_meas)
                z = [meas.linear_vel ; meas.angular_vel]
                R = imu_variance
            end

            Δ = meas.time - last_update

            x̂, P̂ = ekf_step(z, x̂, P̂, Q, R, Δ)
            last_update = meas.time

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
    @info "Starting perception task..."
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

        # process bounding boxes / run ekf / do what you think is good

        perception_state = Percept(0.0, [])
        if isready(perception_state_channel)
            take!(perception_state_channel)
        end
        put!(perception_state_channel, perception_state)
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

function quaternion_to_euler(q)
    yaw = atan(2*q[4]*q[3] + 2*q[1]*q[2], 1 - 2*q[2]^2 - 2*q[3]^2)
    pitch = asin(2*q[4]*q[2] - 2*q[1]*q[3])
    roll = atan(2*q[4]*q[1] + 2*q[2]*q[3], 1 - 2*q[1]^2 - 2*q[2]^2)
    return (roll, pitch, yaw)
end

function test_algorithms(gt_channel,
    localization_state_channel,
    perception_state_channel,
    shutdown_channel,
    ego_vehicle_id)
    @info "Starting testing task..."

    # estimated_vehicle_states = Dict{Int, Tuple{Float64, Union{SimpleVehicleState, LocalizationEstimate}}}()
    gt_vehicle_states = Dict{Int, GroundTruthMeasurement}()

    # only start testing after we have received the first ground truth measurement
    wait(gt_channel)
    first_est = true
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
        if latest_est_ego_state.time < latest_gt_ego_state.time - 0.5
            @warn "Localization algorithm stale."
        else
            estimated_xyz = latest_est_ego_state.position
            true_xyz = latest_gt_ego_state.position

            estimated_v = latest_est_ego_state.linear_vel
            true_v = latest_gt_ego_state.velocity

            estimated_q = latest_est_ego_state.quaternion
            true_q = latest_gt_ego_state.orientation


            position_error = norm(estimated_xyz[1:2] - true_xyz[1:2])
            vel_error = norm(estimated_v[1:2] - true_v[1:2])

            t2 = time()
            if t2 - t > 2.5 || first_est
                @info "------------------------------"
                @info @sprintf "Localization xy position error: %.3f" position_error
                @info @sprintf "x: %.2f | %.2f" estimated_xyz[1] true_xyz[1]
                @info @sprintf "y: %.2f | %.2f" estimated_xyz[2] true_xyz[2]
                # @info "z: $(estimated_xyz[3]) | $(true_xyz[3])"

                @info "orientation (est vs ground truth)"
                @info estimated_q
                @info true_q
                # @info @sprintf "Localization orientation error: %.3f" orientation_error
                # @info @sprintf "roll: %.2f | %.2f" roll_e roll_t
                # @info @sprintf "pitch: %.2f | %.2f" pitch_e pitch_t
                # @info @sprintf "yaw: %.2f | %.2f" yaw_e yaw_t

                @info @sprintf "Localization xy velocity error: %.3f" vel_error
                @info @sprintf "v_x: %.2f | %.2f" estimated_v[1] true_v[1]
                @info @sprintf "v_y: %.2f | %.2f" estimated_v[2] true_v[2]
                # @info "v_z: $(estimated_v[3]) | $(true_v[3])"

                # @info "estimated: $estimated_xyz | true: $true_xyz"
                t = t2
                first_est = false
            end
        end

        # latest_perception_state = fetch(perception_state_channel)
        # last_perception_update = latest_perception_state.last_update
        # vehicles = last_perception_state.x

        # for vehicle in vehicles
        #     xy_position = [vehicle.p1, vehicle.p2]
        #     closest_id = 0
        #     closest_dist = Inf
        #     for (id, gt_vehicle) in gt_vehicle_states
        #         if id == ego_vehicle_id
        #             continue
        #         else
        #             gt_xy_position = gt_vehicle_position[1:2]
        #             dist = norm(gt_xy_position-xy_position)
        #             if dist < closest_dist
        #                 closest_id = id
        #                 closest_dist = dist
        #             end
        #         end
        #     end
        #     paired_gt_vehicle = gt_vehicle_states[closest_id]

        #     # compare estimated to GT

        #     if last_perception_update < paired_gt_vehicle.time - 0.5
        #         @info "Perception upate stale"
        #     else
        #         # compare estimated to true size
        #         estimated_size = [vehicle.l, vehicle.w, vehicle.h]
        #         actual_size = paired_gt_vehicle.size
        #         @info "Estimated size error: $(norm(actual_size-estimated_size))"
        #     end
        # end
    end
    @info "Terminated testing task."
end

function publish_socket_data_to_channels(socket, gps_channel, imu_channel, cam_channel, gt_channel, target_road_segment_channel, shutdown_channel)
    @info "Starting socket data publishing task..."

    while true
        sleep(0.001) # prevent thread from hogging resources & freezing other threads
        isready(shutdown_channel) && break

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

        if isfull(target_road_segment_channel)
            take!(target_road_segment_channel)
        end
        put!(target_road_segment_channel, measurement_msg.target_segment)

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
            elseif meas isa Int
                @info meas

            else
                @info typeof(meas)
                @info meas
            end
        end
    end
    @info "Terminated socket data publishing task."
end

function test_client(host::IPAddr=IPv4(0), port=4444; v_step = 1.0, s_step = π/10, use_keyboard=true)
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

    # wrap worker threads in error monitor to print any errors
    # does not raise an error though
    # so we have to press q to quit and code below will handle killing worker threads
    errormonitor(@async publish_socket_data_to_channels(socket, gps_channel, imu_channel, cam_channel, gt_channel, target_road_segment_channel, shutdown_channel))
    errormonitor(@async localize(map_segments, gps_channel, imu_channel, localization_state_channel, shutdown_channel))
    # errormonitor(@asyc perception(cam_channel, localization_state_channel, perception_state_channel, shutdown_channel))
    if !use_keyboard
        # use gt channel instead of localization channel for testing purposes
        errormonitor(@async decision_making(gt_channel, perception_state_channel, target_road_segment_channel, shutdown_channel, map_segments, socket))
    end
    errormonitor(@async test_algorithms(gt_channel, localization_state_channel, perception_state_channel, shutdown_channel, ego_vehicle_id))

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