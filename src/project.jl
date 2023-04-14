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
    Q = Diagonal(0.01*ones(13))

    # Measurement noise covariance (taken from the generator functions)
    imu_variance = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01].^2)
    gps_variance = Diagonal([1.0, 1.0].^2)

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
                z = [meas.linear_vel ; meas.angular_vel]
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
    @info "Starting decision making task..."
    while true
        sleep(0.001) # prevent thread from hogging resources & freezing other threads
        isready(shutdown_channel) && break

        target_road_segment_id = take!(target_road_segment_channel)

        localization_state = fetch(localization_state_channel)
        position = localization_state.position[1:2] # ground truth for testing
        # position = localization_state.x.position[1:2] # localization estimate

        cur_road_segment_id = cur_map_segment_of_vehicle(position, map)
        @info "Current road segment: $cur_road_segment_id"
        route = shortest_path(cur_road_segment_id, target_road_segment_id, map)

        @info "Route from $cur_road_segment_id to $target_road_segment_id calculated."
        @info "Following the calculated route $route"

        while cur_road_segment_id != target_road_segment_id
            # slower checking since we aren't changing road segments that often
            sleep(0.1) # prevent thread from hogging resources & freezing other threads

            localization_state = fetch(localization_state_channel)
            position = localization_state.position[1:2] # ground truth for testing
            # position = localization_state.x.position[1:2] # localization estimate
            cur_road_segment_id = cur_map_segment_of_vehicle(position, map)

            # TODO: motion planning to follow the route
            # cmd = follow_route(route, cur_road_segment_id, map)
            # serialize(socket, cmd)
        end
        @info "Reached target: $target_road_segment_id."
    end
    @info "Terminated decision making task."
end

function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
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
            position_error = norm(estimated_xyz - true_xyz)
            t2 = time()
            if t2 - t > 5.0
                @info "Localization position error: $position_error"
                @info "estimated: $estimated_xyz | true: $true_xyz"
                t = t2
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

function test_client(host::IPAddr=IPv4(0), port=4444; v_step = 1.0, s_step = π/10)
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
    errormonitor(@async localize(gps_channel, imu_channel, localization_state_channel, shutdown_channel))
    # errormonitor(@asyc perception(cam_channel, localization_state_channel, perception_state_channel, shutdown_channel))
    # errormonitor(@async decision_making(gt_channel, perception_state_channel, target_road_segment_channel, shutdown_channel, map_segments, socket))
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
        cmd = VehicleCommand(steering_angle, target_velocity, controlled)
        serialize(socket, cmd)
    end
end