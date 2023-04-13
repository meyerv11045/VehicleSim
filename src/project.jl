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
struct FullVehicleState
    position::SVector{3, Float64}
    quaternion::SVector{4, Float64}
    linear_vel::SVector{3, Float64}
    angular_vel::SVector{3, Float64}
end

struct LocalizationEstimate
    last_update::Float64
    x::FullVehicleState
end

struct Percept
    last_update::Float64
    x::Vector{SimpleVehicleState}
end

function localize(gps_channel, imu_channel, localization_state_channel, shutdown_channel)
    @info "Starting localization task..."
    # Define the initial state and covariance p(x₀) ~ N(x₀, P₀)
    x₀ = zeros(12)
    x₀[3] = 1 # height of vehicle off the ground
    P₀ = Diagonal(ones(12))  # Initial covariance (uncertainty)

    # Process noise covariance
    Q = Diagonal(0.01*ones(12))

    # Measurement noise covariance (taken from the generator functions)
    imu_variance = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01].^2)
    gps_variance = Diagonal([1.0, 1.0].^2)

    # Initialize the estimate of the state
    x̂ = x₀
    P̂ = P₀

    @info "time to loop"
    while true
        sleep(0.001) # prevent thread from hogging resources & freezing other threads
        if isready(shutdown_channel)
            fetch(shutdown_channel) && throw(error("Shutdown!"))
        end

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

        l = length(fresh_gps_meas)
        l0 = length(fresh_imu_meas)
        # @info "Fresh GPS measurements: $l Fresh IMU Measurements: $l0"

        while length(fresh_gps_meas) > 0 || length(fresh_imu_meas) > 0
            if length(fresh_gps_meas) == length(fresh_imu_meas)
                z = popfirst!(fresh_gps_meas)
            elseif length(fresh_gps_meas) > length(fresh_imu_meas)
                z = popfirst!(fresh_gps_meas)
            else
                z = popfirst!(fresh_imu_meas)
            end

            x̂, P̂ = ekf_step(z, x̂, P̂, imu_variance, gps_variance)

            localization_state = LocalizationEstimate(time(), FullVehicleState(x̂...))
            @info "Localization state: $localization_state"
            if isready(localization_state_channel)
                take!(localization_state_channel)
            end
            put!(localization_state_channel, localization_state)
        end
    end
end

function perception(cam_meas_channel, localization_state_channel, perception_state_channel, shutdown_channel)
    # set up stuff
    while true
        sleep(0.001) # prevent thread from hogging resources & freezing other threads
        if isready(shutdown_channel)
            fetch(shutdown_channel) && throw(error("Shutdown!"))
        end

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
end

function decision_making(localization_state_channel,
        perception_state_channel,
        shutdown_channel,
        map,
        target_road_segment_id,
        socket)
    # do some setup
    while true
        sleep(0.001) # prevent thread from hogging resources & freezing other threads
        if isready(shutdown_channel)
            fetch(shutdown_channel) && throw(error("Shutdown!"))
        end

        latest_localization_state = fetch(localization_state_channel)
        latest_perception_state = fetch(perception_state_channel)

        # figure out what to do ... setup motion planning problem etc
        steering_angle = 0.0
        target_vel = 0.0
        cmd = VehicleCommand(steering_angle, target_vel, true)
        serialize(socket, cmd)
    end
end

function isfull(ch::Channel)
    length(ch.data) ≥ ch.sz_max
end

function test_algorithms(gt_channel,
    localization_state_channel,
    perception_state_channel,
    shutdown_channel,
    ego_vehicle_id)

    estimated_vehicle_states = Dict{Int, Tuple{Float64, Union{SimpleVehicleState, FullVehicleState}}}
    gt_vehicle_states = Dict{Int, GroundTruthMeasurement}

    @info "in the testing algorithm"
    t = time()
    while true
        sleep(0.001) # prevent thread from hogging resources & freezing other threads
        if isready(shutdown_channel)
            fetch(shutdown_channel) && throw(error("Shutdown!"))
        end
        a = isready(gt_channel)
        @info a
        while isready(gt_channel)
            meas = take!(gt_channel)
            id = meas.vehicle_id
            @info "ground truth $meas"
            if meas.time > gt_vehicle_states[id].time
                gt_vehicle_states[id] = meas
            end
        end

        latest_estimated_ego_state = fetch(localization_state_channel)
        latest_true_ego_state = gt_vehicle_states[ego_vehicle_id]
        if latest_estimated_ego_state.last_update < latest_true_ego_state.time - 0.5
            @warn "Localization algorithm stale."
        else
            estimated_xyz = latest_estimated_ego_state.position
            true_xyz = latest_true_ego_state.position
            position_error = norm(estimated_xyz - true_xyz)
            t2 = time()
            if t2 - t > 5.0
                @info "Localization position error: $position_error"
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
end

function publish_socket_data_to_channels(socket, gps_channel, imu_channel, cam_channel, gt_channel, shutdown_channel)
    while true
        sleep(0.001) # prevent thread from hogging resources & freezing other threads
        if isready(shutdown_channel)
            fetch(shutdown_channel) && throw(error("Shutdown!"))
        end

        # read to end of the socket stream to ensure you
        # are looking at the latest messages
        measurement_msg = nothing
        while true
            @async eof(socket)
            if bytesavailable(socket) > 0
                measurement_msg = deserialize(socket)
            else
                break
            end
        end
        if isnothing(measurement_msg)
            continue
        end

        target_map_segment = measurement_msg.target_segment
        ego_vehicle_id = measurement_msg.vehicle_id

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
                @info fetch(gt_channel)
                !isfull(gt_channel) && put!(gt_channel, meas)
                @info fetch(gt_channel)
            end
        end
    end
end

function test_client(host::IPAddr=IPv4(0), port=4444; v_step = 1.0, s_step = π/10)
    socket = Sockets.connect(host, port)
    (peer_host, peer_port) = getpeername(socket)
    msg = deserialize(socket) # Visualization info
    @info msg

    ego_vehicle_id = 1

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)
    shutdown_channel = Channel{Bool}(1)

    localization_state_channel = Channel{LocalizationEstimate}(1)
    perception_state_channel = Channel{Percept}(1)

    # wrap worker threads in error monitor to print any errors
    # does not raise an error though
    # so we have to press q to quit and code below will handle killing worker threads
    errormonitor(@async publish_socket_data_to_channels(socket, gps_channel, imu_channel, cam_channel, gt_channel, shutdown_channel))
    errormonitor(@async localize(gps_channel, imu_channel, localization_state_channel, shutdown_channel))
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