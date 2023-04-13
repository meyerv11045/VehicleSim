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

function localize(gps_channel, imu_channel, localization_state_channel)
    # Define the initial state and covariance p(x₀) ~ N(x₀, P₀)
    x₀ = zeros(12)
    x₀[3] = 1 # height of vehicle off the ground
    P₀ = Diagonal(ones(12))  # Initial covariance (uncertainty)

    # Process noise covariance
    Q = Diagonal(0.01*ones(12))

    # Measurement noise covariance (taken from the generator functions)
    imu_variance = Diagonal([0.01, 0.01, 0.01, 0.01, 0.01, 0.01].^2)
    gps_variance = Diagonal([1.0, 1.0].^2)
    if is_gps_measurement
        R = gps_variance
    else
        R = imu_variance
    end

    # Initialize the estimate of the state
    x̂ = x₀
    P̂ = P₀

    while true
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
            if length(fresh_gps_meas) == length(fresh_imu_meas)
                z = popfirst!(fresh_gps_meas)
            else if length(fresh_gps_meas) > length(fresh_imu_meas)
                z = popfirst!(fresh_gps_meas)
            else
                z = popfirst!(fresh_imu_meas)
            end

            x̂, P̂ = ekf_step(z, x̂, P̂)

            localization_state = LocalizationEstimate(time(), FullVehicleState(x̂...))
            if isready(localization_state_channel)
                take!(localization_state_channel)
            end
            put!(localization_state_channel, localization_state)
        end
    end
end

function perception(cam_meas_channel, localization_state_channel, perception_state_channel)
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
    # set up stuff
    while true
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
        map,
        target_road_segment_id,
        socket)
    # do some setup
    while true
        latest_localization_state = fetch(localization_state_channel)
        latest_perception_state = fetch(perception_state_channel)

        # figure out what to do ... setup motion planning problem etc
        steering_angle = 0.0
        target_vel = 0.0
        cmd = VehicleCommand(steering_angle, target_vel, true)
        serialize(socket, cmd)
    end
end

function isfull(ch:Channel)
    length(ch.data) ≥ ch.sz_max
end


function my_client(host::IPAddr=IPv4(0), port=4444)
    socket = Sockets.connect(host, port)
    map = training_map()

    gps_channel = Channel{GPSMeasurement}(32)
    imu_channel = Channel{IMUMeasurement}(32)
    cam_channel = Channel{CameraMeasurement}(32)
    gt_channel = Channel{GroundTruthMeasurement}(32)

    localization_state_channel = Channel{LocalizationEstimate}(1)
    perception_state_channel = Channel{Percept}(1)

    target_map_segment = 0 # (not a valid segment, will be overwritten by message)
    ego_vehicle_id = 0 # (not a valid id, will be overwritten by message. This is used for discerning ground-truth messages)

    @async while true
        measurement_msg = deserialize(socket)
        target_map_segment = meas.target_segment
        ego_vehicle_id = meas.vehicle_id
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

    @async localize(gps_channel, imu_channel, localization_state_channel)
    @async perception(cam_channel, localization_state_channel, perception_state_channel)
    @async decision_making(localization_state_channel, perception_state_channel, map, map[0], socket)
end


function test_algorithms(gt_channel,
    localization_state_channel,
    perception_state_channel,
    ego_vehicle_id)
estimated_vehicle_states = Dict{Int, Tuple{Float64, Union{SimpleVehicleState, FullVehicleState}}}
gt_vehicle_states = Dict{Int, GroundTruthMeasuremen}

t = time()
while true

    while isready(gt_channel)
        meas = take!(gt_channel)
        id = meas.vehicle_id
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

    latest_perception_state = fetch(perception_state_channel)
    last_perception_update = latest_perception_state.last_update
    vehicles = last_perception_state.x

    for vehicle in vehicles
        xy_position = [vehicle.p1, vehicle.p2]
        closest_id = 0
        closest_dist = Inf
        for (id, gt_vehicle) in gt_vehicle_states
            if id == ego_vehicle_id
                continue
            else
                gt_xy_position = gt_vehicle_position[1:2]
                dist = norm(gt_xy_position-xy_position)
                if dist < closest_dist
                    closest_id = id
                    closest_dist = dist
                end
            end
        end
        paired_gt_vehicle = gt_vehicle_states[closest_id]

        # compare estimated to GT

        if last_perception_update < paired_gt_vehicle.time - 0.5
            @info "Perception upate stale"
        else
            # compare estimated to true size
            estimated_size = [vehicle.l, vehicle.w, vehicle.h]
            actual_size = paired_gt_vehicle.size
            @info "Estimated size error: $(norm(actual_size-estimated_size))"
        end
    end
end
end