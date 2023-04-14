""" Process Model: p(xₖ | xₖ₋₁)
    Given the current state, returns the mean next state
"""
function f(x, Δt)
    position = x[1:3]
    quaternion = x[4:7]
    velocity = x[8:10]
    angular_vel = x[11:13]

    # quaternion update
    r = angular_vel
    mag = norm(r)

    if mag < 1e-5
        # prevent divide by zero -> NaNs
        sᵣ = 1.0
        vᵣ = zeros(3)
    else
        sᵣ = cos(mag*Δt / 2.0)
        vᵣ = sin(mag*Δt / 2.0) * r/mag
    end

    sₙ = quaternion[1]
    vₙ = quaternion[2:4]

    s = sₙ*sᵣ - vₙ'*vᵣ
    v = sₙ*vᵣ+sᵣ*vₙ+vₙ×vᵣ

    new_position = position + Δt * velocity
    new_quaternion = [s; v]
    return [new_position; new_quaternion; velocity; angular_vel]
end

function jac_fx(x, Δ)
    jacobian(state -> f(state, Δ), x)[1]
end

function get_differentiable_imu_transform()
    # replace RotY(0.02) with its values since we can't take derivative through RotY(θ)
    R_imu_in_body = [0.9998 0.0 0.0199987;
                     0.0    1.0      0.0;
                     -0.0199987 0.0 0.9998]
    t_imu_in_body = [0, 0, 0.7]

    T = [R_imu_in_body t_imu_in_body]
end

""" Measurement Model: p(zₖ | xₖ)
    Given a state, returns the mean corresponding measurement
"""
function h(x, output_gps_measurement=true)
    if output_gps_measurement
        # output in GPS frame
        T_body_to_gps = get_gps_transform()
        gps_loc_body = T_body_to_gps*[zeros(3); 1.0]

        xyz_body = x[1:3]
        q_body = x[4:7]
        Tbody = get_body_transform(q_body, xyz_body)
        xyz_gps = Tbody * [gps_loc_body; 1]
        return xyz_gps[1:2]
    else
        # output in IMU frame
        T_body_imu = get_differentiable_imu_transform()
        T_imu_body = invert_transform(T_body_imu)
        R = T_imu_body[1:3,1:3]
        p = T_imu_body[1:3,end]

        v_body = x[8:10]
        ω_body = x[11:13]

        ω_imu = R * ω_body
        v_imu = R * v_body + p × ω_imu

        return [v_imu; ω_imu]
    end
end

function jac_hx(x, output_gps_measurement)
    jacobian(state -> h(state, output_gps_measurement), x)[1]
end

function ekf_step(z, x̂, P̂, Q, R, Δ=0.01)
    is_gps_measurement = length(z) == 2

    # Prediction step
    x̂ = f(x̂, Δ)
    F = jac_fx(x̂, Δ)
    P̂ = F*P̂*F' + Q

    # Update step
    ẑ = h(x̂, is_gps_measurement)
    H = jac_hx(x̂, is_gps_measurement)
    S = H*P̂*H' + R

    inv_s = inv(S)
    K = P̂*H'* inv_s # Kalman gain

    # Update mean and covariance estimate of for xₖ
    x̂ = x̂ + K*(z - ẑ)
    P̂ = (I - K*H)*P̂

    return x̂, P̂
end