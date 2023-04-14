""" Process Model: p(xₖ | xₖ₋₁)
    Given the current state, returns the mean next state """
function f(x, Δ)
    rigid_body_dynamics(x.position, x.quaternion, x.linear_vel, x.angular_vel, Δ)
end

function jac_fx(x, Δ)
    gradient(state -> f(state, Δ), x)[1]
end

""" Measurement Model: p(zₖ | xₖ)
    Given a state, returns the mean corresponding measurement
"""
function h(x, output_gps_measurement=true)
    if output_gps_measurement
        # output in GPS frame
        T_body_to_gps = get_gps_transform()
        gps_loc_body = T_body_to_gps*[zeros(3); 1.0]

        xyz_body = x.position
        q_body = x.quaternion
        Tbody = get_body_transform(q_body, xyz_body)
        xyz_gps = Tbody * [gps_loc_body; 1]
        return xyz_gps[1:2]
    else
        # output in IMU frame
        T_body_imu = get_imu_transform()
        T_imu_body = invert_transform(T_body_imu)
        R = T_imu_body[1:3,1:3]
        p = T_imu_body[1:3,end]

        v_body = x.linear_vel
        ω_body = x.angular_vel

        ω_imu = R * ω_body
        v_imu = R * v_body + p × ω_imu

        return [v_imu; ω_imu]
    end
end

function jac_hx(x, output_gps_measurement)
    gradient(state -> h(state, output_gps_measurement), x)[1]
end

function ekf_step(z, x̂, P̂, Q, R_imu, R_gps, Δ=0.01)
    is_gps_measurement = isa(z, GPSMeasurement)
    if is_gps_measurement
        R = R_gps
    else
        R = R_imu
    end

    # Prediction step
    x̂ = f(x̂, Δ)
    F = jac_fx(x̂, Δ)
    P̂ = F*P̂*F' + Q

    # Update step
    ẑ = h(x̂, is_gps_measurement)
    H = jac_hx(x̂, is_gps_measurement)
    S = H*P̂*H' + R

    K = P̂*H'*inv(S) # Kalman gain

    # Update mean and covariance estimate of for xₖ
    x̂ = x̂ + K*(z - ẑ)
    P̂ = (I - K*H)*P̂

    return x̂, P̂
end