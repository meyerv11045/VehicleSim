""" Process Model: p(xₖ | xₖ₋₁)
    Given the current state, returns the mean next state
"""
function jac_fx(x, Δ)
    # using process model from measurements.jl
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
        return h_gps(x)
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
    is_gps_measurement = length(z) == 3

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