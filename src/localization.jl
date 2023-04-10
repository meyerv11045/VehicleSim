""" Process Model: p(xₖ | xₖ₋₁)
    Given the current state, returns the mean next state """
function f(x, Δ)
    # TODO: look into using rigid_body_dynamics() from measurements.jl
    x + Δ * vcat([x.v1, x.v2, x.v3, x.w_roll, x.w_pitch, x.w_yaw], zeros(6))
end

function jac_fx(x, Δ)
    gradient(state -> f(state, Δ), x)[1]
end

""" Measurement Model: p(zₖ | xₖ)
    Given a state, returns the mean corresponding measurement
"""
function h(x, R, is_gps_measurement=true)
    if is_gps_measurement
        # gps measurement zₖ is expected to be the position of
        # the state with some gaussian noise
        return [x.p1, x.p2] + noise(2, R)
    else
        # imu measurement zₖ is expected to be the linear
        # and angular velocity of the state with some gaussian noise
        return [x.v1, x.v2, x.v3, x.w_roll, x.w_pitch, x.w_yaw] + noise(6, R)
    end
end

function jac_hx(x, is_gps_measurement)
    gradient(state -> h(state, is_gps_measurement), x)[1]
end

function ekf_step(z, x̂, P̂, Δ=0.01)
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

function noise(length, Σ)
    x = randn(length) # x ~ N(0, I)

    L = cholesky(Σ).L

    # Transform the vector of std normal RVs into a vector of RVs with the desired covariance structure
    L * x
end