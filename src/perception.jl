# Process Model
function perception_ekf_step(z, state, covar, Q, R, ego_state, image_width,image_height,focal_length,pixel_len,dt = 0.01)
	x_hat = perception_f(state, dt)
	F = perception_jac_fx(state, dt)
	covar_hat = F * covar * F' + Q
	meas = h1(state, ego_state,image_width,image_height,focal_length,pixel_len)
    z_hat = [meas[1],meas[3],meas[5],meas[7],meas[9],meas[11],meas[13],meas[15]]

    H = [perception_jac_h1top(state,meas[2],ego_state,image_width,image_height,focal_length,pixel_len)
              perception_jac_h1left(state,meas[4],ego_state,image_width,image_height,focal_length,pixel_len)
              perception_jac_h1bottom(state,meas[6],ego_state,image_width,image_height,focal_length,pixel_len)
              perception_jac_h1right(state,meas[8],ego_state,image_width,image_height,focal_length,pixel_len)
              perception_jac_h2top(state,meas[10],ego_state,image_width,image_height,focal_length,pixel_len)
              perception_jac_h2left(state,meas[12],ego_state,image_width,image_height,focal_length,pixel_len)
              perception_jac_h2bottom(state,meas[14],ego_state,image_width,image_height,focal_length,pixel_len)
              perception_jac_h2right(state,meas[16],ego_state,image_width,image_height,focal_length,pixel_len)]

    S = H*covar_hat*H' + R
    inv_s = inv(S)
    K = covar_hat*H'* inv_s # Kalman gain

    x_hat = x_hat + (K*(z - z_hat))'
    covar_hat = (I - K*H)*covar_hat
    
    return x_hat,covar_hat
end

function h1_top(x,corner_num,ego_state,image_width,image_height,focal_len,pixel_len)
    T = [cos(x[3]) -sin(x[3]) 0 x[1]
		 sin(x[3])  cos(x[3]) 0 x[2]
		 0 0 1 x[7]/2
		 0 0 0 1]
    #@info "T"
    #@info T
	corner = []
    meas = []
    i = 1
	for dx in [-x[5] / 2, x[5] / 2]
		for dy in [-x[6] / 2, x[6] / 2]
			for dz in [-x[7] / 2, x[7] / 2]
                if(i == corner_num)
                    #@info "Dx"
                    #@info dx
                    #@info "Dy"
                    #@info dy
                    #@info "Dz"
                    #@info dz
                    corner = T * [dx,dy,dz,1]
                end
                i = i + 1
			end
		end
	end
    #@info "Corner"
    #@info corner
	T_world_body = get_body_transform(ego_state[4:7], ego_state[1:3]) # World -> Base
	T_body_cam1 = get_differentiable_cam_transform(1)
	T_cam_camrot = get_rotated_camera_transform()
	T_body_camrot1 = multiply_transforms(T_body_cam1, T_cam_camrot)
	T_world_camrot1 = multiply_transforms(T_world_body, T_body_camrot1)
	T_camrot1_world = invert_transform(T_world_camrot1)

    other_vehicle_corner = T_camrot1_world * corner
    top = focal_len * other_vehicle_corner[2]/other_vehicle_corner[3]
    top = convert_to_pixel_differentiable(image_height, pixel_len, top)
	top
end
function h1_left(x,corner_num,ego_state,image_width,image_height,focal_len,pixel_len)
    T = [cos(x[3]) -sin(x[3]) 0 x[1]
		 sin(x[3])  cos(x[3]) 0 x[2]
		 0 0 1 x[7]/2
		 0 0 0 1]
    #@info "T"
    #@info T
	corner = []
    meas = []
    i = 1
	for dx in [-x[5] / 2, x[5] / 2]
		for dy in [-x[6] / 2, x[6] / 2]
			for dz in [-x[7] / 2, x[7] / 2]
                if(i == corner_num)
                    #@info "Dx"
                    #@info dx
                    #@info "Dy"
                    #@info dy
                    #@info "Dz"
                    #@info dz
                    corner = T * [dx,dy,dz,1]
                end
                i = i + 1
			end
		end
	end
    #@info "Corner"
    #@info corner
	T_world_body = get_body_transform(ego_state[4:7], ego_state[1:3]) # World -> Base
	T_body_cam1 = get_differentiable_cam_transform(1)
	T_cam_camrot = get_rotated_camera_transform()
	T_body_camrot1 = multiply_transforms(T_body_cam1, T_cam_camrot)
	T_world_camrot1 = multiply_transforms(T_world_body, T_body_camrot1)
	T_camrot1_world = invert_transform(T_world_camrot1)

    other_vehicle_corner = T_camrot1_world * corner
    left = focal_len * other_vehicle_corner[1]/other_vehicle_corner[3]
    left = convert_to_pixel_differentiable(image_width, pixel_len, left)
	left
end

function h1_bottom(x,corner_num,ego_state,image_width,image_height,focal_len,pixel_len)
    T = [cos(x[3]) -sin(x[3]) 0 x[1]
		 sin(x[3])  cos(x[3]) 0 x[2]
		 0 0 1 x[7]/2
		 0 0 0 1]
    #@info "T"
    #@info T
	corner = []
    meas = []
    i = 1
	for dx in [-x[5] / 2, x[5] / 2]
		for dy in [-x[6] / 2, x[6] / 2]
			for dz in [-x[7] / 2, x[7] / 2]
                if(i == corner_num)
                    #@info "Dx"
                    #@info dx
                    #@info "Dy"
                    #@info dy
                    #@info "Dz"
                    #@info dz
                    corner = T * [dx,dy,dz,1]
                end
                i = i + 1
			end
		end
	end
    #@info "Corner"
    #@info corner
	T_world_body = get_body_transform(ego_state[4:7], ego_state[1:3]) # World -> Base
	T_body_cam1 = get_differentiable_cam_transform(1)
	T_cam_camrot = get_rotated_camera_transform()
	T_body_camrot1 = multiply_transforms(T_body_cam1, T_cam_camrot)
	T_world_camrot1 = multiply_transforms(T_world_body, T_body_camrot1)
	T_camrot1_world = invert_transform(T_world_camrot1)

    other_vehicle_corner = T_camrot1_world * corner
    bottom = focal_len * other_vehicle_corner[2]/other_vehicle_corner[3]
    bottom = convert_to_pixel_differentiable(image_height, pixel_len, bottom)
	bottom
end
function h1_right(x,corner_num,ego_state,image_width,image_height,focal_len,pixel_len)
    T = [cos(x[3]) -sin(x[3]) 0 x[1]
		 sin(x[3])  cos(x[3]) 0 x[2]
		 0 0 1 x[7]/2
		 0 0 0 1]
    #@info "T"
    #@info T
	corner = []
    meas = []
    i = 1
	for dx in [-x[5] / 2, x[5] / 2]
		for dy in [-x[6] / 2, x[6] / 2]
			for dz in [-x[7] / 2, x[7] / 2]
                if(i == corner_num)
                    #@info "Dx"
                    #@info dx
                    #@info "Dy"
                    #@info dy
                    #@info "Dz"
                    #@info dz
                    corner = T * [dx,dy,dz,1]
                end
                i = i + 1
			end
		end
	end
    #@info "Corner"
    #@info corner
	T_world_body = get_body_transform(ego_state[4:7], ego_state[1:3]) # World -> Base
	T_body_cam1 = get_differentiable_cam_transform(1)
	T_cam_camrot = get_rotated_camera_transform()
	T_body_camrot1 = multiply_transforms(T_body_cam1, T_cam_camrot)
	T_world_camrot1 = multiply_transforms(T_world_body, T_body_camrot1)
	T_camrot1_world = invert_transform(T_world_camrot1)

    other_vehicle_corner = T_camrot1_world * corner
    right = focal_len * other_vehicle_corner[1]/other_vehicle_corner[3]
    right = convert_to_pixel_differentiable(image_width, pixel_len, right)
	right
end

function h2_top(x,corner_num,ego_state,image_width,image_height,focal_len,pixel_len)
    T = [cos(x[3]) -sin(x[3]) 0 x[1]
		 sin(x[3])  cos(x[3]) 0 x[2]
		 0 0 1 x[7]/2
		 0 0 0 1]
    #@info "T"
    #@info T
	corner = []
    meas = []
    i = 1
	for dx in [-x[5] / 2, x[5] / 2]
		for dy in [-x[6] / 2, x[6] / 2]
			for dz in [-x[7] / 2, x[7] / 2]
                if(i == corner_num)
                    #@info "Dx"
                    #@info dx
                    #@info "Dy"
                    #@info dy
                    #@info "Dz"
                    #@info dz
                    corner = T * [dx,dy,dz,1]
                end
                i = i + 1
			end
		end
	end
    #@info "Corner"
    #@info corner
	T_world_body = get_body_transform(ego_state[4:7], ego_state[1:3]) # World -> Base
	T_body_cam1 = get_differentiable_cam_transform(2)
	T_cam_camrot = get_rotated_camera_transform()
	T_body_camrot1 = multiply_transforms(T_body_cam1, T_cam_camrot)
	T_world_camrot1 = multiply_transforms(T_world_body, T_body_camrot1)
	T_camrot1_world = invert_transform(T_world_camrot1)

    other_vehicle_corner = T_camrot1_world * corner
    top = focal_len * other_vehicle_corner[2]/other_vehicle_corner[3]
    top = convert_to_pixel_differentiable(image_height, pixel_len, top)
	top
end
function h2_left(x,corner_num,ego_state,image_width,image_height,focal_len,pixel_len)
    T = [cos(x[3]) -sin(x[3]) 0 x[1]
		 sin(x[3])  cos(x[3]) 0 x[2]
		 0 0 1 x[7]/2
		 0 0 0 1]
    #@info "T"
    #@info T
	corner = []
    meas = []
    i = 1
	for dx in [-x[5] / 2, x[5] / 2]
		for dy in [-x[6] / 2, x[6] / 2]
			for dz in [-x[7] / 2, x[7] / 2]
                if(i == corner_num)
                    #@info "Dx"
                    #@info dx
                    #@info "Dy"
                    #@info dy
                    #@info "Dz"
                    #@info dz
                    corner = T * [dx,dy,dz,1]
                end
                i = i + 1
			end
		end
	end
    #@info "Corner"
    #@info corner
	T_world_body = get_body_transform(ego_state[4:7], ego_state[1:3]) # World -> Base
	T_body_cam1 = get_differentiable_cam_transform(2)
	T_cam_camrot = get_rotated_camera_transform()
	T_body_camrot1 = multiply_transforms(T_body_cam1, T_cam_camrot)
	T_world_camrot1 = multiply_transforms(T_world_body, T_body_camrot1)
	T_camrot1_world = invert_transform(T_world_camrot1)

    other_vehicle_corner = T_camrot1_world * corner
    left = focal_len * other_vehicle_corner[1]/other_vehicle_corner[3]
    left = convert_to_pixel_differentiable(image_width, pixel_len, left)
	left
end

function h2_bottom(x,corner_num,ego_state,image_width,image_height,focal_len,pixel_len)
    T = [cos(x[3]) -sin(x[3]) 0 x[1]
		 sin(x[3])  cos(x[3]) 0 x[2]
		 0 0 1 x[7]/2
		 0 0 0 1]
    #@info "T"
    #@info T
	corner = []
    meas = []
    i = 1
	for dx in [-x[5] / 2, x[5] / 2]
		for dy in [-x[6] / 2, x[6] / 2]
			for dz in [-x[7] / 2, x[7] / 2]
                if(i == corner_num)
                    #@info "Dx"
                    #@info dx
                    #@info "Dy"
                    #@info dy
                    #@info "Dz"
                    #@info dz
                    corner = T * [dx,dy,dz,1]
                end
                i = i + 1
			end
		end
	end
    #@info "Corner"
    #@info corner
	T_world_body = get_body_transform(ego_state[4:7], ego_state[1:3]) # World -> Base
	T_body_cam1 = get_differentiable_cam_transform(2)
	T_cam_camrot = get_rotated_camera_transform()
	T_body_camrot1 = multiply_transforms(T_body_cam1, T_cam_camrot)
	T_world_camrot1 = multiply_transforms(T_world_body, T_body_camrot1)
	T_camrot1_world = invert_transform(T_world_camrot1)

    other_vehicle_corner = T_camrot1_world * corner
    bottom = focal_len * other_vehicle_corner[2]/other_vehicle_corner[3]
    bottom = convert_to_pixel_differentiable(image_height, pixel_len, bottom)
	bottom
end
function h2_right(x,corner_num,ego_state,image_width,image_height,focal_len,pixel_len)
    T = [cos(x[3]) -sin(x[3]) 0 x[1]
		 sin(x[3])  cos(x[3]) 0 x[2]
		 0 0 1 x[7]/2
		 0 0 0 1]
    #@info "T"
    #@info T
	corner = []
    meas = []
    i = 1
	for dx in [-x[5] / 2, x[5] / 2]
		for dy in [-x[6] / 2, x[6] / 2]
			for dz in [-x[7] / 2, x[7] / 2]
                if(i == corner_num)
                    #@info "Dx"
                    #@info dx
                    #@info "Dy"
                    #@info dy
                    #@info "Dz"
                    #@info dz
                    corner = T * [dx,dy,dz,1]
                end
                i = i + 1
			end
		end
	end
    #@info "Corner"
    #@info corner
	T_world_body = get_body_transform(ego_state[4:7], ego_state[1:3]) # World -> Base
	T_body_cam1 = get_differentiable_cam_transform(2)
	T_cam_camrot = get_rotated_camera_transform()
	T_body_camrot1 = multiply_transforms(T_body_cam1, T_cam_camrot)
	T_world_camrot1 = multiply_transforms(T_world_body, T_body_camrot1)
	T_camrot1_world = invert_transform(T_world_camrot1)

    other_vehicle_corner = T_camrot1_world * corner
    right = focal_len * other_vehicle_corner[1]/other_vehicle_corner[3]
    right = convert_to_pixel_differentiable(image_width, pixel_len, right)
	right
end

function perception_f(x, dt)
	p1 = x[1]
	p2 = x[2]
	theta = x[3]
	velocity = x[4]
	length = x[5]
	width = x[6]
	height = x[7]
	new_p1 = p1 + cos(theta) * velocity * dt
	new_p2 = p2 + sin(theta) * velocity * dt
	return [new_p1; new_p2; theta; velocity; length; width; height]'
end

function h1(x, ego_state,image_width,image_height,focal_len,pixel_len)
    #@info "X"
    #@info x
    T = [cos(x[3]) -sin(x[3]) 0 x[1]
         sin(x[3]) cos(x[3]) 0 x[2]
         0 0 1 ego_state[3]
         0 0 0 1]
    #@info "T"
    #@info T
	corners = []
    meas = []
	for dx in [-x[5] / 2, x[5] / 2]
		for dy in [-x[6] / 2, x[6] / 2]
			for dz in [-x[7] / 2, x[7] / 2]
				push!(corners, T * [dx, dy, dz, 1])
			end
		end
	end
    #@info "Corners"
    #@info corners
    
	T_world_body = get_body_transform(ego_state[4:7], ego_state[1:3]) # World -> Base
	T_body_cam1 = get_cam_transform(1)
	T_body_cam2 = get_cam_transform(2)
	T_cam_camrot = get_rotated_camera_transform()
	T_body_camrot1 = multiply_transforms(T_body_cam1, T_cam_camrot)
	T_body_camrot2 = multiply_transforms(T_body_cam2, T_cam_camrot)
	T_world_camrot1 = multiply_transforms(T_world_body, T_body_camrot1)
	T_world_camrot2 = multiply_transforms(T_world_body, T_body_camrot2)
	T_camrot1_world = invert_transform(T_world_camrot1)
    #@info "T world body"
    #@info T_world_body
    #@info "T body cam1"
    #@info T_body_cam1
    #@info "T cam camrot"
    #@info T_cam_camrot
    #@info "T camrot1 world"
    #@info T_camrot1_world
	T_camrot2_world = invert_transform(T_world_camrot2)
	i = 1
	for (camera_id, transform) in zip((1, 2), (T_camrot1_world, T_camrot2_world))
		other_vehicle_corners = [transform * pt for pt in corners] # Vehicle corners in image frame (not image units)

		left = image_width / 2
		right = -image_width / 2
		top = image_height / 2
		bot = -image_height / 2
        left_i = 0
        right_i = 0
        top_i = 0
        bottom_i = 0
        i = 1
        for corner in other_vehicle_corners
            if corner[3] < focal_len
                break
            end
            px = focal_len*corner[1]/corner[3] # Finds vehicle corner x in pixel space
            py = focal_len*corner[2]/corner[3] # Finds vehicle corner y in pixel space
            left = min(left, px)
            right = max(right, px)
            top = min(top, py)
            bot = max(bot, py)
            if(left == px)
                left_i = i
            end
            if(right == px)
                right_i = i
            end
            if(top == py)
                top_i = i
            end
            if(bot == py)
                bottom_i = i
            end
            i = i + 1
        end
        if top ≈ bot || left ≈ right || top > bot || left > right
            # out of frame
            continue
        else
            # Normalizes the pixels
            top = convert_to_pixel(image_height, pixel_len, top)
            bot = convert_to_pixel(image_height, pixel_len, bot)
            left = convert_to_pixel(image_width, pixel_len, left)
            right = convert_to_pixel(image_width, pixel_len, right)
            push!(meas,top)
            push!(meas,top_i)
            push!(meas,left)
            push!(meas,left_i)
            push!(meas,bot)
            push!(meas,bottom_i)
            push!(meas,right)
            push!(meas,right_i)
        end
	end
	meas
end

function perception_jac_fx(x, Δ)
	jacobian(state -> perception_f(state, Δ), x)[1]
end

function perception_jac_h1top(x,corner_num,ego_state,image_width,image_height,focal_len,pixel_len)
	jacobian(state -> h1_top(state,corner_num,ego_state,image_width,image_height,focal_len,pixel_len), x)[1]
end

function perception_jac_h1left(x,corner_num,ego_state,image_width,image_height,focal_len,pixel_len)
	jacobian(state -> h1_left(state,corner_num,ego_state,image_width,image_height,focal_len,pixel_len), x)[1]
end

function perception_jac_h1bottom(x,corner_num,ego_state,image_width,image_height,focal_len,pixel_len)
	jacobian(state -> h1_bottom(state,corner_num,ego_state,image_width,image_height,focal_len,pixel_len), x)[1]
end

function perception_jac_h1right(x,corner_num,ego_state,image_width,image_height,focal_len,pixel_len)
	jacobian(state -> h1_right(state,corner_num,ego_state,image_width,image_height,focal_len,pixel_len), x)[1]
end

function perception_jac_h2top(x,corner_num,ego_state,image_width,image_height,focal_len,pixel_len)
	jacobian(state -> h2_top(state,corner_num,ego_state,image_width,image_height,focal_len,pixel_len), x)[1]
end

function perception_jac_h2left(x,corner_num,ego_state,image_width,image_height,focal_len,pixel_len)
	jacobian(state -> h2_left(state,corner_num,ego_state,image_width,image_height,focal_len,pixel_len), x)[1]
end

function perception_jac_h2bottom(x,corner_num,ego_state,image_width,image_height,focal_len,pixel_len)
	jacobian(state -> h2_bottom(state,corner_num,ego_state,image_width,image_height,focal_len,pixel_len), x)[1]
end

function perception_jac_h2right(x,corner_num,ego_state,image_width,image_height,focal_len,pixel_len)
	jacobian(state -> h2_right(state,corner_num,ego_state,image_width,image_height,focal_len,pixel_len), x)[1]
end
