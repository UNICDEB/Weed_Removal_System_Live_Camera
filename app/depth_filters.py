import pyrealsense2 as rs

def apply_depth_filters(depth_frame):

    # 1️⃣ Decimation
    decimation = rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 1)
    depth_frame = decimation.process(depth_frame)

    # 2️⃣ Depth to Disparity
    depth_to_disparity = rs.disparity_transform(True)
    depth_frame = depth_to_disparity.process(depth_frame)

    # 3️⃣ Spatial Filter
    spatial = rs.spatial_filter()
    spatial.set_option(rs.option.filter_magnitude, 2)
    spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
    spatial.set_option(rs.option.filter_smooth_delta, 20)
    depth_frame = spatial.process(depth_frame)

    # 4️⃣ Temporal Filter
    temporal = rs.temporal_filter()
    depth_frame = temporal.process(depth_frame)

    # 5️⃣ Disparity to Depth
    disparity_to_depth = rs.disparity_transform(False)
    depth_frame = disparity_to_depth.process(depth_frame)

    # VERY IMPORTANT
    depth_frame = depth_frame.as_depth_frame()

    return depth_frame
