#!/usr/bin/env python3
"""
Point Cloud Registration and Correspondence Script for 3D Gaussian Splatting
This script registers multiple 3D Gaussian point clouds, finds correspondences, and saves residual points.
Preserves all 3D Gaussian Splatting attributes (position, normals, colors, opacity, scale, rotation, etc.)
"""

import numpy as np
import open3d as o3d
import os
from sklearn.neighbors import NearestNeighbors
import argparse
import copy
import plyfile

def construct_gaussian_attributes_list(features_dc_shape, features_rest_shape, scaling_shape, rotation_shape):
    """Construct list of 3D Gaussian Splatting attributes in standard order"""
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels for DC features (usually 3 for RGB)
    for i in range(features_dc_shape[1] * features_dc_shape[2]):  # shape: [N, 1, 3] -> 3 attributes
        l.append('f_dc_{}'.format(i))
    # All channels for rest features (spherical harmonics)
    for i in range(features_rest_shape[1] * features_rest_shape[2]):  # shape: [N, 15, 3] -> 45 attributes
        l.append('f_rest_{}'.format(i))
    l.append('opacity')
    for i in range(scaling_shape[1]):  # shape: [N, 3] -> 3 attributes
        l.append('scale_{}'.format(i))
    for i in range(rotation_shape[1]):  # shape: [N, 4] -> 4 attributes
        l.append('rot_{}'.format(i))
    return l

def load_gaussian_ply_advanced(file_path):
    """Load 3D Gaussian Splatting PLY file using the same method as GaussianModel"""
    print(f"Loading 3D Gaussian PLY from {file_path}")
    
    # Read with plyfile
    plydata = plyfile.PlyData.read(file_path)
    vertex_data = plydata['vertex']
    
    # Get all properties
    properties = [prop.name for prop in vertex_data.properties]
    print(f"Gaussian properties: {properties}")
    
    # Extract basic data
    xyz = np.stack((np.asarray(vertex_data["x"]),
                    np.asarray(vertex_data["y"]),
                    np.asarray(vertex_data["z"])), axis=1)
    opacities = np.asarray(vertex_data["opacity"])[..., np.newaxis]
    
    # Extract DC features (usually f_dc_0, f_dc_1, f_dc_2)
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    if 'f_dc_0' in properties:
        features_dc[:, 0, 0] = np.asarray(vertex_data["f_dc_0"])
    if 'f_dc_1' in properties:
        features_dc[:, 1, 0] = np.asarray(vertex_data["f_dc_1"])
    if 'f_dc_2' in properties:
        features_dc[:, 2, 0] = np.asarray(vertex_data["f_dc_2"])
    
    # Extract rest features (spherical harmonics)
    extra_f_names = [p.name for p in vertex_data.properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
    
    if len(extra_f_names) > 0:
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(vertex_data[attr_name])
        # Reshape to [N, 3, sh_degree_features]
        sh_features_per_channel = len(extra_f_names) // 3
        features_rest = features_extra.reshape((features_extra.shape[0], 3, sh_features_per_channel))
    else:
        # No rest features, create empty array
        features_rest = np.zeros((xyz.shape[0], 3, 0))
    
    # Extract scaling
    scale_names = [p.name for p in vertex_data.properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(vertex_data[attr_name])
    
    # Extract rotation
    rot_names = [p.name for p in vertex_data.properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(vertex_data[attr_name])
    
    # Extract normals (usually zeros in Gaussian Splatting)
    normals = np.zeros_like(xyz)
    if 'nx' in properties and 'ny' in properties and 'nz' in properties:
        normals[:, 0] = np.asarray(vertex_data["nx"])
        normals[:, 1] = np.asarray(vertex_data["ny"])
        normals[:, 2] = np.asarray(vertex_data["nz"])
    
    # Store structured data
    gaussian_data = {
        'xyz': xyz,
        'normals': normals,
        'features_dc': features_dc,  # [N, 3, 1]
        'features_rest': features_rest,  # [N, 3, rest_features]
        'opacities': opacities,
        'scales': scales,
        'rotations': rots
    }
    
    # Create Open3D point cloud for registration algorithms
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Add normals if they exist and are non-zero
    if np.any(normals != 0):
        pcd.normals = o3d.utility.Vector3dVector(normals)
    
    # Convert DC components to RGB colors for visualization
    if features_dc.shape[2] > 0:
        # Convert spherical harmonics DC to RGB
        sh_dc = features_dc[:, :, 0]  # Take the first (and usually only) DC component
        # SH to RGB conversion: RGB = (DC * C0) + 0.5, where C0 = 0.28209479177387814
        C0 = 0.28209479177387814
        colors = sh_dc * C0 + 0.5
        colors = np.clip(colors, 0, 1)  # Ensure values are in [0, 1]
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"Loaded {len(xyz)} Gaussians with structured data")
    print(f"  Features DC shape: {features_dc.shape}")
    print(f"  Features rest shape: {features_rest.shape}")
    print(f"  Scales shape: {scales.shape}")
    print(f"  Rotations shape: {rots.shape}")
    
    return pcd, gaussian_data, properties

def load_point_cloud(file_path):
    """Load point cloud from PLY file, preserving all attributes (wrapper for compatibility)"""
    if file_path.endswith('.ply'):
        try:
            # Try to load as 3D Gaussian Splatting format first
            return load_gaussian_ply_advanced(file_path)
        except:
            # Fallback to standard Open3D loading
            print(f"Loading standard PLY from {file_path}")
            pcd = o3d.io.read_point_cloud(file_path)
            if len(pcd.points) == 0:
                raise ValueError(f"Failed to load point cloud from {file_path}")
            return pcd, None, None
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def analyze_ply_attributes(file_path):
    """Analyze and report detailed 3D Gaussian PLY file attributes"""
    print(f"\n=== Analyzing 3D Gaussian PLY file: {file_path} ===")
    
    try:
        plydata = plyfile.PlyData.read(file_path)
        vertex_data = plydata['vertex']
        
        # Get properties
        properties = [prop.name for prop in vertex_data.properties]
        gaussian_data = np.array(vertex_data.data)
        
        print(f"Number of Gaussians: {len(gaussian_data)}")
        print(f"Properties ({len(properties)}): {properties}")
        
        # Show value ranges for key properties
        if len(gaussian_data) > 0:
            print(f"\nProperty value ranges:")
            for prop in properties:
                values = gaussian_data[prop]
                print(f"  {prop}: [{values.min():.6f}, {values.max():.6f}] (mean: {values.mean():.6f})")
        
        # Check for standard 3D Gaussian properties
        expected_props = ['x', 'y', 'z', 'opacity', 'scale_0', 'scale_1', 'scale_2', 
                         'rot_0', 'rot_1', 'rot_2', 'rot_3']
        missing_props = [prop for prop in expected_props if prop not in properties]
        if missing_props:
            print(f"Missing standard Gaussian properties: {missing_props}")
        else:
            print("✓ All standard 3D Gaussian properties present")
        
        return gaussian_data, properties
        
    except Exception as e:
        print(f"Error analyzing PLY file: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def preprocess_point_cloud(pcd, voxel_size=0.05):
    """Preprocess point cloud: downsample and estimate normals"""
    print(f"Preprocessing point cloud with voxel size {voxel_size}")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """Execute global registration using RANSAC"""
    distance_threshold = voxel_size * 1.5
    print("Executing global registration...")
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    
    return result

def execute_fast_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    """Execute fast global registration"""
    distance_threshold = voxel_size * 0.5
    print("Executing fast global registration...")
    
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    
    return result

def refine_registration(source, target, transformation, voxel_size):
    """Refine registration using ICP"""
    distance_threshold = voxel_size * 0.4
    print("Refining registration with ICP...")
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    return result

def find_correspondences_optimized(source_points, target_points, threshold=0.1, ratio_test=0.8):
    """
    Find point correspondences with optimized matching strategy
    - Uses bidirectional matching for better consistency
    - Applies ratio test to filter ambiguous matches
    - Maximizes common parts between frames
    """
    print(f"Finding optimized correspondences with threshold {threshold}, ratio test {ratio_test}")
    
    # Forward matching: source -> target
    nbrs_target = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(target_points)
    distances_st, indices_st = nbrs_target.kneighbors(source_points)
    
    # Backward matching: target -> source  
    nbrs_source = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(source_points)
    distances_ts, indices_ts = nbrs_source.kneighbors(target_points)
    
    correspondences = []
    residual_indices = []
    matched_target_indices = set()
    
    for i in range(len(source_points)):
        # Get two nearest neighbors in target
        dist1, dist2 = distances_st[i]
        idx1, idx2 = indices_st[i]
        
        # Apply distance threshold
        if dist1 > threshold:
            residual_indices.append(i)
            continue
        
        # Apply ratio test (Lowe's ratio test for better matching)
        if len(distances_st[i]) > 1 and dist2 > 0:
            ratio = dist1 / dist2
            if ratio > ratio_test:
                residual_indices.append(i)
                continue
        
        # Bidirectional consistency check
        best_target_idx = idx1
        if best_target_idx < len(indices_ts):
            # Check if target point maps back to current source point
            back_match_idx = indices_ts[best_target_idx][0]
            back_match_dist = distances_ts[best_target_idx][0]
            
            # Allow some tolerance for bidirectional matching
            if abs(back_match_idx - i) <= 1 or back_match_dist < threshold * 1.5:
                # Avoid duplicate matches in target
                if best_target_idx not in matched_target_indices:
                    correspondences.append((i, best_target_idx, dist1))
                    matched_target_indices.add(best_target_idx)
                else:
                    residual_indices.append(i)
            else:
                residual_indices.append(i)
        else:
            residual_indices.append(i)
    
    print(f"Optimized matching found {len(correspondences)} correspondences, {len(residual_indices)} residual points")
    print(f"Match ratio: {len(correspondences)/(len(correspondences)+len(residual_indices))*100:.1f}%")
    return correspondences, residual_indices

def find_correspondences(source_points, target_points, threshold=0.1):
    """Wrapper for backward compatibility"""
    return find_correspondences_optimized(source_points, target_points, threshold)

def save_gaussian_ply_advanced(gaussian_data, indices, output_path, transformation=None):
    """Save 3D Gaussian Splatting PLY file using the same method as GaussianModel"""
    if len(indices) == 0:
        print("No Gaussians to save")
        return False
    
    # Extract subset of Gaussians
    xyz = gaussian_data['xyz'][indices]
    normals = gaussian_data['normals'][indices]
    features_dc = gaussian_data['features_dc'][indices]  # [N, 3, 1]
    features_rest = gaussian_data['features_rest'][indices]  # [N, 3, rest_features]
    opacities = gaussian_data['opacities'][indices]
    scales = gaussian_data['scales'][indices]
    rots = gaussian_data['rotations'][indices]
    
    # Apply transformation if provided
    if transformation is not None:
        # Transform positions
        ones = np.ones((len(xyz), 1))
        xyz_homo = np.hstack([xyz, ones])
        xyz = (transformation @ xyz_homo.T).T[:, :3]
        
        # Transform normals if they exist and are non-zero
        if np.any(normals != 0):
            # Only apply rotation part to normals
            rotation_matrix = transformation[:3, :3]
            normals = (rotation_matrix @ normals.T).T
    
    # Prepare features according to GaussianModel format
    # Transpose and flatten DC features: [N, 3, 1] -> [N, 1, 3] -> [N, 3]
    f_dc = features_dc.transpose(0, 2, 1).reshape(features_dc.shape[0], -1)
    
    # Transpose and flatten rest features: [N, 3, rest] -> [N, rest, 3] -> [N, rest*3]
    if features_rest.shape[2] > 0:
        f_rest = features_rest.transpose(0, 2, 1).reshape(features_rest.shape[0], -1)
    else:
        f_rest = np.zeros((features_dc.shape[0], 0))
    
    # Construct attribute list based on actual data shapes
    attribute_list = construct_gaussian_attributes_list(
        features_dc.shape, features_rest.shape, scales.shape, rots.shape)
    
    # Create dtype for structured array
    dtype_full = [(attribute, 'f4') for attribute in attribute_list]
    
    # Create structured array
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    
    # Concatenate all attributes in the correct order
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scales, rots), axis=1)
    elements[:] = list(map(tuple, attributes))
    
    # Create PLY element and save
    try:
        el = plyfile.PlyElement.describe(elements, 'vertex')
        plyfile.PlyData([el]).write(output_path)
        print(f"Saved {len(elements)} Gaussians to {output_path}")
        return True
    except Exception as e:
        print(f"Failed to save Gaussians to {output_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_matched_gaussian_pair(source_gaussian, target_gaussian, correspondences, output_dir, source_name, target_name):
    """
    Create matched Gaussian pairs with consistent point count and ordering
    Ensures frame consistency for main PLY files
    """
    if len(correspondences) == 0:
        print("No correspondences found, cannot create matched pair")
        return False, False
    
    # Extract correspondence indices
    source_indices = [corr[0] for corr in correspondences]
    target_indices = [corr[1] for corr in correspondences]
    
    print(f"Creating matched pair with {len(correspondences)} common Gaussians")
    
    # Create matched source part
    matched_source_path = os.path.join(output_dir, f"{source_name}_matched.ply")
    success_source = save_gaussian_ply_advanced(
        source_gaussian, source_indices, matched_source_path
    )
    
    # Create matched target part  
    matched_target_path = os.path.join(output_dir, f"{target_name}_matched.ply")
    success_target = save_gaussian_ply_advanced(
        target_gaussian, target_indices, matched_target_path
    )
    
    if success_source and success_target:
        print(f"✅ Created matched pair:")
        print(f"   Source: {matched_source_path}")
        print(f"   Target: {matched_target_path}")
        print(f"   Common points: {len(correspondences)}")
        
        # Verify point count consistency
        try:
            # Quick verification by reading point counts
            import plyfile
            source_ply = plyfile.PlyData.read(matched_source_path)
            target_ply = plyfile.PlyData.read(matched_target_path) 
            source_count = len(source_ply['vertex'])
            target_count = len(target_ply['vertex'])
            
            if source_count == target_count:
                print(f"✅ Point count verification passed: {source_count} == {target_count}")
            else:
                print(f"⚠️ Point count mismatch: {source_count} != {target_count}")
                
        except Exception as e:
            print(f"⚠️ Could not verify point counts: {e}")
    
    return success_source, success_target

def save_residual_gaussians(gaussian_data, residual_indices, output_path, transformation=None, preserve_color=True):
    """Save residual 3D Gaussians to PLY file with option to preserve original colors"""
    if len(residual_indices) == 0:
        print("No residual Gaussians to save")
        return False
    
    # Create a copy of gaussian data for residual points
    residual_data = {}
    for key in gaussian_data:
        residual_data[key] = gaussian_data[key][residual_indices].copy()
    
    # Optionally modify opacity to make residual points more visible
    # residual_data['opacities'] = np.maximum(residual_data['opacities'], 0.5)
    
    # Preserve original color attributes by default
    if not preserve_color and residual_data['features_dc'].shape[2] > 0:
        # Only change color to red if preserve_color is False
        C0 = 0.28209479177387814
        red_sh = (np.array([1.0, 0.0, 0.0]) - 0.5) / C0  # Convert RGB red to SH
        residual_data['features_dc'][:, :, 0] = red_sh.reshape(1, 3)
        print("Applied red color to residual points")
    else:
        print("Preserved original color attributes for residual points")
    
    return save_gaussian_ply_advanced(residual_data, np.arange(len(residual_indices)), 
                                    output_path, transformation)

def save_point_cloud_preserving_format(pcd, gaussian_data, properties, output_path, transformation=None):
    """Save 3D Gaussian point cloud preserving original format and attributes"""
    if gaussian_data is not None:
        # Save as 3D Gaussian Splatting PLY
        all_indices = np.arange(len(gaussian_data['xyz']))
        return save_gaussian_ply_advanced(gaussian_data, all_indices, output_path, transformation)
    else:
        # Fallback to standard Open3D saving
        points = np.asarray(pcd.points)
        num_points = len(points)
        
        # Create a clean copy
        output_pcd = o3d.geometry.PointCloud()
        output_pcd.points = o3d.utility.Vector3dVector(points)
        
        # Preserve colors if they exist and match point count
        if len(pcd.colors) > 0:
            colors = np.asarray(pcd.colors)
            if len(colors) == num_points:
                output_pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                print(f"Warning: Color count ({len(colors)}) doesn't match point count ({num_points})")
        
        # Preserve normals if they exist and match point count
        if len(pcd.normals) > 0:
            normals = np.asarray(pcd.normals)
            if len(normals) == num_points:
                output_pcd.normals = o3d.utility.Vector3dVector(normals)
            else:
                print(f"Warning: Normal count ({len(normals)}) doesn't match point count ({num_points})")
        
        # Write point cloud
        success = o3d.io.write_point_cloud(output_path, output_pcd)
        if success:
            print(f"Saved point cloud with {num_points} points to {output_path}")
        else:
            print(f"Failed to save point cloud to {output_path}")
        
        return success

def save_correspondences(correspondences, output_path):
    """Save correspondences to text file"""
    with open(output_path, 'w') as f:
        f.write("source_idx,target_idx,distance\n")
        for src_idx, tgt_idx, dist in correspondences:
            f.write(f"{src_idx},{tgt_idx},{dist:.6f}\n")
    print(f"Saved {len(correspondences)} correspondences to {output_path}")

def register_point_clouds_optimized(input_dir, output_dir, voxel_size=0.05, correspondence_threshold=0.08, ratio_test=0.75, preserve_residual_color=True):
    """
    Optimized 3D Gaussian point cloud registration with maximized common parts
    Creates matched pairs and residual files for frame consistency
    
    Args:
        preserve_residual_color: If True, preserve original colors in residual files; if False, mark as red
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load 3D Gaussian point clouds
    ply_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.ply')])
    if len(ply_files) < 2:
        raise ValueError("Need at least 2 PLY files for registration")
    
    print(f"Processing {len(ply_files)} PLY files for registration")
    print(f"Parameters: voxel_size={voxel_size}, correspondence_threshold={correspondence_threshold}, ratio_test={ratio_test}")
    print(f"Residual color preservation: {'Enabled' if preserve_residual_color else 'Disabled (red marking)'}")
    
    point_clouds = []
    gaussian_datasets = []
    properties_list = []
    
    for ply_file in ply_files:
        file_path = os.path.join(input_dir, ply_file)
        pcd, gaussian_data, properties = load_point_cloud(file_path)
        point_clouds.append(pcd)
        gaussian_datasets.append(gaussian_data)
        properties_list.append(properties)
    
    # Use first point cloud as reference
    reference_pcd = point_clouds[0]
    reference_gaussian = gaussian_datasets[0]
    reference_properties = properties_list[0]
    reference_name = ply_files[0][:-4]
    
    # Statistics tracking
    total_correspondences = 0
    total_residuals = 0
    
    # Register each point cloud to reference
    for i, (pcd, gaussian_data, properties) in enumerate(zip(point_clouds[1:], gaussian_datasets[1:], properties_list[1:]), 1):
        current_name = ply_files[i][:-4]
        print(f"Registering {ply_files[i]} to reference {ply_files[0]}")
        
        # Create a copy for processing (preserve original)
        pcd_copy = copy.deepcopy(pcd)
        
        # Preprocess for registration (use downsampled versions)
        source_down, source_fpfh = preprocess_point_cloud(pcd_copy, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(reference_pcd, voxel_size)
        
        # Global registration
        result_ransac = execute_global_registration(
            source_down, target_down, source_fpfh, target_fpfh, voxel_size)
        
        # Refine with ICP
        result_icp = refine_registration(
            source_down, target_down, result_ransac.transformation, voxel_size)
        
        print(f"Registration fitness: {result_icp.fitness:.4f}, RMSE: {result_icp.inlier_rmse:.4f}")
        
        # Apply transformation to the original full-resolution point cloud
        pcd_registered = copy.deepcopy(pcd)
        pcd_registered.transform(result_icp.transformation)
        
        # Find correspondences with optimized algorithm
        source_points = np.asarray(pcd_registered.points)
        target_points = np.asarray(reference_pcd.points)
        
        correspondences, residual_indices = find_correspondences_optimized(
            source_points, target_points, correspondence_threshold, ratio_test)
        
        total_correspondences += len(correspondences)
        total_residuals += len(residual_indices)
        
        # Create matched Gaussian pairs (main PLY files with consistent points)
        if gaussian_data is not None and reference_gaussian is not None:
            success_source, success_target = create_matched_gaussian_pair(
                gaussian_data, reference_gaussian, correspondences, 
                output_dir, current_name, reference_name
            )
            if len(residual_indices) > 0:
                residual_path = os.path.join(output_dir, f"{current_name}_residual.ply")
                save_residual_gaussians(
                    gaussian_data, residual_indices, residual_path, result_icp.transformation, preserve_color=preserve_residual_color
                )
        
        # Save correspondences for analysis
        corr_path = os.path.join(output_dir, f"correspondences_{reference_name}_{current_name}.csv")
        save_correspondences(correspondences, corr_path)
        
        # Save transformation matrix
        transform_path = os.path.join(output_dir, f"transformation_{current_name}.txt")
        np.savetxt(transform_path, result_icp.transformation, fmt='%.6f')
        print(f"💾 Transformation matrix saved to {transform_path}")
        
        # Print frame statistics
        print(f"Frame {i}: original={len(source_points)}, matched={len(correspondences)}, residual={len(residual_indices)}")
    
    # Save reference as matched reference for consistency
    ref_matched_path = os.path.join(output_dir, f"{reference_name}_matched.ply")
    save_point_cloud_preserving_format(reference_pcd, reference_gaussian, reference_properties, 
                                     ref_matched_path)
    
    # Final summary
    print(f"Summary: processed={len(ply_files)}, total_correspondences={total_correspondences}, total_residuals={total_residuals}, match_rate={total_correspondences/(total_correspondences+total_residuals)*100:.1f}%")

def register_point_clouds(input_dir, output_dir, voxel_size=0.05, correspondence_threshold=0.1):
    """Main function with backward compatibility - calls optimized version with color preservation"""
    return register_point_clouds_optimized(input_dir, output_dir, voxel_size, correspondence_threshold, preserve_residual_color=True)

def main():
    parser = argparse.ArgumentParser(description='Optimized 3D Gaussian Splatting Point Cloud Registration')
    parser.add_argument('--input_dir', type=str, default='data_ff/spring', 
                       help='Input directory containing 3D Gaussian PLY files')
    parser.add_argument('--output_dir', type=str, default='output_registration',
                       help='Output directory for results')
    parser.add_argument('--voxel_size', type=float, default=0.05,
                       help='Voxel size for downsampling during registration')
    parser.add_argument('--correspondence_threshold', type=float, default=0.08,
                       help='Distance threshold for correspondences (lower = more common parts)')
    parser.add_argument('--ratio_test', type=float, default=0.75,
                       help='Ratio test threshold for ambiguous matches (lower = stricter)')
    parser.add_argument('--analyze_only', action='store_true',
                       help='Only analyze 3D Gaussian PLY files without performing registration')
    # legacy_mode removed
    parser.add_argument('--mark_residual_red', action='store_true',
                       help='Mark residual points as red instead of preserving original colors')
    
    args = parser.parse_args()
    
    try:
        # Analyze input files if requested
        if args.analyze_only:
            print("=== 3D Gaussian PLY File Analysis Mode ===")
            ply_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.ply')])
            for ply_file in ply_files:
                analyze_ply_attributes(os.path.join(args.input_dir, ply_file))
            return
        
        preserve_color = not args.mark_residual_red
        register_point_clouds_optimized(
            args.input_dir, 
            args.output_dir, 
            args.voxel_size, 
            args.correspondence_threshold,
            args.ratio_test,
            preserve_residual_color=preserve_color
        )
        print("Registration completed.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
