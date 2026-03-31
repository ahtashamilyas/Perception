PROBLEM:
  FoundationPose was detecting YCB dataset objects correctly but failed to
  detect a custom cube (obj_000022).

ROOT CAUSE:
  Unit/Scale mismatch between the cube mesh and YCB meshes.

  - YCB meshes (e.g. obj_000001.ply) store vertices in MILLIMETERS
    (values like -45.03, -50.44, -66.93).
  - The custom cube mesh (obj_000022.ply) stored vertices in METERS
    (values like 0.025, 0.05).
  - The code in integration_test.py (line ~250) unconditionally scales
    all meshes by 1e-3 to convert mm → m:
        loaded_mesh.vertices *= 1e-3
  - This caused the cube (already in meters) to shrink by 1000x, making
    it ~0.025mm — essentially invisible to the pose estimator.

FIX APPLIED:
  Rescaled obj_000022.ply vertex coordinates from meters to millimeters
  (multiplied x, y, z by 1000) so it matches the YCB convention.

  Before: 0.025 0 0 ...     (meters)
  After:  25.0  0 0 ...     (millimeters)

ADDITIONAL NOTE:
  The cube mesh has only 36 vertices / 12 faces — very low compared to
  YCB objects (~10,000 vertices). If detection is still unreliable,
  subdividing the mesh in Blender or MeshLab (or via trimesh) will help.
