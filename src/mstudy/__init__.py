"""Video-based motion and time study for manufacturing operations."""

__version__ = "0.1.0"

# COCO-17 keypoint indices, used throughout for motion measurement.
KP_NOSE = 0
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12

# The joints that actually indicate "working" vs "idle" in manual assembly:
# hands and forearms move during work even when the body is planted.
WORK_KEYPOINTS = [KP_L_ELBOW, KP_R_ELBOW, KP_L_WRIST, KP_R_WRIST]

# Sentinel step labels. Double underscores keep them from colliding with
# user-defined zone names.
LABEL_ABSENT = "__absent__"  # not detected anywhere in frame
LABEL_OUTSIDE = "__outside__"  # detected, but in no defined zone
