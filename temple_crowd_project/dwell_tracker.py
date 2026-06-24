# dwell_tracker.py
# Tracks how long each DeepSORT ID stays visible in the YOLO near-field region.
# DM-Count branch has no interaction with this class whatsoever.

class DwellTracker:

    def __init__(self, fps):
        """
        fps : float — actual video FPS from cap.get(cv2.CAP_PROP_FPS)
               Used to convert frame counts to real seconds.
        """
        self.fps = fps

        # { track_id : entry_frame }
        # Stores the frame number when each ID was first seen
        self.entry_frames = {}

        # { track_id : duration_seconds }
        # Populated when an ID disappears or video ends
        self.completed = {}

    # ------------------------------------------------------------------

    def update(self, active_ids, current_frame):
        """
        Call once per frame after DeepSORT output.

        active_ids    : set of int  — confirmed track IDs visible THIS frame
        current_frame : int         — the current frame_id counter from app.py
        """

        # --- Register any new IDs appearing for the first time ---
        for tid in active_ids:
            if tid not in self.entry_frames:
                self.entry_frames[tid] = current_frame

        # --- Detect IDs that were active last frame but gone now ---
        previously_active = set(self.entry_frames.keys()) - set(self.completed.keys())
        disappeared = previously_active - active_ids

        for tid in disappeared:
            duration = (current_frame - self.entry_frames[tid]) / self.fps
            self.completed[tid] = round(duration, 2)

    # ------------------------------------------------------------------

    def get_live_duration(self, tid, current_frame):
        """
        Returns running duration in seconds for a currently active ID.
        Used for the live overlay on each bounding box.
        """
        if tid in self.entry_frames:
            return round((current_frame - self.entry_frames[tid]) / self.fps, 1)
        return 0.0

    # ------------------------------------------------------------------

    def finalise(self, current_frame):
        """
        Call ONCE after the video loop ends.
        Closes out any IDs still active in the final frame.
        """
        still_active = set(self.entry_frames.keys()) - set(self.completed.keys())

        for tid in still_active:
            duration = (current_frame - self.entry_frames[tid]) / self.fps
            self.completed[tid] = round(duration, 2)

    # ------------------------------------------------------------------

    def report(self):
        """
        Returns a dict with full dwell time statistics.
        Call after finalise().
        """
        if not self.completed:
            return {
                "total_ids"       : 0,
                "mean_dwell_sec"  : 0.0,
                "per_id"          : {}
            }

        durations = list(self.completed.values())
        mean_dwell = round(sum(durations) / len(durations), 2)

        return {
            "total_ids"      : len(self.completed),
            "mean_dwell_sec" : mean_dwell,
            "per_id"         : self.completed
        }