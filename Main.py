from O1_BallTracking.ball_tracker import cBallTracker

### Software: Headis Analysis Framework ###

### Set videopath
#video_path = "C:/Users/Fabian/Documents/Headis Mainz/UniprojektHeadisData/VideoDataAruco/20211001_221603.mp4"
#video_path = "C:/Users/Fabian/Documents/Headis Mainz/UniprojektHeadisData/VideoDataAruco/20211028_144334.mp4"
#video_path = "C:/Users/Fabian/Documents/Headis Mainz/UniprojektHeadisData/VideoDataAruco/20211028_203021.mp4"
#video_path = "C:/Users/Fabian/Documents/Headis Mainz/UniprojektHeadisData/VideoDataAruco/20211028_212444.mp4"
#video_path = "C:/Users/Fabian/Documents/Headis Mainz/UniprojektHeadisData/VideoDataAruco/20211028_224333.mp4"
#video_path = "C:/Users/Fabian/Documents/Headis Mainz/UniprojektHeadisData/VideoDataAruco/20211114_112646.mp4"
#video_path = "C:/Users/Fabian/Documents/Headis Mainz/UniprojektHeadisData/VideoDataAruco/20211114_125247.mp4"
video_path = "C:/Users/Fabian/Documents/Headis Mainz/UniprojektHeadisData/VideoDataAruco/20211114_170239.mp4"
#video_path = "C:/Users/Fabian/Documents/Headis Mainz/UniprojektHeadisData/VideoMarkerLines/20220111_211213.mp4"

### Ball tracking analysis => Data points in 3D (pos x y z, time point)
ball_tracker = cBallTracker()
data_points_df = ball_tracker.get_3d_positions_from_video(video_path)

### Measurement Filtering => Parametrized ball curves in 3D (curve parameters, time start, time end)
parametrized_curves_df = None

### Play visualization => visualize play in blender


### Event Analysis => Event type categorization (rally_id, hit_type_id, player_id, time point, pos x y z, velocity before x y z, velocity after x y z)
event_log_df = None

### Event Visualization => Ball-plate position PDF, Ball-player hit positions PDF, after ball-player hit velocity PDF, ...


### Strength/Weakness Detection =>


### Player Profile => Accuracy, Defense, Offense, Speed, Power