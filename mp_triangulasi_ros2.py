import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
import numpy as np
import cv2
import mediapipe as mp
from sensor_msgs.msg import PointCloud2

class BodyLandmarksPublisher(Node):
    def __init__(self):
        super().__init__('body_landmarks_publisher')
        self.publisher = self.create_publisher(PointCloud2, 'body_landmarks', 10)
        self.timer = self.create_timer(0.1, self.publish_landmarks)

        self.cap = cv2.VideoCapture(0)
        self.mp_holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def publish_landmarks(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame.")
            return

        # Process image
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_holistic.process(image)

        if results.pose_landmarks:
            points = []
            for landmark in results.pose_landmarks.landmark:
                points.append([landmark.x, landmark.y, landmark.z])

            # Publish keypoints as PointCloud2
            self.publisher.publish(self.create_pointcloud2(points))

    def create_pointcloud2(self, points):
        """Create a PointCloud2 message from a list of points."""
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        header = self.get_clock().now().to_msg()
        header.frame_id = 'map'
        points_array = np.array(points, dtype=np.float32)
        return PointCloud2(
            header=header,
            height=1,
            width=len(points_array),
            is_dense=True,
            is_bigendian=False,
            fields=fields,
            point_step=12,
            row_step=12 * len(points_array),
            data=points_array.tobytes(),
        )

    def destroy(self):
        self.cap.release()
        super().destroy()

def main(args=None):
    rclpy.init(args=args)
    node = BodyLandmarksPublisher()
    rclpy.spin(node)
    node.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
