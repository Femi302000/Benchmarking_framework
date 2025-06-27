import rclpy
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener
import rosbag2_py
from rosbag2_py import StorageOptions, ConverterOptions
from tf2_ros import TransformException
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


class BagTfProcessor(Node) :
    def __init__(self) :
        super().__init__('bag_tf_processor_ros2')
        self.tf_buffer = Buffer()
        # In ROS 2, the TransformListener automatically uses the node's clock
        # which is usually the ROS clock (rclpy.clock.Clock)
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def read_tf_from_bag(self, bag_file_path) :
        self.get_logger().info(f"Reading TF from bag: {bag_file_path}")


        # Rosbag2 reader setup
        storage_options = StorageOptions(uri=bag_file_path, storage_id='sqlite3')  # or 'mcap'
        converter_options = ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr"
        )
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)

        # Get topic information to deserialize messages correctly
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic_type.name : topic_type.type for topic_type in topic_types}

        # Filter for TF topics
        tf_topics = ['/tf', '/tf_static']

        storage_filter = rosbag2_py.StorageFilter(topics=tf_topics)
        reader.set_filter(storage_filter)

        # Iterate through messages
        while reader.has_next() :
            topic, data, timestamp = reader.read_next()
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(data, msg_type)

            if topic == '/tf' :
                if isinstance(msg, TFMessage) :
                    for transform_stamped in msg.transforms :
                        self.tf_buffer.set_transform(transform_stamped, "bag_reader")
                elif isinstance(msg, TransformStamped) :
                    # Rarely, a single TransformStamped might be published on /tf
                    self.tf_buffer.set_transform(msg, "bag_reader")
            if topic == '/tf_static' :
                if isinstance(msg, TFMessage) :
                    for transform_stamped in msg.transforms :
                        self.tf_buffer.set_transform_static(transform_stamped, "bag_reader")
                elif isinstance(msg, TransformStamped) :
                    # Rarely, a single TransformStamped might be published on /tf
                    self.tf_buffer.set_transform_static(msg, "bag_reader")
            # You could add other message types here if you need to process them
            # alongside TF data (e.g., for syncing lookups)

        self.get_logger().info("Finished populating TF buffer from bag.")
        del reader  # Close the reader

    def lookup_example(self, target_frame, source_frame, ros_time) :

        try :
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                # 0
                ros_time,
                # Time(),
                timeout=rclpy.duration.Duration(seconds=4.0)  # Timeout for the lookup
            )
            self.get_logger().info(f"Transform at {transform.header.stamp}:")
            self.get_logger().info(
                f"  Translation: x={transform.transform.translation.x}, y={transform.transform.translation.y}, z={transform.transform.translation.z}")
            self.get_logger().info(
                f"  Rotation: x={transform.transform.rotation.x}, y={transform.transform.rotation.y}, z={transform.transform.rotation.z}, w={transform.transform.rotation.w}")
            return transform
        except TransformException as e :
            self.get_logger().warn(f"LookupException: {e}")
            return None


# def main(args=None) :
#     rclpy.init(args=args)
#     bag_tf_processor = BagTfProcessor()
#
#     bag_file_path = '/home/femi/Benchmarking_framework/Data/bag_files/HAM_Airport_2024_08_08_movement_a320_ceo_Germany'  # Replace with your ROS 2 bag directory
#
#     bag_tf_processor.read_tf_from_bag(bag_file_path)
#
#     # Example: Lookup transform at a specific time (in nanoseconds)
#     # You'll need to know a timestamp from your bag file.
#     # E.g., if a message timestamp was 1678886400.123456789 seconds,
#     # the nanoseconds would be 1678886400123456789
#     example_lookup_time_ns = 17231110276796528
#     bag_tf_processor.lookup_example('main_sensor','base_link',Time(seconds=1723111422, nanoseconds=334782665) )
#
#     # Optional: Spin the node for a short period if you have background
#     # processes or need time for cleanup (though not strictly necessary here)
#     # rclpy.spin_once(bag_tf_processor, timeout_sec=0.1)
#
#     bag_tf_processor.destroy_node()
#     rclpy.shutdown()


if __name__ == '__main__' :
    main()