import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped


class SimpleController(Node):

    def __init__(self):
        super().__init__('simple_controller')

        self.publisher_ = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        self.timer = self.create_timer(
            0.1,
            self.control_loop
        )

    def control_loop(self):

        msg = TwistStamped()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'

        msg.twist.linear.x = 0.05
        msg.twist.angular.z = 0.0

        self.publisher_.publish(msg)


def main(args=None):

    rclpy.init(args=args)

    node = SimpleController()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        stop_msg = TwistStamped()
        stop_msg.header.stamp = node.get_clock().now().to_msg()
        stop_msg.header.frame_id = 'base_link'
        stop_msg.twist.linear.x = 0.0
        stop_msg.twist.angular.z = 0.0

        node.publisher_.publish(stop_msg)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
