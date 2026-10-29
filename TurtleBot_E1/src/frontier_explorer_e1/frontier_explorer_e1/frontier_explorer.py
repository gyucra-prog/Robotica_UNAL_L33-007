from collections import deque
import math

import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

class FrontierExplorer(BasicNavigator):
    """Explore unknown map regions using frontier-based navigation."""

    def __init__(self):
        super().__init__()

        self.map_data = None
        self.goal_in_progress = False
        self.failed_frontiers = set()

        map_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            map_qos,
        )

        self.exploration_timer = self.create_timer(
            2.0,
            self.exploration_step,
        )

        self.get_logger().info('Frontier explorer initialized.')

    def map_callback(self, msg):
        """Store the most recent occupancy grid."""
        self.map_data = msg

    @staticmethod
    def is_free(value):
        """Return whether an occupancy value represents free space."""
        return 0 <= value <= 25

    def get_frontiers(self):
        """Find free cells adjacent to unknown cells."""
        if self.map_data is None:
            return []

        grid = self.map_data
        width = grid.info.width
        height = grid.info.height
        data = grid.data

        frontiers = []

        for y in range(1, height - 1):
            for x in range(1, width - 1):
                index = y * width + x

                if not self.is_free(data[index]):
                    continue

                neighbors = (
                    data[(y - 1) * width + x],
                    data[(y + 1) * width + x],
                    data[y * width + x - 1],
                    data[y * width + x + 1],
                )

                if any(value == -1 for value in neighbors):
                    frontiers.append((x, y))

        return frontiers

    @staticmethod
    def group_frontiers(frontiers):
        """Group neighboring frontier cells into connected components."""
        frontier_set = set(frontiers)
        groups = []

        while frontier_set:
            start = frontier_set.pop()
            queue = deque([start])
            group = [start]

            while queue:
                x, y = queue.popleft()

                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue

                        neighbor = (x + dx, y + dy)

                        if neighbor in frontier_set:
                            frontier_set.remove(neighbor)
                            queue.append(neighbor)
                            group.append(neighbor)

            groups.append(group)

        return groups

    def get_robot_pose(self):
        """Get the robot's current pose in the map frame."""
        try:
            return self.getCurrentPose()
        except Exception as exc:
            self.get_logger().warning(
                f'Could not obtain robot pose: {exc}'
            )
            return None

    def frontier_to_pose(self, frontier_group):
        """Convert a frontier group's centroid into a map-frame pose."""
        grid = self.map_data
        resolution = grid.info.resolution
        origin = grid.info.origin

        center_x = sum(cell[0] for cell in frontier_group) / len(
            frontier_group
        )
        center_y = sum(cell[1] for cell in frontier_group) / len(
            frontier_group
        )

        local_x = (center_x + 0.5) * resolution
        local_y = (center_y + 0.5) * resolution

        yaw = 2.0 * math.atan2(
            origin.orientation.z,
            origin.orientation.w,
        )

        map_x = (
            origin.position.x
            + local_x * math.cos(yaw)
            - local_y * math.sin(yaw)
        )
        map_y = (
            origin.position.y
            + local_x * math.sin(yaw)
            + local_y * math.cos(yaw)
        )

        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.pose.position.x = map_x
        goal.pose.position.y = map_y
        goal.pose.orientation.w = 1.0

        return goal

    def exploration_step(self):
        """Select and execute one frontier goal at a time."""
        if self.map_data is None:
            self.get_logger().info(
                'Waiting for map...',
                throttle_duration_sec=5.0,
            )
            return

        if not self.goal_in_progress:
          #  if not self.waitUntilNav2Active():
          #      self.get_logger().info(
          #          'Waiting for Nav2 to become active...',
          #          throttle_duration_sec=5.0,
          #      )
          #      return

            frontiers = self.get_frontiers()

            if not frontiers:
                self.get_logger().info(
                    'No frontiers detected. Exploration complete.'
                )
                self.exploration_timer.cancel()
                return

            groups = self.group_frontiers(frontiers)

            # Ignore very small frontier groups.
            groups = [group for group in groups if len(group) >= 5]

            if not groups:
                self.get_logger().info(
                    'No sufficiently large frontier groups remain.'
                )
                self.exploration_timer.cancel()
                return

            robot_pose = self.get_robot_pose()

            if robot_pose is None:
                return

            robot_x = robot_pose.pose.position.x
            robot_y = robot_pose.pose.position.y

            candidates = []

            for group in groups:
                goal = self.frontier_to_pose(group)

                distance = math.hypot(
                    goal.pose.position.x - robot_x,
                    goal.pose.position.y - robot_y,
                )

                key = (
                    round(goal.pose.position.x, 1),
                    round(goal.pose.position.y, 1),
                )

                if key not in self.failed_frontiers:
                    candidates.append((distance, goal, key))

            if not candidates:
                self.get_logger().warning(
                    'All current frontier targets were previously excluded.'
                )
                self.exploration_timer.cancel()
                return

            candidates.sort(key=lambda item: item[0])

            _, selected_goal, selected_key = candidates[0]

            self.get_logger().info(
                f'Navigating to frontier {selected_key}'
            )

            self.selected_frontier = selected_key
            self.goToPose(selected_goal)
            self.goal_in_progress = True

        else:
            if not self.isTaskComplete():
                return

            result = self.getResult()

            if result == TaskResult.SUCCEEDED:
                self.get_logger().info(
                    'Frontier reached. Searching for another target.'
                )
            else:
                self.get_logger().warning(
                    'Frontier goal failed. Excluding this target.'
                )
                self.failed_frontiers.add(self.selected_frontier)

            self.goal_in_progress = False


def main(args=None):
    """Start the frontier exploration node."""
    rclpy.init(args=args)

    explorer = FrontierExplorer()
    explorer.waitUntilNav2Active()

    try:
        rclpy.spin(explorer)
    except KeyboardInterrupt:
        pass
    finally:
        if explorer.goal_in_progress:
            explorer.cancelTask()

        explorer.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
