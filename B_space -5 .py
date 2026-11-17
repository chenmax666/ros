#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist 
import math
import numpy as np
from nav_msgs.msg import Odometry 
#对应/odom_combined（里程计）
# 激光雷达基础参数（不变）
start_angle = 0
end_angle = 252
L_max_Angle = 63
R_max_Angle = 189
#雷达扫描角度，判断赛道边界
number_fitin = 2.8
lidar_forward = 1.22
error_range = 0.15
distance_range = [0.2, 2.5]
#雷达判断有效距离
base_angle = 90.0

# 距离触发阈值配置（保留核心阈值）
DISTANCE_THRESHOLDS = {
   "zero": 0.00, "first": 3.4, "second": 8.4, "third": 19.7,                      #first 3.0
    "fourth": 22.2, "fifth": 24.0, "sixth": 26.2, "lap_total": 34.8, "traffic_light": 26.5,#第二圈红绿灯 28.5+36=64.5
    "second_lap_pre_stop": 66.3, "second_lap_stop": 67.3  # 第二圈终点停车阈值
}

# B赛道黄胶带段：依据图中约 9.1m 的入口位置和两段各 3.06m 推算。
# 这是初始估值；现场里程计起点/轮径误差会使入口距离略有偏差。
YELLOW_TAPE_START_M = 8.10
YELLOW_TAPE_APPROACH_M = 0.30         # 提前接管成直行，躲开"锥桶已消失"的空档（新增）
YELLOW_TAPE_LEG_LENGTH_M = 2.50            #2.5
YELLOW_TAPE_LENGTH_M = 2 * YELLOW_TAPE_LEG_LENGTH_M
YELLOW_TAPE_SPEED_MPS = 0.45
YELLOW_TAPE_REF_SPEED = 0.45           # 角度就是按这个车速调的；以后改车速会自动保持曲率
YELLOW_TAPE_ENTRY_TURN_END_M = 0.5
YELLOW_TAPE_APEX_TURN_START_M = 2.00            #3.5
YELLOW_TAPE_APEX_TURN_END_M = 3.10
YELLOW_TAPE_EXIT_TURN_START_M = 4.30            #5.20
YELLOW_TAPE_ENTRY_ANGLE = 107.0 # 轻微左转，约 +0.138
YELLOW_TAPE_APEX_ANGLE = 79.0  # V形折点右转，约 -0.160
YELLOW_TAPE_EXIT_ANGLE = 107.0   # 回到上边直道，轻微左转

# 角度→归一化角速度转换（不变）
def python_angle_to_angular_z(python_angle):
    max_angle_range = 45.0
    deviation = python_angle - base_angle
    normalized = deviation / max_angle_range
    return max(min(normalized, 1.0), -1.0)

# 1ServoPID类（不变）
class ServoPID:
    def __init__(self):
        self.lidar_angle = 0.0  
        self.P = 0.55     #0.40         
        self.D = 0.30    #0.20     0.30  
        self.I = 0.0
        self.angle_integral_error = 0.0
        self.integral_limit = 100.0
        self.angle_last_error = 0.0  
        self.first_circle_end_flag = 0
        self.walk_distance = 0.0  # 累加距离（35m时归0）

    def servo_pid_control(self, new_angle_error):
        self.angle_integral_error += new_angle_error
        self.angle_integral_error = max(min(self.angle_integral_error, self.integral_limit), -self.integral_limit)
        output = self.P * new_angle_error + self.D * (new_angle_error - self.angle_last_error) + self.I * self.angle_integral_error
        output = max(min(output, 45.0), -45.0)
        self.angle_last_error = new_angle_error
        return float(output)

# 核心控制节点（修改后）
class LidarCoordinateControlNode(Node):
    def __init__(self):
        super().__init__('lidar_distance_control_node')
        #创建节点
        self.servo = ServoPID()
        
        self.cmd_vel = Twist()
        self.publisher_twist = self.create_publisher(Twist, 'car_cmd_vel', 10)
        self.subscription_scan = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 100)
        #订阅激光雷达
        self.subscription_odom = self.create_subscription(Odometry, '/odom_combined', self.odometry_callback, 10)
        #里程计
        # 1速度参数配置（不变）
        self.default_linear_speed = 0.42
        self.special_linear_speed1 = 0.42
        self.special_angular_speed1 = 102.0
        self.special_angular_speed3 = 110.0            #原本注释了，值是96.5
        self.special_linear_speed5 = 0.42
        self.special_angular_speed5 = 110.0           #99.8
        self.special_linear_speed3 = 0.42
        #self.special_linear_speed7 = 0.0
        #self.special_angular_speed7 = 90.0
        self.special_angular_speed8 = 80.0  # 停车时的角速度
        
        # 模式状态标记（不变）
        self.reached_zero = False
        self.reached_first = False
        self.reached_second = False
        self.reached_third = False
        self.reached_fourth = False
        self.reached_fifth = False
        self.reached_sixth = False
        self.reached_pre_stop = False
        self.reached_stop = False
        self.reached_traffic_light = False
        self.first_lap_completed = False
        self.traffic_light_triggered_once = False
        self.reached_pre_second_lap_stop = False
        self.reached_second_lap_stop = False

        # 红绿灯停车定时器（不变）
        self.stop_timer = None
        self.stop_start_time = None
        self.stop_duration = 3.0

        # 黄胶带段：入口可在线改 + 日志去重 + 卡住检测
        self.declare_parameter('tape_start_m', YELLOW_TAPE_START_M)
        self.tape_start_m = float(self.get_parameter('tape_start_m').value)
        self._tape_phase = None
        self._tape_d = None
        self._tape_t = 0.0
        # 直道加速去抖（连续直行帧数 / 是否已锁存）
        self._straight_cnt = 0
        self._boost_on = False

        self.get_logger().info(
            f"Lidar node started (35m to 0 B ) [tape @{self.tape_start_m:.2f}m]")
    
    # 里程计回调（不变）
    def odometry_callback(self, msg):
        current_timestamp = self.get_clock().now().nanoseconds / 1e9  
        linear_speed = abs(msg.twist.twist.linear.x)
        
        if hasattr(self, 'last_odom_timestamp') and self.last_odom_timestamp is not None:
            dt = current_timestamp - self.last_odom_timestamp  
            if dt > 0:
                self.servo.walk_distance += linear_speed * dt
        
        self.last_odom_timestamp = current_timestamp
        # 打印当前圈距离+真实总距离（新增）
        real_total_dist = (35.0 if self.first_lap_completed else 0.0) + self.servo.walk_distance
        self.get_logger().info(f"juli: {self.servo.walk_distance:.2f}m | : {real_total_dist:.2f}m | : {self.first_lap_completed}")
    
    # 激光雷达回调（不变）
    def lidar_callback(self, data):
        self.lidar_deal(data)
        self.publisher_twist.publish(self.cmd_vel)

    # 重置模式状态（不变）
    def reset_mode_states(self):
        self.reached_zero = False
        self.reached_first = False
        self.reached_second = False
        self.reached_third = False
        self.reached_fourth = False
        self.reached_fifth = False  
        self.reached_sixth = False
        self.reached_traffic_light = False
        self.traffic_light_triggered_once = False
        self.reached_second_lap_stop = False
        self.last_odom_timestamp = None

    # 启动红绿灯停车（不变）
    def start_traffic_light_stop(self):
        if self.stop_timer is None:
            self.stop_start_time = self.get_clock().now().nanoseconds / 1e9
            self.stop_timer = self.create_timer(0.1, self.traffic_light_stop_callback)
            self.get_logger().info("Traffic light stop started (3s)")

    # 红绿灯停车回调（不变）
    def traffic_light_stop_callback(self):
        self.cmd_vel.linear.x = 0.0
        self.cmd_vel.angular.z = python_angle_to_angular_z(base_angle)
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        if current_time - self.stop_start_time >= self.stop_duration:
            self.stop_timer.cancel()
            self.stop_timer = None
            self.reached_traffic_light = False
            self.servo.angle_integral_error = 0.0
            self.get_logger().info("Traffic light stop finished - Resuming")

    # 距离触发逻辑（核心修改：35m归0）
    def check_distance_trigger(self):
        current_dist = self.servo.walk_distance  # 当前圈距离（35m归0后从0开始）
        lap_total = DISTANCE_THRESHOLDS["lap_total"]
        pre_second_lap_stop_thresh = DISTANCE_THRESHOLDS["second_lap_pre_stop"]
        second_lap_stop_thresh = DISTANCE_THRESHOLDS["second_lap_stop"]

        # 第一圈结束逻辑（核心修改：距离归0）
        if not self.first_lap_completed and current_dist >= lap_total:
            self.first_lap_completed = True
            self.servo.walk_distance = 0.0  # 关键：距离直接归0，第二圈从0开始
            self.traffic_light_triggered_once = True 
            self.reset_mode_states()

             # 防红绿灯误触发
            self.get_logger().info("first_lap_completed reset_mode_states")
            return  # 阻断后续判断，彻底防误触发

        # 阶段触发（用当前圈距离，第二圈从0开始）
        if not self.reached_pre_stop and not self.reached_zero and current_dist >= DISTANCE_THRESHOLDS["zero"]:
            self.reached_zero = True
        if not self.reached_pre_stop and not self.reached_first and current_dist >= DISTANCE_THRESHOLDS["first"]:
            self.reached_first = True
        if not self.reached_pre_stop and not self.reached_second and current_dist >= DISTANCE_THRESHOLDS["second"]:
            self.reached_second = True
        if not self.reached_pre_stop and not self.reached_third and current_dist >= DISTANCE_THRESHOLDS["third"]:
            self.reached_third = True
        if not self.reached_pre_stop and not self.reached_fourth and current_dist >= DISTANCE_THRESHOLDS["fourth"]:
            self.reached_fourth = True
        if not self.reached_pre_stop and not self.reached_fifth and current_dist >= DISTANCE_THRESHOLDS["fifth"]:
            self.reached_fifth = True
        if not self.reached_pre_stop and not self.reached_sixth and current_dist >= DISTANCE_THRESHOLDS["sixth"]:
            self.reached_sixth = True
        

        # 红绿灯触发（用当前圈距离，第二圈28.5m才触发）
        if (not self.reached_traffic_light 
            and current_dist >= DISTANCE_THRESHOLDS["traffic_light"] 
            and not self.traffic_light_triggered_once):  # 新增：必须完成第六阶段才触发
            self.reached_traffic_light = True
            self.traffic_light_triggered_once = True
            self.start_traffic_light_stop()
            self.get_logger().info(f"Reached traffic light (: {current_dist:.2f}m) | ")

        # 第二圈70m终点触发停车（判断真实总距离）
        real_total_dist = (35.0 if self.first_lap_completed else 0.0) + current_dist
        if self.first_lap_completed and not self.reached_second_lap_stop:
            if real_total_dist >= second_lap_stop_thresh:
                self.reached_second_lap_stop = True
                self.reached_pre_stop = False
                self.reached_stop = True  
                self.get_logger().info(f"Second lap finished (: {real_total_dist:.2f}m) - Stopping")
                
        if self.first_lap_completed and not self.reached_pre_second_lap_stop:
            if real_total_dist >= pre_second_lap_stop_thresh:
                self.reached_pre_second_lap_stop = True
                self.reached_pre_stop = True
                self.get_logger().info(f"Second lap will finish    (: {real_total_dist:.2f}m) -  pre Stopping")



    def get_yellow_tape_profile(self, current_distance):
        """按里程执行黄胶带段的低速预设转向；返回 (阶段名, 控制角度) 或 None。"""
        # 入口支持在线改：ros2 param set /lidar_distance_control_node tape_start_m 9.20
        try:
            self.tape_start_m = float(self.get_parameter('tape_start_m').value)
        except Exception:
            pass

        # 从 (入口 - 提前量) 开始接管，先直行一小段，把"锥桶已消失"的空档盖住
        offset = current_distance - (self.tape_start_m - YELLOW_TAPE_APPROACH_M)
        if offset < 0.0 or offset >= (YELLOW_TAPE_APPROACH_M + YELLOW_TAPE_LENGTH_M):
            return None
        if offset < YELLOW_TAPE_APPROACH_M:
            return "approach", base_angle
        offset -= YELLOW_TAPE_APPROACH_M

        if offset < YELLOW_TAPE_ENTRY_TURN_END_M:
            return "entry", YELLOW_TAPE_ENTRY_ANGLE
        if offset < YELLOW_TAPE_APEX_TURN_START_M:
            return "first_leg", base_angle
        if offset < YELLOW_TAPE_APEX_TURN_END_M:
            return "v_apex", YELLOW_TAPE_APEX_ANGLE
        if offset < YELLOW_TAPE_EXIT_TURN_START_M:
            return "second_leg", base_angle
        return "exit", YELLOW_TAPE_EXIT_ANGLE

    def yellow_tape_angular_z(self, angle):
        """角速度随车速等比缩放，保证曲率不变。
        现在 REF == SPEED，所以输出和你原来的完全一样；以后改车速就不会把转弯半径也一起缩小。"""
        return python_angle_to_angular_z(angle) * (YELLOW_TAPE_SPEED_MPS / YELLOW_TAPE_REF_SPEED)

    def tape_stuck_watchdog(self):
        """胶带段开环按里程推进：走不动会永远停在当前相位。只报警，不干预控制。"""
        now_s = self.get_clock().now().nanoseconds / 1e9
        if self._tape_d is None:
            self._tape_d, self._tape_t = self.servo.walk_distance, now_s
            return
        if now_s - self._tape_t >= 2.0:
            moved = abs(self.servo.walk_distance - self._tape_d)
            if moved < 0.05:
                self.get_logger().error(
                    f"胶带段 2s 只走了 {moved:.3f}m —— 车没动：检查车速是否进死区/打滑/是否顶住边界")
            self._tape_d, self._tape_t = self.servo.walk_distance, now_s

    def straight_speed_boost(self, angular_z):
        """走直才给 +0.03，但去抖：连续 6 帧 |w|<0.05 才锁存，一帧不直就立刻取消。
        原来是逐帧判断，弯道里 PID 抖一下速度就在 0.36 / 0.39 之间来回跳。"""
        if abs(angular_z) < 0.05:
            self._straight_cnt += 1
        else:
            self._straight_cnt = 0
            self._boost_on = False
        if self._straight_cnt >= 6:
            self._boost_on = True
        return 0.03 if self._boost_on else 0.0

    # 激光雷达数据处理
    def lidar_deal(self, data):
        # 停车模式（优先级最高）
        if self.reached_pre_stop:
            self.reached_sixth = False
            self.reached_fifth = False
            self.reached_fourth = False
            self.reached_third = False
            self.reached_second = False
            self.reached_first = False
            self.reached_zero = False
            self.cmd_vel.linear.x = 0.25
            self.cmd_vel.angular.z = python_angle_to_angular_z(self.special_angular_speed8)
            self.get_logger().info(f"Stopped: Linear=0.25, Angular Z={self.cmd_vel.angular.z:.2f}")
        
        if self.reached_stop:
            self.cmd_vel.linear.x = 0.0
            self.cmd_vel.angular.z = python_angle_to_angular_z(self.special_angular_speed8)
            self.get_logger().info(f"Stopped: Linear=0.0, Angular Z={self.cmd_vel.angular.z:.2f}")
            return
        
        # ===== 黄胶带段：没有锥桶，雷达巡线不可用，在读雷达和跑 PID 之前就接管 =====
        yellow_tape = self.get_yellow_tape_profile(self.servo.walk_distance)
        if yellow_tape is not None and not self.reached_pre_stop:
            tape_phase, tape_angle = yellow_tape
            self.cmd_vel.linear.x = YELLOW_TAPE_SPEED_MPS
            self.cmd_vel.angular.z = self.yellow_tape_angular_z(tape_angle)
            # 清掉雷达 PID 的历史误差，走完胶带回到巡线时不会突然甩一下
            self.servo.angle_last_error = 0.0
            self.servo.angle_integral_error = 0.0
            if tape_phase != self._tape_phase:
                self._tape_phase = tape_phase
                self.get_logger().info(
                    f"Yellow Tape {tape_phase}: Linear={self.cmd_vel.linear.x:.2f}, "
                    f"Angular Z={self.cmd_vel.angular.z:+.3f}, Distance={self.servo.walk_distance:.2f}m")
            self.tape_stuck_watchdog()
            return
        self._tape_phase = None
        
        ranges = data.ranges
        ranges_len = len(ranges)

        if ranges_len == 0 or all(math.isnan(r) or math.isinf(r) for r in ranges):
            # 黄胶带区使用里程计触发的预设轨迹，不要求雷达能看到地面胶带。
            if self.get_yellow_tape_profile(self.servo.walk_distance) is None:
                self.get_logger().warn("Invalid LiDAR data - Forcing stop")
                self.cmd_vel.linear.x = 0.0
                self.cmd_vel.angular.z = python_angle_to_angular_z(base_angle)
                return
            self.get_logger().warn("Invalid LiDAR data in yellow-tape segment - Using scripted control")
        
        L_points = self.lidar_find_point(ranges, ranges_len, start_angle, L_max_Angle, 'L')
        R_points = self.lidar_find_point(ranges, ranges_len, R_max_Angle, end_angle, 'R')

        if L_points and R_points:
            i = 0
            while i < len(R_points):
                if abs(L_points[0][0] - R_points[i][0] - 360) < 35:
                    R_points.pop(i)
                else:
                    i += 1

        if L_points:
            min_y = L_points[0][1]
            i = 0
            while i < len(L_points):
                if L_points[i][1] <= min_y:
                    min_y = L_points[i][1]
                elif len(L_points) > 1:
                    L_points.pop(i)
                    continue
                i += 1
            if L_points:
                _, Left_edge, _ = self.lidar_find_2edge(L_points)
            else:
                Left_edge = [L_max_Angle, lidar_forward]
        else:
            Left_edge = [L_max_Angle, lidar_forward]

        if R_points:
            min_y = R_points[0][1]
            i = 0
            while i < len(R_points):
                if R_points[i][1] <= min_y:
                    min_y = R_points[i][1]
                elif len(R_points) > 1:
                    R_points.pop(i)
                    continue
                i += 1
            if R_points:
                _, Right_edge, _ = self.lidar_find_2edge(R_points)
            else:
                Right_edge = [R_max_Angle, lidar_forward]
        else:
            Right_edge = [R_max_Angle, lidar_forward]

        mid_point = int((Left_edge[0] + (Right_edge[0] - 360)))
        self.servo.lidar_angle = mid_point
        control_angle = self.servo.servo_pid_control(mid_point) + base_angle
        normalized_angular_z = python_angle_to_angular_z(control_angle)
        self.get_logger().debug(f"Mid Angle: {mid_point}, Control Angle: {control_angle:.1f}, Norm Angular Z: {normalized_angular_z:.2f}")

        self.check_distance_trigger()
        yellow_tape_profile = self.get_yellow_tape_profile(self.servo.walk_distance)

        # 模式切换逻辑（完整无缺失）
        if self.reached_traffic_light:
            pass  # 红绿灯停车中
        # 黄胶带段已经在上面提前 return 了，这里不再判断（原来那一段永远走不到）
        elif self.reached_sixth:
            self.cmd_vel.linear.x = self.default_linear_speed
            self.cmd_vel.angular.z = normalized_angular_z
            self.get_logger().info(f"Sixth Mode: Linear={self.cmd_vel.linear.x:.2f}, Angular Z={self.cmd_vel.angular.z:.2f}")
        elif self.reached_fifth and not self.reached_sixth:
            self.cmd_vel.linear.x = self.special_linear_speed5 +0.04
            self.cmd_vel.angular.z = python_angle_to_angular_z(self.special_angular_speed5)
            self.get_logger().info(f"Fifth Mode: Linear={self.cmd_vel.linear.x:.2f}, Angular Z={self.cmd_vel.angular.z:.2f}")
          #以上注释掉
        elif self.reached_fourth and not self.reached_fifth:
            self.cmd_vel.linear.x = self.default_linear_speed
            self.cmd_vel.angular.z = normalized_angular_z
            self.get_logger().info(
                f"Fourth Mode (PID Straight): Linear={self.cmd_vel.linear.x:.2f}, Angular Z={self.cmd_vel.angular.z:.2f}")
        elif self.reached_third and not self.reached_fourth:
            self.cmd_vel.linear.x = self.special_linear_speed3 + 0.03
            self.cmd_vel.angular.z = python_angle_to_angular_z(self.special_angular_speed3)
            self.get_logger().info(
                f"Third Mode (Left Turn): Linear={self.cmd_vel.linear.x:.2f}, Angular Z={self.cmd_vel.angular.z:.2f}")
        elif self.reached_second and not self.reached_third:
            self.cmd_vel.angular.z = normalized_angular_z
            # 走直才 +0.03，但去抖：连续 6 帧直行才加，一进弯立刻取消
            self.cmd_vel.linear.x = self.default_linear_speed + self.straight_speed_boost(self.cmd_vel.angular.z)
            self.get_logger().info(f"Second Mode: Linear={self.cmd_vel.linear.x:.2f}, Angular Z={self.cmd_vel.angular.z:.2f}")
        elif self.reached_first and not self.reached_second:
            self.cmd_vel.linear.x = self.special_linear_speed1
            self.cmd_vel.angular.z = python_angle_to_angular_z(self.special_angular_speed1)
            self.get_logger().info(f"First Mode: Linear={self.cmd_vel.linear.x:.2f}, Angular Z={self.cmd_vel.angular.z:.2f}")
        elif self.reached_zero and not self.reached_second:
            self.cmd_vel.linear.x = self.default_linear_speed
            self.cmd_vel.angular.z = normalized_angular_z
            self.get_logger().info(f"Default Mode: Linear={self.cmd_vel.linear.x:.2f}, Angular Z={self.cmd_vel.angular.z:.2f}")

    def lidar_find_2edge(self, points):
        if points[0][1] < lidar_forward or len(points) == 1:
            return points[0], points[0], points[0]
        else:
            for i in range(1, len(points)):
                if points[i][1] < lidar_forward:
                    denominator = (points[i-1][0] - points[i][0])
                    if denominator == 0:
                        return points[i-1], points[i-1], points[i]
                    
                    k = (points[i-1][1] - points[i][1]) / denominator
                    b = points[i-1][1] - k * points[i-1][0]
                    self.get_logger().debug(f"Edge Calculation: k={k:.2f}, b={b:.2f}")
                    edge_angle = int(lidar_forward / k - b / k)
                    return points[i-1], [edge_angle, lidar_forward], points[i]
        
        return points[0], points[0], points[0]

    def lidar_find_point(self, ranges, ranges_len, start, end, lor_r):
        points = []
        if lor_r == 'L':
            max_rgi = min(63 * 4 - 1, ranges_len - 1)
            rgi = 0
            while rgi <= max_rgi:
                rgi_true = rgi % ranges_len
                if not (math.isnan(ranges[rgi_true]) or math.isinf(ranges[rgi_true])):
                    if distance_range[0] < ranges[rgi_true] < distance_range[1]:
                        start_i = rgi
                        while (rgi <= max_rgi and abs(ranges[rgi_true] - ranges[(rgi_true + 1) % ranges_len]) < error_range):
                            rgi_true = (rgi_true + 1) % ranges_len
                            rgi += 1
                        if abs(rgi - start_i) > number_fitin:
                            points.append([int(rgi / 2.8), ranges[rgi_true]])
                        rgi += 1
                        continue
                rgi += 1

        elif lor_r == 'R':
            min_rgi = max(189 * 4 + 1, 0)
            max_rgi = min(251 * 4 - 1, ranges_len - 1)
            rgi = max_rgi
            while rgi >= min_rgi:
                rgi_true = rgi % ranges_len
                if not (math.isnan(ranges[rgi_true]) or math.isinf(ranges[rgi_true])):
                    if distance_range[0] < ranges[rgi_true] < distance_range[1]:
                        start_i = rgi
                        while (rgi >= min_rgi and abs(ranges[rgi_true] - ranges[(rgi_true - 1) % ranges_len]) < error_range):
                            rgi_true = (rgi_true - 1) % ranges_len
                            rgi -= 1
                        if abs(rgi - start_i) > number_fitin:
                            points.append([int(rgi / 2.8), ranges[rgi_true]])
                        rgi -= 1
                        continue
                rgi -= 1

        return points

def main(args=None):
    rclpy.init(args=args)
    node = LidarCoordinateControlNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        stop_cmd = Twist()
        stop_cmd.linear.x = 0.0
        stop_cmd.angular.z = python_angle_to_angular_z(base_angle)
        node.publisher_twist.publish(stop_cmd)
        node.get_logger().info("Node stopped by user - Robot forced stop")
    finally:
        if node.stop_timer is not None:
            node.stop_timer.cancel()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
