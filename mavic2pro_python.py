import math
import sys
import numpy as np
import cv2
from controller import Robot, Camera, GPS, InertialUnit, Gyro, Compass, Motor, LED, Keyboard
import struct
from car_classifier import predict_image
from PIL import Image

class Car_class:
    def do_it():
        prediction, confidence = predict_image('C:/Users/chuva/AppData/Local/Programs/Webots/projects/robots/dji/mavic/controllers/mavic2pro_python/im.png')
        if prediction == 1:
            print(f'Машина обнаружена! Уверенность: {confidence:.2%}')
        else:
            print(f'Машины нет. Уверенность: {(1-confidence):.2%}')

class LandingAnalyzer:
   
    def __init__(self):
        self.good_surface_threshold = 200
        self.min_uniform_area = 0.6
        self.sand_lower_1 = np.array([10, 30, 80])    # светлый песок
        self.sand_upper_1 = np.array([30, 200, 255])
        
        self.sand_lower_2 = np.array([0, 20, 100])    # очень светлый песок
        self.sand_upper_2 = np.array([20, 150, 255])
        
        # Асфальт: низкая насыщенность, средняя яркость
        self.asphalt_lower = np.array([0, 0, 20])
        self.asphalt_upper = np.array([180, 80, 150])
        
        # Темные объекты (машины, тени): очень низкая яркость
        self.dark_lower = np.array([0, 0, 0])
        self.dark_upper = np.array([180, 255, 80])
        
        # Пороги
        self.min_sand_ratio = 0.6          # минимум 60% песка для посадки
        self.max_asphalt_ratio = 0.15      # максимум 15% асфальта
        self.max_dark_ratio = 0.10         # максимум 10% темных объектов
        self.edge_threshold = 0.3         # максимальная плотность краев было 0.03
        
    def analyze_landing_zone(self, image_data, camera_width, camera_height):
        if image_data is None:
            return None, 0, False
        
        try:

            image = np.frombuffer(image_data, np.uint8)
            image = image.reshape((camera_height, camera_width, 4))
            

            bgr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            
            h, w = hsv_image.shape[:2]
            margin_h, margin_w = int(h * 0.15), int(w * 0.15)
            center_hsv = hsv_image[margin_h:h-margin_h, margin_w:w-margin_w]
            
            # Песок 
            sand_mask_1 = cv2.inRange(center_hsv, self.sand_lower_1, self.sand_upper_1)
            sand_mask_2 = cv2.inRange(center_hsv, self.sand_lower_2, self.sand_upper_2)
            sand_mask = cv2.bitwise_or(sand_mask_1, sand_mask_2)
            
            # Асфалт
            asphalt_mask = cv2.inRange(center_hsv, self.asphalt_lower, self.asphalt_upper)
            
           
            dark_mask = cv2.inRange(center_hsv, self.dark_lower, self.dark_upper)
            
           
            total_pixels = center_hsv.shape[0] * center_hsv.shape[1]
            sand_ratio = np.sum(sand_mask > 0) / total_pixels
            asphalt_ratio = np.sum(asphalt_mask > 0) / total_pixels
            dark_ratio = np.sum(dark_mask > 0) / total_pixels
            
            gray = cv2.cvtColor(bgr_image[margin_h:h-margin_h, margin_w:w-margin_w], 
                               cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            sand_regions = sand_mask.copy()

            kernel = np.ones((5, 5), np.uint8)
            sand_regions = cv2.morphologyEx(sand_regions, cv2.MORPH_CLOSE, kernel)
            sand_regions = cv2.morphologyEx(sand_regions, cv2.MORPH_OPEN, kernel)
            
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sand_regions)
            

            has_large_sand_area = False
            if num_labels > 1:  # 0 это фон ес чо
                largest_area = np.max(stats[1:, cv2.CC_STAT_AREA])
                if largest_area > total_pixels * 0.4:
                    has_large_sand_area = True
            

            quality_score = 0
            
            if sand_ratio >= self.min_sand_ratio:
                quality_score += 40
            else:
                quality_score += int(40 * (sand_ratio / self.min_sand_ratio))
            
            if asphalt_ratio <= self.max_asphalt_ratio:
                quality_score += 25
            else:
                penalty = int(25 * (1 - min(asphalt_ratio, 0.5) / 0.5))
                quality_score += max(0, penalty)
            
            if dark_ratio <= self.max_dark_ratio:
                quality_score += 20
            else:
                penalty = int(20 * (1 - min(dark_ratio, 0.3) / 0.3))
                quality_score += max(0, penalty)
            
            if has_large_sand_area:
                quality_score += 15
            elif num_labels > 1:
                quality_score += int(15 * (largest_area / (total_pixels * 0.4)))
            
            if edge_density > self.edge_threshold:
                edge_penalty = min(10, int(10 * (edge_density / 0.1)))
                quality_score -= edge_penalty
            

            is_suitable = (
                quality_score >= 70 and                    # общая оценка хорошая
                sand_ratio >= self.min_sand_ratio and      # достаточно песка
                asphalt_ratio <= self.max_asphalt_ratio and # мало асфальта
                dark_ratio <= self.max_dark_ratio and       # нет темных объектов
                edge_density <= self.edge_threshold         # нет четких объектов
            )
            
            center_coords = (0.0, 0.0)
            

            print(f"Анализ поверхности:")
            print(f'Качество {quality_score}')
            print(f"  Песок: {sand_ratio*100:.1f}% (мин: {self.min_sand_ratio*100:.0f}%)")
            print(f"  Асфальт: {asphalt_ratio*100:.1f}% (макс: {self.max_asphalt_ratio*100:.0f}%)")
            print(f"  Темные объекты: {dark_ratio*100:.1f}% (макс: {self.max_dark_ratio*100:.0f}%)")
            print(f"  Плотность краев: {edge_density:.4f} (макс: {self.edge_threshold:.4f})")
            print(f"  Оценка: {quality_score}/100, Подходит: {is_suitable}")
            
            return center_coords, quality_score, is_suitable
            
        except Exception as e:
            print(f"Ошибка анализа: {e}")
            import traceback
            traceback.print_exc()
            return None, 0, False

class EmergencyLandingController:
    
    def __init__(self, robot, motors, gps, imu, gyro, camera, landing_analyzer):
        self.robot = robot
        self.motors = motors
        self.gps = gps
        self.imu = imu
        self.gyro = gyro
        self.camera = camera
        self.landing_analyzer = landing_analyzer
        

        self.search_speed = 1.5           # скорость поиска
        self.landing_speed = 0.3          # скорость снижения
        self.min_altitude = 0.2           # минимальная высота для посадки
        self.hover_altitude = 5.0         #высота зависания для поиска
        self.analysis_interval = 0.5      # интервал анализа поверхности
        self.threshold_speed = 0.1
        self.descending_start_time = 100
   
        self.is_landing = False
        self.landing_phase = "preparing"  # preparing -> searching -> hovering -> descending -> completed
        self.last_analysis_time = 0
        self.landing_target = (0, 0)
        self.target_confidence = 0
        self.hover_start_time = 0
        self.search_start_time = 0
        self.surface_suitable = False
        self.stable_detections = 0
        self.required_stable_detections = 3
        

        self.search_pitch = -1.0   # наклон для движения вперед при поиске
        self.max_search_time = 500.0  # максимальное время поиска
        
    def start_landing(self):
        self.is_landing = True
        self.landing_phase = "preparing"
        self.search_start_time = self.robot.getTime()
        self.last_analysis_time = self.robot.getTime()
        self.stable_detections = 0
        print("Начиная посадку...")
        print("Фаза:Подготовка, работа камеры")
        
    def update_landing(self, current_altitude):
        if not self.is_landing:
            return None
            
        current_time = self.robot.getTime()
        
        if self.landing_phase == "preparing":
            if current_time - self.search_start_time > 1.0:
                self.landing_phase = "searching"
                print("Фаза:Поск подходящей поверхности")
            return self.hover_altitude
            
        # фаза поиска
        elif self.landing_phase == "searching":
            if current_time - self.last_analysis_time > self.analysis_interval:
                self.analyze_current_surface()
                self.last_analysis_time = current_time
                
            search_time = current_time - self.search_start_time
            if search_time > self.max_search_time:
                print("Время вышло, сажусь немедлено")
                self.landing_phase = "hovering"
                self.hover_start_time = current_time
                return self.hover_altitude
                
            if self.surface_suitable and self.stable_detections >= self.required_stable_detections:
                print("Поверхность найдена, готовлюсь к зависанию")
                self.landing_phase = "hovering"
                self.hover_start_time = current_time
                
            return self.hover_altitude
            

        elif self.landing_phase == "hovering":
            if current_time - self.hover_start_time > 5.0:
                self.landing_phase = "descending"
                self.descending_start_time = current_time
                print("Фаза: Начинаю контролируемую посадку")
            return self.hover_altitude
            
        # Фаза снижения
        elif self.landing_phase == "descending":

            descent_rate = self.landing_speed * (current_time - self.last_analysis_time)
            new_altitude = current_altitude - descent_rate
            vertical_speed = self.gps.getSpeed()
            
            if new_altitude <= self.min_altitude or (abs(vertical_speed) < self.threshold_speed and current_time - self.descending_start_time > 7.0):
                self.complete_landing()
                return 0.0
                

            if current_time - self.last_analysis_time > self.analysis_interval:
                self.analyze_current_surface()
                self.last_analysis_time = current_time
                
            return new_altitude
            
        return None
        
    def analyze_current_surface(self):
        image_data = self.camera.getImage()
        if image_data:
            width = self.camera.getWidth()
            height = self.camera.getHeight()
            
            image = Image.new("RGB", (width, height))
            
            for y in range(height):
                for x in range(width):
                    index = (y * width + x) * 4
                    b = image_data[index]
                    g = image_data[index + 1]
                    r = image_data[index + 2]
                    
                    image.putpixel((x, y), (r, g, b))
            

            image.save('C:/Users/chuva/AppData/Local/Programs/Webots/projects/robots/dji/mavic/controllers/mavic2pro_python/im.png', 'PNG')
            print(f"Изображение сохранено")
            Car_class.do_it()
            target, score, is_suitable = self.landing_analyzer.analyze_landing_zone(
                image_data, self.camera.getWidth(), self.camera.getHeight()
            )
            
            if target:
                self.landing_target = target
                self.target_confidence = score
                
                if is_suitable:
                    self.stable_detections += 1
                    self.surface_suitable = True
                    print(f"ПОдходящее место для посадки! Confirmations: {self.stable_detections}/{self.required_stable_detections}")
                else:
                    self.stable_detections = max(0, self.stable_detections - 1)
                    if self.stable_detections == 0:
                        self.surface_suitable = False
                        
    def get_search_control(self):
        # ИСПРАВИТЬ: движение вперед ТОЛЬКО во время фазы поиска
        if self.landing_phase == "searching":
            return 0.0, self.search_pitch, 0.0  # roll, pitch, yaw
        return 0.0, 0.0, 0.0
        
    def apply_landing_correction(self, roll_input, pitch_input, yaw_input):
        if not self.is_landing:
            return roll_input, pitch_input, yaw_input
            
        # во время поиска делаю только движение вперед
        if self.landing_phase == "searching":
            search_roll, search_pitch, search_yaw = self.get_search_control()
            return (roll_input + search_roll, 
                   pitch_input + search_pitch, 
                   yaw_input + search_yaw)
        
        elif self.landing_phase == "hovering":
            return roll_input, pitch_input, yaw_input  # стаблизацию не трогать!
                   
        elif self.landing_phase == "descending":
            return roll_input, pitch_input, yaw_input
                   
        return roll_input, pitch_input, yaw_input
        
    def complete_landing(self):
        self.is_landing = False
        self.landing_phase = "completed"
        
        print("Посадка выполнена, выключаю моторы")
        

class Mavic2ProController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        self.K_VERTICAL_THRUST = 68.5  # с этой тягой дрон взлетает
        self.K_VERTICAL_OFFSET = 0.6   # вертикальное смещение для стабилизации
        self.K_VERTICAL_P = 3.0        # P-константа вертикального PID
        self.K_ROLL_P = 50.0          # P-константа крена PID
        self.K_PITCH_P = 30.0         # P-константа тангажа PID
        

        self.target_altitude = 3.0     # целевая высота
        self.emergency_mode = False    # режим аварийной посадки
        

        self.initialize_devices()
        
        self.landing_analyzer = LandingAnalyzer()
        self.emergency_controller = EmergencyLandingController(
            self.robot, self.motors, self.gps, self.imu, self.gyro,
            self.camera, self.landing_analyzer
        )
        
        print("Запуск дрона...")
        
    def initialize_devices(self):
        try:
            # камера
            self.camera = self.robot.getDevice("camera")
            self.camera.enable(self.timestep)
            self.camera_width = self.camera.getWidth()
            self.camera_height = self.camera.getHeight()
            
            # светодиоды
            self.front_left_led = self.robot.getDevice("front left led")
            self.front_right_led = self.robot.getDevice("front right led")
            
            # сенсоры
            self.imu = self.robot.getDevice("inertial unit")
            self.imu.enable(self.timestep)
            self.gps = self.robot.getDevice("gps")
            self.gps.enable(self.timestep)
            self.compass = self.robot.getDevice("compass")
            self.compass.enable(self.timestep)
            self.gyro = self.robot.getDevice("gyro")
            self.gyro.enable(self.timestep)
            
            # клавиатура (не забыть исправить клавиши
            self.keyboard = self.robot.getKeyboard()
            self.keyboard.enable(self.timestep)
            
            # моторы камеры
            self.camera_roll_motor = self.robot.getDevice("camera roll")
            self.camera_pitch_motor = self.robot.getDevice("camera pitch")
            
            # моторы пропеллеров
            self.front_left_motor = self.robot.getDevice("front left propeller")
            self.front_right_motor = self.robot.getDevice("front right propeller")
            self.rear_left_motor = self.robot.getDevice("rear left propeller")
            self.rear_right_motor = self.robot.getDevice("rear right propeller")
            
            # настройка моторов пропеллеров (velocity mode)
            self.motors = [
                self.front_left_motor,
                self.front_right_motor,
                self.rear_left_motor,
                self.rear_right_motor
            ]
            
            for motor in self.motors:
                motor.setPosition(float('inf'))
                motor.setVelocity(1.0)
                
        except Exception as e:
            print(f"Ошибка инициализации какого-то девайса: {e}")
            sys.exit(1)
            
    def clamp(self, value, low, high):

        return max(low, min(value, high))
        
    def wait(self, seconds):
        start_time = self.robot.getTime()
        while self.robot.step(self.timestep) != -1:
            if self.robot.getTime() - start_time > seconds:
                break
                
    def display_instructions(self):

        print('стрелочки для движения, стелочи с шифтом - снижение и взлет, f для имитации аварии')
        
    def process_keyboard_input(self):
        roll_disturbance = 0.0
        pitch_disturbance = 0.0
        yaw_disturbance = 0.0
        
        key = self.keyboard.getKey()
        while key > 0:
            if key == ord('F') or key == ord('f'):
                if not self.emergency_mode:
                    self.activate_emergency_landing()
            elif key == Keyboard.UP:
                pitch_disturbance = -2.0
            elif key == Keyboard.DOWN:
                pitch_disturbance = 2.0
            elif key == Keyboard.RIGHT:
                yaw_disturbance = -1.3
            elif key == Keyboard.LEFT:
                yaw_disturbance = 1.3
            elif key == (Keyboard.SHIFT + Keyboard.RIGHT):
                roll_disturbance = -1.0
            elif key == (Keyboard.SHIFT + Keyboard.LEFT):
                roll_disturbance = 1.0
            elif key == (Keyboard.SHIFT + Keyboard.UP):
                self.target_altitude += 0.05
                print(f"целевая высота: {self.target_altitude:.3f} [m]")
            elif key == (Keyboard.SHIFT + Keyboard.DOWN):
                self.target_altitude -= 0.05
                print(f"целевая высота: {self.target_altitude:.3f} [m]")
                
            key = self.keyboard.getKey()
            
        return roll_disturbance, pitch_disturbance, yaw_disturbance
        
    def activate_emergency_landing(self):

        if self.emergency_mode:
            return
            
        self.emergency_mode = True
        print("Произошла авария, начинаю поиск места посадки")
        

        self.camera_pitch_motor.setPosition(1.57)  # 90 градусов вниз, 1 для установки под углом
        
        self.emergency_controller.start_landing()
        
    def run(self):
        self.wait(1.0)
        self.display_instructions()
        
        while self.robot.step(self.timestep) != -1:
            current_time = self.robot.getTime()
            
            roll, pitch, yaw = self.imu.getRollPitchYaw()
            altitude = self.gps.getValues()[2]
            roll_velocity, pitch_velocity, yaw_velocity = self.gyro.getValues()
            
            # Мигаем светодиодами, потому что красиво
            if self.emergency_mode:
                led_state = int(current_time * 4) % 2  # Быстрее в аварийном режиме
            else:
                led_state = int(current_time) % 2
                
            self.front_left_led.set(led_state)
            self.front_right_led.set(not led_state)
            
            # стабилизация камеры (только в нормальном режиме)
            if not self.emergency_mode:
                self.camera_roll_motor.setPosition(-0.115 * roll_velocity)
                self.camera_pitch_motor.setPosition(-0.1 * pitch_velocity)
                
            # обработка клавиатуры
            roll_disturbance, pitch_disturbance, yaw_disturbance = self.process_keyboard_input()
            
            # обновление управления посадкой
            current_target_altitude = self.target_altitude
            if self.emergency_mode:
                landing_target_altitude = self.emergency_controller.update_landing(altitude)
                if landing_target_altitude is not None:
                    current_target_altitude = landing_target_altitude
                    
                # отладочная информация
                phase = self.emergency_controller.landing_phase
                if phase == "preparing":
                    print(f"Подготовка к посадке: {altitude:.2f}m - Camera adjustment")
                elif phase == "searching":
                    search_time = current_time - self.emergency_controller.search_start_time
                    print(f"Поиск нужной поверхности: {altitude:.2f}m, time={search_time:.1f}s")
                elif phase == "hovering":
                    remaining = 2.0 - (current_time - self.emergency_controller.hover_start_time)
                    print(f"Зависание перед снижением: {altitude:.2f}m, starting in {remaining:.1f}s")
                elif phase == "descending":
                    print(f"Высота при снижении: {altitude:.2f}m")
                    
            # вычисление управляющих сигналов
            roll_input = self.K_ROLL_P * self.clamp(roll, -1.0, 1.0) + roll_velocity + roll_disturbance
            pitch_input = self.K_PITCH_P * self.clamp(pitch, -1.0, 1.0) + pitch_velocity + pitch_disturbance
            yaw_input = yaw_disturbance
            
            # коррекция управления для посадки
            if self.emergency_mode:
                roll_input, pitch_input, yaw_input = self.emergency_controller.apply_landing_correction(
                    roll_input, pitch_input, yaw_input
                )
                
            clamped_difference_altitude = self.clamp(
                current_target_altitude - altitude + self.K_VERTICAL_OFFSET, -1.0, 1.0
            )
            vertical_input = self.K_VERTICAL_P * math.pow(clamped_difference_altitude, 3.0)
            
            # управление моторами
            front_left_motor_input = self.K_VERTICAL_THRUST + vertical_input - roll_input + pitch_input - yaw_input
            front_right_motor_input = self.K_VERTICAL_THRUST + vertical_input + roll_input + pitch_input + yaw_input
            rear_left_motor_input = self.K_VERTICAL_THRUST + vertical_input - roll_input - pitch_input + yaw_input
            rear_right_motor_input = self.K_VERTICAL_THRUST + vertical_input + roll_input - pitch_input - yaw_input
            
            # применение скоростей к моторам
            self.front_left_motor.setVelocity(front_left_motor_input)
            self.front_right_motor.setVelocity(-front_right_motor_input)
            self.rear_left_motor.setVelocity(-rear_left_motor_input)
            self.rear_right_motor.setVelocity(rear_right_motor_input)
            
            # проверка завершения посадки
            if self.emergency_mode and not self.emergency_controller.is_landing:
                print("Аварийная посадка выполнена успешно!")
                
                # Постепенное снижение оборотов вместо резкого отключения
                for i in range(50):  # 50 для плавности, можно меньше
                    reduced_thrust = self.K_VERTICAL_THRUST * (1 - i/50)
                    self.front_left_motor.setVelocity(reduced_thrust)
                    self.front_right_motor.setVelocity(-reduced_thrust)
                    self.rear_left_motor.setVelocity(-reduced_thrust)
                    self.rear_right_motor.setVelocity(reduced_thrust)
                    
                    if self.robot.step(self.timestep) == -1:
                        break
                        
                for motor in self.motors:
                    motor.setVelocity(0.0)
                    
                self.emergency_mode = False
                print("Моторы выключены, посадка завершена")
                
               
                self.camera_pitch_motor.setPosition(0.0)
                break

if __name__ == "__main__":
    controller = Mavic2ProController()
    controller.run()