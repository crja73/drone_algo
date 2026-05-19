class LandingAnalyzer:
   
    def __init__(self):
        self.good_surface_threshold = 200
        self.min_uniform_area = 0.6
        self.sand_lower_1 = np.array([10, 30, 80])    # светлый песок
        self.sand_upper_1 = np.array([30, 200, 255])
        
        self.sand_lower_2 = np.array([0, 20, 100])    # очень светлый песок
        self.sand_upper_2 = np.array([20, 150, 255])
        
        self.sand_lower_3 = np.array([5, 20, 40])     # песок при низкой освещённости
        self.sand_upper_3 = np.array([35, 150, 85])
        
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
            # Преобразование изображения
            image = np.frombuffer(image_data, np.uint8)
            image = image.reshape((camera_height, camera_width, 4))
            
            # Конвертация BGRA -> BGR -> HSV
            bgr_image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            # Применение CLAHE для нормализации освещения
            lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            lab = cv2.merge([l, a, b])
            bgr_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
            
            # Анализ центральной области (игнорируем края кадра)
            h, w = hsv_image.shape[:2]
            margin_h, margin_w = int(h * 0.15), int(w * 0.15)
            center_hsv = hsv_image[margin_h:h-margin_h, margin_w:w-margin_w]
            
            # Создание масок для разных типов поверхностей
            # Песок (три маски для разных оттенков и условий освещения)
            sand_mask_1 = cv2.inRange(center_hsv, self.sand_lower_1, self.sand_upper_1)
            sand_mask_2 = cv2.inRange(center_hsv, self.sand_lower_2, self.sand_upper_2)
            sand_mask_3 = cv2.inRange(center_hsv, self.sand_lower_3, self.sand_upper_3)
            sand_mask = cv2.bitwise_or(sand_mask_1, sand_mask_2)
            sand_mask = cv2.bitwise_or(sand_mask, sand_mask_3)
            
            # Асфальт
            asphalt_mask = cv2.inRange(center_hsv, self.asphalt_lower, self.asphalt_upper)
            
            # Темные объекты (машины, тени)
            dark_mask = cv2.inRange(center_hsv, self.dark_lower, self.dark_upper)
            
            # Вычисление процентного соотношения
            total_pixels = center_hsv.shape[0] * center_hsv.shape[1]
            sand_ratio = np.sum(sand_mask > 0) / total_pixels
            asphalt_ratio = np.sum(asphalt_mask > 0) / total_pixels
            dark_ratio = np.sum(dark_mask > 0) / total_pixels
            
            # Анализ краев (для проверки на объекты типа машин)
            gray = cv2.cvtColor(bgr_image[margin_h:h-margin_h, margin_w:w-margin_w], 
                               cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Проверка однородности песка (песок должен быть относительно однородным)
            sand_regions = sand_mask.copy()
            # Морфологическая операция для удаления шума
            kernel = np.ones((5, 5), np.uint8)
            sand_regions = cv2.morphologyEx(sand_regions, cv2.MORPH_CLOSE, kernel)
            sand_regions = cv2.morphologyEx(sand_regions, cv2.MORPH_OPEN, kernel)
            
            # Анализ связных компонент песка
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(sand_regions)
            
            # Проверка на наличие крупных однородных областей песка
            has_large_sand_area = False
            if num_labels > 1:  # 0 это фон
                largest_area = np.max(stats[1:, cv2.CC_STAT_AREA])
                if largest_area > total_pixels * 0.4:  # хотя бы 40% одной областью
                    has_large_sand_area = True
            
            # Система оценки (0-100)
            quality_score = 0
            
            # 1. Оценка за наличие песка (0-40 баллов)
            if sand_ratio >= self.min_sand_ratio:
                quality_score += 40
            else:
                quality_score += int(40 * (sand_ratio / self.min_sand_ratio))
            
            # 2. Штраф за асфальт (0-25 баллов)
            if asphalt_ratio <= self.max_asphalt_ratio:
                quality_score += 25
            else:
                penalty = int(25 * (1 - min(asphalt_ratio, 0.5) / 0.5))
                quality_score += max(0, penalty)
            
            # 3. Штраф за темные объекты (0-20 баллов)
            if dark_ratio <= self.max_dark_ratio:
                quality_score += 20
            else:
                penalty = int(20 * (1 - min(dark_ratio, 0.3) / 0.3))
                quality_score += max(0, penalty)
            
            # 4. Оценка за однородность (0-15 баллов)
            if has_large_sand_area:
                quality_score += 15
            elif num_labels > 1:
                quality_score += int(15 * (largest_area / (total_pixels * 0.4)))
            
            # 5. Штраф за края (объекты) (вычитаем до 10 баллов)
            if edge_density > self.edge_threshold:
                edge_penalty = min(10, int(10 * (edge_density / 0.1)))
                quality_score -= edge_penalty
            
            # Определение пригодности для посадки
            is_suitable = (
                quality_score >= 70 and                    # общая оценка хорошая
                sand_ratio >= self.min_sand_ratio and      # достаточно песка
                asphalt_ratio <= self.max_asphalt_ratio and # мало асфальта
                dark_ratio <= self.max_dark_ratio and       # нет темных объектов
                edge_density <= self.edge_threshold         # нет четких объектов
            )
            
            center_coords = (0.0, 0.0)
            
            # Вывод детальной информации
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