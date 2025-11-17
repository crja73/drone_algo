from car_classifier import predict_image

prediction, confidence = predict_image('C:/Users/chuva/Documents/Python Scripts/binary_car_classieir/dataset/val/car/10.png')
if prediction == 1:
    print(f'Машина обнаружена! Уверенность: {confidence:.2%}')
else:
    print(f'Машины нет. Уверенность: {(1-confidence):.2%}')
