from test_optimizers import (
    test_momentum,
    test_nesterov,
    test_adagrad,
    test_rmsprop,
    test_adam,
    test_adamw,
    test_neuron_training,
    compare_optimizers_visualization,
    rosenbrock_comparison
)


def main():
    print("\n" + "=" * 60)
    print("ЗАДАНИЕ 3: РЕАЛИЗАЦИЯ ОПТИМИЗАТОРОВ")
    print("=" * 60 + "\n")
    
    # Запускаем все тесты
    test_momentum()
    test_nesterov()
    test_adagrad()
    test_rmsprop()
    test_adam()
    test_adamw()
    
    # Тест обучения нейрона
    results = test_neuron_training()
    
    # Визуализация
    compare_optimizers_visualization(results)
    
    # Сравнение на функции Розенброка
    rosenbrock_comparison()


if __name__ == "__main__":
    main()