import numpy as np
from layers import (
    Linear, BatchNorm, Dropout, ReLU, Sigmoid, Softmax,
    CrossEntropyLoss, Sequential, Adam
)


def test_linear():
    """Тест Linear слоя."""
    print("=" * 60)
    print("Тест Linear")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Тест 1: Форма выхода
    linear = Linear(10, 5)
    x = np.random.randn(32, 10)
    out = linear.forward(x)
    assert out.shape == (32, 5), f"Expected shape (32, 5), got {out.shape}"
    
    # Тест 2: Градиентная проверка
    linear = Linear(3, 2)
    x = np.random.randn(4, 3)
    out = linear.forward(x)
    
    grad_output = np.random.randn(4, 2)
    grad_input = linear.backward(grad_output)
    
    assert grad_input.shape == x.shape, f"Grad input shape mismatch"
    assert linear.grads['weight'].shape == linear.params['weight'].shape
    assert linear.grads['bias'].shape == linear.params['bias'].shape
    
    # Тест 3: Численная проверка градиента
    eps = 1e-5
    x = np.random.randn(2, 3)
    linear = Linear(3, 2)
    
    # Forward и backward
    out = linear.forward(x)
    grad_out = np.ones_like(out)
    grad_in = linear.backward(grad_out)
    
    # Численный градиент по входу
    numerical_grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_plus = x.copy()
            x_plus[i, j] += eps
            out_plus = linear.forward(x_plus)
            
            x_minus = x.copy()
            x_minus[i, j] -= eps
            out_minus = linear.forward(x_minus)
            
            numerical_grad[i, j] = np.sum(out_plus - out_minus) / (2 * eps)
    
    assert np.allclose(grad_in, numerical_grad, atol=1e-4), \
        f"Gradient check failed for Linear input"
    
    print("Все тесты Linear пройдены")
    print()


def test_batchnorm():
    """Тест BatchNorm слоя."""
    print("=" * 60)
    print("Тест BatchNorm")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Тест 1: Нормализация в режиме обучения
    bn = BatchNorm(4)
    x = np.random.randn(32, 4) * 5 + 3  # Смещённые данные
    
    out = bn.forward(x)
    
    # После BatchNorm среднее ≈ beta (0), std ≈ gamma (1)
    assert np.allclose(np.mean(out, axis=0), bn.params['beta'], atol=1e-5), \
        "Mean should be close to beta"
    assert np.allclose(np.std(out, axis=0), bn.params['gamma'], atol=0.1), \
        "Std should be close to gamma"
    
    # Тест 2: Режим инференса
    bn.eval()
    x_test = np.random.randn(8, 4) * 5 + 3
    out_test = bn.forward(x_test)
    
    # Running stats должны использоваться
    assert out_test.shape == x_test.shape
    
    # Тест 3: Обратный проход
    bn.train()
    x = np.random.randn(16, 4)
    out = bn.forward(x)
    grad_output = np.random.randn(16, 4)
    grad_input = bn.backward(grad_output)
    
    assert grad_input.shape == x.shape
    assert 'gamma' in bn.grads
    assert 'beta' in bn.grads
    
    print("Все тесты BatchNorm пройдены")
    print()


def test_dropout():
    """Тест Dropout слоя."""
    print("=" * 60)
    print("Тест Dropout")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Тест 1: Доля обнулённых элементов
    dropout = Dropout(p=0.5)
    x = np.ones((1000, 100))
    
    out = dropout.forward(x)
    
    # Примерно 50% должны быть обнулены
    zero_ratio = np.mean(out == 0)
    assert 0.4 < zero_ratio < 0.6, f"Expected ~50% zeros, got {zero_ratio*100:.1f}%"
    
    # Тест 2: Масштабирование (inverted dropout)
    # Среднее должно сохраняться примерно
    assert 0.8 < np.mean(out) < 1.2, f"Mean should be close to 1, got {np.mean(out)}"
    
    # Тест 3: Режим инференса (без dropout)
    dropout.eval()
    x = np.random.randn(32, 64)
    out = dropout.forward(x)
    assert np.allclose(out, x), "In eval mode, output should equal input"
    
    # Тест 4: Обратный проход
    dropout.train()
    x = np.random.randn(32, 64)
    out = dropout.forward(x)
    grad_output = np.ones_like(out)
    grad_input = dropout.backward(grad_output)
    
    # Градиент должен быть 0 где вход был обнулён
    assert np.all((grad_input == 0) == (out == 0))
    
    print("Все тесты Dropout пройдены")
    print()


def test_relu():
    """Тест ReLU активации."""
    print("=" * 60)
    print("Тест ReLU")
    print("=" * 60)
    
    relu = ReLU()
    
    # Тест 1: Базовая функциональность
    x = np.array([-2, -1, 0, 1, 2])
    out = relu.forward(x)
    expected = np.array([0, 0, 0, 1, 2])
    assert np.allclose(out, expected), f"Expected {expected}, got {out}"
    
    # Тест 2: Градиент
    grad_output = np.ones_like(x)
    grad_input = relu.backward(grad_output)
    expected_grad = np.array([0, 0, 0, 1, 1])
    assert np.allclose(grad_input, expected_grad), f"Expected {expected_grad}, got {grad_input}"
    
    # Тест 3: Численная проверка градиента
    np.random.seed(42)
    x = np.random.randn(10)
    relu.forward(x)
    grad_out = np.random.randn(10)
    grad_in = relu.backward(grad_out)
    
    eps = 1e-5
    numerical_grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        out_plus = np.maximum(0, x_plus)
        
        x_minus = x.copy()
        x_minus[i] -= eps
        out_minus = np.maximum(0, x_minus)
        
        numerical_grad[i] = np.sum(grad_out * (out_plus - out_minus)) / (2 * eps)
    
    # Проверяем только там, где x далеко от 0
    mask = np.abs(x) > 0.1
    assert np.allclose(grad_in[mask], numerical_grad[mask], atol=1e-4)
    
    print("Все тесты ReLU пройдены")
    print()


def test_sigmoid():
    """Тест Sigmoid активации."""
    print("=" * 60)
    print("Тест Sigmoid")
    print("=" * 60)
    
    sigmoid = Sigmoid()
    
    # Тест 1: Известные значения
    x = np.array([0])
    out = sigmoid.forward(x)
    assert np.allclose(out, 0.5), f"sigmoid(0) should be 0.5, got {out}"
    
    # Тест 2: Диапазон выходов
    x = np.array([-10, -1, 0, 1, 10])
    out = sigmoid.forward(x)
    assert np.all(out > 0) and np.all(out < 1), "Sigmoid output should be in (0, 1)"
    
    # Тест 3: Градиент в точке 0
    x = np.array([0.0])
    sigmoid.forward(x)
    grad_out = np.array([1.0])
    grad_in = sigmoid.backward(grad_out)
    # sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25
    assert np.allclose(grad_in, 0.25), f"Expected 0.25, got {grad_in}"
    
    # Тест 4: Численная проверка
    np.random.seed(42)
    x = np.random.randn(10)
    sigmoid.forward(x)
    grad_out = np.random.randn(10)
    grad_in = sigmoid.backward(grad_out)
    
    eps = 1e-5
    numerical_grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        out_plus = 1 / (1 + np.exp(-x_plus))
        
        x_minus = x.copy()
        x_minus[i] -= eps
        out_minus = 1 / (1 + np.exp(-x_minus))
        
        numerical_grad[i] = np.sum(grad_out * (out_plus - out_minus)) / (2 * eps)
    
    assert np.allclose(grad_in, numerical_grad, atol=1e-4)
    
    print("Все тесты Sigmoid пройдены")
    print()


def test_softmax():
    """Тест Softmax активации."""
    print("=" * 60)
    print("Тест Softmax")
    print("=" * 60)
    
    softmax = Softmax()
    
    # Тест 1: Сумма вероятностей = 1
    x = np.random.randn(32, 10)
    out = softmax.forward(x)
    sums = np.sum(out, axis=1)
    assert np.allclose(sums, 1.0), "Softmax outputs should sum to 1"
    
    # Тест 2: Все значения положительные
    assert np.all(out > 0), "Softmax outputs should be positive"
    
    # Тест 3: Численная стабильность (большие значения)
    x_large = np.array([[1000, 1001, 1002]])
    out_large = softmax.forward(x_large)
    assert not np.any(np.isnan(out_large)), "Softmax should handle large values"
    assert np.allclose(np.sum(out_large), 1.0)
    
    # Тест 4: Градиент
    np.random.seed(42)
    x = np.random.randn(4, 5)
    out = softmax.forward(x)
    grad_out = np.random.randn(4, 5)
    grad_in = softmax.backward(grad_out)
    
    assert grad_in.shape == x.shape
    
    # Численная проверка градиента
    eps = 1e-5
    numerical_grad = np.zeros_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_plus = x.copy()
            x_plus[i, j] += eps
            # Пересоздаём softmax для чистого forward
            sm = Softmax()
            out_plus = sm.forward(x_plus)
            
            x_minus = x.copy()
            x_minus[i, j] -= eps
            sm = Softmax()
            out_minus = sm.forward(x_minus)
            
            numerical_grad[i, j] = np.sum(grad_out * (out_plus - out_minus)) / (2 * eps)
    
    assert np.allclose(grad_in, numerical_grad, atol=1e-4), \
        f"Softmax gradient check failed"
    
    print("Все тесты Softmax пройдены")
    print()


def test_cross_entropy():
    """Тест Cross-Entropy Loss."""
    print("=" * 60)
    print("Тест CrossEntropyLoss")
    print("=" * 60)
    
    ce_loss = CrossEntropyLoss()
    
    # Тест 1: Известное значение
    logits = np.array([[0, 0, 0]])  # Равные логиты
    targets = np.array([0])
    loss = ce_loss.forward(logits, targets)
    # Для равных логитов: -log(1/3) = log(3) ≈ 1.0986
    assert np.allclose(loss, np.log(3), atol=1e-4)
    
    # Тест 2: Идеальное предсказание
    logits = np.array([[100, 0, 0]])  # Уверенное предсказание класса 0
    targets = np.array([0])
    loss = ce_loss.forward(logits, targets)
    assert loss < 0.01, "Loss should be very small for correct confident prediction"
    
    # Тест 3: Градиент
    np.random.seed(42)
    logits = np.random.randn(8, 5)
    targets = np.random.randint(0, 5, 8)
    
    loss = ce_loss.forward(logits, targets)
    grad = ce_loss.backward()
    
    assert grad.shape == logits.shape
    
    print("Все тесты CrossEntropyLoss пройдены")
    print()


def test_sequential():
    """Тест Sequential модели."""
    print("=" * 60)
    print("Тест Sequential")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Создаём простую сеть
    model = Sequential([
        Linear(10, 32),
        BatchNorm(32),
        ReLU(),
        Dropout(0.2),
        Linear(32, 5)
    ])
    
    # Тест 1: Forward pass
    x = np.random.randn(16, 10)
    out = model.forward(x)
    assert out.shape == (16, 5), f"Expected shape (16, 5), got {out.shape}"
    
    # Тест 2: Backward pass
    grad_output = np.random.randn(16, 5)
    grad_input = model.backward(grad_output)
    assert grad_input.shape == x.shape
    
    # Тест 3: Режимы train/eval
    model.eval()
    out_eval = model.forward(x)
    model.train()
    
    assert out_eval.shape == (16, 5)
    
    print("Все тесты Sequential пройдены")
    print()


def test_full_training():
    """Полный тест обучения на синтетических данных."""
    print("=" * 60)
    print("Тест полного цикла обучения")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Генерируем данные для классификации
    n_samples = 200
    n_features = 10
    n_classes = 3
    
    # Создаём разделимые классы
    X = np.random.randn(n_samples, n_features)
    true_weights = np.random.randn(n_features, n_classes)
    logits = np.dot(X, true_weights)
    y = np.argmax(logits + np.random.randn(*logits.shape) * 0.5, axis=1)
    
    # Создаём модель
    model = Sequential([
        Linear(n_features, 32),
        BatchNorm(32),
        ReLU(),
        Dropout(0.1),
        Linear(32, 16),
        ReLU(),
        Linear(16, n_classes)
    ])
    
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(learning_rate=0.01)
    
    # Обучение
    losses = []
    accuracies = []
    epochs = 100
    
    for epoch in range(epochs):
        model.train()
        
        # Forward
        logits = model.forward(X)
        loss = loss_fn.forward(logits, y)
        
        # Backward
        grad = loss_fn.backward()
        model.backward(grad)
        
        # Update
        model.update_params(optimizer)
        
        # Метрики
        model.eval()
        pred_logits = model.forward(X)
        predictions = np.argmax(pred_logits, axis=1)
        accuracy = np.mean(predictions == y)
        
        losses.append(loss)
        accuracies.append(accuracy)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch + 1}/{epochs}: Loss = {loss:.4f}, Accuracy = {accuracy:.2%}")
    
    assert losses[-1] < losses[0], "Loss should decrease during training"
    assert accuracies[-1] > 0.7, f"Final accuracy should be > 70%, got {accuracies[-1]:.2%}"
    
    print(f"\nФинальные метрики: Loss = {losses[-1]:.4f}, Accuracy = {accuracies[-1]:.2%}")
    print("Тест полного цикла обучения пройден")
    print()
    
    return losses, accuracies

