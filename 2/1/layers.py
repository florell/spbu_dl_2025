import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional


# =============================================================================
# Базовый класс оптимизатора (из задания 3 прошлой работы)
# =============================================================================

class Optimizer(ABC):
    """Абстрактный базовый класс для всех оптимизаторов."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.state: Dict = {}
    
    @abstractmethod
    def update(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        pass
    
    def reset(self):
        self.state = {}


class Adam(Optimizer):
    """Adam оптимизатор."""
    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
    
    def update(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if param_name not in self.state:
            self.state[param_name] = {
                'm': np.zeros_like(param),
                'v': np.zeros_like(param)
            }
        
        if param_name == list(self.state.keys())[0] or len(self.state) == 1:
            self.t += 1
        
        m = self.state[param_name]['m']
        v = self.state[param_name]['v']
        
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * grad ** 2
        
        self.state[param_name]['m'] = m
        self.state[param_name]['v'] = v
        
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        
        return param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def reset(self):
        super().reset()
        self.t = 0


class SGD(Optimizer):
    """SGD с опциональным momentum."""
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
    
    def update(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if self.momentum > 0:
            if param_name not in self.state:
                self.state[param_name] = {'velocity': np.zeros_like(param)}
            
            v = self.state[param_name]['velocity']
            v = self.momentum * v + grad
            self.state[param_name]['velocity'] = v
            return param - self.learning_rate * v
        
        return param - self.learning_rate * grad


# =============================================================================
# Базовый класс слоя
# =============================================================================

class Layer(ABC):
    """
    Абстрактный базовый класс для всех слоёв нейронной сети.
    
    Каждый слой должен реализовать:
    - forward: прямой проход
    - backward: обратный проход (вычисление градиентов)
    """
    
    def __init__(self):
        self.training = True
        self.params: Dict[str, np.ndarray] = {}
        self.grads: Dict[str, np.ndarray] = {}
        self.cache: Dict = {}
    
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход."""
        pass
    
    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Обратный проход. Возвращает градиент по входу."""
        pass
    
    def train(self):
        """Переключение в режим обучения."""
        self.training = True
    
    def eval(self):
        """Переключение в режим инференса."""
        self.training = False
    
    def update_params(self, optimizer: Optimizer, layer_name: str):
        """Обновление параметров слоя с помощью оптимизатора."""
        for name, param in self.params.items():
            if name in self.grads:
                full_name = f"{layer_name}_{name}"
                self.params[name] = optimizer.update(full_name, param, self.grads[name])


# =============================================================================
# Задача 1: BatchNorm
# =============================================================================

class BatchNorm(Layer):
    """
    Batch Normalization слой.
    
    Нормализует входные данные по мини-батчу для стабилизации обучения.
    
    Формулы:
        μ = mean(x, axis=0)                    - среднее по батчу
        σ^2 = var(x, axis=0)                   - дисперсия по батчу
        x̂ = (x - μ) / sqrt(σ^2 + ε)           - нормализация
        y = γ * x̂ + β                         - масштабирование и сдвиг
    
    При инференсе используются накопленные running_mean и running_var.
    
    Args:
        num_features: количество признаков (размерность входа)
        epsilon: малая константа для численной стабильности
        momentum: коэффициент для обновления running statistics
    """
    
    def __init__(self, num_features: int, epsilon: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        
        # Обучаемые параметры
        self.params['gamma'] = np.ones(num_features)   # масштаб
        self.params['beta'] = np.zeros(num_features)   # сдвиг
        
        # Running statistics для инференса
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Прямой проход BatchNorm.
        
        Args:
            x: входной тензор shape (batch_size, num_features)
        
        Returns:
            нормализованный тензор той же формы
        """
        if self.training:
            # Вычисляем статистики по батчу
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            
            # Обновляем running statistics (экспоненциальное скользящее среднее)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Используем статистики батча
            mean = batch_mean
            var = batch_var
        else:
            # При инференсе используем накопленные статистики
            mean = self.running_mean
            var = self.running_var
        
        # Нормализация
        self.cache['x'] = x
        self.cache['mean'] = mean
        self.cache['var'] = var
        
        x_centered = x - mean
        std = np.sqrt(var + self.epsilon)
        x_norm = x_centered / std
        
        self.cache['x_centered'] = x_centered
        self.cache['std'] = std
        self.cache['x_norm'] = x_norm
        
        # Масштабирование и сдвиг
        out = self.params['gamma'] * x_norm + self.params['beta']
        
        return out
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Обратный проход BatchNorm.
        
        Градиенты вычисляются по формулам:
        dL/dγ = sum(dL/dy * x̂)
        dL/dβ = sum(dL/dy)
        dL/dx = (1/N) * γ/σ * (N*dL/dx̂ - sum(dL/dx̂) - x̂*sum(dL/dx̂*x̂))
        """
        x_norm = self.cache['x_norm']
        std = self.cache['std']
        gamma = self.params['gamma']
        
        N = grad_output.shape[0]
        
        # Градиенты по параметрам
        self.grads['gamma'] = np.sum(grad_output * x_norm, axis=0)
        self.grads['beta'] = np.sum(grad_output, axis=0)
        
        # Градиент по x_norm
        dx_norm = grad_output * gamma
        
        # Градиент по входу через цепочку производных
        # Используем упрощённую формулу для полного градиента
        dx = (1.0 / N) * (1.0 / std) * (
            N * dx_norm 
            - np.sum(dx_norm, axis=0) 
            - x_norm * np.sum(dx_norm * x_norm, axis=0)
        )
        
        return dx


# =============================================================================
# Задача 2: Linear
# =============================================================================

class Linear(Layer):
    """
    Полносвязный линейный слой.
    
    Выполняет линейное преобразование: y = x @ W + b
    
    Args:
        in_features: размерность входа
        out_features: размерность выхода
        bias: использовать ли смещение
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        
        # Инициализация весов (Xavier)
        # Помогает поддерживать дисперсию активаций
        limit = np.sqrt(6.0 / (in_features + out_features))
        self.params['weight'] = np.random.uniform(-limit, limit, (in_features, out_features))
        
        if bias:
            self.params['bias'] = np.zeros(out_features)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Прямой проход линейного слоя.
        
        Args:
            x: входной тензор shape (batch_size, in_features)
        
        Returns:
            выходной тензор shape (batch_size, out_features)
        """
        self.cache['input'] = x
        
        out = np.dot(x, self.params['weight'])
        if self.use_bias:
            out = out + self.params['bias']
        
        return out
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Обратный проход линейного слоя.
        
        Градиенты:
        dL/dW = x^T @ dL/dy
        dL/db = sum(dL/dy, axis=0)
        dL/dx = dL/dy @ W^T
        """
        x = self.cache['input']
        
        # Градиент по весам: (in_features, batch) @ (batch, out_features)
        self.grads['weight'] = np.dot(x.T, grad_output)
        
        # Градиент по смещению
        if self.use_bias:
            self.grads['bias'] = np.sum(grad_output, axis=0)
        
        # Градиент по входу: (batch, out_features) @ (out_features, in_features)
        grad_input = np.dot(grad_output, self.params['weight'].T)
        
        return grad_input


# =============================================================================
# Задача 3: Dropout 
# =============================================================================

class Dropout(Layer):
    """
    Dropout слой для регуляризации.
    
    Случайно обнуляет элементы входа с вероятностью p во время обучения.
    При инференсе пропускает вход без изменений.
    
    Используется инвертированный dropout: выход масштабируется на 1/(1-p) 
    во время обучения, чтобы при инференсе не нужно было масштабировать.
    
    Args:
        p: вероятность обнуления элемента (dropout rate)
    """
    
    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0 <= p < 1:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Прямой проход Dropout.
        
        Args:
            x: входной тензор любой формы
        
        Returns:
            тензор с обнулёнными элементами (при обучении)
        """
        if self.training and self.p > 0:
            # Создаём маску: 1 с вероятностью (1-p), 0 с вероятностью p
            self.cache['mask'] = (np.random.rand(*x.shape) > self.p).astype(np.float64)
            # Inverted dropout: масштабируем на 1/(1-p)
            scale = 1.0 / (1.0 - self.p)
            return x * self.cache['mask'] * scale
        
        return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Обратный проход Dropout.
        
        Градиент проходит только через те элементы, которые не были обнулены.
        """
        if self.training and self.p > 0:
            scale = 1.0 / (1.0 - self.p)
            return grad_output * self.cache['mask'] * scale
        
        return grad_output


# =============================================================================
# Задача 4: Функции активации
# =============================================================================

class ReLU(Layer):
    """
    Rectified Linear Unit активация.
    
    f(x) = max(0, x)
    
    Простейшая и одна из самых популярных функций активации.
    Решает проблему затухающих градиентов для положительных значений.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Прямой проход ReLU.
        
        Args:
            x: входной тензор любой формы
        
        Returns:
            тензор с применённым ReLU
        """
        self.cache['input'] = x
        return np.maximum(0, x)
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Обратный проход ReLU.
        
        Градиент: 1 если x > 0, иначе 0
        """
        x = self.cache['input']
        grad_input = grad_output * (x > 0).astype(np.float64)
        return grad_input


class Sigmoid(Layer):
    """
    Логистическая активация.
    
    σ(x) = 1 / (1 + exp(-x))
    
    Преобразует значения в диапазон (0, 1).
    Часто используется в выходном слое для бинарной классификации.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Прямой проход Sigmoid.
        
        Args:
            x: входной тензор любой формы
        
        Returns:
            тензор со значениями в (0, 1)
        """
        # Численно стабильная версия
        out = np.where(
            x >= 0,
            1 / (1 + np.exp(-x)),
            np.exp(x) / (1 + np.exp(x))
        )
        self.cache['output'] = out
        return out
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Обратный проход Sigmoid.
        
        Производная: σ'(x) = σ(x) * (1 - σ(x))
        """
        sigmoid_out = self.cache['output']
        grad_input = grad_output * sigmoid_out * (1 - sigmoid_out)
        return grad_input


class Softmax(Layer):
    """
    Softmax активация.
    
    softmax(x)_i = exp(x_i) / sum(exp(x_j))
    
    Преобразует вектор логитов в распределение вероятностей.
    Используется в выходном слое для многоклассовой классификации.
    
    Args:
        axis: ось, по которой применяется softmax (по умолчанию -1, последняя)
    """
    
    def __init__(self, axis: int = -1):
        super().__init__()
        self.axis = axis
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Прямой проход Softmax.
        
        Args:
            x: входной тензор, обычно shape (batch_size, num_classes)
        
        Returns:
            тензор вероятностей той же формы
        """
        # Вычитаем максимум для численной стабильности
        x_shifted = x - np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        out = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
        self.cache['output'] = out
        return out
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Обратный проход Softmax.
        
        Для softmax якобиан имеет вид:
        ∂softmax_i/∂x_j = softmax_i * (δ_ij - softmax_j)
        
        где δ_ij - символ Кронекера.
        
        Упрощённая формула для вектора градиентов:
        dL/dx = softmax * (dL/dy - sum(dL/dy * softmax))
        """
        softmax_out = self.cache['output']
        
        # Эффективное вычисление без явного построения якобиана
        # dL/dx_i = sum_j(dL/dy_j * ∂y_j/∂x_i)
        # = sum_j(dL/dy_j * y_j * (δ_ij - y_i))
        # = dL/dy_i * y_i - y_i * sum_j(dL/dy_j * y_j)
        # = y_i * (dL/dy_i - sum_j(dL/dy_j * y_j))
        
        sum_term = np.sum(grad_output * softmax_out, axis=self.axis, keepdims=True)
        grad_input = softmax_out * (grad_output - sum_term)
        
        return grad_input


# =============================================================================
# Вспомогательные классы: функции потерь
# =============================================================================

class CrossEntropyLoss:
    """
    Cross-Entropy Loss для многоклассовой классификации.
    
    Объединяет LogSoftmax и NLLLoss для численной стабильности.
    
    L = -1/N * sum(log(softmax(x)_y))
    
    где y - истинный класс.
    """
    
    def __init__(self):
        self.cache = {}
    
    def forward(self, logits: np.ndarray, targets: np.ndarray) -> float:
        """
        Вычисление loss.
        
        Args:
            logits: предсказания модели shape (batch_size, num_classes)
            targets: истинные метки shape (batch_size,) - индексы классов
        
        Returns:
            значение loss (скаляр)
        """
        batch_size = logits.shape[0]
        
        # Численно стабильный softmax
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        softmax_out = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        self.cache['softmax'] = softmax_out
        self.cache['targets'] = targets
        self.cache['batch_size'] = batch_size
        
        # Cross-entropy: -log(p_correct)
        # Клиппинг для численной стабильности
        eps = 1e-10
        correct_probs = softmax_out[np.arange(batch_size), targets]
        loss = -np.mean(np.log(correct_probs + eps))
        
        return loss
    
    def backward(self) -> np.ndarray:
        """
        Градиент cross-entropy loss по логитам.
        
        Для cross-entropy с softmax градиент упрощается до:
        dL/dx_i = softmax_i - 1{i == target}
        """
        softmax = self.cache['softmax']
        targets = self.cache['targets']
        batch_size = self.cache['batch_size']
        
        grad = softmax.copy()
        grad[np.arange(batch_size), targets] -= 1
        grad /= batch_size
        
        return grad


class BCELoss:
    """
    Binary Cross-Entropy Loss для бинарной классификации.
    
    L = -1/N * sum(y*log(p) + (1-y)*log(1-p))
    """
    
    def __init__(self):
        self.cache = {}
    
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Args:
            predictions: предсказания после sigmoid, shape (batch_size,) или (batch_size, 1)
            targets: истинные метки (0 или 1), той же формы
        """
        eps = 1e-10
        predictions = np.clip(predictions, eps, 1 - eps)
        
        self.cache['predictions'] = predictions
        self.cache['targets'] = targets
        
        loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
        return loss
    
    def backward(self) -> np.ndarray:
        """
        Градиент BCE по предсказаниям (до sigmoid).
        
        Для BCE + sigmoid: dL/dx = predictions - targets
        """
        predictions = self.cache['predictions']
        targets = self.cache['targets']
        batch_size = len(targets.flatten())
        
        # Градиент по выходу sigmoid: (p - y) / (p * (1-p))
        # Но если мы хотим градиент по входу sigmoid, то
        # dL/dx = dL/dp * dp/dx = dL/dp * p(1-p) = p - y
        grad = (predictions - targets) / batch_size
        
        return grad


# =============================================================================
# Sequential модель для объединения слоёв
# =============================================================================

class Sequential:
    """
    Контейнер для последовательного применения слоёв.
    
    Пример:
        model = Sequential([
            Linear(784, 128),
            BatchNorm(128),
            ReLU(),
            Dropout(0.5),
            Linear(128, 10)
        ])
    """
    
    def __init__(self, layers: List[Layer]):
        self.layers = layers
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Прямой проход через все слои."""
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Обратный проход через все слои (в обратном порядке)."""
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output
    
    def train(self):
        """Переключение всех слоёв в режим обучения."""
        for layer in self.layers:
            layer.train()
    
    def eval(self):
        """Переключение всех слоёв в режим инференса."""
        for layer in self.layers:
            layer.eval()
    
    def update_params(self, optimizer: Optimizer):
        """Обновление параметров всех слоёв."""
        for i, layer in enumerate(self.layers):
            layer.update_params(optimizer, f"layer_{i}")
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

