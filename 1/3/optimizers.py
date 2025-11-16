import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

class Optimizer(ABC):
    """Абстрактный базовый класс для всех оптимизаторов."""
    
    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate
        self.state: Dict = {}  # Состояние оптимизатора для каждого параметра
    
    @abstractmethod
    def update(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """
        Обновляет параметр на основе градиента.
        
        Args:
            param_name: Имя параметра (для хранения состояния)
            param: Текущее значение параметра
            grad: Градиент по параметру
            
        Returns:
            Обновленное значение параметра
        """
        pass
    
    def reset(self):
        """Сбрасывает состояние оптимизатора."""
        self.state = {}

class Momentum(Optimizer):    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
    
    def update(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if param_name not in self.state:
            self.state[param_name] = {'velocity': np.zeros_like(param)}
        
        v = self.state[param_name]['velocity']
        
        # обновляем скорость: v = β * v + g
        v = self.momentum * v + grad
        self.state[param_name]['velocity'] = v
        
        # обновляем параметр
        return param - self.learning_rate * v

class Nesterov(Optimizer):    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
    
    def update(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if param_name not in self.state:
            self.state[param_name] = {'velocity': np.zeros_like(param)}
        
        v = self.state[param_name]['velocity']
        
        # обновляем скорость
        v_new = self.momentum * v + self.learning_rate * grad
        self.state[param_name]['velocity'] = v_new
        
        # используем "заглядывание вперед"
        # θ = θ - (β * v_new + lr * grad)
        return param - (self.momentum * v_new + self.learning_rate * grad)

class Adagrad(Optimizer):    
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.epsilon = epsilon
    
    def update(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if param_name not in self.state:
            self.state[param_name] = {'G': np.zeros_like(param)}
        
        G = self.state[param_name]['G']
        
        # накапливаем квадраты градиентов
        G = G + grad ** 2
        self.state[param_name]['G'] = G
        
        # адаптивное обновление
        return param - self.learning_rate * grad / (np.sqrt(G) + self.epsilon)

class RMSProp(Optimizer):
    def __init__(self, learning_rate: float = 0.01, decay: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.decay = decay
        self.epsilon = epsilon
    
    def update(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if param_name not in self.state:
            self.state[param_name] = {'E_g2': np.zeros_like(param)}
        
        E_g2 = self.state[param_name]['E_g2']
        
        # экспоненциальное скользящее среднее квадратов градиентов
        E_g2 = self.decay * E_g2 + (1 - self.decay) * grad ** 2
        self.state[param_name]['E_g2'] = E_g2
        
        # адаптивное обновление
        return param - self.learning_rate * grad / (np.sqrt(E_g2) + self.epsilon)

class Adam(Optimizer):    
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9, 
                 beta2: float = 0.999, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # счетчик шагов для bias correction
    
    def update(self, param_name: str, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        if param_name not in self.state:
            self.state[param_name] = {
                'm': np.zeros_like(param),  # первый момент
                'v': np.zeros_like(param)   # второй момент
            }
        
        # увеличиваем счетчик только один раз за шаг оптимизации
        # (предполагаем, что update вызывается для всех параметров последовательно)
        if param_name == list(self.state.keys())[0] or len(self.state) == 1:
            self.t += 1
        
        m = self.state[param_name]['m']
        v = self.state[param_name]['v']
        
        # обновляем моменты
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * grad ** 2
        
        self.state[param_name]['m'] = m
        self.state[param_name]['v'] = v
        
        # коррекция смещения
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        
        # обновление параметра
        return param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def reset(self):
        super().reset()
        self.t = 0

class AdamW(Optimizer):
    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
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
        
        # обновляем моменты без weight decay в градиенте!
        m = self.beta1 * m + (1 - self.beta1) * grad
        v = self.beta2 * v + (1 - self.beta2) * grad ** 2
        
        self.state[param_name]['m'] = m
        self.state[param_name]['v'] = v
        
        # коррекция смещения
        m_hat = m / (1 - self.beta1 ** self.t)
        v_hat = v / (1 - self.beta2 ** self.t)
        
        # decoupled weight decay: применяем напрямую к весам
        # θ = θ * (1 - lr * λ) - lr * m̂ / (sqrt(v̂) + ε)
        param_decayed = param * (1 - self.learning_rate * self.weight_decay)
        return param_decayed - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
    
    def reset(self):
        super().reset()
        self.t = 0


class Neuron:
    """Нейрон с сигмоидной активацией и NLL loss"""
    
    def __init__(self, n_inputs: int):
        self.weights = np.random.randn(n_inputs) * 0.1
        self.bias = 0.0
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Прямой проход: z = X @ w + b, a = sigmoid(z)"""
        self.X = X
        self.z = np.dot(X, self.weights) + self.bias
        self.y_pred = self.sigmoid(self.z)
        return self.y_pred
    
    def compute_nll(self, y: np.ndarray) -> float:
        """NLL = -mean(y*log(p) + (1-y)*log(1-p))"""
        eps = 1e-7
        return -np.mean(y * np.log(self.y_pred + eps) + 
                       (1 - y) * np.log(1 - self.y_pred + eps))
    
    def compute_gradients(self, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Градиенты для NLL + sigmoid: error = y_pred - y"""
        error = self.y_pred - y
        grad_weights = np.dot(self.X.T, error) / len(y)
        grad_bias = np.mean(error)
        return grad_weights, grad_bias
    
    def train_step(self, X: np.ndarray, y: np.ndarray, optimizer: Optimizer) -> float:
        """Один шаг обучения."""
        self.forward(X)
        nll = self.compute_nll(y)
        grad_w, grad_b = self.compute_gradients(y)
        
        self.weights = optimizer.update('weights', self.weights, grad_w)
        self.bias = optimizer.update('bias', np.array([self.bias]), np.array([grad_b]))[0]
        
        return nll
    
    def train(self, X: np.ndarray, y: np.ndarray, optimizer: Optimizer,
              epochs: int = 100, verbose: bool = False) -> List[float]:
        """Обучение на нескольких эпохах."""
        nll_history = []
        for epoch in range(epochs):
            nll = self.train_step(X, y, optimizer)
            nll_history.append(nll)
            if verbose and (epoch + 1) % (epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, NLL: {nll:.4f}")
        return nll_history

