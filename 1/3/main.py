import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Callable
import matplotlib.pyplot as plt


# =============================================================================
# Базовый класс оптимизатора
# =============================================================================

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


# =============================================================================
# 1. Momentum
# =============================================================================

class Momentum(Optimizer):
    """
    Momentum (Импульс)
    
    ИДЕЯ АЛГОРИТМА:
    ---------------
    Представим шарик, катящийся по холмистой поверхности. Обычный градиентный
    спуск - это шарик без массы, который движется строго по направлению
    наибольшего уклона. Momentum добавляет шарику "массу" и "инерцию".
    
    Проблема ванильного градиентного спуска: в узких оврагах (когда функция
    потерь имеет разную кривизну в разных направлениях) градиент колеблется
    поперек оврага, а не движется вдоль него к минимуму.
    
    Решение Momentum: накапливаем скорость - экспоненциально
    взвешенное скользящее среднее градиентов. Колебания поперек оврага
    взаимно компенсируются, а движение вдоль оврага усиливается.
    
    Формулы:
        v_t = β * v_{t-1} + (1 - β) * g_t     (или v_t = β * v_{t-1} + g_t)
        θ_t = θ_{t-1} - lr * v_t
    
    где β (momentum) обычно 0.9, g_t - текущий градиент.
    
    Преимущества:
    - Ускоряет сходимость в направлениях с постоянным градиентом
    - Сглаживает колебания в направлениях с переменным знаком градиента
    - Помогает преодолевать локальные минимумы за счет "разгона"
    """
    
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


# =============================================================================
# 2. NAG
# =============================================================================

class Nesterov(Optimizer):
    """
    Nesterov Accelerated Gradient (NAG)
    
    ИДЕЯ АЛГОРИТМА:
    ---------------
    Nesterov - это "умный" Momentum. Представим: вы катите шарик с горы,
    и он набирает скорость. С обычным Momentum вы смотрите под ноги и решаете,
    куда двигаться. С Nesterov вы "заглядываете вперед" - сначала делаете
    шаг по инерции, а потом смотрите, какой там градиент.
    
    Интуиция: если мы всё равно собираемся двигаться по инерции, давайте
    вычислим градиент не в текущей точке, а в той, куда мы приземлимся
    после движения по инерции. Это дает предвидение и позволяет
    корректировать направление заранее.
    
    Формулы:
        θ_look_ahead = θ_{t-1} - β * v_{t-1}    <- смотрим вперед
        g_t = ∇L(θ_look_ahead)                  <- градиент в будущей точке
        v_t = β * v_{t-1} + lr * g_t
        θ_t = θ_{t-1} - v_t
    
    Эквивалентная форма используется для удобства реализации:
        v_t = β * v_{t-1} + lr * g_t
        θ_t = θ_{t-1} - (β * v_t + lr * g_t)
    
    Преимущества:
    - Более быстрая сходимость, чем у обычного Momentum
    - Лучше "тормозит" при приближении к минимуму
    - Теоретически оптимальная сходимость для выпуклых функций
    """
    
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


# =============================================================================
# 3. Adagrad
# =============================================================================

class Adagrad(Optimizer):
    """
    Adaptive Gradient (Adagrad)
    
    ИДЕЯ АЛГОРИТМА:
    ---------------
    Разные параметры модели могут иметь разную "важность" и частоту обновления.
    Например, в NLP редкие слова встречаются нечасто, но когда встречаются -
    их эмбеддинги нужно сильно обновить. Частые слова обновляются часто,
    но помалу.
    
    Adagrad адаптирует learning rate для каждого параметра индивидуально:
    параметры с большими накопленными градиентами получают меньший lr,
    а параметры с маленькими градиентами - больший lr.
    
    Формулы:
        G_t = G_{t-1} + g_t^2                    <- накапливаем квадраты градиентов
        θ_t = θ_{t-1} - lr / (sqrt(G_t) + ε) * g_t    <- адаптивный learning rate
    
    где ε ≈ 1e-8 для численной стабильности.
    
    Преимущества:
    - Автоматическая адаптация lr для каждого параметра
    - Хорошо работает с разреженными градиентами
    - Не требует ручной настройки lr для разных параметров
    
    Недостатки:
    - G_t монотонно растет, поэтому lr со временем стремится к 0
    - Обучение может "заморозиться" на поздних этапах
    """
    
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


# =============================================================================
# 4. RMSProp
# =============================================================================

class RMSProp(Optimizer):
    """
    Root Mean Square Propagation (RMSProp)
    
    ИДЕЯ АЛГОРИТМА:
    ---------------
    RMSProp решает главную проблему Adagrad - монотонное убывание lr.
    Вместо накопления всех квадратов градиентов, RMSProp использует
    экспоненциально взвешенное скользящее среднее EWMA.
    
    Интуиция: нас интересуют не все градиенты с начала обучения, а только
    недавние. Старые градиенты "забываются" благодаря экспоненциальному
    затуханию. Это позволяет lr адаптироваться к текущей "ситуации" на
    поверхности функции потерь.
    
    Формулы:
        E[g^2]_t = β * E[g^2]_{t-1} + (1-β) * g_t^2    <- скользящее среднее
        θ_t = θ_{t-1} - lr / (sqrt(E[g^2]_t) + ε) * g_t
    
    где β (decay rate) обычно 0.9 или 0.99.
    
    Преимущества:
    - Не имеет проблемы "замораживания" Adagrad
    - Хорошо работает в нестационарных условиях когда оптимальный lr меняется
    - Отлично подходит для RNN и других рекуррентных архитектур
    """
    
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


# =============================================================================
# 5. Adam
# =============================================================================

class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam)
    
    ИДЕЯ АЛГОРИТМА:
    ---------------
    Adam объединяет лучшее из Momentum и RMSProp:
    - От Momentum: экспоненциальное среднее градиентов (первый момент m)
    - От RMSProp: экспоненциальное среднее квадратов градиентов (второй момент v)
    
    Интуиция: m_t приближает E[g] (среднее направление), v_t приближает E[g^2]
    (дисперсию градиентов). Деление m на sqrt(v) нормализует шаг: большие
    градиенты "приглушаются", маленькие - усиливаются.
    
    Важная деталь - bias correction:
    В начале обучения m и v инициализированы нулями, поэтому они смещены
    к нулю. Коррекция компенсирует это: m̂ = m / (1 - β1^t).
    
    Формулы:
        m_t = β1 * m_{t-1} + (1-β1) * g_t       <- первый момент
        v_t = β2 * v_{t-1} + (1-β2) * g_t²      <- второй момент
        m̂_t = m_t / (1 - β1^t)                  <- коррекция смещения
        v̂_t = v_t / (1 - β2^t)                  <- коррекция смещения
        θ_t = θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε)

    Типичные значения: β1=0.9, β2=0.999, ε=1e-8
    
    Преимущества:
    - Комбинирует преимущества Momentum и адаптивных методов
    - Работает "из коробки" для большинства задач
    - Инвариантен к масштабу градиентов
    - Стандарт в глубоком обучении
    """
    
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


# =============================================================================
# 6. AdamW
# =============================================================================

class AdamW(Optimizer):
    """
    Adam with Decoupled Weight Decay (AdamW)
    
    ИДЕЯ АЛГОРИТМА:
    ---------------
    AdamW исправляет важную проблему в Adam при использовании L2-регуляризации.
    
    Проблема: в Adam (и SGD) L2-регуляризацию обычно добавляют к градиенту:
        g_t = ∇L(θ) + λ*θ
    
    Но Adam масштабирует градиенты. Это означает, что штраф за большие 
    веса тоже масштабируется, что приводит к непредсказуемому
    поведению регуляризации.
    
    Решение AdamW: разделить weight decay и градиентное обновление.
    Weight decay применяется напрямую к весам, минуя адаптивное масштабирование:
    
    Формулы:
        m_t, v_t - как в Adam
        θ_t = θ_{t-1} - lr * (m̂_t / (sqrt(v̂_t) + ε) + λ * θ_{t-1})
        
    или эквивалентно:
        θ_t = (1 - lr * λ) * θ_{t-1} - lr * m̂_t / (sqrt(v̂_t) + ε)
    
    Преимущества:
    - Правильная регуляризация в адаптивных оптимизаторах
    - Лучшая генерализация на практике
    - Стандарт для обучения трансформеров (BERT, GPT и т.д.)
    - Более предсказуемое поведение weight decay
    
    Важно: λ в AdamW - это НЕ то же самое, что λ в L2! При одинаковых
    значениях эффект разный. Типичные значения для AdamW: λ = 0.01-0.1
    """
    
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


# =============================================================================
# Нейрон из задания 1
# =============================================================================

# =============================================================================
# Нейрон из задания 1 (с сигмоидой и NLL loss)
# =============================================================================

class Neuron:
    """Нейрон с сигмоидной активацией и NLL loss (как в задании 1)."""
    
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

# =============================================================================
# ТЕСТЫ
# =============================================================================

def test_momentum():
    """Тест Momentum оптимизатора."""
    print("=" * 60)
    print("Тест Momentum")
    print("=" * 60)
    
    opt = Momentum(learning_rate=0.1, momentum=0.9)
    
    # Тест 1: Базовое обновление
    param = np.array([1.0, 2.0])
    grad = np.array([0.1, 0.2])
    
    new_param = opt.update('test', param, grad)
    # v = 0.9 * 0 + 0.1 = [0.1, 0.2]
    # param = [1, 2] - 0.1 * [0.1, 0.2] = [0.99, 1.98]
    expected = param - 0.1 * grad
    assert np.allclose(new_param, expected), f"Expected {expected}, got {new_param}"
    
    # Тест 2: Накопление импульса
    new_param2 = opt.update('test', new_param, grad)
    # v = 0.9 * [0.1, 0.2] + [0.1, 0.2] = [0.19, 0.38]
    expected_v = 0.9 * grad + grad
    expected2 = new_param - 0.1 * expected_v
    assert np.allclose(new_param2, expected2), f"Expected {expected2}, got {new_param2}"
    
    # Тест 3: Ускорение при постоянном градиенте
    # При постоянном градиенте скорость должна расти
    opt.reset()
    param = np.array([10.0])
    const_grad = np.array([1.0])
    velocities = []
    
    for _ in range(10):
        param = opt.update('accel', param, const_grad)
        velocities.append(opt.state['accel']['velocity'][0])
    
    # Скорость должна монотонно расти (к пределу 1/(1-0.9) = 10)
    assert all(velocities[i] < velocities[i+1] for i in range(len(velocities)-1)), \
        "Velocity should increase with constant gradient"
    
    print("Все тесты Momentum пройдены!")
    print()


def test_nesterov():
    """Тест Nesterov оптимизатора."""
    print("=" * 60)
    print("Тест Nesterov")
    print("=" * 60)
    
    opt = Nesterov(learning_rate=0.1, momentum=0.9)
    
    # Тест 1: Nesterov должен делать большие шаги, чем Momentum
    # при постоянном градиенте
    param = np.array([10.0])
    grad = np.array([1.0])
    
    opt_mom = Momentum(learning_rate=0.1, momentum=0.9)
    
    param_n = param.copy()
    param_m = param.copy()
    
    for _ in range(5):
        param_n = opt.update('test', param_n, grad)
        param_m = opt_mom.update('test', param_m, grad)
    
    # Nesterov должен продвинуться дальше
    assert param_n[0] < param_m[0], "Nesterov should make faster progress"
    
    # Тест 2: Проверка "заглядывания вперед"
    opt.reset()
    param = np.array([5.0])
    
    # Первый шаг
    new_param = opt.update('look', param, grad)
    v = opt.state['look']['velocity']
    
    # Nesterov использует: param - (β * v_new + lr * grad)
    expected_v = 0.1 * grad  # lr * grad
    expected = param - (0.9 * expected_v + 0.1 * grad)
    assert np.allclose(new_param, expected), f"Expected {expected}, got {new_param}"
    
    print("Все тесты Nesterov пройдены!")
    print()


def test_adagrad():
    """Тест Adagrad оптимизатора."""
    print("=" * 60)
    print("Тест Adagrad")
    print("=" * 60)
    
    opt = Adagrad(learning_rate=1.0, epsilon=1e-8)
    
    # Тест 1: Убывающий learning rate
    param = np.array([1.0])
    grad = np.array([1.0])
    
    steps = []
    prev_param = param.copy()
    
    for i in range(5):
        new_param = opt.update('decay', prev_param, grad)
        step = abs(prev_param[0] - new_param[0])
        steps.append(step)
        prev_param = new_param
    
    # Шаги должны уменьшаться
    assert all(steps[i] > steps[i+1] for i in range(len(steps)-1)), \
        "Step size should decrease over time"
    
    # Тест 2: Адаптивность к разным масштабам градиентов
    opt.reset()
    param = np.array([1.0, 1.0])
    
    # Разные масштабы градиентов
    grad1 = np.array([10.0, 0.1])
    
    new_param = opt.update('scale', param, grad1)
    
    # Параметр с большим градиентом должен измениться меньше (относительно)
    change = abs(param - new_param)
    # После нормализации изменения должны быть ближе друг к другу
    # (но не равны из-за разных абсолютных величин градиентов)
    
    # Тест 3: Проверка формулы G
    opt.reset()
    param = np.array([1.0])
    grad = np.array([2.0])
    
    opt.update('g_test', param, grad)
    assert np.allclose(opt.state['g_test']['G'], np.array([4.0])), \
        "G should be grad^2"
    
    opt.update('g_test', param, grad)
    assert np.allclose(opt.state['g_test']['G'], np.array([8.0])), \
        "G should accumulate"
    
    print("Все тесты Adagrad пройдены!")
    print()


def test_rmsprop():
    """Тест RMSProp оптимизатора."""
    print("=" * 60)
    print("Тест RMSProp")
    print("=" * 60)
    
    opt = RMSProp(learning_rate=0.1, decay=0.9, epsilon=1e-8)
    
    # Тест 1: Скользящее среднее
    param = np.array([1.0])
    grad = np.array([1.0])
    
    opt.update('ema', param, grad)
    # E[g^2] = 0.9 * 0 + 0.1 * 1 = 0.1
    assert np.allclose(opt.state['ema']['E_g2'], np.array([0.1])), \
        "EMA should be 0.1 after first step"
    
    opt.update('ema', param, grad)
    # E[g^2] = 0.9 * 0.1 + 0.1 * 1 = 0.19
    assert np.allclose(opt.state['ema']['E_g2'], np.array([0.19])), \
        "EMA should be 0.19 after second step"
    
    # Тест 2: Не замораживается как Adagrad
    opt.reset()
    opt_ada = Adagrad(learning_rate=0.1)
    
    param = np.array([10.0])
    grad = np.array([1.0])
    
    for _ in range(100):
        param_rms = opt.update('nf', param, grad)
        param_ada = opt_ada.update('nf', param, grad)
        param = param_rms  # Используем RMSProp param для следующей итерации
    
    # RMSProp должен продолжать делать значимые шаги
    final_step_rms = abs(opt.learning_rate * grad[0] / (np.sqrt(opt.state['nf']['E_g2'][0]) + opt.epsilon))
    final_step_ada = abs(opt_ada.learning_rate * grad[0] / (np.sqrt(opt_ada.state['nf']['G'][0]) + opt_ada.epsilon))
    
    assert final_step_rms > final_step_ada * 5, \
        "RMSProp should maintain larger steps than Adagrad"
    
    print("Все тесты RMSProp пройдены!")
    print()


def test_adam():
    """Тест Adam оптимизатора."""
    print("=" * 60)
    print("Тест Adam")
    print("=" * 60)
    
    opt = Adam(learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8)
    
    # Тест 1: Bias correction
    param = np.array([1.0])
    grad = np.array([1.0])
    
    opt.update('bias', param, grad)
    
    m = opt.state['bias']['m']
    v = opt.state['bias']['v']
    
    # Без коррекции: m = 0.1, v = 0.001
    assert np.allclose(m, np.array([0.1])), "m should be 0.1"
    assert np.allclose(v, np.array([0.001])), "v should be 0.001"
    
    # С коррекцией: m_hat = 0.1 / (1 - 0.9) = 1.0
    #               v_hat = 0.001 / (1 - 0.999) = 1.0
    
    # Тест 2: Комбинация momentum и адаптивности
    opt.reset()
    param = np.array([5.0, 5.0])
    
    # Градиенты разного масштаба, но с momentum
    for _ in range(10):
        grad = np.array([10.0, 0.1])
        param = opt.update('combo', param, grad)
    
    # Оба параметра должны уменьшиться
    assert param[0] < 5.0, "First param should decrease"
    assert param[1] < 5.0, "Second param should decrease"
    
    # Тест 3: Счетчик шагов
    opt.reset()
    assert opt.t == 0, "Counter should start at 0"
    opt.update('t1', np.array([1.0]), np.array([1.0]))
    assert opt.t == 1, "Counter should be 1"
    
    print("Все тесты Adam пройдены")
    print()


def test_adamw():
    """Тест AdamW оптимизатора."""
    print("=" * 60)
    print("Тест AdamW")
    print("=" * 60)
    
    # Тест 1: Decoupled weight decay
    opt_adamw = AdamW(learning_rate=0.1, weight_decay=0.1, beta1=0.9, beta2=0.999)
    opt_adam = Adam(learning_rate=0.1, beta1=0.9, beta2=0.999)
    
    param_w = np.array([10.0])
    param_a = np.array([10.0])
    grad = np.array([0.0])  # Нулевой градиент - только weight decay
    
    new_param_w = opt_adamw.update('wd', param_w, grad)
    new_param_a = opt_adam.update('wd', param_a, grad)
    
    # AdamW должен уменьшить параметр даже при нулевом градиенте
    # new = 10 * (1 - 0.1 * 0.1) = 10 * 0.99 = 9.9
    assert new_param_w[0] < param_w[0], "AdamW should decay weights"
    assert np.allclose(new_param_w, np.array([9.9])), f"Expected 9.9, got {new_param_w}"
    
    # Adam без L2 в градиенте не должен менять параметр (или минимально из-за bias correction)
    # При нулевом градиенте Adam почти не меняет параметр
    
    # Тест 2: Weight decay независим от адаптивного scaling
    opt_adamw.reset()
    param = np.array([5.0])
    
    # Несколько шагов с одинаковым градиентом
    for _ in range(5):
        param = opt_adamw.update('scale_wd', param, np.array([1.0]))
    
    # Параметр должен уменьшаться
    assert param[0] < 5.0, "Parameter should decrease with positive gradient"
    
    # Тест 3: Сравнение с Adam + L2 (они должны отличаться)
    opt_adamw.reset()
    opt_adam.reset()
    
    param_w = np.array([10.0])
    param_a = np.array([10.0])
    
    for _ in range(10):
        grad = np.array([1.0])
        grad_with_l2 = grad + 0.01 * param_a  # L2 добавлен к градиенту
        
        param_w = opt_adamw.update('compare', param_w, grad)
        param_a = opt_adam.update('compare', param_a, grad_with_l2)
    
    # Результаты должны отличаться
    assert not np.allclose(param_w, param_a), \
        "AdamW and Adam+L2 should produce different results"
    
    print("Все тесты AdamW пройдены")
    print()



def test_neuron_training():
    """Тест обучения нейрона с разными оптимизаторами."""
    print("=" * 60)
    print("Тест обучения нейрона (sigmoid + NLL)")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Генерируем данные для бинарной классификации
    n_samples = 100
    n_features = 2
    
    X = np.random.randn(n_samples, n_features)
    true_weights = np.array([1.0, -0.5])
    y = (np.dot(X, true_weights) > 0).astype(float)
    
    optimizers = {
        'Momentum': Momentum(learning_rate=0.5, momentum=0.9),
        'Nesterov': Nesterov(learning_rate=0.5, momentum=0.9),
        'Adagrad': Adagrad(learning_rate=1.0),
        'RMSProp': RMSProp(learning_rate=0.1),
        'Adam': Adam(learning_rate=0.1),
        'AdamW': AdamW(learning_rate=0.1, weight_decay=0.01)
    }
    
    results = {}
    
    for name, opt in optimizers.items():
        np.random.seed(42)
        neuron = Neuron(n_features)
        nll_history = neuron.train(X, y, opt, epochs=100, verbose=False)
        results[name] = nll_history
        print(f"{name}: Final NLL = {nll_history[-1]:.4f}")
    
    for name, history in results.items():
        assert history[-1] < history[0], f"{name} should decrease NLL"
    
    print("Тест обучения нейрона пройден")
    print()
    
    return results


def compare_optimizers_visualization(results: dict):
    """Визуализация сравнения оптимизаторов."""
    plt.figure(figsize=(12, 6))
    
    for name, losses in results.items():
        plt.plot(losses, label=name, linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('NLL Loss', fontsize=12) 
    plt.title('Сравнение оптимизаторов на задаче классификации (sigmoid + NLL)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/optimizers_comparison.png', dpi=150)
    plt.close()
    print("График сохранен: optimizers_comparison.png")


def rosenbrock_comparison():
    """
    Сравнение оптимизаторов на функции Розенброка.
    
    f(x, y) = (a - x)^2 + b(y - x^2)^2
    
    Это классический тест для оптимизаторов:
    - Узкий изогнутый овраг
    - Глобальный минимум в (a, a^2) = (1, 1) при стандартных параметрах
    """
    print("=" * 60)
    print("Сравнение на функции Розенброка")
    print("=" * 60)
    
    a, b = 1, 100
    
    def rosenbrock(x, y):
        return (a - x) ** 2 + b * (y - x ** 2) ** 2
    
    def rosenbrock_grad(x, y):
        dx = -2 * (a - x) - 4 * b * x * (y - x ** 2)
        dy = 2 * b * (y - x ** 2)
        return np.array([dx, dy])
    
    optimizers = {
        'Momentum': Momentum(learning_rate=0.0001, momentum=0.9),
        'Nesterov': Nesterov(learning_rate=0.0001, momentum=0.9),
        'Adagrad': Adagrad(learning_rate=0.1),
        'RMSProp': RMSProp(learning_rate=0.001, decay=0.9),
        'Adam': Adam(learning_rate=0.01),
        'AdamW': AdamW(learning_rate=0.01, weight_decay=0.0001)
    }
    
    trajectories = {}
    n_steps = 1000
    
    for name, opt in optimizers.items():
        param = np.array([-1.0, 1.0])  # Начальная точка
        trajectory = [param.copy()]
        
        for _ in range(n_steps):
            grad = rosenbrock_grad(param[0], param[1])
            param = opt.update('rosenbrock', param, grad)
            trajectory.append(param.copy())
        
        trajectories[name] = np.array(trajectory)
        final_val = rosenbrock(param[0], param[1])
        distance = np.sqrt((param[0] - 1) ** 2 + (param[1] - 1) ** 2)
        print(f"{name}: Final point = ({param[0]:.4f}, {param[1]:.4f}), "
              f"f = {final_val:.6f}, distance to optimum = {distance:.4f}")
    
    # Визуализация
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Создаем контурный график
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-1, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = rosenbrock(X, Y)
    
    for ax, (name, traj) in zip(axes, trajectories.items()):
        ax.contour(X, Y, Z, levels=np.logspace(0, 3, 20), cmap='viridis', alpha=0.5)
        ax.plot(traj[:, 0], traj[:, 1], 'r-', linewidth=1, alpha=0.7)
        ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=10, label='Start')
        ax.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=15, label='End')
        ax.plot(1, 1, 'k*', markersize=15, label='Optimum')
        ax.set_title(name, fontsize=12)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 3)
    
    plt.suptitle('Траектории оптимизаторов на функции Розенброка', fontsize=14)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/rosenbrock_comparison.png', dpi=150)
    plt.close()
    print("\nГрафик траекторий сохранен: rosenbrock_comparison.png")


# =============================================================================
# Главная функция
# =============================================================================

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
    
    print("\n" + "=" * 60)
    print("ВСЕ ТЕСТЫ УСПЕШНО ПРОЙДЕНЫ")
    print("=" * 60)


if __name__ == "__main__":
    main()