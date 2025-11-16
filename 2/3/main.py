"""
Реализация трансформаций изображений

Классы трансформаций для аугментации данных.
"""

import numpy as np
from PIL import Image
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union
import random
import math

class Tensor:
    """Простая эмуляция torch.Tensor для демонстрации."""
    
    def __init__(self, data, dtype=None):
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float32 if dtype is None else dtype)
        else:
            self.data = np.array(data, dtype=np.float32 if dtype is None else dtype)
    
    @property
    def shape(self):
        return self.data.shape
    
    @property 
    def dtype(self):
        return self.data.dtype
    
    def numpy(self):
        return self.data
    
    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"
    
    def permute(self, *dims):
        return Tensor(np.transpose(self.data, dims))
    
    def __getitem__(self, idx):
        return Tensor(self.data[idx])


def tensor(data, dtype=None):
    """Создание тензора, аналог torch.tensor"""
    return Tensor(data, dtype)


class BaseTransform(ABC):
    """
    Базовый класс для всех трансформаций.
    
    Args:
        p: вероятность применения трансформации (0.0 - 1.0)
    """
    
    def __init__(self, p: float = 1.0):
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"Вероятность должна быть в диапазоне [0, 1], получено: {p}")
        self.p = p
    
    @abstractmethod
    def apply(self, image: Image.Image) -> Image.Image:
        """
        Применяет трансформацию к изображению.
        Должен быть реализован в подклассах.
        """
        pass
    
    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Вызывает трансформацию с вероятностью p.
        
        Args:
            image: входное PIL.Image
            
        Returns:
            трансформированное или исходное изображение
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"Ожидается PIL.Image, получено: {type(image)}")
        
        if random.random() < self.p:
            return self.apply(image)
        return image


class RandomCrop(BaseTransform):
    """
    Случайное кадрирование изображения.
    
    Args:
        p: вероятность применения
        size: размер выходного изображения (height, width) или int для квадрата
        padding: паддинг перед кадрированием, опционально
    """
    
    def __init__(self, p: float = 1.0, size: Union[int, Tuple[int, int]] = None, 
                 padding: int = 0, **kwargs):
        super().__init__(p)
        
        if size is None:
            raise ValueError("Параметр 'size' обязателен для RandomCrop")
        
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = tuple(size)
        
        self.padding = padding
    
    def apply(self, image: Image.Image) -> Image.Image:
        """Применяет случайное кадрирование."""
        if self.padding > 0:
            new_width = image.width + 2 * self.padding
            new_height = image.height + 2 * self.padding
            
            # Определяем цвет паддинга (чёрный для RGB, серый для grayscale)
            if image.mode == 'RGB':
                padded = Image.new('RGB', (new_width, new_height), (0, 0, 0))
            elif image.mode == 'RGBA':
                padded = Image.new('RGBA', (new_width, new_height), (0, 0, 0, 255))
            else:
                padded = Image.new(image.mode, (new_width, new_height), 0)
            
            padded.paste(image, (self.padding, self.padding))
            image = padded
        
        width, height = image.size
        crop_height, crop_width = self.size
        
        # Проверяем, что изображение достаточно большое
        if width < crop_width or height < crop_height:
            raise ValueError(
                f"Изображение ({width}x{height}) меньше размера кадрирования "
                f"({crop_width}x{crop_height})"
            )
        
        # Случайные координаты верхнего левого угла
        left = random.randint(0, width - crop_width)
        top = random.randint(0, height - crop_height)
        
        # Кадрируем
        return image.crop((left, top, left + crop_width, top + crop_height))


class RandomRotate(BaseTransform):
    """
    Случайный поворот изображения.
    
    Args:
        p: вероятность применения
        degrees: диапазон углов поворота. 
                 Если int/float - от -degrees до +degrees.
                 Если tuple - (min_angle, max_angle)
        expand: расширять ли изображение для вмещения повёрнутого
        fill: цвет заполнения фона
    """
    
    def __init__(self, p: float = 1.0, degrees: Union[float, Tuple[float, float]] = 45,
                 expand: bool = False, fill: Union[int, Tuple[int, ...]] = 0, **kwargs):
        super().__init__(p)
        
        if isinstance(degrees, (int, float)):
            self.degrees = (-abs(degrees), abs(degrees))
        else:
            self.degrees = tuple(degrees)
        
        self.expand = expand
        self.fill = fill
    
    def apply(self, image: Image.Image) -> Image.Image:
        """Применяет случайный поворот."""
        angle = random.uniform(self.degrees[0], self.degrees[1])

        if image.mode == 'RGB' and isinstance(self.fill, int):
            fill_color = (self.fill, self.fill, self.fill)
        elif image.mode == 'RGBA' and isinstance(self.fill, int):
            fill_color = (self.fill, self.fill, self.fill, 255)
        else:
            fill_color = self.fill
        
        # Поворачиваем с использованием билинейной интерполяции
        return image.rotate(
            angle, 
            resample=Image.BILINEAR,
            expand=self.expand,
            fillcolor=fill_color
        )


# =============================================================================
# RandomZoom
# =============================================================================

class RandomZoom(BaseTransform):
    """
    Случайное масштабирование изображения.
    
    Args:
        p: вероятность применения
        scale: диапазон масштабирования (min_scale, max_scale)
               scale > 1 - увеличение (zoom in)
               scale < 1 - уменьшение (zoom out)
    """
    
    def __init__(self, p: float = 1.0, scale: Tuple[float, float] = (0.8, 1.2), **kwargs):
        super().__init__(p)
        
        if not isinstance(scale, (tuple, list)) or len(scale) != 2:
            raise ValueError("scale должен быть кортежем (min_scale, max_scale)")
        
        if scale[0] <= 0 or scale[1] <= 0:
            raise ValueError("Значения scale должны быть положительными")
        
        self.scale = tuple(scale)
    
    def apply(self, image: Image.Image) -> Image.Image:
        """Применяет случайное масштабирование."""
        width, height = image.size
        
        scale_factor = random.uniform(self.scale[0], self.scale[1])
        
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Масштабируем изображение
        scaled = image.resize((new_width, new_height), Image.BILINEAR)
        
        if scale_factor > 1:
            # Zoom in - кадрируем центральную часть
            left = (new_width - width) // 2
            top = (new_height - height) // 2
            return scaled.crop((left, top, left + width, top + height))
        else:
            # Zoom out - добавляем паддинг
            if image.mode == 'RGB':
                result = Image.new('RGB', (width, height), (0, 0, 0))
            elif image.mode == 'RGBA':
                result = Image.new('RGBA', (width, height), (0, 0, 0, 255))
            else:
                result = Image.new(image.mode, (width, height), 0)
            
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            result.paste(scaled, (left, top))
            return result


class ToTensor:
    """
    Конвертирует PIL.Image в Tensor.
    
    Преобразования:
    - PIL.Image -> numpy array -> Tensor
    - Значения нормализуются к [0, 1]
    - Формат меняется с (H, W, C) на (C, H, W)
    """
    
    def __call__(self, image: Image.Image) -> Tensor:
        """
        Конвертирует изображение в тензор.
        
        Args:
            image: входное PIL.Image
            
        Returns:
            Tensor формата (C, H, W) с значениями в [0, 1]
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"Ожидается PIL.Image, получено: {type(image)}")
        
        np_image = np.array(image, dtype=np.float32)
        
        # Нормализуем к [0, 1]
        np_image = np_image / 255.0
        
        # Меняем формат с (H, W, C) на (C, H, W)
        if np_image.ndim == 3:
            np_image = np.transpose(np_image, (2, 0, 1))
        elif np_image.ndim == 2:
            # Grayscale - добавляем размерность канала
            np_image = np_image[np.newaxis, :, :]
        
        return Tensor(np_image)


class Compose:
    """
    Композиция трансформаций.
    
    Последовательно применяет список трансформаций к изображению.
    
    Args:
        transforms: список трансформаций (экземпляры BaseTransform или ToTensor)
    """
    
    def __init__(self, transforms: List[Union[BaseTransform, ToTensor]]):
        if not isinstance(transforms, list):
            raise TypeError("transforms должен быть списком")
        
        self.transforms = transforms
    
    def __call__(self, image: Image.Image) -> Union[Image.Image, Tensor]:
        """
        Применяет все трансформации последовательно.
        
        Args:
            image: входное PIL.Image
            
        Returns:
            трансформированное изображение (PIL.Image или Tensor)
        """
        for transform in self.transforms:
            image = transform(image)
        return image
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t.__class__.__name__},'
        format_string += '\n)'
        return format_string


def create_test_image(size: Tuple[int, int] = (100, 100), mode: str = 'RGB') -> Image.Image:
    """Создаёт тестовое изображение с паттерном."""
    if mode == 'RGB':
        img = Image.new('RGB', size, (128, 128, 128))
        pixels = img.load()
        
        # Добавляем паттерн для визуальной проверки трансформаций
        for i in range(0, size[0], 10):
            for j in range(size[1]):
                pixels[i, j] = (255, 0, 0)  # Красные вертикальные линии
        
        for j in range(0, size[1], 10):
            for i in range(size[0]):
                if pixels[i, j] != (255, 0, 0):
                    pixels[i, j] = (0, 255, 0)  # Зелёные горизонтальные линии
        
        # Белый квадрат в центре для отслеживания позиции
        center_x, center_y = size[0] // 2, size[1] // 2
        for i in range(center_x - 10, center_x + 10):
            for j in range(center_y - 10, center_y + 10):
                if 0 <= i < size[0] and 0 <= j < size[1]:
                    pixels[i, j] = (255, 255, 255)
        
        return img
    else:
        img = Image.new('L', size, 128)
        return img


def test_base_transform():
    """Тест базового класса трансформации."""
    print("=" * 60)
    print("Тест BaseTransform")
    print("=" * 60)
    
    # Тест 1: Проверка валидации вероятности
    try:
        class DummyTransform(BaseTransform):
            def apply(self, image):
                return image
        
        DummyTransform(p=1.5)
        assert False, "Должно быть исключение для p > 1"
    except ValueError:
        print("   Тест 1: Валидация p > 1")
    
    try:
        DummyTransform(p=-0.1)
        assert False, "Должно быть исключение для p < 0"
    except ValueError:
        print("   Тест 2: Валидация p < 0")
    
    # Тест 3: Проверка типа входных данных
    class DummyTransform(BaseTransform):
        def apply(self, image):
            return image
    
    t = DummyTransform(p=1.0)
    try:
        t("not an image")
        assert False, "Должно быть исключение для неверного типа"
    except TypeError:
        print("   Тест 3: Валидация типа входных данных")
    
    print()


def test_random_crop():
    """Тест RandomCrop."""
    print("=" * 60)
    print("Тест RandomCrop")
    print("=" * 60)
    
    img = create_test_image((100, 100))
    
    # Тест 1: Корректный размер выхода
    crop = RandomCrop(p=1.0, size=50)
    result = crop(img)
    assert result.size == (50, 50), f"Ожидался размер (50, 50), получен {result.size}"
    print("   Тест 1: Корректный размер выхода")
    
    # Тест 2: Размер как tuple
    crop = RandomCrop(p=1.0, size=(30, 40))
    result = crop(img)
    assert result.size == (40, 30), f"Ожидался размер (40, 30), получен {result.size}"
    print("   Тест 2: Размер как tuple")
    
    # Тест 3: С паддингом
    crop = RandomCrop(p=1.0, size=50, padding=10)
    result = crop(img)
    assert result.size == (50, 50)
    print("   Тест 3: Кадрирование с паддингом")
    
    # Тест 4: Граничный случай - размер равен изображению
    crop = RandomCrop(p=1.0, size=100)
    result = crop(img)
    assert result.size == (100, 100)
    print("   Тест 4: Граничный случай (размер = изображение)")
    
    # Тест 5: Исключение при слишком большом размере
    crop = RandomCrop(p=1.0, size=150)
    try:
        result = crop(img)
        assert False, "Должно быть исключение"
    except ValueError:
        print("   Тест 5: Исключение при размере > изображения")
    
    # Тест 6: p=0 - изображение не меняется
    crop = RandomCrop(p=0.0, size=50)
    result = crop(img)
    assert result.size == img.size, "При p=0 размер должен сохраниться"
    print("   Тест 6: p=0 не применяет трансформацию")
    
    print()


def test_random_rotate():
    """Тест RandomRotate."""
    print("=" * 60)
    print("Тест RandomRotate")
    print("=" * 60)
    
    img = create_test_image((100, 100))
    
    # Тест 1: Базовый поворот
    rotate = RandomRotate(p=1.0, degrees=45)
    result = rotate(img)
    assert isinstance(result, Image.Image)
    print("   Тест 1: Базовый поворот")
    
    # Тест 2: Поворот с expand=False сохраняет размер
    rotate = RandomRotate(p=1.0, degrees=30, expand=False)
    result = rotate(img)
    assert result.size == img.size, f"Размер должен сохраниться: {result.size} != {img.size}"
    print("   Тест 2: expand=False сохраняет размер")
    
    # Тест 3: Поворот с expand=True может изменить размер
    rotate = RandomRotate(p=1.0, degrees=45, expand=True)
    result = rotate(img)
    # При повороте на 45° размер увеличивается
    assert isinstance(result, Image.Image)
    print("   Тест 3: expand=True работает")
    
    # Тест 4: Диапазон углов как tuple
    rotate = RandomRotate(p=1.0, degrees=(-10, 30))
    result = rotate(img)
    assert isinstance(result, Image.Image)
    print("   Тест 4: Диапазон углов как tuple")
    
    # Тест 5: p=0
    rotate = RandomRotate(p=0.0, degrees=90)
    result = rotate(img)
    # При p=0 изображение должно быть идентичным
    assert list(result.getdata()) == list(img.getdata()), "При p=0 изображение не должно меняться"
    print("   Тест 5: p=0 не применяет трансформацию")
    
    print()


def test_random_zoom():
    """Тест RandomZoom."""
    print("=" * 60)
    print("Тест RandomZoom")
    print("=" * 60)
    
    img = create_test_image((100, 100))
    
    # Тест 1: Zoom сохраняет размер
    zoom = RandomZoom(p=1.0, scale=(0.8, 1.2))
    result = zoom(img)
    assert result.size == img.size, f"Размер должен сохраниться: {result.size}"
    print("   Тест 1: Zoom сохраняет размер")
    
    # Тест 2: Zoom in (scale > 1)
    zoom = RandomZoom(p=1.0, scale=(1.5, 1.5))  # Фиксированный zoom
    result = zoom(img)
    assert result.size == img.size
    print("   Тест 2: Zoom in работает")
    
    # Тест 3: Zoom out (scale < 1)
    zoom = RandomZoom(p=1.0, scale=(0.5, 0.5))
    result = zoom(img)
    assert result.size == img.size
    print("   Тест 3: Zoom out работает")
    
    # Тест 4: Валидация параметров
    try:
        zoom = RandomZoom(p=1.0, scale=(-0.5, 1.0))
        assert False, "Должно быть исключение"
    except ValueError:
        print("   Тест 4: Валидация отрицательного scale")
    
    # Тест 5: p=0
    zoom = RandomZoom(p=0.0, scale=(0.5, 1.5))
    result = zoom(img)
    assert list(result.getdata()) == list(img.getdata())
    print("   Тест 5: p=0 не применяет трансформацию")
    
    print()


def test_to_tensor():
    """Тест ToTensor."""
    print("=" * 60)
    print("Тест ToTensor")
    print("=" * 60)
    
    # Тест 1: RGB изображение
    img = create_test_image((100, 100), mode='RGB')
    to_tensor = ToTensor()
    result = to_tensor(img)
    
    assert isinstance(result, Tensor), f"Ожидался Tensor, получен {type(result)}"
    assert result.shape == (3, 100, 100), f"Ожидалась форма (3, 100, 100), получена {result.shape}"
    print("   Тест 1: RGB -> Tensor (C, H, W)")
    
    # Тест 2: Значения в диапазоне [0, 1]
    assert result.data.min() >= 0.0 and result.data.max() <= 1.0, \
        "Значения должны быть в [0, 1]"
    print("   Тест 2: Нормализация к [0, 1]")
    
    # Тест 3: Grayscale изображение
    img_gray = create_test_image((50, 50), mode='L')
    result_gray = to_tensor(img_gray)
    assert result_gray.shape == (1, 50, 50), f"Ожидалась форма (1, 50, 50), получена {result_gray.shape}"
    print("   Тест 3: Grayscale -> Tensor (1, H, W)")
    
    # Тест 4: Валидация типа
    try:
        to_tensor("not an image")
        assert False, "Должно быть исключение"
    except TypeError:
        print("   Тест 4: Валидация типа входных данных")
    
    print()


def test_compose():
    """Тест Compose."""
    print("=" * 60)
    print("Тест Compose")
    print("=" * 60)
    
    img = create_test_image((100, 100))
    
    # Тест 1: Композиция трансформаций
    transform = Compose([
        RandomCrop(p=1.0, size=80),
        RandomRotate(p=1.0, degrees=15),
    ])
    result = transform(img)
    assert isinstance(result, Image.Image)
    assert result.size == (80, 80)
    print("   Тест 1: Композиция Crop + Rotate")
    
    # Тест 2: С ToTensor в конце
    transform = Compose([
        RandomCrop(p=1.0, size=50),
        ToTensor()
    ])
    result = transform(img)
    assert isinstance(result, Tensor)
    assert result.shape == (3, 50, 50)
    print("   Тест 2: Композиция с ToTensor")
    
    # Тест 3: Все трансформации
    transform = Compose([
        RandomCrop(p=1.0, size=80),
        RandomRotate(p=1.0, degrees=10),
        RandomZoom(p=1.0, scale=(0.9, 1.1)),
        ToTensor()
    ])
    result = transform(img)
    assert isinstance(result, Tensor)
    print("   Тест 3: Полная композиция")
    
    # Тест 4: Пустой список
    transform = Compose([])
    result = transform(img)
    assert result == img
    print("   Тест 4: Пустая композиция")
    
    # Тест 5: repr
    transform = Compose([
        RandomCrop(p=0.5, size=50),
        RandomRotate(p=0.5, degrees=30),
        ToTensor()
    ])
    repr_str = repr(transform)
    assert 'Compose' in repr_str
    assert 'RandomCrop' in repr_str
    print("   Тест 5: __repr__ работает")
    
    print()


def test_reproducibility():
    """Тест воспроизводимости (с фиксированным seed)."""
    print("=" * 60)
    print("Тест воспроизводимости")
    print("=" * 60)
    
    img = create_test_image((100, 100))
    
    # Тест 1: RandomCrop воспроизводим
    random.seed(42)
    crop = RandomCrop(p=1.0, size=50)
    result1 = crop(img)
    
    random.seed(42)
    result2 = crop(img)
    
    assert list(result1.getdata()) == list(result2.getdata()), \
        "RandomCrop должен быть воспроизводим с одинаковым seed"
    print("   Тест 1: RandomCrop воспроизводим")
    
    # Тест 2: RandomRotate воспроизводим
    random.seed(42)
    rotate = RandomRotate(p=1.0, degrees=45)
    result1 = rotate(img)
    
    random.seed(42)
    result2 = rotate(img)
    
    assert list(result1.getdata()) == list(result2.getdata()), \
        "RandomRotate должен быть воспроизводим"
    print("   Тест 2: RandomRotate воспроизводим")
    
    # Тест 3: RandomZoom воспроизводим
    random.seed(42)
    zoom = RandomZoom(p=1.0, scale=(0.8, 1.2))
    result1 = zoom(img)
    
    random.seed(42)
    result2 = zoom(img)
    
    assert list(result1.getdata()) == list(result2.getdata()), \
        "RandomZoom должен быть воспроизводим"
    print("   Тест 3: RandomZoom воспроизводим")
    
    # Тест 4: Compose воспроизводим
    random.seed(42)
    transform = Compose([
        RandomCrop(p=1.0, size=80),
        RandomRotate(p=1.0, degrees=30),
        RandomZoom(p=1.0, scale=(0.9, 1.1)),
    ])
    result1 = transform(img)
    
    random.seed(42)
    result2 = transform(img)
    
    assert list(result1.getdata()) == list(result2.getdata()), \
        "Compose должен быть воспроизводим"
    print("   Тест 4: Compose воспроизводим")
    
    print()


def test_probability():
    """Тест вероятностного применения трансформаций."""
    print("=" * 60)
    print("Тест вероятностного применения")
    print("=" * 60)
    
    img = create_test_image((100, 100))
    n_trials = 1000
    
    # Тест 1: p=0 никогда не применяет
    crop = RandomCrop(p=0.0, size=50)
    for _ in range(100):
        result = crop(img)
        assert result.size == img.size, "При p=0 трансформация не должна применяться"
    print("   Тест 1: p=0 никогда не применяет")
    
    # Тест 2: p=1 всегда применяет
    crop = RandomCrop(p=1.0, size=50)
    for _ in range(100):
        result = crop(img)
        assert result.size == (50, 50), "При p=1 трансформация должна применяться всегда"
    print("   Тест 2: p=1 всегда применяет")
    
    # Тест 3: Проверка распределения для p=0.5
    random.seed(42)
    crop = RandomCrop(p=0.5, size=50)
    applied_count = 0
    
    for _ in range(n_trials):
        result = crop(img)
        if result.size == (50, 50):
            applied_count += 1
    
    ratio = applied_count / n_trials
    # Должно быть примерно 50% с допуском
    assert 0.4 < ratio < 0.6, f"Ожидалось ~50%, получено {ratio*100:.1f}%"
    print(f"   Тест 3: p=0.5 применяет в {ratio*100:.1f}% случаев (ожидалось ~50%)")
    
    print()


def visualize_transforms():
    """Визуализация трансформаций."""
    import matplotlib.pyplot as plt
    
    img = create_test_image((200, 200))
    
    # Настраиваем трансформации
    transforms = {
        'Original': None,
        'RandomCrop(80)': RandomCrop(p=1.0, size=80),
        'RandomRotate(30°)': RandomRotate(p=1.0, degrees=30),
        'RandomZoom(0.7-1.3)': RandomZoom(p=1.0, scale=(0.7, 1.3)),
    }
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    random.seed(42)
    
    for idx, (name, transform) in enumerate(transforms.items()):
        if transform is None:
            result = img
        else:
            result = transform(img)
        
        axes[idx].imshow(result)
        axes[idx].set_title(name, fontsize=11)
        axes[idx].axis('off')
    
    # Compose примеры
    random.seed(42)
    compose = Compose([
        RandomCrop(p=1.0, size=150),
        RandomRotate(p=1.0, degrees=20),
    ])
    axes[4].imshow(compose(img))
    axes[4].set_title('Compose: Crop+Rotate', fontsize=11)
    axes[4].axis('off')
    
    random.seed(42)
    compose = Compose([
        RandomZoom(p=1.0, scale=(1.2, 1.2)),
        RandomRotate(p=1.0, degrees=15),
    ])
    axes[5].imshow(compose(img))
    axes[5].set_title('Compose: Zoom+Rotate', fontsize=11)
    axes[5].axis('off')
    
    random.seed(42)
    compose = Compose([
        RandomCrop(p=1.0, size=150),
        RandomRotate(p=1.0, degrees=25),
        RandomZoom(p=1.0, scale=(0.9, 1.1)),
    ])
    axes[6].imshow(compose(img))
    axes[6].set_title('Compose: All', fontsize=11)
    axes[6].axis('off')
    
    # ToTensor визуализация
    to_tensor = ToTensor()
    tensor_result = to_tensor(img)
    # Конвертируем обратно для отображения
    tensor_img = np.transpose(tensor_result.data, (1, 2, 0))
    axes[7].imshow(tensor_img)
    axes[7].set_title(f'ToTensor\nshape: {tensor_result.shape}', fontsize=11)
    axes[7].axis('off')
    
    plt.suptitle('Демонстрация трансформаций изображений', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('transforms_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("График сохранён: transforms_demo.png")


# =============================================================================
# Главная функция
# =============================================================================

def main():
    test_base_transform()
    test_random_crop()
    test_random_rotate()
    test_random_zoom()
    test_to_tensor()
    test_compose()
    test_reproducibility()
    test_probability()
    
    print("Визуализация трансформаций")
    visualize_transforms()


if __name__ == "__main__":
    main()