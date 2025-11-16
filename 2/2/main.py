"""
Медианный фильтр на чистом PyTorch.

Для каждого пикселя берётся окрестность размера kernel_size x kernel_size,
все значения сортируются и выбирается медианное значение.

Использование: только torch, без torch.nn
"""

import torch
import matplotlib.pyplot as plt


def median_filter(image: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """
    Медианный фильтр на чистом PyTorch.
    
    Алгоритм:
    1. Паддинг изображения для обработки краёв (reflect padding)
    2. Извлечение патчей (окрестностей) для каждого пикселя с помощью unfold
    3. Сортировка значений в каждом патче
    4. Выбор медианного значения (элемент с индексом kernel_size^2 // 2)
    
    Args:
        image: входное изображение shape (H, W) или (C, H, W) или (B, C, H, W)
        kernel_size: размер ядра фильтра
    
    Returns:
        отфильтрованное изображение той же формы
    """
    # Сохраняем исходную размерность
    original_ndim = image.ndim
    
    # Приводим к формату (B, C, H, W)
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
    elif image.ndim == 3:
        image = image.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
    
    B, C, H, W = image.shape
    
    # Паддинг для сохранения размера
    pad = kernel_size // 2
    
    # Паддинг с отражением для краёв
    # torch.nn.functional.pad не используем, реализуем вручную
    # Создаём паддинг через индексирование
    
    # Паддинг по горизонтали 
    left_pad = image[:, :, :, 1:pad+1].flip(dims=[3])
    right_pad = image[:, :, :, -(pad+1):-1].flip(dims=[3])
    image_h_padded = torch.cat([left_pad, image, right_pad], dim=3)
    
    # Паддинг по вертикали
    top_pad = image_h_padded[:, :, 1:pad+1, :].flip(dims=[2])
    bottom_pad = image_h_padded[:, :, -(pad+1):-1, :].flip(dims=[2])
    image_padded = torch.cat([top_pad, image_h_padded, bottom_pad], dim=2)
    
    # Извлекаем патчи с помощью unfold
    # unfold по высоте: (B, C, H_padded, W_padded) -> (B, C, H, kernel_size, W_padded)
    patches = image_padded.unfold(2, kernel_size, 1)
    # unfold по ширине: -> (B, C, H, W, kernel_size, kernel_size)  
    patches = patches.unfold(3, kernel_size, 1)
    
    # Берём только нужную часть (первые H x W патчей)
    patches = patches[:, :, :H, :W, :, :]
    
    # Reshape для сортировки: (B, C, H, W, kernel_size * kernel_size)
    patches = patches.contiguous().view(B, C, H, W, -1)
    
    # Сортируем и берём медиану
    sorted_patches, _ = torch.sort(patches, dim=-1)
    median_idx = kernel_size * kernel_size // 2
    output = sorted_patches[..., median_idx]
    
    # Возвращаем к исходной размерности
    if original_ndim == 2:
        output = output.squeeze(0).squeeze(0)
    elif original_ndim == 3:
        output = output.squeeze(0)
    
    return output


def add_salt_pepper_noise(image: torch.Tensor, amount: float = 0.05) -> torch.Tensor:
    """Добавляет импульсный шум (соль и перец)."""
    noisy = image.clone()
    
    # Белые точки
    salt_mask = torch.rand_like(image) < amount / 2
    noisy[salt_mask] = 1.0
    
    # Чёрные точки
    pepper_mask = torch.rand_like(image) < amount / 2
    noisy[pepper_mask] = 0.0
    
    return noisy


def create_test_image(size: int = 256) -> torch.Tensor:
    """Создаёт тестовое изображение с различными паттернами."""
    image = torch.zeros(size, size)
    
    # Градиент
    for i in range(size):
        image[i, :size//3] = i / size
    
    # Круг
    center = (size // 2, size // 2)
    radius = size // 4
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    circle_mask = ((x - center[1])**2 + (y - center[0])**2) < radius**2
    image[circle_mask] = 0.8
    
    # Квадрат
    image[size//4:size//4+size//6, 2*size//3:2*size//3+size//6] = 0.5
    
    # Линии
    image[::20, :] = 0.3
    
    return image


def demo_median_filter():
    """Демонстрация медианного фильтра с размерами ядра 3, 5, 10."""
    
    torch.manual_seed(42)
    
    # Создаём тестовое изображение
    original = create_test_image(256)
    
    # Добавляем шум
    noisy = add_salt_pepper_noise(original, amount=0.1)
    
    # Применяем медианный фильтр с разными размерами ядра
    kernel_sizes = [3, 5, 10]
    filtered_images = {}
    
    
    for k in kernel_sizes:
        print(f"\nПрименение медианного фильтра с ядром {k}x{k}...")
        filtered = median_filter(noisy, k)
        filtered_images[k] = filtered
        
        # Вычисляем метрики
        mse_noisy = torch.mean((noisy - original) ** 2).item()
        mse_filtered = torch.mean((filtered - original) ** 2).item()
        print(f"  MSE зашумлённого: {mse_noisy:.6f}")
        print(f"  MSE после фильтра: {mse_filtered:.6f}")
        print(f"  Улучшение: {mse_noisy/mse_filtered:.2f}x")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Верхний ряд: оригинал, зашумлённое, разница
    axes[0, 0].imshow(original.numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Оригинал', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(noisy.numpy(), cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('С шумом (salt & pepper 10%)', fontsize=12)
    axes[0, 1].axis('off')
    
    # Показываем шум
    noise_diff = torch.abs(noisy - original)
    axes[0, 2].imshow(noise_diff.numpy(), cmap='hot', vmin=0, vmax=1)
    axes[0, 2].set_title('Шум (абсолютная разница)', fontsize=12)
    axes[0, 2].axis('off')
    
    for idx, k in enumerate(kernel_sizes):
        axes[1, idx].imshow(filtered_images[k].numpy(), cmap='gray', vmin=0, vmax=1)
        mse = torch.mean((filtered_images[k] - original) ** 2).item()
        axes[1, idx].set_title(f'Медианный фильтр {k}x{k}\nMSE: {mse:.6f}', fontsize=12)
        axes[1, idx].axis('off')
    
    plt.suptitle('Демонстрация медианного фильтра', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('median_filter_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nГрафик сохранён: median_filter_demo.png")
    
    return original, noisy, filtered_images


def test_median_filter():
    """Тесты корректности медианного фильтра."""
    print("=" * 60)
    print("Тесты медианного фильтра")
    print("=" * 60)
    
    # Тест 1: Константное изображение
    const_img = torch.ones(10, 10) * 0.5
    result = median_filter(const_img, 3)
    assert torch.allclose(result, const_img), "Константное изображение должно остаться неизменным"
    print("  Тест 1: Константное изображение")
    
    # Тест 2: Удаление одиночного выброса
    img_with_outlier = torch.zeros(5, 5)
    img_with_outlier[2, 2] = 1.0
    result = median_filter(img_with_outlier, 3)
    assert result[2, 2] == 0.0, "Одиночный выброс должен быть удалён"
    print("  Тест 2: Удаление одиночного выброса")
    
    # Тест 3: Сохранение формы
    for shape in [(32, 32), (3, 32, 32), (2, 3, 32, 32)]:
        img = torch.rand(shape)
        result = median_filter(img, 3)
        assert result.shape == img.shape, f"Форма должна сохраняться для {shape}"
    print("  Тест 3: Сохранение формы")
    
    # Тест 4: Проверка значения медианы
    img = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float32)
    result = median_filter(img, 3)
    assert result[1, 1] == 5.0, "Медиана центрального пикселя должна быть 5"
    print("  Тест 4: Корректное значение медианы")
    
    # Тест 5-6: Разные размеры ядра
    img = torch.rand(50, 50)
    for k in [5, 10]:
        result = median_filter(img, k)
        assert result.shape == img.shape
        print(f"  Тест: Ядро {k}x{k}")
    
    print("\nВсе тесты пройдены")


if __name__ == "__main__":
    test_median_filter()
    print()
    demo_median_filter()