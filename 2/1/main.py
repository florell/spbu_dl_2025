import matplotlib.pyplot as plt
from tests import (
    test_linear, test_batchnorm, test_dropout, test_relu,
    test_sigmoid, test_softmax, test_cross_entropy,
    test_sequential, test_full_training
)


def visualize_training(losses, accuracies, filename='training_curves.png'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(losses)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(accuracies)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'{filename}', dpi=150)
    plt.close()
    print(f"График сохранён: {filename}")


def main():
    print("Реализация слоёв в рамках фреймворка оптимизаторов")
    
    test_linear()           
    test_batchnorm()       
    test_dropout()         
    test_relu()            
    test_sigmoid()       
    test_softmax()         
    test_cross_entropy()  
    test_sequential()    
    
    losses, accuracies = test_full_training()
    visualize_training(losses, accuracies)


if __name__ == "__main__":
    main()
