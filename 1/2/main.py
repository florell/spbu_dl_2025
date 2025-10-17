import torch
import unittest


class Node:
    def __init__(self, data, _children=(), _op=''):
        # конвертируем данные в torch.tensor для корректной работы
        if isinstance(data, (int, float)):
            self.data = torch.tensor(float(data), requires_grad=False)
        else:
            self.data = data
        
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Node(data={self.data.item()}, grad={self.grad})"

    def __add__(self, other):
        # поддержка добавления скаляров
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data + other.data, (self, other), '+')
        
        def _backward():
            # градиент суммы распределяется на оба слагаемых
            self.grad += out.grad
            other.grad += out.grad
        
        out._backward = _backward
        return out

    def __radd__(self, other):
        # для поддержки other + self
        return self + other

    def __mul__(self, other):
        # поддержка умножения на скаляры
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other), '*')
        
        def _backward():
            # градиент произведения: d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data.item() * out.grad
            other.grad += self.data.item() * out.grad
        
        out._backward = _backward
        return out

    def __rmul__(self, other):
        # для поддержки other * self
        return self * other

    def relu(self):
        out = Node(torch.relu(self.data), (self,), 'ReLU')
        
        def _backward():
            # градиент ReLU: 1 если x > 0, иначе 0
            self.grad += (self.data.item() > 0) * out.grad
        
        out._backward = _backward
        return out

    def backward(self):
        # топологическая сортировка графа вычислений
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # градиент выходного узла равен 1
        self.grad = 1
        
        # обратный проход в обратном топологическом порядке
        for node in reversed(topo):
            node._backward()


class TestAutograd(unittest.TestCase):
    
    def test_addition(self):
        """Тест сложения"""
        a = Node(2)
        b = Node(3)
        c = a + b
        c.backward()
        
        self.assertEqual(c.data.item(), 5)
        self.assertEqual(a.grad, 1)
        self.assertEqual(b.grad, 1)
    
    def test_multiplication(self):
        """Тест умножения"""
        a = Node(2)
        b = Node(3)
        c = a * b
        c.backward()
        
        self.assertEqual(c.data.item(), 6)
        self.assertEqual(a.grad, 3)
        self.assertEqual(b.grad, 2)
    
    def test_relu_positive(self):
        """Тест ReLU с положительным значением"""
        a = Node(5)
        b = a.relu()
        b.backward()
        
        self.assertEqual(b.data.item(), 5)
        self.assertEqual(a.grad, 1)
    
    def test_relu_negative(self):
        """Тест ReLU с отрицательным значением"""
        a = Node(-5)
        b = a.relu()
        b.backward()
        
        self.assertEqual(b.data.item(), 0)
        self.assertEqual(a.grad, 0)
    
    def test_combined_operations(self):
        """Тест из примера задания"""
        a = Node(2)
        b = Node(-3)
        c = Node(10)
        d = a + b * c
        e = d.relu()
        e.backward()
        
        self.assertEqual(a.data.item(), 2)
        self.assertEqual(b.data.item(), -3)
        self.assertEqual(c.data.item(), 10)
        self.assertEqual(d.data.item(), -28)
        self.assertEqual(e.data.item(), 0)
        
        self.assertEqual(a.grad, 0)
        self.assertEqual(b.grad, 0)
        self.assertEqual(c.grad, 0)
        self.assertEqual(e.grad, 1)
    
    def test_complex_graph(self):
        """Тест сложного вычислительного графа"""
        a = Node(2)
        b = Node(3)
        c = Node(4)
        
        d = a * b  # 6
        e = d + c  # 10
        f = e.relu()  # 10
        g = f * Node(2)  # 20
        g.backward()
        
        self.assertEqual(g.data.item(), 20)
        self.assertEqual(a.grad, 6)  # 2 * 3
        self.assertEqual(b.grad, 4)  # 2 * 2
        self.assertEqual(c.grad, 2)  # 2 * 1
    
    def test_multiple_uses(self):
        """Тест переменной, используемой несколько раз"""
        a = Node(2)
        b = a + a  # b = 2a = 4
        c = b * a  # c = b*a = 2a*a = 2a^2= 8
        c.backward()
        
        self.assertEqual(c.data.item(), 8)
        # dc/da = d(2a^2)/da = 4a = 4*2 = 8, da/da = 1
        self.assertEqual(a.grad, 8)
    
    def test_relu_zero(self):
        """Тест ReLU с нулевым значением"""
        a = Node(0)
        b = a.relu()
        b.backward()
        
        self.assertEqual(b.data.item(), 0)
        self.assertEqual(a.grad, 0)
    
    def test_scalar_operations(self):
        """Тест операций со скалярами"""
        a = Node(5)
        b = a + 3  # 8
        c = b * 2  # 16
        c.backward()
        
        self.assertEqual(c.data.item(), 16)
        self.assertEqual(a.grad, 2)


if __name__ == "__main__":
    print("=== Пример из задания ===")
    a = Node(2)
    b = Node(-3)
    c = Node(10)
    d = a + b * c
    e = d.relu()
    e.backward()
    
    print(f"a: {a}")
    print(f"b: {b}")
    print(f"c: {c}")
    print(f"d: {d}")
    print(f"e: {e}")
    
    print("\n=== Дополнительный пример ===")
    x = Node(3)
    y = Node(4)
    z = x * y + Node(2)
    w = z.relu()
    w.backward()
    
    print(f"x: {x}")
    print(f"y: {y}")
    print(f"z: {z}")
    print(f"w: {w}")
    
    print("\n=== Запуск тестов ===")
    unittest.main(argv=[''], verbosity=2, exit=False)