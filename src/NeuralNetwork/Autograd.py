import math
import numpy as np

class Scalar:
    """
    Kelas untuk menyimpan nilai skalar dan gradiennya.
    Kelas ini digunakan dalam perhitungan otomatisasi diferensiasi (autograd),
    memungkinkan propagasi mundur (backpropagation) untuk menghitung turunan.
    """

    def __init__(self, value: int | float, parents: tuple = (), operation: str = ''):
        """
        Inisialisasi objek Scalar value.
        
        Parameters:
        - value (float): Nilai skalar yang disimpan.
        - parents (tuple): Node sebelumnya dalam computational graph.
        - operation (str): Operasi yang menghasilkan nilai ini
        """
        if isinstance(value, (int, float, np.integer, np.floating)):
            self.value = value
            self.grad = 0  # Gradien default adalah nol
            self._backward: callable = lambda: None  # Fungsi untuk backpropagation
            self._parents = set(parents)  # Menyimpan node asal
            self._operation = operation  # Menyimpan operasi yang dilakukan parents
        else:
            raise TypeError("Value should be integer or float")

    #! Basic getter/setter
    def get_value(self) -> int | float:
        """
        Getter untuk value
        """
        return self.value

    def get_backward(self) -> callable:
        """
        Call fungsi _backward()

        Returns:
            callable: Fungsi yang akan dipanggil
        """
        return self._backward

    def set_backward(self, _backward: callable):
        """
        Set fungsi backward

        Args:
            _backward (callable): fungsi backward yang akan di-set
        """
        self._backward = _backward

    def do_backward(self):
        """
        Do the backward function
        """
        self._backward()

    def get_parents(self):
        """
        Return parents of the Scalar value

        Returns:
            parents: parents of the Scalar
        """
        return self._parents

    def get_operation(self) -> str:
        """
        Return the operation that creating this Scalar value. '' if not exists

        Returns:
            str: Operation creating this scalar
        """
        return self._operation

    #! Basic math operator
    def __add__(self, other) -> 'Scalar':
        """ Operator penjumlahan antara dua objek Scalar. Dipanggil otomatis di self + other"""
        other = other if isinstance(other, Scalar) else Scalar(other)
        result = Scalar(self.value + other.value, (self, other), '+')

        def _backward():
            self.grad += result.grad
            other.grad += result.grad
        result._backward = _backward

        return result

    def __mul__(self, other) -> 'Scalar':
        """ Operator perkalian antara dua objek Scalar. Dipanggil otomatis di self * other"""
        other = other if isinstance(other, Scalar) else Scalar(other)
        result = Scalar(self.value * other.value, (self, other), '*')

        def _backward():
            self.grad += other.value * result.grad
            other.grad += self.value * result.grad
        result._backward = _backward

        return result

    def __pow__(self, exponent) -> 'Scalar':
        """ Operator pemangkatan. Dipanggil otomatis di self ** other"""
        exponent = exponent.value if isinstance(exponent, Scalar) else exponent
        assert isinstance(exponent, (float, int)), "Nilai harus bilangan"
        result = Scalar(self.value ** exponent, (self,), f'**{exponent}')

        def _backward():
            # Turunan x^m = m * x^(m-1)
            self.grad += (exponent * self.value ** (exponent - 1)) * result.grad
        result._backward = _backward

        return result

    def __neg__(self) -> 'Scalar':
        """ Operator negasi (-self). """
        return self * -1

    def __sub__(self, other) -> 'Scalar':
        """ Operator pengurangan (self - other). """
        return self + (-other)

    def __rsub__(self, other) -> 'Scalar':
        """ Operator pengurangan untuk other - self. """
        return other + (-self)

    def __radd__(self, other) -> 'Scalar':
        """ Operator penjumlahan untuk other + self. """
        return self + other

    def __rmul__(self, other) -> 'Scalar':
        """ Operator perkalian untuk other * self. """
        return self * other

    def __truediv__(self, other) -> 'Scalar':
        """ Operator pembagian (self / other). """
        return self * (other ** -1)

    def __rtruediv__(self, other) -> 'Scalar':
        """ Operator pembagian untuk other / self. """
        return other * (self ** -1)

    def __repr__(self):
        """ Representasi string dari objek Scalar. """
        return f"Scalar(value={self.value}, grad={self.grad})"

    def log(self) -> 'Scalar':
        """ Compute natural logarithm of the Scalar """
        assert self.value > 0, "Log input must be positive"
        out = Scalar(math.log(self.value), (self,), 'log')

        def _backward():
            self.grad += (1 / self.value) * out.grad
        out.set_backward(_backward)

        return out

    def __mod__(self, other: 'Scalar') -> 'Scalar':
        """
        Operasi modulo

        Args:
            other (Scalar): nilai lain yang akan dimodulo

        Returns:
            Scalar
        """
        return Scalar(self.value % other.value)


    def __rmod__(self, other: 'Scalar') -> 'Scalar':
        """
        Operasi modulo

        Args:
            other (Scalar): nilai lain yang akan dimodulo

        Returns:
            Scalar
        """
        return Scalar(other.value % self.value)


    #! Activation function
    def relu(self) -> 'Scalar':
        """ Fungsi aktivasi ReLU (Rectified Linear Unit). Dikalikan dengan nilai out.grad
          karena chain rule (dz/dx = dz/dy * dy/dx)
        f(x) = max(0, x). Turunan: if x > 0 return 1, else return 0
        """
        result = Scalar(0 if self.value <= 0 else self.value, (self,), 'ReLU')

        def _backward():
            if result.value > 0:
                self.grad += 1 * result.grad
        result.set_backward(_backward)

        return result

    def sigmoid(self) -> 'Scalar':
        """
        Fungsi aktivasi sigmoid: f(x) = 1/(1 + e^(-x))
        Turunan: o * (1 - o)
        """
        out = Scalar(1 / (1 + math.exp(-self.value)), (self,), 'sigmoid')

        def _backward():
            self.grad += (out.value * (1 - out.value)) * out.grad
        out.set_backward(_backward)

        return out

    def linear(self) -> 'Scalar':
        """
        Fungsi aktivasi linear

        Returns:
            float/int/Scalar: Scalar value hasil linear
        """
        out = Scalar(self.value, (self,), 'linear')

        def _backward():
            self.grad += 1 * out.grad
        out.set_backward(_backward)

        return out

    def tanh(self) -> 'Scalar':
        """
        Fungsi aktivasi tanh: f(x) = tanh(x)
        Turunan: 1 - tanh(x) ** 2

        Returns:
            int | float: Scalar value hasil tanh
        """
        out = Scalar(math.tanh(self.value), (self,), 'tanh')

        def _backward():
            self.grad += (1 - out.value ** 2) * out.grad
        out.set_backward(_backward)

        return out

    def backward(self):
        """
        Melakukan propagasi mundur untuk menghitung gradien dari nilai ini terhadap
        semua nilai dalam computational graph yang berkontribusi terhadapnya.
        """
        topo_order: list[Scalar] = []
        visited = set()

        def build_topology(node: Scalar):
            if node not in visited:
                visited.add(node)
                for parent in node.get_parents():
                    build_topology(parent)
                topo_order.append(node)

        build_topology(self)
        self.grad = 1

        for node in reversed(topo_order):
            node.do_backward()
