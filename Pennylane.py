import pennylane as qml
from pennylane import numpy as np

dev1 = qml.device("default.qubit", wires = 1)

@qml.qnode(dev1)
def circuit(phi1,phi2):
    qml.RX(phi1, wires = 0)
    qml.RY(phi2, wires = 0)
    return qml.expval(qml.PauliZ(0))

def cost(x,y):
    return np.sin(np.abs(circuit(x,y))) - 1

dcost = qml.grad(cost, argnum = [0 , 1])

