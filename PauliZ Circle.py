import pennylane as qml
import numpy as np
import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from datetime import datetime

X, Y, Z = qml.PauliX(0), qml.PauliY(0), qml.PauliZ(0)

omega = 2 * jnp.pi * 5.

# To generate a time-dependent ``ParametrizedHamiltonian``, we multiply a ``callable``
# and an ``Operator``.
# Due to PennyLane convention, the callable has to have the signature (p, t).
# Here, the only parameter that we control is the phase ``p`` in the sinusodial.
def amp(nu):
    def wrapped(p, t):
        return jnp.pi * jnp.sin(nu*t + p)
    return wrapped

H = -omega/2 * qml.PauliZ(0)
H += amp(omega) * qml.PauliY(0)

# We generate a qnode that evolves the qubit state according to the time-dependent
# Hamiltonian H.
@jax.jit
@qml.qnode(qml.device("default.qubit", wires=1), interface="jax")
def trajectory(params, t):
    qml.evolve(H)((params,), t, return_intermediate=True)
    return [qml.expval(op) for op in [X, Y, Z]]

# By setting ``return_intermediate=True``, we can output all intermediate time steps.
# We compute the time series for 10000 samples for the phase equal to 0 and pi/2, respectively.
ts = jnp.linspace(0., 1., 10000)
res0 = trajectory(0., ts)
res1 = trajectory(jnp.pi/2, ts)

# We plot the evolution in the Bloch sphere.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(*res0, "-", label="$\\phi=0$")
ax.plot(*res1, "-", label="$\\phi=\\pi/2$")
ax.legend()