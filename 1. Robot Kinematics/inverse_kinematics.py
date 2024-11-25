import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models

r2 = tf.keras.models.load_model('fk_r2.keras')
r3 = tf.keras.models.load_model('fk_r3.keras')

# Forward kinematics functions
def forward_kinematics_r2(q):
    q = np.array([q], dtype=np.float32)
    return r2.predict(q, verbose=0)[0]

def forward_kinematics_r3(q):
    q = np.array([q], dtype=np.float32)
    return r3.predict(q, verbose=0)[0]

# Jacobian calculation functions
def Jacobian_r2(q):
    q = tf.Variable([q], dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(q)
        y = r2(q)
    return np.reshape(tape.jacobian(y, q), (3, 2))

def Jacobian_r3(q):
    q = tf.Variable([q], dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(q)
        y = r3(q)
    return np.reshape(tape.jacobian(y, q), (3, 3))

# Inverse kinematics function
max_iterations = 10     # Maximum number of iterations
tolerance = 0.001       # Error threshold for convergence

def inverse_kinematics_r2(target_pose, initial_guess):
    lambda_val = 0.01 # Initial damping factor
    q = np.array(initial_guess, dtype=np.float32)
    for i in range(1, max_iterations+1):
        current_pose = forward_kinematics_r2(q)
        error = target_pose - current_pose
        error_norm = np.linalg.norm(error)
        #if i % (max_iterations/10) == 0:
            #print(f'########## Iteration {i} ##########')
            #print(f'Current pose: {current_pose}')
            #print(f'Error = {error_norm:.5f}')
        if np.linalg.norm(error) < tolerance:
            #print(f'Converged in {i+1} iterations to {q}')
            #print(f'Reached end-effector pose: {current_pose}')
            #print(f'Original desired pose: {target_pose}')
            return q

        # Update rule
        J = Jacobian_r2(q)
        JTJ = J.T @ J + lambda_val*np.eye(J.shape[1]) # Regularized Jacobian
        delta_q = np.linalg.inv(JTJ) @ J.T @ error    # LM step

        # Tentative update
        q_new = q + delta_q
        new_error = np.linalg.norm(target_pose - forward_kinematics_r2(q_new))
        if new_error < error_norm:
            lambda_val /= 10 # Trust the step more
            q = q_new        # Accept the update
        else:
            lambda_val *= 10 # Trust the step less

    #print(f'[r2] LM could not converge within {max_iterations} iterations')
    return q

# Plot robot function
l1, l2 = 0.1, 0.1
def plot_2r_robot(q):
    q1, q2 = q
    x0, y0 = 0, 0
    x1, y1 = l1*np.cos(q1), l1*np.sin(q1)
    x2, y2 = x1 + l2*np.cos(q1+q2), y1 + l2*np.sin(q1+q2)
    plt.figure(figsize=(6, 6))
    plt.plot([x0, x1], [y0, y1], '-o', label='Link 1', linewidth=4)
    plt.plot([x1, x2], [y1, y2], '-o', label='Link 2', linewidth=4)
    
    # Plot end-effector
    plt.plot(x2, y2, 'ro', label='End-Effector')
    
    # Set plot limits and labels
    plt.xlim(-0.25, 0.25)
    plt.ylim(-0.25, 0.25)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('[r2] "learned" Inverse kinematics')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()
