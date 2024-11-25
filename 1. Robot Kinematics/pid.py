import numpy as np

def pid_control(target_angles, current_angles, dt, error_sum, previous_error, Kp, Ki, Kd):
    control_signals = []
    for i in range(len(target_angles)):
        error = target_angles[i] - current_angles[i]
        error_sum[i] += error * dt
        error_derivative = (error - previous_error[i]) / dt
        #print(f'Error for joint {i+1} = {error}')
        #print(f'Error_sum for joint {i+1} = {error_sum[i]}')
        #print(f'Error_derivative for joint {i+1} = {error_derivative}')

        control_signal = (Kp[i]*error) + (Ki[i]*error_sum[i]) + (Kd[i]*error_derivative)
        #print(f'Control signal for joint {i+1}: {control_signal}')
        control_signals.append(control_signal)
        
        previous_error[i] = error
    return control_signals
