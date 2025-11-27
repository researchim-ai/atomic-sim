import torch
import torch.nn as nn

class ControlSystem1D(nn.Module):
    def __init__(self, device='cpu', dtype=torch.float64):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        # State
        self.rod_position = 0.2 # 0.0 (out) to 1.0 (in)
        self.boron_concentration = 1000.0 # ppm
        
        # Targets & Auto Flags
        self.target_power = 3000.0 # MW (will be overwritten by sim)
        self.auto_power = False # PID for rods
        self.auto_boron = False # Shim control
        
        # PID Params (Power)
        self.kp = 0.01
        self.ki = 0.005
        self.kd = 0.05
        self.integral_error = 0.0
        self.prev_error = 0.0
        
        # Shim Params (Boron)
        self.shim_deadband = 0.05 # Allow rod deviation +/- 5% from ref
        self.rod_ref = 0.2 # We want rods at ~20% inserted for bite.
        self.boron_rate = 2.0 # ppm/s
        
    def step(self, dt, target_power, current_power, manual=False):
        if manual:
            # In manual mode, user sets rods/boron directly via UI setters on this object
            # Reset integral error to avoid windup when switching back
            self.integral_error = 0.0
            self.prev_error = 0.0
            return
            
        # --- Auto Power Control (Rods) ---
        if self.auto_power:
            error = target_power - current_power
            self.integral_error += error * dt
            derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
            
            # PID output = required reactivity change
            # But we control rod speed.
            # If error > 0 (Power too low) -> Move rods OUT (decrease position)
            
            control_signal = self.kp * error + self.ki * self.integral_error + self.kd * derivative
            
            # Map signal to rod speed
            # Signal > 0 => Need Power UP => Rod Speed < 0 (Withdraw)
            rod_speed = -control_signal * 0.01 # Scaling factor
            
            # Clamp speed
            rod_speed = max(-0.1, min(0.1, rod_speed))
            
            self.rod_position = max(0.0, min(1.0, self.rod_position + rod_speed * dt))
            self.prev_error = error
            
        # --- Auto Boron Control (Shim) ---
        if self.auto_boron:
            # Logic: Keep rods at reference position
            # If Rods > Ref (Too deep): We need to withdraw them. To allow withdrawal while keeping power const, we must ADD poison? 
            # Balance: Rho_total = 0.
            # Rho_rods + Rho_boron = const.
            
            rod_dev = self.rod_position - self.rod_ref
            
            if rod_dev > self.shim_deadband: 
                # Rods too deep (0.5 > 0.2). Need to pull out.
                # Action: Add Boron. Power drops. Auto-rod pulls rods out.
                self.boron_concentration += self.boron_rate * dt
            elif rod_dev < -self.shim_deadband:
                # Rods too high (0.1 < 0.2). Need to push in.
                # Action: Dilute Boron (Remove poison). Power rises. Auto-rod pushes rods in.
                self.boron_concentration -= self.boron_rate * dt
                
            self.boron_concentration = max(0.0, self.boron_concentration)
            
        return # State is updated internally
