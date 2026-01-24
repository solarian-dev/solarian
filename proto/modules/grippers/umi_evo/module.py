from proto.robots.module import Module


class UMIEvoGripperModule(Module):
    
    def initialize(self):
        super().initialize()

        # Disable torque for each arm motor.
        self.write_actuators("Torque_Enable", 0)
        self.write_actuators("Lock", 0)
        
        self.write_actuators("Return_Delay_Time", 0)
        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        self.write_actuators("Maximum_Acceleration", 254)
        self.write_actuators("Acceleration", 254)
        
        # Mode=0 for Position Control
        self.set_actuators_position_mode()
        # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
        # self.write_actuators("P_Coefficient", 16)
        # self.write_actuators("P_Coefficient", 8)
        # self.write_actuators("P_Coefficient", 32)
        # # Set I_Coefficient and D_Coefficient to default value 0 and 32
        # self.write_actuators("I_Coefficient", 0)
        # self.write_actuators("D_Coefficient", 16)
        self.write_actuators("P_Coefficient", 32)
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        self.write_actuators("I_Coefficient", 2)
        self.write_actuators("D_Coefficient", 2)
        
        # enable torque
        self.write_actuators("Torque_Enable", 1)
        self.write_actuators("Lock", 1)

        # self.write_actuators_pos({"gripper": 0})

   