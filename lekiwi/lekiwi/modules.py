from dataclasses import asdict
from lerobot.common.motors import MotorCalibration

from proto.robots.module import Module
from proto.common.configs import ModuleConfig
from proto.robots.component_bus import ComponentBus
from proto.conf import settings


class BaseModule(Module):
    def __init__(self, cfg: ModuleConfig, name: str, component_bus: ComponentBus):
        super().__init__(cfg, name, component_bus)

    def setup(self):
        # disable torque
        self.write_actuators("Torque_Enable", 0)
        self.write_actuators("Lock", 0)

        self.write_actuators("Return_Delay_Time", 0)
        # Set Maximum_Acceleration to 254 to speedup acceleration and deceleration of
        # the motors. Note: this configuration is not in the official STS3215 Memory Table
        self.write_actuators("Maximum_Acceleration", 254)
        self.write_actuators("Acceleration", 254)

        # set to velocity mode TODO: use enum
        self.write_actuators("Operating_Mode", 1)

        # enable torque
        self.write_actuators("Torque_Enable", 1)
        self.write_actuators("Lock", 1)


    def calibrate(self):
        # no calibration needed
        pass

    def is_calibrated(self) -> bool:
        return True


class ArmModule(Module):
    def __init__(self, cfg: ModuleConfig, name: str, component_bus: ComponentBus):
        super().__init__(cfg, name, component_bus)

    def setup(self):

        if not self.is_calibrated():
            self.calibrate()

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
        self.write_actuators("P_Coefficient", 12)
        # Set I_Coefficient and D_Coefficient to default value 0 and 32
        self.write_actuators("I_Coefficient", 0)
        self.write_actuators("D_Coefficient", 32)
        
        # enable torque
        self.write_actuators("Torque_Enable", 1)
        self.write_actuators("Lock", 1)

    def is_calibrated(self) -> bool:
        if self.calibration is None:
            return False
        return self.calibration == self.read_calibration()
    
    def read_calibration(self) -> dict[str, MotorCalibration]:
        actuator_names = list(self.actuators.keys())
        offsets = self.component_bus.read("Homing_Offset", actuator_names, normalize=False)
        mins = self.component_bus.read("Min_Position_Limit", actuator_names, normalize=False)
        maxes = self.component_bus.read("Max_Position_Limit", actuator_names, normalize=False)
        
        calibration = {}
        for actuator_name in actuator_names:
            calibration[actuator_name] = MotorCalibration(
                id=self.actuators[actuator_name].cfg.id,
                drive_mode=0,
                homing_offset=offsets[actuator_name],
                range_min=mins[actuator_name],
                range_max=maxes[actuator_name],
            )

        return calibration

    def calibrate(self) -> None:

        print(f"\nRunning calibration of {self}")

        # group actuators by port
        actuators_by_port = {}
        for actuator_name, actuator in self.actuators.items():
            if actuator.cfg.port not in actuators_by_port:
                actuators_by_port[actuator.cfg.port] = []
            actuators_by_port[actuator.cfg.port].append(actuator_name)
        
        print("Disabling torque...")
        for port, actuators in actuators_by_port.items():
            self.component_bus.component_bus_by_port[port].disable_torque(actuators)

        print("Setting position mode...")
        for port, actuators in actuators_by_port.items():
            self.component_bus.component_bus_by_port[port].set_position_mode(actuators)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = {}
        for port, actuators in actuators_by_port.items():
            homing_offsets.update(self.component_bus.component_bus_by_port[port].set_half_turn_homings(actuators))

        # actuators that can rotate 360 degrees
        full_turn_motor = [settings.ActuatorName.WRIST_ROLL]

        # actuators that can't rotate 360 degrees
        unknown_range_motors = [motor for motor in self.actuators.keys() if motor not in full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )

        range_mins = {}
        range_maxes = {}
        range_mins[full_turn_motor] = 0
        range_maxes[full_turn_motor] = 4095

        # get ranges of motion for each actuator
        for port, actuators in actuators_by_port.items():
            range_mins, range_maxes = self.component_bus.component_bus_by_port[port].record_ranges_of_motion(unknown_range_motors)
            range_mins.update(range_mins)
            range_maxes.update(range_maxes)
            

        self.calibration = {}
        for port, actuators in actuators_by_port.items():
            port_calibration = {}
            for actuator_name in actuators:
                port_calibration[actuator_name] = MotorCalibration(
                    id=self.actuators[actuator_name].cfg.id,
                    drive_mode=0,
                    homing_offset=homing_offsets[actuator_name],
                    range_min=range_mins[actuator_name],
                    range_max=range_maxes[actuator_name],
                )
            self.component_bus.component_bus_by_port[port].write_calibration(port_calibration)
            self.calibration[port] = port_calibration

        self._save_calibration()
        print("Calibration saved to", self.cfg.calibration_file)


class LeaderArmModule(Module):
    def __init__(self, cfg: ModuleConfig, name: str, component_bus: ComponentBus):
        super().__init__(cfg, name, component_bus)

    def is_calibrated(self) -> bool:
        if self.calibration is None:
            return False
        return self.calibration == self.read_calibration()
    
    def read_calibration(self) -> dict[str, MotorCalibration]:
        actuator_names = list(self.actuators.keys())
        offsets = self.component_bus.read("Homing_Offset", actuator_names, normalize=False)
        mins = self.component_bus.read("Min_Position_Limit", actuator_names, normalize=False)
        maxes = self.component_bus.read("Max_Position_Limit", actuator_names, normalize=False)
        drive_modes = self.component_bus.read("Drive_Mode", actuator_names, normalize=False)

        calibration = {}
        for actuator_name in actuator_names:
            calibration[actuator_name] = MotorCalibration(
                id=self.actuators[actuator_name].cfg.id,
                drive_mode=drive_modes[actuator_name],
                homing_offset=offsets[actuator_name],
                range_min=mins[actuator_name],
                range_max=maxes[actuator_name],
            )

        return calibration
    
    
    def setup(self):
        if not self.is_calibrated():
            self.calibrate()

        # Disable torque for all actuators
        self.disable_actuators_torque()
        
        EXTENDED_POSITION = 4
        CURRENT_POSITION = 5

        self.write_actuators("Operating_Mode", 
            {
                settings.ActuatorName.SHOULDER_PAN: EXTENDED_POSITION, 
                settings.ActuatorName.SHOULDER_LIFT: EXTENDED_POSITION, 
                settings.ActuatorName.ELBOW_FLEX: EXTENDED_POSITION, 
                settings.ActuatorName.WRIST_FLEX: EXTENDED_POSITION, 
                settings.ActuatorName.WRIST_ROLL: EXTENDED_POSITION
            })

        self.write_actuators("Return_Delay_Time", 0)

        self.write_actuators("Operating_Mode", {settings.ActuatorName.GRIPPER: CURRENT_POSITION})

        # Enable torque for the gripper and set position
        print(f"Setting gripper open angle...")
        self.enable_actuators_torque([settings.ActuatorName.GRIPPER])
        self.write_actuators_pos({settings.ActuatorName.GRIPPER: self.cfg.gripper_open_degree}, normalize=True)

    def calibrate(self) -> None:
        print(f"\nRunning calibration of {self.name}")

        # group actuators by port
        actuators_by_port = {}
        for actuator_name, actuator in self.actuators.items():
            if actuator.cfg.port not in actuators_by_port:
                actuators_by_port[actuator.cfg.port] = []
            actuators_by_port[actuator.cfg.port].append(actuator_name)
        
        print("Disabling torque...")
        self.disable_actuators_torque()

        print("Setting position mode...")
        self.set_actuators_position_mode()

        self.write_actuators("Drive_Mode", 0)

        drive_modes = {motor: 1 if motor in [settings.ActuatorName.SHOULDER_PAN.value, settings.ActuatorName.WRIST_FLEX.value, settings.ActuatorName.WRIST_ROLL.value] else 0 for motor in self.actuators.keys()}

        input(f"Move {self.name} to the middle of its range of motion and press ENTER....")
        homing_offsets = {}
        for port, actuators in actuators_by_port.items():
            homing_offsets.update(self.component_bus.component_bus_by_port[port].set_half_turn_homings(actuators))

        # actuators that can rotate 360 degrees
        full_turn_motor = [settings.ActuatorName.WRIST_ROLL.value, settings.ActuatorName.SHOULDER_PAN.value]

        # actuators that can't rotate 360 degrees
        unknown_range_motors = [motor for motor in self.actuators.keys() if motor not in full_turn_motor]
        print(
            f"Move all joints except sequentially through their"
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )

        range_mins = {}
        range_maxes = {}
        for motor in full_turn_motor:
            range_mins[motor] = 0
            range_maxes[motor] = 4095

        # get ranges of motion for each actuator
        for port, actuators in actuators_by_port.items():
            mins, maxes = self.component_bus.component_bus_by_port[port].record_ranges_of_motion(unknown_range_motors)
            range_mins.update(mins)
            range_maxes.update(maxes)

        self.calibration = {}
        for port, actuators in actuators_by_port.items():
            port_calibration = {}
            for actuator_name in actuators:
                calib = MotorCalibration(
                    id=self.actuators[actuator_name].cfg.id,
                    drive_mode=drive_modes[actuator_name],
                    homing_offset=homing_offsets[actuator_name],
                    range_min=range_mins[actuator_name],
                    range_max=range_maxes[actuator_name],
                )
                port_calibration[actuator_name] = calib
                self.calibration[actuator_name] = asdict(calib)
            self.component_bus.component_bus_by_port[port].write_calibration(port_calibration)
            
        self._save_calibration(self.calibration)
        print("Calibration saved to ", self.cfg.calibration_file)


    
        
