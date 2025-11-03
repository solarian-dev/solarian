import time
import serial.tools.list_ports
from omegaconf import DictConfig, OmegaConf

from proto.management import BaseCommand, CommandRegistry
from proto.common.utils import get_class
from proto.conf import settings


@CommandRegistry.register('detect-device')
class DetectDeviceCommand(BaseCommand):
    help = 'Detects devices connected to the computer (port and name).'

    def handle(self, cfg: DictConfig) -> None:
         

        def list_ports():
            return {port.device: port.description for port in serial.tools.list_ports.comports()}

        print("Unplug your device, then press Enter...")
        input()
        before = list_ports()

        print("\nDevices before:")
        for dev, desc in before.items():
            print(f"  {dev}: {desc}")

        print("\nNow plug in your device, wait a second, then press Enter...")
        input()
        after = list_ports()

        print("\nDevices after:")
        for dev, desc in after.items():
            print(f"  {dev}: {desc}")

        # Find newly added ports
        new_ports = set(after.keys()) - set(before.keys())

        if new_ports:
            print("\nNew device(s) detected:")
            for dev in new_ports:
                print(f"  {dev}: {after[dev]}")
        else:
            print("\nNo new device detected.")
