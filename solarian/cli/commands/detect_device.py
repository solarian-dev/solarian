import time
import serial.tools.list_ports
from omegaconf import DictConfig, OmegaConf

from solarian.cli import BaseCommand, CommandRegistry
from solarian.utils import get_class


@CommandRegistry.register('detect-device')
class DetectDeviceCommand(BaseCommand):
    help = "Detect a serial device by plugging it in (prints port and name)"
    description = """
    Lists the serial ports before and after you plug a device in, and names the
    one that appeared. Use it to find the port for an actuator bus or a servo
    chain, then put that path in the deployment profile as actuator_port_<module>.

    Interactive: it waits for Enter with the device unplugged, then again once it
    is connected.
    """
    usage = ""
    examples = ("solarian detect-device",)

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
