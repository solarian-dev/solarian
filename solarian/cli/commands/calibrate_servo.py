"""``solarian calibrate-servo`` — prepare one servo for use, any brand.

A servo out of the box is not addressable (fresh units share the factory ID) and
its shaft sits wherever it was left, which is how a joint ends up with the 0/CPR
encoder seam inside its travel. This walks brand → port → servo and then hands
over to that brand's bus class to set the ID, reset the registers that matter,
and park the shaft at mid-encoder so the horn can be fitted at mid-travel.

Needs no project and no Hydra config: the servo is not a module component yet.
"""

from solarian.cli import BaseCommand, CommandRegistry
from solarian.core.actuator_bus_registry import available_bus_names, get_bus_class


def _yes(prompt: str, default: bool = True) -> bool:
    raw = input(f"{prompt} {'[Y/n]' if default else '[y/N]'} ").strip().lower()
    if not raw:
        return default
    return raw in ("y", "yes")


def _servo_label(servo: dict) -> str:
    label = f"id={servo['id']}  model={servo['model']}"
    if servo.get("baudrate"):
        label += f"  baudrate={servo['baudrate']}"
    return label


def select(kind: str, options: list, label=str):
    """Prompt for one of *options*, auto-selecting when there is only one.

    Returns the chosen element, or ``None`` if the user declined / there is
    nothing to choose from.
    """
    if not options:
        print(f"No {kind} found.")
        return None
    if len(options) == 1:
        print(f"Using the only {kind} found: {label(options[0])}")
        return options[0]

    print(f"\n=== Select {kind} ===")
    for idx, option in enumerate(options, 1):
        print(f"{idx}. {label(option)}")
    raw = input(f"{kind.capitalize()} number (1-{len(options)}), or Enter to stop: ").strip()
    if raw.isdigit() and 1 <= int(raw) <= len(options):
        return options[int(raw) - 1]
    if raw:
        print("Invalid selection.")
    return None


@CommandRegistry.register("calibrate-servo")
class CalibrateServoCommand(BaseCommand):
    help = "Prepare a servo for use: set its ID, reset registers, and centre its range."
    description = """
    Walks brand -> port -> servo, then sets the servo's ID, resets the registers
    that matter and parks the shaft at mid-encoder so the horn can be fitted at
    mid-travel. Loops so a whole arm can be numbered in one sitting.

    Run it before the servo is part of a module: it needs no project and no yaml.
    Connect one new servo at a time, since fresh units share the factory ID and
    collide on a shared bus.
    """
    usage = ""
    examples = ("solarian calibrate-servo",)

    def handle(self, cfg) -> None:
        del cfg  # no project config: the servo is not a declared component yet

        brand = select("servo brand", available_bus_names())
        if brand is None:
            return
        bus_class = get_bus_class(brand)

        try:
            port = select("port", bus_class.available_ports())
        except NotImplementedError as exc:
            print(f"{exc}. Nothing to do for {brand}.")
            return
        if port is None:
            return

        while True:
            # Re-scanned every round: an ID or baud-rate change invalidates it.
            try:
                found = bus_class.scan_port(port)
            except NotImplementedError as exc:
                print(f"{exc}. Nothing to do for {brand}.")
                return

            if not found:
                print(
                    f"\nNo {brand} servos responded on {port}.\n"
                    "  - Check power and the data cable.\n"
                    "  - Brand-new servos on one chain all share the factory ID and\n"
                    "    collide; connect only one new servo at a time.\n"
                )
                if not _yes("Scan again?", default=True):
                    return
                continue

            servo = select("servo", found, label=_servo_label)
            if servo is None:
                return

            bus_class.provision_servo(port, servo)

            if not _yes("\nSet up another servo?", default=True):
                print("\nDone.")
                return
