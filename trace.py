import sys
import inspect
from pathlib import Path
from types import FrameType
import os
from datetime import datetime



PROJECT_DIR = Path(__file__).parent.resolve()


timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
log_file_path = PROJECT_DIR.parent / "logs" / "trace" / f"{timestamp}.log"
os.makedirs(log_file_path.parent, exist_ok=True)
print(log_file_path)
log_file = sys.stderr #open(log_file_path, "w", encoding="utf-8")

depth = 0

def is_project_file(filename:Path):
    return (
        filename.is_relative_to(PROJECT_DIR)
        and ".venv" not in filename.parts
    )

def get_function_name(frame:FrameType) -> str:
    module = frame.f_globals.get("__name__", "")
    function = frame.f_code.co_name

    # Mètode d'una instància
    self = frame.f_locals.get("self")

    if self is not None:
        cls = type(self).__qualname__
        return f"{module}.{cls}.{function}"

    # Funció normal
    return f"{module}.{function}"

def trace(frame:FrameType, event, arg):
    global depth, log_file

    if event not in ("call", "return"):
        return trace

    filename = Path(frame.f_code.co_filename).resolve()

    # Ignorar funcions que no pertanyen al nostre projecte
    if not is_project_file(filename):
        return trace

    if event == "call":
        args = inspect.getargvalues(frame)

        arguments = []

        for name in args.args:
            arguments.append(f"{name}={args.locals[name]!r}")

        if args.varargs:
            arguments.append(
                f"*{args.varargs}={args.locals[args.varargs]!r}"
            )

        if args.keywords:
            arguments.append(
                f"**{args.keywords}={args.locals[args.keywords]!r}"
            )

        log_file.write(
            "|   " * depth
            + f"{frame.f_code.co_name}({', '.join(arguments)})\n"
        )

        depth += 1

    elif event == "return":
        depth -= 1

        if arg is not None:
            log_file.write(
                "|   " * depth
                + f"return {arg!r}\n"
            )


    return trace




sys.settrace(trace)

# Executem el programa
from train import main
main()

sys.settrace(None)
log_file.close()