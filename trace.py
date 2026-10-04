import sys
import inspect
from pathlib import Path
from types import FrameType
import os
from datetime import datetime
import re


PROJECT_DIR = Path(__file__).parent.resolve()


timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
log_file_path = PROJECT_DIR / "logs" / "trace" / f"{timestamp}.log"
os.makedirs(log_file_path.parent, exist_ok=True)
print(log_file_path)
log_file = open(log_file_path, "w", encoding="utf-8")

depth = 0

stack = []

def is_project_file(filename:str):
    if filename.startswith("<"):
        return False
    filename = Path(filename).resolve()
    return (
        filename.is_relative_to(PROJECT_DIR)
        and ".venv" not in filename.parts
    )

def is_valid_function(frame:FrameType):
    return not (frame.f_code.co_name.startswith("<"))

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


def my_repr(obj):
    try:
        ret = repr(obj)
        ret = re.sub(r"\s+", r" ", ret)
        if len(ret) > 500:
            ret = ret[:500] + "…"
        return ret
    except Exception:
        return f"<unrepresentable object of type {type(obj).__name__}>"

def trace(frame:FrameType, event, arg):
    global depth, log_file, stack

    if event not in ("call", "return"):
        return trace

    filename = frame.f_code.co_filename

    # Ignorar funcions que no pertanyen al nostre projecte
    if not is_project_file(filename):
        return trace

    if event == "call":
        if not is_valid_function(frame):
            return trace
        args = inspect.getargvalues(frame)

        arguments = []

        for name in args.args:
            arguments.append(f"{name}={my_repr(args.locals[name])}")

        if args.varargs:
            arguments.append(
                f"*{args.varargs}={my_repr(args.locals[args.varargs])}"
            )

        if args.keywords:
            arguments.append(
                f"**{args.keywords}={my_repr(args.locals[args.keywords])}"
            )

        signature = f"{get_function_name(frame)}({', '.join(arguments)})"
        log_file.write(
            "    " * depth
            + signature
        )

        stack.append(signature)

        depth += 1
        log_file.flush()

    elif event == "return":
        if not is_valid_function(frame):
            return trace
        depth -= 1
        if arg is not None:
            log_file.write(
                "    " * depth
                + f"`-> {my_repr(arg)}\n"
            )
        log_file.flush()


    return trace





# Executem el programa
sys.settrace(trace)
from train import main
main()
sys.settrace(None)
log_file.close()