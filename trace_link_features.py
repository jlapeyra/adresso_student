import sys
import inspect
import ast
import os
import re

from pathlib import Path
from datetime import datetime
from functools import lru_cache
from types import FrameType

PROJECT_DIR = Path(__file__).parent.resolve()
TRACE_FILE = Path(__file__).resolve()

timestamp = datetime.now().strftime("%y%m%d-%H%M%S")
log_file_path = PROJECT_DIR / "logs" / "trace" / f"{timestamp}.log"
os.makedirs(log_file_path.parent, exist_ok=True)
print(log_file_path)
log_file = open(log_file_path, "w", encoding="utf-8")
flush_pending = False

def log(s :str):
    global flush_pending
    log_file.write(s)
    if not s.endswith('\n'):
        log_file.write('\n')
    flush_pending = True

depth = 0
stack = []
tracked_ids = set()
tracked_vars = {}
last_line = {}
pending_returns = {}
tracking_started = False


@lru_cache(maxsize=None)
def is_project_file(filename: str) -> bool:
    if filename.startswith("<"):
        return False
    try:
        path = Path(filename).resolve()
    except Exception:
        return False
    return (
        path.is_relative_to(PROJECT_DIR)
        and ".venv" not in path.parts
        and path != TRACE_FILE
    )


def is_valid_function(frame: FrameType) -> bool:
    return not frame.f_code.co_name.startswith("<")


def get_function_name(frame: FrameType) -> str:
    module = frame.f_globals.get("__name__", "")
    function = frame.f_code.co_name

    self = frame.f_locals.get("self")
    if self is not None:
        cls = type(self).__qualname__
        return f"{module}.{cls}.{function}"

    return f"{module}.{function}"


def my_repr(obj, truncate_length=80, truncate_mode="fixed"):
    if truncate_mode == "proportional":
        try:
            truncate_length *= max(1, min(10, len(obj)))
        except:
            pass
    try:
        ret = repr(obj)
        ret = re.sub(r"\s+", " ", ret)
        if len(ret) > truncate_length:
            ret = ret[:truncate_length] + "…"
        if type(obj).__name__ not in ret and type(obj) not in (int, float, str, dict, list, dict, set):
            ret = f'<{type(obj).__name__} {ret}>'
        return ret
    except Exception:
        return f"<unrepresentable object of type {type(obj).__name__}>"


def is_initial_function(frame: FrameType) -> bool:
    file, _ = os.path.splitext(os.path.basename(frame.f_code.co_filename))
    return (
        file == "adresso_dataset"
        and frame.f_code.co_name == "build_merged_df"
    )


def get_target_names(node):
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, (ast.Tuple, ast.List)):
        result = set()
        for element in node.elts:
            result |= get_target_names(element)
        return result
    if isinstance(node, ast.Starred):
        return get_target_names(node.value)
    return set()


def get_load_names(node):
    result = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            result.add(child.id)
    return result


@lru_cache(maxsize=None)
def analyse_source(filename: str):
    line_info = {}
    call_targets = {}

    try:
        source = Path(filename).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=filename)
    except Exception:
        return line_info, call_targets

    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            if isinstance(node, ast.Assign):
                value = node.value
                writes = set()
                for target in node.targets:
                    writes |= get_target_names(target)
                reads = get_load_names(value)

            elif isinstance(node, ast.AnnAssign):
                value = node.value
                writes = get_target_names(node.target)
                reads = get_load_names(value)

            else:
                value = node.value
                writes = get_target_names(node.target)
                reads = get_load_names(value)
                reads |= writes

            line_info[node.end_lineno] = (reads, writes)

            contains_call = any(
                isinstance(child, ast.Call)
                for child in ast.walk(value)
            )

            if contains_call:
                targets = set()

                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        targets |= get_target_names(target)
                elif isinstance(node, ast.AnnAssign):
                    targets |= get_target_names(node.target)
                elif isinstance(node, ast.AugAssign):
                    targets |= get_target_names(node.target)

                for lineno in range(node.lineno, node.end_lineno + 1):
                    call_targets.setdefault(lineno, set()).update(targets)

    return line_info, call_targets


def get_line_info(filename, lineno):
    line_info, _ = analyse_source(filename)
    return line_info.get(lineno)


def get_call_targets(filename, lineno):
    _, call_targets = analyse_source(filename)
    return call_targets.get(lineno, set())


def print_stack():
    #log('print_stack')
    for entry in stack:
        if entry["relevant"] and not entry["printed"]:
            log(
                "    " * entry["depth"]
                + entry["signature"]
                + "\n"
            )
            entry["printed"] = True


def mark_frame_relevant(frame_id):
    #log('mark_frame_relevant')
    for i, entry in enumerate(stack):
        if entry["frame_id"] == frame_id:
            for ancestor in stack[:i + 1]:
                ancestor["relevant"] = True
            print_stack()
            return

def resolve_pending_returns(frame):
    frame_id = id(frame)
    pending = pending_returns.get(frame_id)

    if not pending:
        return set()

    newly_tracked = set()
    remaining = []

    for object_id, target_names in pending:
        matched = False

        for name in target_names:
            if name not in frame.f_locals:
                continue

            value = frame.f_locals[name]

            if id(value) != object_id:
                continue

            tracked_ids.add(object_id)
            tracked_vars.setdefault(frame_id, set()).add(name)

            newly_tracked.add(name)
            matched = True

            #log(f'resolve_pending_returns: {newly_tracked=}')
            mark_frame_relevant(frame_id)

        if not matched:
            remaining.append((object_id, target_names))

    if remaining:
        pending_returns[frame_id] = remaining
    else:
        pending_returns.pop(frame_id, None)

    return newly_tracked

def register_return(frame, value):
    if value is None:
        return

    frame_id = id(frame)

    relevant = (
        frame_id in tracked_vars
        or is_initial_function(frame)
    )

    if not relevant:
        return

    object_id = id(value)
    tracked_ids.add(object_id)

    caller = frame.f_back

    if caller is None:
        return

    caller_id = id(caller)

    targets = get_call_targets(
        caller.f_code.co_filename,
        caller.f_lineno,
    )

    if not targets:
        return

    pending_returns.setdefault(caller_id, []).append(
        (object_id, set(targets))
    )

def get_source_line(filename, lineno):
    #log(f'{lineno=}, {filename=}')
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    if 0 <= lineno-1 < len(lines):
        line = lines[lineno-1]
        line = re.sub(r'\s+', ' ', line.strip())
        return line
    else:
        return '[source line not found]'
    # _, _, source_lines = analyse_source(filename)
    # return source_lines.get(lineno, "").strip()

def process_executed_line(
    frame,
    lineno,
    newly_tracked=None,
):
    #log('process_executed_line')
    frame_id = id(frame)

    if newly_tracked is None:
        newly_tracked = set()

    tracked = tracked_vars.get(frame_id)

    if not tracked:
        return

    info = get_line_info(
        frame.f_code.co_filename,
        lineno,
    )
    #log(f'{info=}')

    if info is None:
        return

    reads, writes = info
    #log(f'process_executed_line: {reads=}, {writes=}')

    dependencies = reads & tracked
    #log(f'process_executed_line: {dependencies=}')
    #log(f'process_executed_line: kk')

    # Variables introduced by a function return
    # are also valid tracking dependencies.
    dependencies |= newly_tracked & writes
    #log(f'process_executed_line: tt')
    #log(f'process_executed_line: {dependencies=}')

    if not dependencies:
        return

    mark_frame_relevant(frame_id)

    source_line = get_source_line(
        frame.f_code.co_filename,
        lineno,
    )

    #log(f'{source_line=}')

    for name in writes:
        #log(f'{name=}')
        if name not in frame.f_locals:
            continue

        obj = frame.f_locals[name]
        obj_id = id(obj)

        tracked_ids.add(obj_id)
        tracked.add(name)

        log(
            "    " * len(stack)
            + f"[TRACK] {source_line} -> "
            + f"{name} = {my_repr(obj)}\n"
        )


def get_function_signature(frame: FrameType) -> str:
    args = inspect.getargvalues(frame)
    arguments = []

    for name in args.args:
        if name in args.locals:
            arguments.append(
                f"{name}={my_repr(args.locals[name])}"
            )

    if args.varargs:
        arguments.append(
            f"*{args.varargs}="
            f"{my_repr(args.locals[args.varargs], truncate_mode='proportional')}"
        )

    if args.keywords:
        arguments.append(
            f"**{args.keywords}="
            f"{my_repr(args.locals[args.keywords], truncate_mode='proportional')}"
        )

    return (
        f"{get_function_name(frame)}"
        f"({', '.join(arguments)})"
    )


def trace(frame: FrameType, event, arg):
    global depth
    global tracking_started
    global flush_pending

    try:
        filename = frame.f_code.co_filename

        if not is_project_file(filename):
            return None

        if not is_valid_function(frame):
            return None

        frame_id = id(frame)

        if event == "call":
            initial = is_initial_function(frame)

            if not tracking_started:
                signature = get_function_signature(frame)

                stack.append({
                    "frame_id": frame_id,
                    "frame": frame,
                    "signature": signature,
                    "depth": depth,
                    "printed": False,
                    "relevant": False,
                })

                depth += 1
                return trace

            relevant_names = set()

            if initial:
                relevant_names.add("<build_merged_df>")
            else:
                for name, value in frame.f_locals.items():
                    if id(value) in tracked_ids:
                        relevant_names.add(name)

            if not relevant_names:
                return None

            if initial:
                tracked_vars.setdefault(frame_id, set())
            else:
                tracked_vars[frame_id] = (
                    relevant_names - {"<build_merged_df>"}
                )

            signature = get_function_signature(frame)

            stack.append({
                "frame_id": frame_id,
                "frame": frame,
                "signature": signature,
                "depth": depth,
                "printed": False,
                "relevant": False,
            })

            depth += 1

            if not initial:
                mark_frame_relevant(frame_id)

            return trace

        if event == "line":
            if not tracking_started:
                last_line[frame_id] = frame.f_lineno
                return trace

            newly_tracked = resolve_pending_returns(frame)

            previous_line = last_line.get(frame_id)

            #log(f'trace {newly_tracked=}, {previous_line=}')

            if previous_line is not None:
                process_executed_line(
                    frame,
                    previous_line,
                    newly_tracked,
                )

            last_line[frame_id] = frame.f_lineno

            return trace

        if event == "return":
            relevant = (
                frame_id in tracked_vars
                or is_initial_function(frame)
            )

            if relevant:
                if arg is not None:
                    if is_initial_function(frame):
                        tracking_started = True

                    register_return(frame, arg)

                if stack:
                    entry = stack[-1]

                    if (
                        entry["frame_id"] == frame_id
                        and entry["relevant"]
                        and arg is not None
                    ):
                        log(
                            "    " * depth
                            + f"RETURN {my_repr(arg)}\n"
                        )

            if stack and stack[-1]["frame_id"] == frame_id:
                stack.pop()
                depth -= 1

            tracked_vars.pop(frame_id, None)
            last_line.pop(frame_id, None)
            return trace

        return trace
    
    except Exception as e:
        log("--- Error processing trace:", e, '---')
    finally:
        if flush_pending:
            log_file.flush()
            flush_pending = False


sys.settrace(trace)

try:
    from train import main
    main()
finally:
    sys.settrace(None)
    log_file.close()