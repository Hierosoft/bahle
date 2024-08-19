import os
import sly
import sys

from bahle.basicinterpreter import BasicInterpreter


def print_title():
    print("Cman's simple BASIC v0.1")
    print('2019 Cheaterman')

    try:
        import psutil
        print(f'\n{psutil.virtual_memory().available / 2**30:.3f} GB free')

    except ImportError:
        pass


def run_file(path):
    print_title()

    interpreter = BasicInterpreter()
    parent = os.path.dirname(path)
    os.chdir(parent)
    lines = []
    with open(path, 'r') as stream:
        for raw in stream:
            lines.append(raw.rstrip("\r\n"))
    line_index = -1
    for line in lines:
        line_index += 1
        if not line.strip():
            # Avoid "SyntaxError: Unexpected EOF"
            continue
        try:
            interpreter.interpret(line)
        except SyntaxError as exception:
            print('"{}" line {}: {}: {}'
                  .format(path, line_index+1,
                          type(exception).__name__, exception))
        except sly.lex.LexError as exception:
            print('"{}" line {}: {}: {}'
                  .format(path, line_index+1,
                          type(exception).__name__, exception))
        except KeyboardInterrupt:
            if interpreter.running_program:
                print(f'Break in {interpreter.last_program_lineno}')

    print('Bye!')
    return 0


def main_interactive():
    print_title()

    interpreter = BasicInterpreter()

    while True:
        line = ''

        try:
            print('\nReady')

            while not line:
                line = input()

        except KeyboardInterrupt:
            print()
            break

        except EOFError:
            break

        try:
            interpreter.interpret(line)

        except SyntaxError as exception:
            print(type(exception).__name__ + ':', exception)

        except KeyboardInterrupt:
            if interpreter.running_program:
                print(f'Break in {interpreter.last_program_lineno}')

    print('Bye!')
    return 0


def main():
    path = None
    if len(sys.argv) > 1:
        path = sys.argv[1]
    if path:
        return run_file(path)
    return main_interactive()


if __name__ == '__main__':
    sys.exit(main())
